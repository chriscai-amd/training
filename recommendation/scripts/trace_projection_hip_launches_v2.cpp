// Diagnostic LD_PRELOAD wrapper, version 2. Copies names while API arguments
// are live, and joins returned function handles to later module launches.
// Signatures checked against hip_runtime_api.h and hip_ext.h; opaque handles
// remain pointers and hipError_t remains the ABI's integer return value.
// No GPU synchronization, numerical operation, or argument modification.
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <time.h>
#include <sys/uio.h>
#include <unistd.h>

namespace {
using Handle = void*;
std::mutex lock;
std::map<Handle, std::string> names;
uint64_t next_launch = 0;
uint64_t next_copy = 0;
int log_fd = -2;
constexpr size_t capture_limit = 4096;

std::string quote(const std::string& s) {
  std::string out = "\"";
  const char* hex = "0123456789abcdef";
  for (unsigned char c : s) {
    if (c >= 32 && c < 127 && c != '"' && c != '\\') out += c;
    else {
      out += "\\u00";
      out += hex[c >> 4]; out += hex[c & 15];
    }
  }
  return out + "\"";
}
std::string pointer(Handle p) {
  std::ostringstream out; out << p; return quote(out.str());
}
bool tracing_enabled() { return std::getenv("PROJECTION_HIP_LAUNCH_TRACE") != nullptr; }

// Only read our own host address space through a fault-reporting syscall. This
// adds no HIP call, GPU readback, pointer-attribute query, or synchronization.
// Each caller provides a protocol-defined object size bounded by 4096 bytes.
bool read_host(void* destination, const void* source, size_t bytes) {
  if (bytes == 0) return true;
  if (!source || bytes > capture_limit) return false;
  iovec local{destination, bytes};
  iovec remote{const_cast<void*>(source), bytes};
  ssize_t result;
  do { result = process_vm_readv(getpid(), &local, 1, &remote, 1, 0); }
  while (result < 0 && errno == EINTR);
  return result == static_cast<ssize_t>(bytes);
}
std::string hex_bytes(const unsigned char* data, size_t size) {
  static const char* digits = "0123456789abcdef";
  std::string out; out.reserve(size * 2);
  for (size_t i = 0; i < size; ++i) {
    out += digits[data[i] >> 4]; out += digits[data[i] & 15];
  }
  return out;
}
std::string capture_extra(void** params, void** extra) {
  auto skip = [](const char* reason) {
    return "{\"status\":\"skipped\",\"reason\":" + quote(reason) + "}";
  };
  if (!tracing_enabled()) return skip("tracing_disabled");
  if (params) return skip("kernel_params_sizes_unknown");
  if (!extra) return skip("no_extra");
  // HIP runtime header defines POINTER=1, SIZE=2, END=3. Accept exactly
  // POINTER/value + SIZE/value + END, in either pair order, with no duplicates.
  // An unknown tag is never followed, and no pointee is read until the full
  // known format and both bounds have been validated.
  Handle buffer = nullptr, size_pointer = nullptr;
  bool have_buffer = false, have_size = false, ended = false;
  for (size_t index = 0; index <= 4;) {
    Handle tag = nullptr;
    if (!read_host(&tag, extra + index, sizeof(tag))) return skip("extra_tag_unreadable");
    uintptr_t key = reinterpret_cast<uintptr_t>(tag);
    if (key == 3) { ended = true; break; }
    if (key != 1 && key != 2) return skip("unknown_extra_tag");
    if (index == 4) return skip("extra_format_too_long");
    if ((key == 1 && have_buffer) || (key == 2 && have_size)) return skip("duplicate_extra_tag");
    Handle value = nullptr;
    if (!read_host(&value, extra + index + 1, sizeof(value))) return skip("extra_value_unreadable");
    if (key == 1) { buffer = value; have_buffer = true; }
    else { size_pointer = value; have_size = true; }
    index += 2;
  }
  if (!ended || !have_buffer || !have_size) return skip("incomplete_extra_format");
  size_t bytes = 0;
  if (!read_host(&bytes, size_pointer, sizeof(bytes))) return skip("buffer_size_unreadable");
  std::string prefix = "\"buffer_pointer\":" + pointer(buffer) +
      ",\"buffer_size\":" + std::to_string(bytes);
  if (bytes > capture_limit) return "{\"status\":\"skipped\",\"reason\":\"buffer_exceeds_4096\"," + prefix + "}";
  unsigned char copied[capture_limit];
  if (!read_host(copied, buffer, bytes)) return "{\"status\":\"skipped\",\"reason\":\"buffer_unreadable\"," + prefix + "}";
  return "{\"status\":\"captured\"," + prefix + ",\"buffer_hex\":" + quote(hex_bytes(copied, bytes)) + "}";
}
uint64_t timestamp() {
  timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
void emit(const std::string& record) {  // Called with lock held.
  if (log_fd == -2) {
    const char* path = std::getenv("PROJECTION_HIP_LAUNCH_TRACE");
    log_fd = path ? open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600) : -1;
    if (path && log_fd < 0) { std::perror("projection trace open"); std::abort(); }
  }
  if (log_fd < 0) return;
  std::string line = "{\"pid\":" + std::to_string(getpid()) +
      ",\"monotonic_ns\":" + std::to_string(timestamp()) + "," + record + "}\n";
  size_t offset = 0;
  while (offset < line.size()) {
    ssize_t n = write(log_fd, line.data() + offset, line.size() - offset);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) { std::perror("projection trace write"); std::abort(); }
    offset += size_t(n);
  }
}
template<class T> T original(const char* name) {
  auto result = reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
  if (!result) { std::fprintf(stderr, "Missing next HIP symbol: %s\n", name); std::abort(); }
  return result;
}
uint64_t begin(const char* api, Handle f, uint32_t gx, uint32_t gy, uint32_t gz,
               uint32_t bx, uint32_t by, uint32_t bz, size_t shared, Handle stream,
               void** params, void** extra) {
  const std::string arguments = capture_extra(params, extra);
  std::lock_guard<std::mutex> guard(lock);
  uint64_t id = ++next_launch;
  const auto found = names.find(f);
  emit("\"event\":\"launch_begin\",\"api\":" + quote(api) +
       ",\"launch_id\":" + std::to_string(id) + ",\"function\":" + pointer(f) +
       ",\"name\":" + quote(found == names.end() ? "" : found->second) +
       ",\"grid_or_global\":[" + std::to_string(gx) + "," + std::to_string(gy) + "," + std::to_string(gz) +
       "],\"block_or_local\":[" + std::to_string(bx) + "," + std::to_string(by) + "," + std::to_string(bz) +
       "],\"shared_bytes\":" + std::to_string(shared) + ",\"stream\":" + pointer(stream) +
       ",\"kernel_params_pointer\":" + pointer(params) + ",\"extra_pointer\":" + pointer(extra) +
       ",\"argument_capture\":" + arguments);
  return id;
}
void end(uint64_t id, int result) {
  std::lock_guard<std::mutex> guard(lock);
  emit("\"event\":\"launch_return\",\"launch_id\":" + std::to_string(id) +
       ",\"hip_error\":" + std::to_string(result));
}
}

extern "C" int hipModuleGetFunction(Handle* f, Handle module, const char* name) {
  using Fn = int (*)(Handle*, Handle, const char*);
  static Fn call = original<Fn>("hipModuleGetFunction");
  // Copy BEFORE the call, not at profiler finalization or after caller cleanup.
  const std::string copied = name ? name : "";
  int result = call(f, module, name);
  std::lock_guard<std::mutex> guard(lock);
  if (result == 0 && f) names[*f] = copied;
  emit("\"event\":\"get_function_return\",\"module\":" + pointer(module) +
       ",\"function\":" + pointer(result == 0 && f ? *f : nullptr) +
       ",\"name\":" + quote(copied) + ",\"hip_error\":" + std::to_string(result));
  return result;
}

extern "C" int hipModuleLaunchKernel(Handle f, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned shared, Handle stream,
    void** params, void** extra) {
  using Fn = int (*)(Handle, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                    unsigned, Handle, void**, void**);
  static Fn call = original<Fn>("hipModuleLaunchKernel");
  const int before_errno = errno;
  uint64_t id = begin("hipModuleLaunchKernel", f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
  errno = before_errno;
  int result = call(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
  const int result_errno = errno;
  end(id, result); errno = result_errno; return result;
}

extern "C" int hipExtModuleLaunchKernel(Handle f, uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz, size_t shared, Handle stream,
    void** params, void** extra, Handle start, Handle stop, uint32_t flags) {
  using Fn = int (*)(Handle, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                    size_t, Handle, void**, void**, Handle, Handle, uint32_t);
  static Fn call = original<Fn>("hipExtModuleLaunchKernel");
  const int before_errno = errno;
  uint64_t id = begin("hipExtModuleLaunchKernel", f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
  errno = before_errno;
  int result = call(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra, start, stop, flags);
  const int result_errno = errno;
  end(id, result); errno = result_errno; return result;
}

extern "C" int hipMemcpyAsync(void* destination, const void* source, size_t bytes,
    int kind, Handle stream) {
  using Fn = int (*)(void*, const void*, size_t, int, Handle);
  static Fn call = original<Fn>("hipMemcpyAsync");
  // hipMemcpyHostToDevice is exactly 1 in installed HIP headers. Default,
  // D2H, D2D and H2H directions are never dereferenced or captured here.
  if (!tracing_enabled() || kind != 1 || bytes > capture_limit)
    return call(destination, source, bytes, kind, stream);
  const int before_errno = errno;
  unsigned char copied[capture_limit];
  bool captured = read_host(copied, source, bytes);
  uint64_t id;
  {
    std::lock_guard<std::mutex> guard(lock);
    id = ++next_copy;
    emit("\"event\":\"h2d_copy_begin\",\"api\":\"hipMemcpyAsync\",\"copy_id\":" +
         std::to_string(id) + ",\"destination\":" + pointer(destination) +
         ",\"source\":" + pointer(const_cast<void*>(source)) +
         ",\"size_bytes\":" + std::to_string(bytes) + ",\"kind\":1,\"stream\":" + pointer(stream) +
         ",\"source_capture\":{\"status\":" + quote(captured ? "captured" : "unreadable") +
         (captured ? ",\"source_hex\":" + quote(hex_bytes(copied, bytes)) : "") + "}");
  }
  errno = before_errno;
  int result = call(destination, source, bytes, kind, stream);
  const int result_errno = errno;
  {
    std::lock_guard<std::mutex> guard(lock);
    emit("\"event\":\"h2d_copy_return\",\"copy_id\":" + std::to_string(id) +
         ",\"hip_error\":" + std::to_string(result));
  }
  errno = result_errno;
  return result;
}
