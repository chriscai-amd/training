// Diagnostic LD_PRELOAD wrapper. Copies names while the real API's arguments
// are live, and joins returned function handles to later module launches.
// Signatures checked against hip_runtime_api.h and hip_ext.h; opaque handles
// remain pointers and hipError_t remains the ABI's integer return value.
// No GPU synchronization, numerical operation, or argument modification.
#include <cstdint>
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
#include <unistd.h>

namespace {
using Handle = void*;
std::mutex lock;
std::map<Handle, std::string> names;
uint64_t next_launch = 0;
int log_fd = -2;

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
               uint32_t bx, uint32_t by, uint32_t bz, size_t shared, Handle stream) {
  std::lock_guard<std::mutex> guard(lock);
  uint64_t id = ++next_launch;
  const auto found = names.find(f);
  emit("\"event\":\"launch_begin\",\"api\":" + quote(api) +
       ",\"launch_id\":" + std::to_string(id) + ",\"function\":" + pointer(f) +
       ",\"name\":" + quote(found == names.end() ? "" : found->second) +
       ",\"grid_or_global\":[" + std::to_string(gx) + "," + std::to_string(gy) + "," + std::to_string(gz) +
       "],\"block_or_local\":[" + std::to_string(bx) + "," + std::to_string(by) + "," + std::to_string(bz) +
       "],\"shared_bytes\":" + std::to_string(shared) + ",\"stream\":" + pointer(stream));
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
  uint64_t id = begin("hipModuleLaunchKernel", f, gx, gy, gz, bx, by, bz, shared, stream);
  int result = call(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
  end(id, result); return result;
}

extern "C" int hipExtModuleLaunchKernel(Handle f, uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz, size_t shared, Handle stream,
    void** params, void** extra, Handle start, Handle stop, uint32_t flags) {
  using Fn = int (*)(Handle, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                    size_t, Handle, void**, void**, Handle, Handle, uint32_t);
  static Fn call = original<Fn>("hipExtModuleLaunchKernel");
  uint64_t id = begin("hipExtModuleLaunchKernel", f, gx, gy, gz, bx, by, bz, shared, stream);
  int result = call(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra, start, stop, flags);
  end(id, result); return result;
}
