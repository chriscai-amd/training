// Opt-in, exact-symbol native ELF control for direct-DX experiments.
// This composes the frozen v2 tracer; its launch and argument-copy code is
// unchanged. No on-disk HIP library is replaced. Both controlled arms use the
// same loader, host-side context checks, and tracing path.
#define hipModuleGetFunction projection_v2_get_function
#define hipModuleLaunchKernel projection_v2_launch_kernel
#define hipExtModuleLaunchKernel projection_v2_ext_launch_kernel
#define hipMemcpyAsync projection_v2_memcpy_async
#include "trace_projection_hip_launches_v2.cpp"
#undef hipModuleGetFunction
#undef hipModuleLaunchKernel
#undef hipExtModuleLaunchKernel
#undef hipMemcpyAsync

#include <openssl/evp.h>
#include <sys/stat.h>
#include <vector>

namespace {
constexpr char target_symbol[] =
    "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x240x128_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_AGNTAB0_AFEM1_AFEM1_ASEM1_BL1_BS1_CD1_1_CLR0_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA0_DTVB0_DTVMXSA0_DTVMXSB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA1_GRVWB1_GSU0_GSUAMB_GLS0_HPLR1_ISA1250_ICIW1_IU1_K1_LDSSI0_LDSTI0_LBSPPA512_LBSPPB256_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA8_LPB8_LPMXSA0_LPMXSB0_LPM0_LRVWn1_LWPMn1_MIAV1_MIWT2_15_MXLITDM_MXSFIMS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB1_ONLL1_PAP0_PGL0_PGR2_PLR1_PKA1_SGROB0_SIA4_SS0_SPO0_SRVW0_SSO0_SVW8_SK3_SKFTR0_SKFDPO0_SKWS0_SKXCCM0_SNLL1_SIP1_SGRO0_TDMI3_TDMIM0_TDMLWS0_TDMPLB0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS1_TLDSMn1_ULSGRO0_USL1_USLMX0_UDFMAC0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA2_VWB1_WSGRA0_WSGRB0_WS32_WG64_2_1_WGMXCC1";
constexpr char original_sha[] = "b323216507600606e076ab3546e81a931809008627c77d7f0d5a85ffa6cf1542";
constexpr char reuse_off_sha[] = "7d3fe969381c072e1c28d18491ab1f5ea1de51b186c73cbad3cda32c406db69d";
constexpr size_t elf_bytes = 23585704;
constexpr int fatal_exit = 125;
const pid_t library_pid = getpid();
std::mutex override_mutex;
thread_local bool in_replacement = false;
bool configured = false, enabled = false, loaded = false;
std::string mode, elf_path, trace_path, required_sha;
Handle original_module = nullptr, original_function = nullptr;
Handle replacement_module = nullptr, replacement_function = nullptr, owner_context = nullptr;
int owner_device = -1;
uint64_t retrieval_id = 0;
std::vector<unsigned char> image_bytes;

std::string environment(const char* name) {
  const char* value = std::getenv(name);
  return value ? value : "";
}

void override_emit(const std::string& message) {
  std::lock_guard<std::mutex> guard(lock);
  emit(message);
}

[[noreturn]] void fail(const std::string& reason) {
  override_emit("\"event\":\"native_override_fatal\",\"reason\":" + quote(reason) +
                ",\"mode\":" + quote(mode) + ",\"target\":" + quote(target_symbol));
  std::fprintf(stderr, "Native projection override failed closed: %s\n", reason.c_str());
  _exit(fatal_exit);
}

void process_guard() {
  // Check before either mutex: locks inherited across fork may be owned by a
  // vanished thread. No inherited replacement handle is used in a child.
  if (getpid() != library_pid) {
    static const char message[] = "Native projection override forbids inherited process state\n";
    const ssize_t ignored = write(STDERR_FILENO, message, sizeof(message) - 1);
    (void)ignored;
    _exit(fatal_exit);
  }
  if (in_replacement) fail("reentrant_interposed_HIP_call_during_replacement");
}

void configuration_guard() {  // override_mutex held.
  const auto current_mode = environment("PROJECTION_NATIVE_OVERRIDE");
  const auto current_path = environment("PROJECTION_NATIVE_OVERRIDE_ELF");
  const auto current_trace = environment("PROJECTION_HIP_LAUNCH_TRACE");
  if (!configured) {
    configured = true;
    mode = current_mode; elf_path = current_path; trace_path = current_trace;
    enabled = !mode.empty();
    if (!enabled && !elf_path.empty()) fail("ELF_path_without_opt_in_mode");
    if (enabled) {
      if (mode != "original" && mode != "a_reuse_off") fail("unknown_override_mode");
      if (elf_path.empty() || elf_path[0] != '/' || trace_path.empty())
        fail("absolute_ELF_path_and_trace_path_required");
      required_sha = mode == "original" ? original_sha : reuse_off_sha;
    }
  }
  if (mode != current_mode || elf_path != current_path || trace_path != current_trace)
    fail("override_configuration_changed_after_first_API_call");
  if (enabled) {
    for (const char* name : {"AMDGCN_USE_BUFFER_OPS", "TRITON_FULL_AUTOTUNE", "TRITON_ALLOW_PIPELINING"})
      if (environment(name) != "0") fail(std::string(name) + "_must_equal_zero");
  }
}

std::string sha256(const std::vector<unsigned char>& bytes) {
  unsigned char digest[EVP_MAX_MD_SIZE]; unsigned size = 0;
  if (EVP_Digest(bytes.data(), bytes.size(), digest, &size, EVP_sha256(), nullptr) != 1 || size != 32)
    fail("SHA256_computation_failed");
  return hex_bytes(digest, size);
}

void read_verified_image() {  // override_mutex held, before module load.
  int fd = open(elf_path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) fail("ELF_open_failed");
  struct stat info{};
  if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size != static_cast<off_t>(elf_bytes)) {
    close(fd); fail("ELF_requires_exact_regular_file_size");
  }
  image_bytes.resize(elf_bytes);
  size_t offset = 0;
  while (offset < image_bytes.size()) {
    ssize_t count = read(fd, image_bytes.data() + offset, image_bytes.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) { close(fd); fail("ELF_read_failed_or_truncated"); }
    offset += static_cast<size_t>(count);
  }
  unsigned char extra;
  ssize_t tail;
  do { tail = read(fd, &extra, 1); } while (tail < 0 && errno == EINTR);
  close(fd);
  if (tail != 0) fail("ELF_size_changed_during_read");
  const auto actual_sha = sha256(image_bytes);
  override_emit("\"event\":\"native_override_image_verified\",\"mode\":" + quote(mode) +
                ",\"path\":" + quote(elf_path) + ",\"bytes\":" + std::to_string(image_bytes.size()) +
                ",\"sha256\":" + quote(actual_sha) + ",\"required_sha256\":" + quote(required_sha) +
                ",\"matches\":" + (actual_sha == required_sha ? "true" : "false"));
  if (actual_sha != required_sha) fail("ELF_SHA256_mismatch");
}

std::pair<Handle, int> current_context() {
  using GetContext = int (*)(Handle*);
  using GetDevice = int (*)(int*);
  static GetContext get_context = original<GetContext>("hipCtxGetCurrent");
  static GetDevice get_device = original<GetDevice>("hipGetDevice");
  Handle context = nullptr; int device = -1;
  const int context_result = get_context(&context);
  const int device_result = get_device(&device);
  if (context_result != 0 || device_result != 0 || !context || device < 0)
    fail("current_HIP_context_or_device_unavailable");
  return {context, device};
}

void context_guard() {
  const auto current = current_context();
  if (current.first != owner_context || current.second != owner_device)
    fail("replacement_context_or_device_changed");
}

void launch_guard(Handle function) {
  process_guard();
  std::lock_guard<std::mutex> guard(override_mutex);
  configuration_guard();
  if (!enabled || !loaded) return;
  if (function == original_function) fail("original_target_handle_bypassed_replacement");
  if (function == replacement_function) context_guard();
}
}

extern "C" int hipModuleGetFunction(Handle* function, Handle module, const char* name) {
  process_guard();
  const std::string copied_name = name ? name : "";
  bool replace;
  {
    std::lock_guard<std::mutex> guard(override_mutex);
    configuration_guard();
    replace = enabled && copied_name == target_symbol;
  }
  // Exactly one ordinary lookup for every caller invocation, including the
  // target. The frozen tracer logs its result and original function handle.
  int result = projection_v2_get_function(function, module, name);
  const int original_errno = errno;
  if (!replace) return result;
  std::lock_guard<std::mutex> guard(override_mutex);
  configuration_guard();
  const uint64_t id = ++retrieval_id;
  // hipBLASLt may search several modules, receiving HIP_ERROR_NOT_FOUND=500
  // before finding this symbol. Preserve unsuccessful searches exactly: they
  // return no validated target function and must not trigger module loading.
  if (result != 0) {
    override_emit("\"event\":\"native_override_original_lookup_error\",\"retrieval_id\":" + std::to_string(id) +
                  ",\"original_module\":" + pointer(module) + ",\"target\":" + quote(copied_name) +
                  ",\"hip_error\":" + std::to_string(result) + ",\"forwarded_as_original_error\":true");
    errno = original_errno;
    return result;
  }
  if (!function || !*function) fail("original_target_success_with_null_handle");
  const Handle observed_original = *function;
  if (loaded) {
    if (module != original_module || observed_original != original_function)
      fail("original_target_module_or_handle_changed");
    context_guard();
  } else {
    read_verified_image();
    const auto current = current_context();
    owner_context = current.first; owner_device = current.second;
    original_module = module; original_function = observed_original;
    using Load = int (*)(Handle*, const void*);
    using Get = int (*)(Handle*, Handle, const char*);
    static Load load_image = original<Load>("hipModuleLoadData");
    static Get get_function = original<Get>("hipModuleGetFunction");
    in_replacement = true;
    int load_result = load_image(&replacement_module, image_bytes.data());
    in_replacement = false;
    override_emit("\"event\":\"native_override_module_load_return\",\"hip_error\":" + std::to_string(load_result) +
                  ",\"module\":" + pointer(replacement_module) + ",\"image_pointer\":" + pointer(image_bytes.data()) +
                  ",\"sha256\":" + quote(required_sha) + ",\"bytes\":" + std::to_string(image_bytes.size()) +
                  ",\"context\":" + pointer(owner_context) + ",\"device\":" + std::to_string(owner_device));
    if (load_result != 0 || !replacement_module || replacement_module == module)
      fail("replacement_module_load_failed_or_aliased");
    if (sha256(image_bytes) != required_sha) fail("module_load_mutated_verified_host_ELF_bytes");
    context_guard();
    in_replacement = true;
    int get_result = get_function(&replacement_function, replacement_module, target_symbol);
    in_replacement = false;
    override_emit("\"event\":\"native_override_GetFunction_return\",\"hip_error\":" + std::to_string(get_result) +
                  ",\"module\":" + pointer(replacement_module) + ",\"function\":" + pointer(replacement_function) +
                  ",\"target\":" + quote(target_symbol) + ",\"sha256\":" + quote(required_sha));
    if (get_result != 0 || !replacement_function || replacement_function == original_function)
      fail("replacement_GetFunction_failed_or_aliased");
    context_guard();
    loaded = true;
  }
  *function = replacement_function;
  {
    std::lock_guard<std::mutex> trace_guard(lock);
    names[replacement_function] = copied_name;
    emit("\"event\":\"native_override_function_return\",\"retrieval_id\":" + std::to_string(id) +
         ",\"original_module\":" + pointer(module) + ",\"original_function\":" + pointer(observed_original) +
         ",\"replacement_module\":" + pointer(replacement_module) + ",\"replacement_function\":" + pointer(replacement_function) +
         ",\"target\":" + quote(copied_name) + ",\"mode\":" + quote(mode) +
         ",\"sha256\":" + quote(required_sha) + ",\"context\":" + pointer(owner_context) +
         ",\"device\":" + std::to_string(owner_device) + ",\"hip_error\":0");
  }
  errno = original_errno;
  return result;
}

extern "C" int hipModuleLaunchKernel(Handle f, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned shared, Handle stream, void** params, void** extra) {
  const int saved_errno = errno;
  launch_guard(f); errno = saved_errno;
  return projection_v2_launch_kernel(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}

extern "C" int hipExtModuleLaunchKernel(Handle f, uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz, size_t shared, Handle stream, void** params, void** extra,
    Handle start, Handle stop, uint32_t flags) {
  const int saved_errno = errno;
  launch_guard(f); errno = saved_errno;
  return projection_v2_ext_launch_kernel(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra, start, stop, flags);
}

extern "C" int hipMemcpyAsync(void* destination, const void* source, size_t bytes, int kind, Handle stream) {
  process_guard();
  const int saved_errno = errno;
  {
    std::lock_guard<std::mutex> guard(override_mutex);
    configuration_guard();
  }
  errno = saved_errno;
  return projection_v2_memcpy_async(destination, source, bytes, kind, stream);
}

extern "C" int hipModuleUnload(Handle module) {
  process_guard();
  const int saved_errno = errno;
  {
    std::lock_guard<std::mutex> guard(override_mutex);
    configuration_guard();
    if (enabled && loaded && module == replacement_module) fail("replacement_module_unload_forbidden");
  }
  using Fn = int (*)(Handle);
  static Fn call = original<Fn>("hipModuleUnload");
  errno = saved_errno;
  return call(module);
}

extern "C" int hipDeviceReset() {
  process_guard();
  const int saved_errno = errno;
  {
    std::lock_guard<std::mutex> guard(override_mutex);
    configuration_guard();
    if (enabled && loaded) fail("device_reset_with_live_replacement_forbidden");
  }
  using Fn = int (*)();
  static Fn call = original<Fn>("hipDeviceReset");
  errno = saved_errno;
  return call();
}
