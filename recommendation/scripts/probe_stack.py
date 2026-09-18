import time, torch, triton, triton.language as tl

@triton.jit
def _add(X, Y, O, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    tl.store(O + i, tl.load(X + i, mask=m) + tl.load(Y + i, mask=m), mask=m)

def main():
    print("device     :", torch.cuda.get_device_name(0))
    print("torch      :", torch.__version__, "| hip", torch.version.hip)
    print("triton     :", triton.__version__, "|", triton.__file__)
    a = torch.randn(4096, 4096, device="cuda"); b = torch.randn(4096, 4096, device="cuda")
    t = time.time(); c = a @ b; torch.cuda.synchronize()
    print("matmul     : finite=%s  %.2fs" % (bool(torch.isfinite(c).all()), time.time() - t))
    N = 1 << 20
    x = torch.randn(N, device="cuda"); y = torch.randn(N, device="cuda"); o = torch.empty_like(x)
    t = time.time(); _add[(N // 1024,)](x, y, o, N, BLOCK=1024); torch.cuda.synchronize()
    print("triton JIT : max_err=%.3g  compile+run=%.1fs" % ((o - (x + y)).abs().max().item(), time.time() - t))

if __name__ == "__main__":
    main()
