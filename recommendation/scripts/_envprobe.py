import sys, os, subprocess, json
out = {}
out["python"] = sys.version.split()[0]
try:
    import torch
    out["torch"] = torch.__version__
    out["torch.version.hip"] = getattr(torch.version, "hip", None)
    out["torch.git"] = getattr(torch.version, "git_version", None)[:12] if getattr(torch.version,"git_version",None) else None
    out["torch.file"] = torch.__file__
except Exception as e:
    out["torch"] = f"ERR {e}"
try:
    import triton
    out["triton"] = triton.__version__
    out["triton.file"] = triton.__file__
except Exception as e:
    out["triton"] = f"ERR {e}"
for f in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
    if os.path.exists(f):
        out[f] = open(f).read().strip()
for cmd, key in ((["hipcc","--version"],"hipcc"), (["/opt/rocm/llvm/bin/clang","--version"],"rocm-llvm")):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out[key] = (r.stdout or r.stderr).strip().splitlines()[0][:100]
    except Exception as e:
        out[key] = f"ERR {type(e).__name__}"
print("###JSON###")
print(json.dumps(out, indent=1, sort_keys=True))
