"""Where HRX / Loom and the ROCm runtime live, for the generator tools (engine/hrx-env.sh sets the same for shells).

YAH_HRX         the HRX checkout: the pinned revision plus engine/hrx/patches (engine/hrx/bootstrap.sh); default external/hrx
YAH_HRX_BUILD   its cmake build tree; default $YAH_HRX/build/cmake
YAH_ROCM        the extracted ROCm Core SDK 10.0.0 runtime packages (engine/hrx-env.sh --fetch); default external/rocm10
YAH_LOOM_HOME   a different HRX / Loom tree for diagnosis only (e.g. an experimental compiler); overrides YAH_HRX
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
HRX = os.environ.get("YAH_LOOM_HOME") or os.environ.get("YAH_HRX") or os.path.join(ROOT, "external", "hrx")
BUILD = os.environ.get("YAH_HRX_BUILD") or os.path.join(HRX, "build", "cmake")
ROCM = os.environ.get("YAH_ROCM") or os.path.join(ROOT, "external", "rocm10")
LIBHSA = os.path.join(ROCM, "x_runtime/opt/rocm/core-10.0/lib")

IREE_RUN_LOOM = os.path.join(BUILD, "loom/src/loom/tools/iree-run-loom/iree-run-loom")
LOOM_COMPILE = os.path.join(BUILD, "loom/src/loom/tools/loom-compile/loom-compile")
XDNA_RUN = os.path.join(BUILD, "experimental/xdna/iree-xdna-run")


def env():
    """os.environ plus the HSA runtime HRX loads and the libraries its tools link."""
    e = dict(os.environ)
    e["IREE_HAL_AMDGPU_LIBHSA_PATH"] = LIBHSA
    libs = [os.path.join(BUILD, "libhrx/src/binding/hip"), os.path.join(BUILD, "libhrx/src/libhrx"), LIBHSA,
            os.path.join(ROCM, "x_llvm/opt/rocm/core-10.0/lib/llvm/lib"),
            os.path.join(ROCM, "x_sysdeps/opt/rocm/core-10.0/lib/rocm_sysdeps/lib"), "/opt/rocm/lib"]
    if e.get("LD_LIBRARY_PATH"):
        libs.append(e["LD_LIBRARY_PATH"])
    e["LD_LIBRARY_PATH"] = ":".join(libs)
    return e
