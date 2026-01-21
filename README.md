# GPGPU (OpenGL) vs OpenCL

A benchmark comparing **OpenGL Compute Shaders** against **OpenCL** on the same hardware.

## The Benchmark
* **Task:** $1024 \times 1024$ Matrix Multiplication ($C = A \times B$).
* **Hardware:** Sony Vaio SVF15N2C5E (i7-4500U, Intel HD 4400, NVIDIA GT 735M).

## Results

### Performance Comparison

| Device | API | Time | Speedup vs CPU |
|:-------|:----|-----:|---------------:|
| CPU (i7-4500U) | OpenCL (POCL) | ~5300 ms | 1× |
| Intel HD 4400 | OpenGL (Mesa) | ~227 ms | 23× |
| Intel HD 4400 | OpenCL (Beignet) | ~307 ms | 17× |
| NVIDIA GT 735M | OpenGL | ~149 ms | 36× |
| NVIDIA GT 735M | OpenCL | ~157 ms | 34× |

### Key Findings

1. **OpenGL ≈ OpenCL on NVIDIA:** When running both APIs on the GT 735M, performance is nearly identical (~5% difference), confirming that OpenGL Compute Shaders incur no significant penalty compared to OpenCL.

2. **Mesa outperforms Beignet on Intel:** OpenGL via Mesa (~227 ms) is 35% faster than OpenCL via Beignet (~307 ms) on the HD 4400. Beignet is a legacy unmaintained driver with known performance limitations.

3. **GPU vs CPU:** Both GPUs provide 17-36× speedup over the CPU for this compute-bound workload.

### Notes

* The benchmark dynamically adjusts OpenCL work group sizes based on device limits (Beignet's HD 4400 is limited to 512 work items vs 1024 on NVIDIA).
* Rusticl (Mesa's modern OpenCL) requires the `iris` driver (Gen8+/Broadwell), so it's unavailable for Haswell.
* The legacy Intel OpenCL runtime (NEO) doesn't support Haswell on modern Linux.

## Prerequisites

* **OpenCL Runtime:**
    * `nvidia-driver-470` (for NVIDIA)
    * `beignet-opencl-icd` (for Intel Haswell GPU)
    * `pocl-opencl-icd` (for CPU fallback)
    * `ocl-icd-opencl-dev` (Headers/Loader)

## Build & Run

```bash
# Install dependencies
sudo apt install ocl-icd-opencl-dev opencl-headers clinfo pocl-opencl-icd

# Build
cd examples/002_OpenCL
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg.cmake
cmake --build build --config Release
```

**Run on integrated Intel GPU (default):**

```bash
./build/gpgpu_vs_cl
```

**Run on NVIDIA (via PRIME offload):**

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./build/gpgpu_vs_cl
```
