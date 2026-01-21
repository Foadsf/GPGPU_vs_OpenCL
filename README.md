# GPGPU (OpenGL) vs OpenCL

A benchmark comparing **OpenGL Compute Shaders** against **OpenCL** on the same hardware.

## The Benchmark
* **Task:** $1024 \times 1024$ Matrix Multiplication ($C = A \times B$).
* **Hardware:** Sony Vaio SVF15N2C5E (i7-4500U, Intel HD 4400, NVIDIA GT 735M).

## Results

### 1. Performance Comparison (NVIDIA GT 735M)
When running both APIs on the dedicated GPU, the performance is identical.

| API | Time (ms) | Notes |
| :--- | :--- | :--- |
| **OpenGL Compute** | **~150.1 ms** | Easier to integrate with graphics. |
| **OpenCL** | **~153.3 ms** | Dedicated compute API. |

**Conclusion:** OpenGL Compute Shaders incur **no performance penalty** compared to OpenCL for this workload.

### 2. Device Comparison
| Device | API | Time | Speedup vs CPU |
| :--- | :--- | :--- | :--- |
| **CPU (i7-4500U)** | OpenCL (POCL) | ~6048 ms | 1x |
| **Intel HD 4400** | OpenGL | ~226 ms | 26x |
| **NVIDIA GT 735M** | OpenGL | ~150 ms | 40x |

*(Note: The legacy Intel OpenCL driver for Haswell is not available on modern Linux, so the Intel GPU was benchmarked via OpenGL only.)*

## Prerequisites
* **OpenCL Runtime:**
    * `nvidia-driver-470` (for NVIDIA)
    * `pocl-opencl-icd` (for CPU)
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

**Run on NVIDIA:**

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./build/gpgpu_vs_cl

```
