# GPGPU Benchmark: OpenGL vs OpenCL

A direct performance comparison of **OpenGL 4.6 Compute Shaders** vs **OpenCL 3.0** on the same GPU.

## The Benchmark
* **Task:** $1024 \times 1024$ Matrix Multiplication.
* **Device:** Forces execution on the Discrete GPU (NVIDIA/AMD) using driver exports.
* **Implementation:** * **OpenGL:** Compute Shader (`#version 430`).
    * **OpenCL:** ND-Range Kernel.

## Typical Results (NVIDIA RTX A2000)
* **OpenGL:** ~15 ms
* **OpenCL:** ~10 ms
* **Verdict:** OpenCL has lower driver overhead for pure compute tasks, but OpenGL is competitive.

## Prerequisites
* [Scoop](https://scoop.sh/)
* Visual Studio (MSVC)

## Build
```powershell
vcpkg install opencl glad glfw3 --triplet=x64-windows
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE="$env:USERPROFILE/scoop/apps/vcpkg/current/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Release
.\build\Release\gpgpu_vs_cl.exe

```
