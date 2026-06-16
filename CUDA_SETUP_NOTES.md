# CUDA Setup Notes

This note summarizes the preparation work done before starting the actual CUDA renderer implementation.

## Starting Point

- Created a separate CUDA study repository from the original CPU ray tracer.
- Reset the CUDA repository to the early "first rendered image" state of the original project.
- Kept the scene small enough to make CUDA migration easier.

## CPU Baseline

The CPU renderer was reduced to a lightweight debug render:

- Image size: 200 x 112
- Samples per pixel: 5
- Max depth: 10
- Total primary samples: 112,000

Measured result:

- Render time: 5.88546 seconds
- Pixels/sec: 3,805.99
- Primary samples/sec: 19,029.9

Timing was added in `camera::render()` using `std::chrono`.

## CUDA Toolkit

Installed CUDA Toolkit 13.3.

Confirmed `nvcc`:

```powershell
nvcc --version
```

Result:

```text
Cuda compilation tools, release 13.3, V13.3.33
```

`nvcc` is the NVIDIA CUDA compiler. It compiles `.cu` files and device code that runs on the GPU.

## Visual Studio Compiler

CUDA on Windows also needs the Microsoft C++ compiler, `cl.exe`.

In a normal PowerShell session, `cl` was not visible. However, it worked after loading the Visual Studio developer environment:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && cl && nvcc --version'
```

Confirmed environment:

- Visual Studio 2026 Community
- MSVC 19.51
- CUDA 13.3

## CMake Changes

Updated `CMakeLists.txt` so CUDA is optional:

- `USE_CUDA` option added.
- CMake checks whether a CUDA compiler exists.
- If CUDA is found, CUDA language support is enabled.
- If CUDA is not found, the project still builds as CPU-only.

This keeps the project usable even on machines without CUDA.

## NMake CUDA Build

Because the installed CMake did not provide a `Visual Studio 18 2026` generator, the project was configured with `NMake Makefiles` inside the Visual Studio developer environment.

Configure:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && cmake -S . -B build_nmake_cuda -G "NMake Makefiles"'
```

Build:

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && cmake --build build_nmake_cuda'
```

Confirmed:

```text
CUDA support enabled
[100%] Built target inOneWeekend
```

## Current Status

- CPU renderer still runs.
- CUDA compiler is detected through the Visual Studio developer environment.
- No CUDA rendering code has been implemented yet.
- Next step: add a minimal `.cu` file and render a simple GPU-generated gradient image.
