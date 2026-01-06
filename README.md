# Raytracing-Voxilizer-Vulkan-Intresection

Vulkan RayTracing Intersection + voxelization project for my **Bachelor’s Thesis** at **TUM**.

Built on top of:

- [nvpro-samples / nvpro_core](https://github.com/nvpro-samples/nvpro_core)
- [vk_raytracing_tutorial_KHR](https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR)
- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader)


## Requirements

- **Vulkan SDK** (latest recommended)
- **CMake** (3.9)
- **C++23** compiler (MSVC/Clang/GCC)
- **GPU + driver with Vulkan Ray Tracing support**  (e.g. RTX-class NVIDIA, or any GPU supporting `VK_KHR_ray_tracing_pipeline`)
- **Git**

> Windows: the SDK installer sets `VULKAN_SDK` automatically.  


## Get the code

Clone NVIDIA’s core utilities first:

```bash
git clone --recursive --shallow-submodules https://github.com/nvpro-samples/nvpro_core.git
```

Then clone this repository:

```bash
git clone https://github.com/MatBayern/Raytracing-Voxilizer-Vulkan-Intresection.git
```

## Build

### Windows (Visual Studio)

```bash
cd Raytracing-Voxilizer-Vulkan-Intresection
mkdir build
cd build
cmake ..
```

> If CMake can’t find `nvpro_core`, pass `-DNVPRO_CORE_DIR=/absolute/path/to/nvpro_core`.

---

## Run

From your build output directory (adjust the binary name/path for your platform):

```bash
RaytracingVoxilizerVulkan.exe <Path to obj file> <Voxlesize>
```

## Troubleshooting

- **Ray tracing extensions missing** (`VK_ERROR_EXTENSION_NOT_PRESENT`)  
  Update your GPU driver and ensure support for `VK_KHR_ray_tracing_pipeline`, `VK_KHR_acceleration_structure`, and related extensions.
- **Vulkan SDK not found**  
  Verify the SDK is installed and `VULKAN_SDK` is set in your environment.
- **`nvpro_core` not found**  
  Provide `-DNVPRO_CORE_DIR=/path/to/nvpro_core` or set an env var with that path.
- **Stale build artifacts**  
  Try a clean build: delete the `build/` folder and reconfigure.
- **No startup project found**  
You need to select vk_src_KHR as the startprojcet

## Acknowledgements

- NVIDIA **nvpro-samples** and **vk_raytracing_tutorial_KHR** for reference implementations.

## License

This repo includes third-party components with their own licenses (see their repos).
