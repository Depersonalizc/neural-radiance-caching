# Neural Radiance Caching

An implementation of [Real-time Neural Radiance Caching for Path Tracing](https://research.nvidia.com/publication/2021-06_real-time-neural-radiance-caching-path-tracing) with OptiX 8 and CUDA API.

![NRC hero image](./nrc/res/nrc.png)
*Rendered in 2K resolution with an RTX 4080 Laptop GPU*

## How can I run this?

> *Tested on Windows with Visual Studio 2022.*
> 
> *Building for Linux, or Windows without VS, is possible, but you may have to manually set up the third-party dependencies.*

### Prerequisites

- **OptiX SDK 7 or 8, and a supported NVIDIA GPU**.
- **CUDA Toolkit**: Tested with CUDA 12.4, but any recent version compatible with your OptiX SDK should work.
- **CMake 3.26 or higher**.
- **Visual Studio 2017 or higher**.
- **MDL SDK 1.8**: Required for PBR materials. We recommend downloading the binary [here](https://developer.nvidia.com/mdl-sdk-get-started).
- Additional third-party dependencies set up by the `3rdparty.cmd` script.

### Build & Run

We have included a `CMakeLists.txt` for the project and a `3rdparty.cmd` script to install the additional dependencies. 

You can build the project by following these steps:

- Launch the *x64 Native Tools Command Prompt for VS 2017/2019/2022*. (We will assume this as the shell environment for all commands below.)

- `cd` into the project root directory, and run `3rdparty.cmd`.

- Set the environment variable `MDL_SDK_PATH` to point to your MDL library folder (containing an `include` folder.)

  - Alternatively, place your MDL library folder in the `3rdparty` folder with the name `MDL_SDK`.

- To generate a solution file for the project, type

  ```cmd
  mkdir build && cd build
  cmake -G "Visual Studio <version>" ..
  ```

  where `<version>` is `15 2017` for VS 2017, `16 2019` for VS 2019, and `17 2022` for VS 2022, resp.

To run the renderer,

- Set the working directory of the application to `$(TargetDir)` in Visual Studio (where the executable is located.)

- In the working directory, create a symbolic link to the `data` folder by typing

  ```cmd
  mklink /D .\data <actual_data_folder>
  ```
  :warning: You might need admin privilege for this on Windows :\  (at least sudo is coming soon...)

- Try out the Cornell Box (with friends! 🐉🐇) scene with the command line arguments:

  ```cmd
  -s data/system_mdl_cornell.txt -d data/scene_mdl_cornell_friends.txt
  ```
## Demo

https://github.com/Depersonalizc/neural-radiance-caching/assets/29355340/d442713d-f261-4a35-9368-0c1b15655b82

https://github.com/Depersonalizc/neural-radiance-caching/assets/29355340/962b69b6-2be0-4075-8d4b-8b9c934761e0

