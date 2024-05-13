cmake_minimum_required(VERSION 3.17)

if (MDL_SDK_FOUND)
  add_subdirectory( MDL_renderer )
  add_subdirectory( neural_radiance_caching_mdl )
else()
  message(FATAL_ERROR "The Neural Radiance Caching demo requires the MDL_SDK. Set the MDL_SDK_PATH variable. Build aborted.") 
endif()
