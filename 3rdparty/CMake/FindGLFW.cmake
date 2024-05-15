# Looks for environment variable:
# GLFW_PATH 

# Sets the variables:
# GLFW_INCLUDE_DIR
# GLFW_LIBRARIES
# GLFW_FOUND

set(GLFW_PATH $ENV{GLFW_PATH})

# If there was no environment variable override for the GLFW_PATH
# try finding it inside the local 3rdparty path.
if ("${GLFW_PATH}" STREQUAL "")
  set(GLFW_PATH "${LOCAL_3RDPARTY}/glfw")
endif()

# message("GLFW_PATH = " "${GLFW_PATH}")

find_path( GLFW_INCLUDE_DIR "GLFW/glfw3.h"
  PATHS /usr/include ${GLFW_PATH}/include )

# message("GLFW_INCLUDE_DIR = " "${GLFW_INCLUDE_DIR}")

# GLFW as download from glfw_sourceforge comes with precompiled libraries per MSVS version.
if (WIN32)
  if("${MSVC_VERSION}" VERSION_EQUAL "1900") # MSVS 2015
    set(MSVS_VERSION "2015")
  elseif(("${MSVC_VERSION}" VERSION_GREATER_EQUAL "1910") AND ("${MSVC_VERSION}" VERSION_LESS_EQUAL "1919")) # MSVS 2017
    set(MSVS_VERSION "2017")
  elseif(("${MSVC_VERSION}" VERSION_GREATER_EQUAL "1920") AND ("${MSVC_VERSION}" VERSION_LESS_EQUAL "1929")) # MSVS 2019
    set(MSVS_VERSION "2019")
  elseif(("${MSVC_VERSION}" VERSION_GREATER_EQUAL "1930") AND ("${MSVC_VERSION}" VERSION_LESS_EQUAL "1939")) # MSVS 2022
    set(MSVS_VERSION "2022")
  endif()
  set(GLFW_LIBRARY_DIR ${GLFW_PATH}/lib-vc${MSVS_VERSION})
  # message("GLFW_LIBRARY_DIR = " "${GLFW_LIBRARY_DIR}")
endif()

find_library(GLFW_LIBRARIES
  NAMES glfw3 glfw
  PATHS /usr/lib64 ${GLFW_LIBRARY_DIR} )

message("GLFW_LIBRARIES = " "${GLFW_LIBRARIES}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(GLFW DEFAULT_MSG GLFW_INCLUDE_DIR GLFW_LIBRARIES)

# message("GLFW_FOUND = " "${GLFW_FOUND}")

if (GLFW_FOUND)
  if (WIN32)
    # Assume glfw.dll is in the same directory as glfw.lib
    cmake_path(REPLACE_EXTENSION GLFW_LIBRARIES "dll" 
               OUTPUT_VARIABLE GLFW_LIBRARIES_SHARED)
    find_file(GLFW_LIBRARIES_SHARED ${GLFW_LIBRARIES_SHARED} REQUIRED)
  else()
    set(GLFW_LIBRARIES_SHARED ${GLFW_LIBRARIES})
  endif()

  message("GLFW_LIBRARIES_SHARED = " ${GLFW_LIBRARIES_SHARED})

  add_library(glfw SHARED IMPORTED)
  set_target_properties(glfw PROPERTIES
    IMPORTED_LOCATION ${GLFW_LIBRARIES_SHARED}
    IMPORTED_IMPLIB ${GLFW_LIBRARIES}
    INTERFACE_INCLUDE_DIRECTORIES ${GLFW_INCLUDE_DIR} 
  )
endif()

mark_as_advanced(GLFW_INCLUDE_DIR GLFW_LIBRARIES GLFW_LIBRARIES_SHARED)