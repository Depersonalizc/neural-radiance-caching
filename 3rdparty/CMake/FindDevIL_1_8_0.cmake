# Looks for the environment variable:
# DEVIL_PATH

# Sets the variables:
# IL_INCLUDE_DIR
# IL_LIBRARIES
# ILU_LIBRARIES
# ILUT_LIBRARIES

# DevIL_1_8_0_FOUND

# Note that the DevIL 1.8.0 version is missing support for EXR images!

set(DEVIL_PATH $ENV{DEVIL_PATH})

if ("${DEVIL_PATH}" STREQUAL "")
  # Try finding it inside the 3rdparty folder.
  set(DEVIL_PATH "${LOCAL_3RDPARTY}/devil_1_8_0")
endif()

# message("DEVIL_PATH = " "${DEVIL_PATH}")

find_path( IL_INCLUDE_DIR IL/il.h
  PATHS /usr/include ${DEVIL_PATH}/include )

# message("IL_INCLUDE_DIR = " "${IL_INCLUDE_DIR}")

set(DEVIL_LIBRARY_DIR "${DEVIL_PATH}/lib/x64/Release")

find_library( IL_LIBRARIES
  NAMES DevIL IL
  PATHS ${DEVIL_LIBRARY_DIR} )

message("IL_LIBRARIES = " "${IL_LIBRARIES}")

# find_library( ILU_LIBRARIES
#   NAMES ILU
#   PATHS ${DEVIL_LIBRARY_DIR} )

# message("ILU_LIBRARIES = " "${ILU_LIBRARIES}")

# find_library( ILUT_LIBRARIES
#   NAMES ILUT
#   PATHS ${DEVIL_LIBRARY_DIR} )

# message("ILUT_LIBRARIES = " "${ILUT_LIBRARIES}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(DevIL_1_8_0
  DEFAULT_MSG 
  IL_INCLUDE_DIR IL_LIBRARIES 
  # ILU_LIBRARIES ILUT_LIBRARIES
)

# message("DevIL_1_8_0_FOUND = " "${DevIL_1_8_0_FOUND}")

if (DevIL_1_8_0_FOUND)
  if (WIN32)
    # Assume DevIL.dll is in the same directory as DevIL.lib
    cmake_path(REPLACE_EXTENSION IL_LIBRARIES "dll" 
               OUTPUT_VARIABLE IL_LIBRARIES_SHARED)
    find_file(IL_LIBRARIES_SHARED ${IL_LIBRARIES_SHARED} REQUIRED)
  else()
    set(IL_LIBRARIES_SHARED ${IL_LIBRARIES})
  endif()

  message("IL_LIBRARIES_SHARED = " ${IL_LIBRARIES_SHARED})

  add_library(il SHARED IMPORTED)
  set_target_properties(il PROPERTIES
    IMPORTED_LOCATION ${IL_LIBRARIES_SHARED}
    IMPORTED_IMPLIB ${IL_LIBRARIES}
    INTERFACE_INCLUDE_DIRECTORIES ${IL_INCLUDE_DIR} 
  )
endif()

mark_as_advanced(
  IL_INCLUDE_DIR IL_LIBRARIES IL_LIBRARIES_SHARED
  # ILU_LIBRARIES ILUT_LIBRARIES 
)
