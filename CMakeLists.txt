cmake_minimum_required(VERSION 3.9.6 FATAL_ERROR)
project(Raytracing-Voxilizer-Vulkan)

# ------------------------------------------------------------
# Locate nvpro_core (same logic as upstream)
# - If nvpro_core lives elsewhere, pass -DBASE_DIRECTORY=/path/to/parent
#   where nvpro_core/cmake/setup.cmake is located.
# ------------------------------------------------------------
if(NOT BASE_DIRECTORY)
  find_path(
    BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/..
      ${CMAKE_CURRENT_SOURCE_DIR}/../..
    REQUIRED
    DOC "Directory containing nvpro_core"
  )
endif()

if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/utilities.cmake)
else()
  message(FATAL_ERROR
    "Could not find nvpro_core. Set BASE_DIRECTORY to folder containing nvpro_core")
endif()

# This variable is used by the sample's own CMake to find /common etc.
set(TUTO_KHR_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Nice-to-have MSVC warnings suppression (matches upstream)
if(MSVC)
  add_definitions(/wd26812)  # 'enum class' over 'enum'
  add_definitions(/wd26451)  # 4->8 byte cast warning
endif()

# ------------------------------------------------------------
# Shared packages/libs used by the sample (provided by nvpro_core)
# ------------------------------------------------------------
_add_package_VulkanSDK()
_add_package_ImGUI()
_add_nvpro_core_lib()

# Copy sample media next to output, just like the original root CMake
message(STATUS "COPY ${CMAKE_CURRENT_SOURCE_DIR}/media  to  ${OUTPUT_PATH}")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/media DESTINATION ${OUTPUT_PATH})

# ------------------------------------------------------------
# Build ONLY the 'src' sample
# (uses its own CMakeLists.txt which expects TUTO_KHR_DIR and the nvpro_core macros)
# ------------------------------------------------------------
add_subdirectory(src)

# ------------------------------------------------------------
# Optional: install media for Release/Debug to match upstream layout
# ------------------------------------------------------------
install(DIRECTORY "media" CONFIGURATIONS Release DESTINATION "bin_${ARCH}")
install(DIRECTORY "media" CONFIGURATIONS Debug   DESTINATION "bin_${ARCH}_debug")
