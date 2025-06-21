cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(ePIC_Toy_Detector)

# Set CMake policies for modern behavior
if(POLICY CMP0069)
    cmake_policy(SET CMP0069 NEW)  # Link time optimization support
endif()

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)  # find_package uses PackageName_ROOT variables
endif()

# Set C++17 as minimum standard for broader compatibility
# Individual targets will use target_compile_features for explicit requirements
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compiler optimization and multithreading support
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")

# Enable multithreading support
find_package(Threads REQUIRED)

# Set build type to Release by default for optimal performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Add performance optimization options for multithreading
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Enable link-time optimization if supported
    include(CheckIPOSupported)
    check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT error)
    if(LTO_SUPPORTED)
        message(STATUS "Link-time optimization supported - enabling for better performance")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "Link-time optimization not supported: ${error}")
    endif()
    
    # Add additional optimization flags for multithreaded performance
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -funroll-loops")
        message(STATUS "Added performance optimization flags for ${CMAKE_CXX_COMPILER_ID}")
    endif()
endif()

# Find Geant4 package with multithreading support
find_package(Geant4 REQUIRED ui_all vis_all)

# Check if Geant4 was built with multithreading support
if(Geant4_multithreaded_FOUND)
    message(STATUS "Geant4 multithreading support: ENABLED")
    add_definitions(-DG4MULTITHREADED)
else()
    message(STATUS "Geant4 multithreading support: DISABLED")
    message(WARNING "Geant4 was not built with multithreading support. Consider rebuilding Geant4 with -DGEANT4_BUILD_MULTITHREADED=ON for optimal performance.")
endif()

# Try to find ROOT using different methods
# Method 1: Use ROOT's provided scripts
execute_process(
  COMMAND bash -c "which root-config || echo NOTFOUND"
  OUTPUT_VARIABLE ROOT_CONFIG_EXECUTABLE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE ROOT_CONFIG_RESULT
  TIMEOUT 10
)

if(ROOT_CONFIG_EXECUTABLE AND NOT "${ROOT_CONFIG_EXECUTABLE}" STREQUAL "NOTFOUND")
  message(STATUS "Found ROOT config at: ${ROOT_CONFIG_EXECUTABLE}")
  
  # Check if root-config is actually executable
  execute_process(
    COMMAND ${ROOT_CONFIG_EXECUTABLE} --version
    OUTPUT_VARIABLE ROOT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE ROOT_VERSION_RESULT
    ERROR_QUIET
    TIMEOUT 10
  )
  
  if(ROOT_VERSION_RESULT EQUAL 0)
    message(STATUS "ROOT version: ${ROOT_VERSION}")
    
    # Check ROOT version for ABI compatibility
    if(ROOT_VERSION VERSION_LESS "6.20.0")
        message(WARNING "ROOT version ${ROOT_VERSION} is older than recommended (6.20+). Consider upgrading to avoid potential ABI issues.")
    elseif(ROOT_VERSION VERSION_GREATER_EQUAL "6.30.0")
        message(WARNING "ROOT version ${ROOT_VERSION} is very new. Monitor for potential compatibility issues.")
    endif()
    
    # If root-config works correctly, get configuration details
    execute_process(
      COMMAND ${ROOT_CONFIG_EXECUTABLE} --prefix
      OUTPUT_VARIABLE ROOT_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      TIMEOUT 10
    )
    
    execute_process(
      COMMAND ${ROOT_CONFIG_EXECUTABLE} --cflags
      OUTPUT_VARIABLE ROOT_CFLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      TIMEOUT 10
    )
    
    execute_process(
      COMMAND ${ROOT_CONFIG_EXECUTABLE} --libs
      OUTPUT_VARIABLE ROOT_LIBS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      TIMEOUT 10
    )
    
    execute_process(
      COMMAND ${ROOT_CONFIG_EXECUTABLE} --incdir
      OUTPUT_VARIABLE ROOT_INCLUDE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      TIMEOUT 10
    )
    
    execute_process(
      COMMAND ${ROOT_CONFIG_EXECUTABLE} --libdir
      OUTPUT_VARIABLE ROOT_LIBRARY_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      TIMEOUT 10
    )
  else()
    message(STATUS "root-config found but not executable, will try alternative methods")
    set(ROOT_CONFIG_EXECUTABLE "")
  endif()
  
  if(ROOT_PREFIX AND ROOT_INCLUDE_DIR AND ROOT_LIBRARY_DIR)
    message(STATUS "ROOT prefix: ${ROOT_PREFIX}")
    message(STATUS "ROOT include dir: ${ROOT_INCLUDE_DIR}")
    message(STATUS "ROOT library dir: ${ROOT_LIBRARY_DIR}")
    
    include_directories(${ROOT_INCLUDE_DIR})
    link_directories(${ROOT_LIBRARY_DIR})
    
    set(ROOT_FOUND TRUE)
  else()
    message(WARNING "Failed to get ROOT configuration from root-config, trying fallback methods")
    set(ROOT_CONFIG_EXECUTABLE "")
  endif()
else()
  # Method 2: Standard CMake find_package
  find_package(ROOT QUIET)
  
  if(NOT ROOT_FOUND)
    # Method 3: Try explicit paths if using snap
    if(EXISTS "/snap/root-framework/current/usr/local")
      set(ROOT_PREFIX "/snap/root-framework/current/usr/local")
      set(ROOT_INCLUDE_DIR "${ROOT_PREFIX}/include")
      set(ROOT_LIBRARY_DIR "${ROOT_PREFIX}/lib")
      include_directories(${ROOT_INCLUDE_DIR})
      link_directories(${ROOT_LIBRARY_DIR})
      set(ROOT_FOUND TRUE)
      
      message(STATUS "Using ROOT from snap at: ${ROOT_PREFIX}")
    endif()
  endif()
endif()

# Final fallback: allow manual specification of ROOT paths
if(NOT ROOT_FOUND)
  # Check for common ROOT installation locations
  foreach(dir "/usr/local" "/opt/root" "/usr" "/opt" "/snap/root-framework/current/usr/local")
    if(EXISTS "${dir}/include/root" OR EXISTS "${dir}/include/ROOT" OR EXISTS "${dir}/include/TROOT.h")
      set(ROOT_PREFIX "${dir}")
      
      # Try to determine include and lib directories
      if(EXISTS "${dir}/include/root")
        set(ROOT_INCLUDE_DIR "${dir}/include/root")
      elseif(EXISTS "${dir}/include/ROOT")
        set(ROOT_INCLUDE_DIR "${dir}/include/ROOT") 
      else()
        set(ROOT_INCLUDE_DIR "${dir}/include")
      endif()
      
      if(EXISTS "${dir}/lib/root")
        set(ROOT_LIBRARY_DIR "${dir}/lib/root")
      else()
        set(ROOT_LIBRARY_DIR "${dir}/lib")
      endif()
      
      include_directories(${ROOT_INCLUDE_DIR})
      link_directories(${ROOT_LIBRARY_DIR})
      set(ROOT_FOUND TRUE)
      
      message(STATUS "Using ROOT found in: ${dir}")
      break()
    endif()
  endforeach()
  
  # Allow user to manually specify ROOT paths via environment variables
  if(NOT ROOT_FOUND AND DEFINED ENV{ROOTSYS})
    set(ROOT_PREFIX $ENV{ROOTSYS})
    set(ROOT_INCLUDE_DIR "${ROOT_PREFIX}/include")
    set(ROOT_LIBRARY_DIR "${ROOT_PREFIX}/lib")
    include_directories(${ROOT_INCLUDE_DIR})
    link_directories(${ROOT_LIBRARY_DIR})
    set(ROOT_FOUND TRUE)
    message(STATUS "Using ROOT from ROOTSYS environment variable: ${ROOT_PREFIX}")
  endif()
  
  # If still not found, exit with error
  if(NOT ROOT_FOUND)
    message(STATUS "Could not find ROOT automatically.")
    # Instead of failing, continue with dummy settings and warn
    message(WARNING "Will attempt to build without ROOT paths. You may need to set ROOT paths manually.")
    set(ROOT_FOUND TRUE)
    set(ROOT_LIBRARIES "")
  endif()
endif()

include(${Geant4_USE_FILE})

# Set include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Find Ceres Solver with version checking to prevent ABI mismatches
find_package(Ceres 2.0 REQUIRED)
if(Ceres_FOUND)
    message(STATUS "Found Ceres Solver version ${Ceres_VERSION}")
    message(STATUS "Ceres include dirs: ${CERES_INCLUDE_DIRS}")
    message(STATUS "Ceres libraries: ${CERES_LIBRARIES}")
    
    # Check for potential ABI compatibility issues
    if(Ceres_VERSION VERSION_LESS "2.0.0")
        message(WARNING "Ceres version ${Ceres_VERSION} is older than recommended (2.0+). Consider upgrading to avoid potential ABI issues.")
    elseif(Ceres_VERSION VERSION_GREATER_EQUAL "3.0.0")
        message(WARNING "Ceres version ${Ceres_VERSION} is very new. Monitor for potential compatibility issues.")
    endif()
else()
    message(FATAL_ERROR "Ceres Solver not found! Please install Ceres Solver version 2.0 or later.")
endif()

# Add Ceres include directories
include_directories(${CERES_INCLUDE_DIRS})

# Find source files
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

# Define ROOT libraries needed - only if ROOT was found properly
if(ROOT_FOUND AND NOT "${ROOT_LIBS}" STREQUAL "")
  # Use the libraries returned by root-config if available, and add Minuit
  set(ROOT_LIBRARIES ${ROOT_LIBS} -lMinuit)
else()
  # Fall back to common ROOT libraries including math libraries for fitting
  if(ROOT_FOUND)
    message(STATUS "Using default ROOT libraries")
    set(ROOT_LIBRARIES "-lCore -lRIO -lHist -lTree -lGraf -lMinuit -lMathCore -lMatrix")
  else()
    set(ROOT_LIBRARIES "")
  endif()
endif()

message(STATUS "ROOT Libraries: ${ROOT_LIBRARIES}")

# Create the executable with enhanced performance settings
add_executable(epicChargeSharing epicChargeSharing.cc ${sources} ${headers})

# Set explicit C++ standard requirements for this target
target_compile_features(epicChargeSharing PRIVATE cxx_std_17)

# Link libraries including threading support
target_link_libraries(epicChargeSharing 
    ${Geant4_LIBRARIES} 
    ${ROOT_LIBRARIES} 
    ${CERES_LIBRARIES}
    Threads::Threads
)

# Set target-specific properties for optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_target_properties(epicChargeSharing PROPERTIES
        COMPILE_FLAGS "-O3 -march=native -mtune=native"
    )
    
    # Enable interprocedural optimization for this target if supported
    if(LTO_SUPPORTED)
        set_property(TARGET epicChargeSharing PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

# Copy macro files
file(GLOB MACRO_FILES
  "macros/*.mac"
)
file(COPY ${MACRO_FILES} DESTINATION ${PROJECT_BINARY_DIR})

# Copy data files
file(GLOB DATA_FILES
  "*.dat"
)
file(COPY ${DATA_FILES} DESTINATION ${PROJECT_BINARY_DIR})

add_custom_target(ePIC DEPENDS epicChargeSharing)

# Add custom target for performance build
add_custom_target(performance
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config Release
    COMMENT "Building with maximum performance optimizations"
)

# Print summary of multithreading capabilities
message(STATUS "=== MULTITHREADING SUMMARY ===")
message(STATUS "Geant4 MT support: ${Geant4_multithreaded_FOUND}")
message(STATUS "Threading library: ${CMAKE_THREAD_LIBS_INIT}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Performance optimizations: ENABLED")
    if(LTO_SUPPORTED)
        message(STATUS "Link-time optimization: ENABLED")
    else()
        message(STATUS "Link-time optimization: DISABLED")
    endif()
else()
    message(STATUS "Performance optimizations: DISABLED (Debug build)")
endif()
message(STATUS "==============================")
