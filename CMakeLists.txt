cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(ePIC_Toy_Detector)

# Set C++17 standard (more widely supported than C++20)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Geant4 package
find_package(Geant4 REQUIRED ui_all vis_all)

# Try to find ROOT using different methods
# Method 1: Use ROOT's provided scripts
execute_process(
  COMMAND bash -c "which root-config || echo NOTFOUND"
  OUTPUT_VARIABLE ROOT_CONFIG_EXECUTABLE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE ROOT_CONFIG_RESULT
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
  )
  
  if(ROOT_VERSION_RESULT EQUAL 0)
    # If root-config works correctly, get configuration details
    execute_process(
      COMMAND ${ROOT_CONFIG_EXECUTABLE} --prefix
      OUTPUT_VARIABLE ROOT_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    execute_process(
      COMMAND ${ROOT_CONFIG_EXECUTABLE} --cflags
      OUTPUT_VARIABLE ROOT_CFLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    execute_process(
      COMMAND ${ROOT_CONFIG_EXECUTABLE} --libs
      OUTPUT_VARIABLE ROOT_LIBS
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    execute_process(
      COMMAND ${ROOT_CONFIG_EXECUTABLE} --incdir
      OUTPUT_VARIABLE ROOT_INCLUDE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    execute_process(
      COMMAND ${ROOT_CONFIG_EXECUTABLE} --libdir
      OUTPUT_VARIABLE ROOT_LIBRARY_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  else()
    message(STATUS "root-config found but not executable, will try alternative methods")
    set(ROOT_CONFIG_EXECUTABLE "")
  endif()
  
  message(STATUS "ROOT prefix: ${ROOT_PREFIX}")
  message(STATUS "ROOT include dir: ${ROOT_INCLUDE_DIR}")
  message(STATUS "ROOT library dir: ${ROOT_LIBRARY_DIR}")
  
  include_directories(${ROOT_INCLUDE_DIR})
  link_directories(${ROOT_LIBRARY_DIR})
  
  set(ROOT_FOUND TRUE)
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

# Find source files
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

# Define ROOT libraries needed - only if ROOT was found properly
if(ROOT_FOUND AND NOT "${ROOT_LIBS}" STREQUAL "")
  # Use the libraries returned by root-config if available
  set(ROOT_LIBRARIES ${ROOT_LIBS})
else()
  # Fall back to common ROOT libraries
  if(ROOT_FOUND)
    message(STATUS "Using default ROOT libraries")
    set(ROOT_LIBRARIES "-lCore -lRIO -lHist -lTree -lGraf")
  else()
    set(ROOT_LIBRARIES "")
  endif()
endif()

message(STATUS "ROOT Libraries: ${ROOT_LIBRARIES}")

# Create the executable
add_executable(epicToy epicToy.cc ${sources} ${headers})
target_link_libraries(epicToy ${Geant4_LIBRARIES} ${ROOT_LIBRARIES})

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

add_custom_target(ePIC DEPENDS epicToy)
