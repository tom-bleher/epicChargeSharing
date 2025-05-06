cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(ePIC_Toy_Detector)

# Find Geant4 package
find_package(Geant4 REQUIRED ui_all vis_all)

# Try to find ROOT using different methods
# Method 1: Use ROOT's provided scripts
execute_process(
  COMMAND which root-config
  OUTPUT_VARIABLE ROOT_CONFIG_EXECUTABLE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(ROOT_CONFIG_EXECUTABLE)
  message(STATUS "Found ROOT config at: ${ROOT_CONFIG_EXECUTABLE}")
  
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

if(NOT ROOT_FOUND)
  message(FATAL_ERROR "Could not find ROOT! Please make sure ROOT is properly installed.")
endif()

include(${Geant4_USE_FILE})

# Set include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Find source files
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

# Define ROOT libraries needed
set(ROOT_LIBRARIES -lCore -lRIO -lHist -lTree -lGraf)

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
