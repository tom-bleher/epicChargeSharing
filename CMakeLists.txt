cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(epicChargeSharing)

# Set CMake policies for modern behavior
if(POLICY CMP0069)
    cmake_policy(SET CMP0069 NEW)  # Link time optimization support
endif()

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)  # find_package uses PackageName_ROOT variables
endif()

# Set C++20 as minimum standard to match ROOT requirements
# Individual targets will use target_compile_features for explicit requirements
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Accumulate compile and link options for the main target
set(_project_compile_options)
set(_project_link_options)
set(_project_release_compile_options)

# Encourage consistent diagnostics
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    list(APPEND _project_compile_options -Wall -Wextra -Wpedantic)
endif()

# Enable multithreading support
find_package(Threads REQUIRED)

# Set build type to Release by default for optimal performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Release by default")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Optional fast-math/native CPU optimizations (opt-in)
option(EC_FAST_MATH "Enable fast-math" OFF)
set(EC_MARCH "native" CACHE STRING "CPU architecture to tune/optimize for (e.g., native, znver4, skylake, neoverse-n1)")
option(EC_PROFILE "Enable profiling flags (-pg, frame pointers)" OFF)
# Back-compat alias requested in report
option(PROFILE "Alias for EC_PROFILE" OFF)
if(PROFILE)
    set(EC_PROFILE ON CACHE BOOL "Enable profiling flags (-pg, frame pointers)" FORCE)
endif()
if(EC_FAST_MATH)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        list(APPEND _project_compile_options
             -ffast-math
             -fno-math-errno
             -ffp-contract=fast
             "-march=${EC_MARCH}"
             "-mtune=${EC_MARCH}")
        message(STATUS "EC_FAST_MATH enabled: -ffast-math -fno-math-errno -ffp-contract=fast -march=${EC_MARCH} -mtune=${EC_MARCH}")
    else()
        message(WARNING "EC_FAST_MATH is enabled, but not supported for ${CMAKE_CXX_COMPILER_ID}")
    endif()
endif()

if(EC_PROFILE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        list(APPEND _project_compile_options -pg -fno-omit-frame-pointer)
        list(APPEND _project_link_options -pg)
        message(STATUS "EC_PROFILE enabled: adding -pg and -fno-omit-frame-pointer")
    else()
        message(WARNING "EC_PROFILE requested, but unsupported for ${CMAKE_CXX_COMPILER_ID}")
    endif()
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
        list(APPEND _project_release_compile_options -funroll-loops)
        message(STATUS "Added -funroll-loops for ${CMAKE_CXX_COMPILER_ID}")
    endif()
endif()

# Find Eigen3 for optimized matrix operations (header-only library)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")

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

set(ROOT_COMPONENTS Core RIO Hist Tree Graf Minuit MathCore Matrix)

# Help CMake locate ROOT installations such as the snap package
set(_root_hint_paths)
if(DEFINED ROOT_DIR)
    list(APPEND _root_hint_paths "${ROOT_DIR}")
endif()

find_program(ROOT_CONFIG_EXECUTABLE
    NAMES root-config
    HINTS "$ENV{ROOTSYS}/bin" /snap/bin
    DOC "Path to the root-config helper from ROOT")

if(ROOT_CONFIG_EXECUTABLE)
    execute_process(
        COMMAND ${ROOT_CONFIG_EXECUTABLE} --prefix
        OUTPUT_VARIABLE _root_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_root_prefix)
        set(_root_prefix_real "${_root_prefix}")
        if(NOT IS_ABSOLUTE "${_root_prefix_real}")
            get_filename_component(_root_prefix_real "${_root_prefix_real}" ABSOLUTE)
        endif()
        set(_candidate_root_cmake_paths
            "${_root_prefix_real}/cmake"
            "${_root_prefix_real}/lib/cmake/ROOT")
        foreach(_candidate IN LISTS _candidate_root_cmake_paths)
            if(EXISTS "${_candidate}/ROOTConfig.cmake")
                list(APPEND _root_hint_paths "${_candidate}")
                if(NOT DEFINED ROOT_DIR)
                    set(ROOT_DIR "${_candidate}" CACHE PATH "Path to ROOTConfig.cmake detected via root-config" FORCE)
                endif()
            endif()
        endforeach()
        list(APPEND CMAKE_PREFIX_PATH "${_root_prefix_real}")
    endif()
endif()

if(_root_hint_paths)
    list(REMOVE_DUPLICATES _root_hint_paths)
    find_package(ROOT CONFIG REQUIRED COMPONENTS ${ROOT_COMPONENTS} HINTS ${_root_hint_paths})
else()
    find_package(ROOT CONFIG REQUIRED COMPONENTS ${ROOT_COMPONENTS})
endif()

set(ROOT_LIB_TARGETS)
foreach(component IN LISTS ROOT_COMPONENTS)
    list(APPEND ROOT_LIB_TARGETS ROOT::${component})
endforeach()

include(${Geant4_USE_FILE})

# ============================================================================
# EDM4hep Support (optional)
# ============================================================================
option(WITH_EDM4HEP "Enable EDM4hep output format for EICrecon compatibility" OFF)

if(WITH_EDM4HEP)
    # Find PODIO (EDM4hep's I/O backend)
    find_package(podio 1.0 REQUIRED)
    message(STATUS "PODIO version: ${podio_VERSION}")

    # Find EDM4hep
    find_package(EDM4HEP REQUIRED)
    message(STATUS "EDM4HEP version: ${EDM4HEP_VERSION}")

    message(STATUS "EDM4hep output: ENABLED")
else()
    message(STATUS "EDM4hep output: DISABLED (use -DWITH_EDM4HEP=ON to enable)")
endif()

# Consolidate project sources
set(epicChargeSharing_sources
    ${PROJECT_SOURCE_DIR}/src/ActionInitialization.cc
    ${PROJECT_SOURCE_DIR}/src/ChargeSharingCalculator.cc
    ${PROJECT_SOURCE_DIR}/src/DetectorConstruction.cc
    ${PROJECT_SOURCE_DIR}/src/EventAction.cc
    ${PROJECT_SOURCE_DIR}/src/PhysicsList.cc
    ${PROJECT_SOURCE_DIR}/src/PrimaryGenerator.cc
    ${PROJECT_SOURCE_DIR}/src/RunAction.cc
    ${PROJECT_SOURCE_DIR}/src/SteppingAction.cc
    ${PROJECT_SOURCE_DIR}/src/RootHelpers.cc
    ${PROJECT_SOURCE_DIR}/src/RootIO.cc
    ${PROJECT_SOURCE_DIR}/src/EDM4hepIO.cc
    # ${PROJECT_SOURCE_DIR}/src/RNTupleIO.cc  # Disabled due to ROOT version incompatibility
)

# Create the executable with enhanced performance settings
add_executable(epicChargeSharing ${PROJECT_SOURCE_DIR}/epicChargeSharing.cc)
target_sources(epicChargeSharing PRIVATE ${epicChargeSharing_sources})

# Set explicit C++ standard requirements for this target
target_compile_features(epicChargeSharing PRIVATE cxx_std_20)

target_include_directories(epicChargeSharing
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

# Link libraries including threading support and Eigen3
target_link_libraries(epicChargeSharing
    PRIVATE
        ${Geant4_LIBRARIES}
        Threads::Threads
        ${ROOT_LIB_TARGETS}
        Eigen3::Eigen
        $<$<BOOL:${WITH_EDM4HEP}>:podio::podio>
        $<$<BOOL:${WITH_EDM4HEP}>:podio::podioRootIO>
        $<$<BOOL:${WITH_EDM4HEP}>:EDM4HEP::edm4hep>
)

# Add EDM4hep compile definition if enabled
if(WITH_EDM4HEP)
    target_compile_definitions(epicChargeSharing PRIVATE WITH_EDM4HEP)
endif()

if(_project_compile_options)
    target_compile_options(epicChargeSharing PRIVATE ${_project_compile_options})
endif()

if(_project_release_compile_options)
    foreach(flag IN LISTS _project_release_compile_options)
        target_compile_options(epicChargeSharing PRIVATE $<$<CONFIG:Release>:${flag}>)
    endforeach()
endif()

if(_project_link_options)
    string(REPLACE ";" " " _link_flags "${_project_link_options}")
    set_property(TARGET epicChargeSharing APPEND_STRING PROPERTY LINK_FLAGS " ${_link_flags}")
endif()

# Provide the project source dir to the code for locating resources (e.g., ROOT macros)
target_compile_definitions(epicChargeSharing PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Set target-specific properties for optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
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

# Print summary of build configuration
message(STATUS "=== BUILD CONFIGURATION SUMMARY ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Geant4 MT support: ${Geant4_multithreaded_FOUND}")
message(STATUS "Threading library: ${CMAKE_THREAD_LIBS_INIT}")
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
if(WITH_EDM4HEP)
    message(STATUS "EDM4hep output: ENABLED (PODIO ${podio_VERSION}, EDM4HEP ${EDM4HEP_VERSION})")
else()
    message(STATUS "EDM4hep output: DISABLED")
endif()
message(STATUS "===================================")

# ============================================================================
# Testing Support (optional)
# ============================================================================
option(BUILD_TESTING "Build unit tests" OFF)

if(BUILD_TESTING)
    message(STATUS "Building with tests enabled")
    add_subdirectory(tests)
endif()

# ============================================================================
# Documentation Support
# ============================================================================
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/docs/Doxyfile
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
    message(STATUS "Doxygen found - 'make docs' available")
else()
    message(STATUS "Doxygen not found - documentation generation disabled")
endif()
