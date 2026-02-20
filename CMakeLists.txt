cmake_minimum_required(VERSION 3.16...3.31)
project(epicChargeSharing VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# ============================================================================
# Options
# ============================================================================
option(WITH_EDM4HEP  "Enable EDM4hep output format"               ON)
option(BUILD_TESTING "Build unit tests"                            OFF)

# ============================================================================
# Dependencies
# ============================================================================
find_package(Geant4 REQUIRED ui_all vis_all)
find_package(ROOT CONFIG REQUIRED COMPONENTS Core RIO Hist Tree Graf Minuit MathCore Matrix)
find_package(Eigen3 REQUIRED CONFIG)
find_package(Threads REQUIRED)

if(WITH_EDM4HEP)
    find_package(podio 1.0 REQUIRED)
    find_package(EDM4HEP REQUIRED)
endif()

# ============================================================================
# Main target
# ============================================================================
add_executable(epicChargeSharing
    epicChargeSharing.cc
    src/ActionInitialization.cc
    src/ChargeSharingCalculator.cc
    src/DetectorConstruction.cc
    src/EventAction.cc
    src/GaussianFitter.cc
    src/PhysicsList.cc
    src/PrimaryGenerator.cc
    src/RunAction.cc
    src/RuntimeConfig.cc
    src/SteppingAction.cc
    src/RootHelpers.cc
    src/RootIO.cc
    src/EDM4hepIO.cc
)

target_include_directories(epicChargeSharing PRIVATE include core)

target_link_libraries(epicChargeSharing
    PRIVATE
        # Geant4_LIBRARIES is already a list of Geant4::* imported targets
        # assembled by find_package() for the specific ui/vis drivers available.
        ${Geant4_LIBRARIES}
        ROOT::Core ROOT::RIO ROOT::Hist ROOT::Tree
        ROOT::Graf ROOT::Minuit ROOT::MathCore ROOT::Matrix
        Eigen3::Eigen
        Threads::Threads
        $<$<BOOL:${WITH_EDM4HEP}>:podio::podio>
        $<$<BOOL:${WITH_EDM4HEP}>:podio::podioRootIO>
        $<$<BOOL:${WITH_EDM4HEP}>:EDM4HEP::edm4hep>
)

target_compile_definitions(epicChargeSharing
    PRIVATE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        $<$<BOOL:${Geant4_multithreaded_FOUND}>:G4MULTITHREADED>
        $<$<BOOL:${WITH_EDM4HEP}>:WITH_EDM4HEP>
)

# Warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(epicChargeSharing PRIVATE -Wall -Wextra -Wpedantic)
    # Suppress #warning directives from ROOT headers (C++ standard mismatch)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(epicChargeSharing PRIVATE -Wno-\#warnings)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(epicChargeSharing PRIVATE -Wno-cpp)
    endif()
endif()

# LTO for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _lto_ok)
    if(_lto_ok)
        set_property(TARGET epicChargeSharing PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

# Copy runtime resources
file(GLOB _mac_files "macros/*.mac")
file(GLOB _dat_files "*.dat")
if(_mac_files)
    file(COPY ${_mac_files} DESTINATION ${PROJECT_BINARY_DIR})
endif()
if(_dat_files)
    file(COPY ${_dat_files} DESTINATION ${PROJECT_BINARY_DIR})
endif()

add_custom_target(ePIC DEPENDS epicChargeSharing)

# ============================================================================
# Testing
# ============================================================================
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Developer tools (optional)
# ============================================================================
find_program(CLANG_TIDY_EXE NAMES clang-tidy
    HINTS /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin)
find_program(CLANG_FORMAT_EXE NAMES clang-format
    HINTS /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin)

file(GLOB_RECURSE _all_sources src/*.cc include/*.hh include/*.h epicChargeSharing.cc)

if(CLANG_TIDY_EXE)
    add_custom_target(tidy
        COMMAND ${CLANG_TIDY_EXE} -p ${CMAKE_BINARY_DIR} ${_all_sources}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Running clang-tidy" VERBATIM)
endif()

if(CLANG_FORMAT_EXE)
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXE} -i ${_all_sources}
        COMMENT "Formatting sources" VERBATIM)
    add_custom_target(format-check
        COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror ${_all_sources}
        COMMENT "Checking format" VERBATIM)
endif()

find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/docs/Doxyfile
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
        COMMENT "Generating docs" VERBATIM)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "epicChargeSharing configuration:")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Geant4 MT  : ${Geant4_multithreaded_FOUND}")
message(STATUS "  EDM4hep    : ${WITH_EDM4HEP}")
message(STATUS "  Testing    : ${BUILD_TESTING}")
