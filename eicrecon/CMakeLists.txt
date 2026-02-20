cmake_minimum_required(VERSION 3.16)

project(chargeSharingRecon LANGUAGES CXX)

# Add JANA cmake directory to module path (eic-shell installs to non-standard location)
list(APPEND CMAKE_PREFIX_PATH "$ENV{JANA_HOME}/lib/JANA/cmake")

# Find EICrecon which brings in JANA, EDM4hep, EDM4eic, DD4hep, etc.
find_package(EICrecon REQUIRED)
find_package(DD4hep REQUIRED COMPONENTS DDCore)

# Find ROOT for Gaussian fitting and monitoring histograms/trees
find_package(ROOT REQUIRED COMPONENTS Core MathCore Minuit2 Hist Tree)

add_library(chargeSharingRecon SHARED
    src/ChargeSharingReconFactory.cc
    src/ChargeSharingReconPlugin.cc
    src/ChargeSharingReconstructor.cc
    src/ChargeSharingMonitor.cc
)

target_include_directories(chargeSharingRecon
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../core
        ${EICrecon_INCLUDE_DIR}
        ${EICrecon_INCLUDE_DIR}/EICrecon
        # ROOT include dirs propagate automatically via ROOT::* target links
)

target_link_libraries(chargeSharingRecon
    PRIVATE
        ${JANA_LIBRARIES}
        EDM4HEP::edm4hep
        EDM4EIC::edm4eic
        DD4hep::DDCore
        ROOT::Core
        ROOT::MathCore
        ROOT::Minuit2
        ROOT::Hist
        ROOT::Tree
        fmt::fmt
)

set_target_properties(chargeSharingRecon PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    PREFIX ""  # JANA plugins don't use 'lib' prefix
)

install(TARGETS chargeSharingRecon
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)

