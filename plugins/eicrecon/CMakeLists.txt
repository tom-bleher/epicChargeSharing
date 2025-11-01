cmake_minimum_required(VERSION 3.16)

project(chargeSharingRecon LANGUAGES CXX)

find_package(JANA2 REQUIRED)
find_package(EDM4HEP REQUIRED)
find_package(EDM4EIC REQUIRED)
find_package(DD4hep REQUIRED)

add_library(chargeSharingRecon SHARED
    src/ChargeSharingReconFactory.cc
    src/ChargeSharingReconPlugin.cc
    src/ChargeSharingReconstructor.cc
)

target_include_directories(chargeSharingRecon
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(chargeSharingRecon
    PRIVATE
        JANA2::jana2
        EDM4HEP::edm4hep
        EDM4EIC::edm4eic
        DD4hep::DDCore
)

set_target_properties(chargeSharingRecon PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

install(TARGETS chargeSharingRecon
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins)

