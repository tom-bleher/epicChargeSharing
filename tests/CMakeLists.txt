add_executable(test_charge_sharing_core test_charge_sharing_core.cpp)
target_include_directories(test_charge_sharing_core PRIVATE
    ${CMAKE_SOURCE_DIR}/eicrecon/include)
target_compile_features(test_charge_sharing_core PRIVATE cxx_std_20)

add_test(NAME ChargeSharingCore COMMAND test_charge_sharing_core)
