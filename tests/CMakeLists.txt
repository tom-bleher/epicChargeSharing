add_executable(test_charge_sharing_core test_charge_sharing_core.cpp)
target_include_directories(test_charge_sharing_core PRIVATE
    ${CMAKE_SOURCE_DIR}/core)
target_compile_features(test_charge_sharing_core PRIVATE cxx_std_20)

add_test(NAME ChargeSharingCore COMMAND test_charge_sharing_core)

# GaussianFit tests (requires ROOT for the fitting backend)
add_executable(test_gaussian_fit test_gaussian_fit.cpp)
target_include_directories(test_gaussian_fit PRIVATE
    ${CMAKE_SOURCE_DIR}/core)
target_compile_features(test_gaussian_fit PRIVATE cxx_std_20)
target_link_libraries(test_gaussian_fit PRIVATE
    ROOT::Core ROOT::RIO ROOT::Hist ROOT::Minuit ROOT::MathCore ROOT::Matrix)

add_test(NAME GaussianFit COMMAND test_gaussian_fit)
