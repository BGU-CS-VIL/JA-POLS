# CMake generated Testfile for 
# Source directory: /Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA
# Build directory: /Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(GrassmannAveragesPCA_test-1 "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/build/GrassmannAveragesPCA_test")
set_tests_properties(GrassmannAveragesPCA_test-1 PROPERTIES  _BACKTRACE_TRIPLES "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/CMakeLists.txt;431;add_test;/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/CMakeLists.txt;0;")
if("${CTEST_CONFIGURATION_TYPE}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(GrassmannAveragesPCA_test-2 "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/build/GrassmannAveragesPCA_test_with_files" "--" "--data" "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/mat_test.csv" "--basis_vectors" "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/mat_test_init_vectors.csv" "--expected_result" "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/mat_test_desired_output.csv")
  set_tests_properties(GrassmannAveragesPCA_test-2 PROPERTIES  _BACKTRACE_TRIPLES "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/CMakeLists.txt;448;add_test;/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/CMakeLists.txt;0;")
endif("${CTEST_CONFIGURATION_TYPE}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
