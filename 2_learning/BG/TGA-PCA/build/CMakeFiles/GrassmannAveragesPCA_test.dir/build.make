# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.15.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.15.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/build"

# Include any dependencies generated for this target.
include CMakeFiles/GrassmannAveragesPCA_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/GrassmannAveragesPCA_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GrassmannAveragesPCA_test.dir/flags.make

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_main.cpp.o: CMakeFiles/GrassmannAveragesPCA_test.dir/flags.make
CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_main.cpp.o: ../test/test_main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_main.cpp.o -c "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_main.cpp"

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_main.cpp" > CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_main.cpp.i

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_main.cpp" -o CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_main.cpp.s

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca.cpp.o: CMakeFiles/GrassmannAveragesPCA_test.dir/flags.make
CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca.cpp.o: ../test/test_grassmannpca.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca.cpp.o -c "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_grassmannpca.cpp"

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_grassmannpca.cpp" > CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca.cpp.i

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_grassmannpca.cpp" -o CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca.cpp.s

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca_trimming.cpp.o: CMakeFiles/GrassmannAveragesPCA_test.dir/flags.make
CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca_trimming.cpp.o: ../test/test_grassmannpca_trimming.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca_trimming.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca_trimming.cpp.o -c "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_grassmannpca_trimming.cpp"

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca_trimming.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca_trimming.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_grassmannpca_trimming.cpp" > CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca_trimming.cpp.i

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca_trimming.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca_trimming.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_grassmannpca_trimming.cpp" -o CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca_trimming.cpp.s

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_simplepca.cpp.o: CMakeFiles/GrassmannAveragesPCA_test.dir/flags.make
CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_simplepca.cpp.o: ../test/test_simplepca.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_simplepca.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_simplepca.cpp.o -c "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_simplepca.cpp"

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_simplepca.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_simplepca.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_simplepca.cpp" > CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_simplepca.cpp.i

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_simplepca.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_simplepca.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_simplepca.cpp" -o CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_simplepca.cpp.s

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_row_proxy.cpp.o: CMakeFiles/GrassmannAveragesPCA_test.dir/flags.make
CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_row_proxy.cpp.o: ../test/test_row_proxy.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_row_proxy.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_row_proxy.cpp.o -c "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_row_proxy.cpp"

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_row_proxy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_row_proxy.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_row_proxy.cpp" > CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_row_proxy.cpp.i

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_row_proxy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_row_proxy.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_row_proxy.cpp" -o CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_row_proxy.cpp.s

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_k_first.cpp.o: CMakeFiles/GrassmannAveragesPCA_test.dir/flags.make
CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_k_first.cpp.o: ../test/test_k_first.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_k_first.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_k_first.cpp.o -c "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_k_first.cpp"

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_k_first.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_k_first.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_k_first.cpp" > CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_k_first.cpp.i

CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_k_first.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_k_first.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/test/test_k_first.cpp" -o CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_k_first.cpp.s

# Object files for target GrassmannAveragesPCA_test
GrassmannAveragesPCA_test_OBJECTS = \
"CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_main.cpp.o" \
"CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca.cpp.o" \
"CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca_trimming.cpp.o" \
"CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_simplepca.cpp.o" \
"CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_row_proxy.cpp.o" \
"CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_k_first.cpp.o"

# External object files for target GrassmannAveragesPCA_test
GrassmannAveragesPCA_test_EXTERNAL_OBJECTS =

GrassmannAveragesPCA_test: CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_main.cpp.o
GrassmannAveragesPCA_test: CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca.cpp.o
GrassmannAveragesPCA_test: CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_grassmannpca_trimming.cpp.o
GrassmannAveragesPCA_test: CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_simplepca.cpp.o
GrassmannAveragesPCA_test: CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_row_proxy.cpp.o
GrassmannAveragesPCA_test: CMakeFiles/GrassmannAveragesPCA_test.dir/test/test_k_first.cpp.o
GrassmannAveragesPCA_test: CMakeFiles/GrassmannAveragesPCA_test.dir/build.make
GrassmannAveragesPCA_test: libgrassmann_averages.a
GrassmannAveragesPCA_test: /Applications/boost_1_71_0/stage/lib/libboost_unit_test_framework.a
GrassmannAveragesPCA_test: /Applications/boost_1_71_0/stage/lib/libboost_chrono.a
GrassmannAveragesPCA_test: /Applications/boost_1_71_0/stage/lib/libboost_system.a
GrassmannAveragesPCA_test: /Applications/boost_1_71_0/stage/lib/libboost_thread.a
GrassmannAveragesPCA_test: /Applications/boost_1_71_0/stage/lib/libboost_date_time.a
GrassmannAveragesPCA_test: CMakeFiles/GrassmannAveragesPCA_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable GrassmannAveragesPCA_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GrassmannAveragesPCA_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GrassmannAveragesPCA_test.dir/build: GrassmannAveragesPCA_test

.PHONY : CMakeFiles/GrassmannAveragesPCA_test.dir/build

CMakeFiles/GrassmannAveragesPCA_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/GrassmannAveragesPCA_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/GrassmannAveragesPCA_test.dir/clean

CMakeFiles/GrassmannAveragesPCA_test.dir/depend:
	cd "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA" "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA" "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/build" "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/build" "/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/2_learning/BG/TGA-PCA/build/CMakeFiles/GrassmannAveragesPCA_test.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/GrassmannAveragesPCA_test.dir/depend

