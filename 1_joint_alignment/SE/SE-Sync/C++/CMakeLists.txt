cmake_minimum_required(VERSION 3.13)
project(C__)

set(CMAKE_CXX_STANDARD 14)

include_directories(SE-Sync/include)
include_directories(SE-Sync/include/SESync)

add_executable(C__
        cmake-build-debug/CMakeCache.txt
        examples/CMakeLists.txt
        examples/main.cpp
        SE-Sync/include/SESync/RelativePoseMeasurement.h
        SE-Sync/include/SESync/SESync.h
        SE-Sync/include/SESync/SESync_types.h
        SE-Sync/include/SESync/SESync_utils.h
        SE-Sync/include/SESync/SESyncProblem.h
        SE-Sync/include/SESync/StiefelProduct.h
        SE-Sync/src/SESync.cpp
        SE-Sync/src/SESync_utils.cpp
        SE-Sync/src/SESyncProblem.cpp
        SE-Sync/src/StiefelProduct.cpp
        CMakeLists.txt)
