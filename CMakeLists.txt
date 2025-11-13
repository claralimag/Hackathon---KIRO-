cmake_minimum_required(VERSION 3.10)
project(KiroSolver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_BUILD_TYPE Release)

add_executable(kiro_solver kiro_solver.cpp)