cmake_minimum_required(VERSION 3.16)
project(pyexuber LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

include(FetchContent)

# Pinned to the exact exubercore tag exuber vendors -- bump both together,
# each as its own reviewed PR (see exubercore/CONTRIBUTING.md once it
# exists).
set(EXUBERCORE_TAG "v0.1.0" CACHE STRING "exubercore tag to build against")
FetchContent_Declare(
  exubercore
  GIT_REPOSITORY https://github.com/kvasilopoulos/exubercore.git
  GIT_TAG ${EXUBERCORE_TAG}
)
set(EXUBERCORE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(exubercore)

# carma vendors its own pybind11 (extern/pybind11) and wires it up via its
# CMakeLists -- pulling that in here rather than a separate
# find_package(pybind11), which conflicted with carma's copy.
FetchContent_Declare(
  carma
  GIT_REPOSITORY https://github.com/RUrlus/carma.git
  GIT_TAG v0.7.0
)
FetchContent_MakeAvailable(carma)

# CMake's FindArmadillo module (what exubercore's CMakeLists uses) finds the
# Armadillo library itself but doesn't propagate its own LAPACK/BLAS
# dependency to consumers -- link them explicitly here, otherwise a static
# libarmadillo.a leaves LAPACK symbols (ssytrf_ etc.) undefined at this
# module's final link step.
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

pybind11_add_module(_core src/bindings.cpp)
target_link_libraries(_core PRIVATE exubercore::exubercore carma::carma ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

install(TARGETS _core DESTINATION exuber)
