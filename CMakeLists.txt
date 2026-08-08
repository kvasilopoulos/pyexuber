cmake_minimum_required(VERSION 3.16)
project(pyexuber LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# exubercore builds as a static library by default (fine for its own
# standalone use/tests), but we link it into this Python extension module,
# which is a shared object -- static libs pulled into a .so need PIC on
# Linux or the final link fails with "recompile with -fPIC". Set before
# fetching exubercore so its target picks this up as its default.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(pybind11 CONFIG REQUIRED)

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

# CMake's FindArmadillo module (what exubercore's CMakeLists uses) finds the
# Armadillo library itself but doesn't propagate its own LAPACK/BLAS
# dependency to consumers -- link them explicitly here, otherwise a static
# libarmadillo.a leaves LAPACK symbols (ssytrf_ etc.) undefined at this
# module's final link step.
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

pybind11_add_module(_core src/bindings.cpp)
target_link_libraries(_core PRIVATE exubercore::exubercore ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

install(TARGETS _core DESTINATION exuber)
