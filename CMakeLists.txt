cmake_minimum_required(VERSION 3.8)
project(StereoMatch LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
  

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

find_package(PythonInterp 3 REQUIRED)

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_torch_dir.py
  OUTPUT_VARIABLE FOUND_TORCH_DIR)

string(STRIP ${FOUND_TORCH_DIR} FOUND_TORCH_DIR)
set(Torch_DIR ${FOUND_TORCH_DIR})

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

find_package(CUDA REQUIRED)
find_package(PythonLibs 3 REQUIRED)
if (SKBUILD)
  find_package(PythonExtensions REQUIRED)
endif(SKBUILD)

execute_process(
	COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_torch_cxx11_abi.py
	OUTPUT_VARIABLE TORCH_CXX11_ABI)
string(STRIP ${TORCH_CXX11_ABI} TORCH_CXX11_ABI)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
set(CMAKE_CXX_FLAGS -fPIC)

include_directories(${CMAKE_SOURCE_DIR}/include)

find_program(iwyu_path NAMES include-what-you-use iwyu)

# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_subdirectory(src)
