cmake_minimum_required(VERSION 3.10)

project(ocl LANGUAGES C)

find_package(OpenCL REQUIRED)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_DEBUG")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(ocl STATIC)

target_sources(ocl PRIVATE
    ocl.c
)

target_include_directories(ocl PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../api
    ${CMAKE_CURRENT_LIST_DIR}/../logging
)

target_link_libraries(ocl
	OpenCL::OpenCL
)
