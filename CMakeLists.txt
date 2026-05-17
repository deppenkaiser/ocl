cmake_minimum_required(VERSION 3.10)
include_guard(GLOBAL)

project(ocl LANGUAGES C)

find_package(OpenCL REQUIRED)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_DEBUG")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(ocl STATIC)
add_subdirectory(../logging ${CMAKE_BINARY_DIR}/logging_from_ocl)
add_subdirectory(../threading ${CMAKE_BINARY_DIR}/threading_from_ocl)
add_subdirectory(../api	${CMAKE_BINARY_DIR}/api_from_ocl)

target_sources(ocl PRIVATE
	ocl.c
	kernels.c
)

target_include_directories(ocl PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ocl
	OpenCL::OpenCL
	logging
	threading
	api
)
