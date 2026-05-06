#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/opencl.h>
#include <stdbool.h>

#define OCL_MAX_PLATFORMS 3
#define OCL_MAX_DEVICES 3
#define OCL_MAX_KERNELS 3

typedef struct ocl_platform_info
{
	char name[256];
	char vendor[256];
	char version[256];
} *ocl_platform_info_t;

typedef struct ocl_platforms
{
	cl_platform_id ids[OCL_MAX_PLATFORMS];
	struct ocl_platform_info info[OCL_MAX_PLATFORMS];
	cl_uint count;
} *ocl_platforms_t;

typedef struct ocl_device_info
{
	char name[256];
	char version[256];
} *ocl_device_info_t;

typedef struct ocl_devices
{
    cl_device_id ids[OCL_MAX_DEVICES];
	struct ocl_device_info info[OCL_MAX_DEVICES];
	cl_uint count;
} *ocl_devices_t;

typedef struct ocl_program
{
	cl_program binary;
	const char* source;
	cl_kernel kernels[OCL_MAX_KERNELS];
} *ocl_program_t;

typedef struct ocl_core
{
	struct ocl_platforms platforms;
	struct ocl_devices devices;
	struct ocl_program program;
    cl_context context;
	cl_command_queue queue;
} *ocl_core_t;

typedef struct ocl_image_operation
{
	cl_mem image;
	cl_uint width;
	cl_uint height;
	cl_uint pitch_bytes;
	size_t size_bytes;
} *ocl_image_operation_t;

typedef cl_uint ocl_histogram_t[256];

bool ocl_initialize(ocl_core_t ocl);
bool ocl_compile(ocl_core_t ocl);
void ocl_deinitialize(ocl_core_t ocl);
cl_mem ocl_create_input_buffer_from_memory(ocl_core_t ocl, uint8_t* data, size_t size_bytes);
cl_mem ocl_create_output_buffer(ocl_core_t ocl, size_t size_bytes);
const char* ocl_get_source_subtract_images();
void ocl_set_parameter_subtract_images(cl_kernel kernel, ocl_image_operation_t parameter, cl_mem b, cl_mem result);
void ocl_set_parameter_histogram(cl_kernel kernel, ocl_image_operation_t parameter, cl_mem result);
const char* ocl_get_sources();
bool ocl_load_kernels(ocl_core_t ocl);
