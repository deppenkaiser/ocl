#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/opencl.h>

#define OCL_MAX_PLATFORMS 3
#define OCL_MAX_DEVICES 3

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

typedef struct ocl_devices
{
    cl_device_id ids[OCL_MAX_DEVICES];
	cl_uint count;
} *ocl_devices_t;

typedef struct ocl_core
{
	struct ocl_platforms platforms;
	struct ocl_devices devices;
} *ocl_core_t;
