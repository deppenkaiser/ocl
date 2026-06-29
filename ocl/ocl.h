#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/opencl.h>
#include <stdbool.h>

#define OCL_MAX_PLATFORMS 3
#define OCL_MAX_DEVICES 3

typedef enum
{
	OCL_KERNEL_SUBTRACT_IMAGES = 0,
	OCL_KERNEL_HISTOGRAM,
	OCL_KERNEL_BRIGHTEST_SPOT,
	OCL_KERNEL_MATVEC_BF16,
	OCL_KERNEL_MATVEC_BF16_FUSED,
	OCL_KERNEL_MATVEC_F32,
	OCL_KERNEL_SD_ATTENTION_F32,
	OCL_KERNEL_SD_OUTPUT_PROJ_F32,
	OCL_KERNEL_SD_ATTENTION_OUT_F32,
	OCL_KERNEL_SD_NORM_QKV_F32,
	OCL_KERNEL_IWT_UPDATE,
	OCL_KERNEL_COUNT
} ocl_kernel_t;

#define OCL_MAX_KERNELS OCL_KERNEL_COUNT

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

/* Buffer-Typen */
typedef enum
{
	OCL_BUF_READ_ONLY,
	OCL_BUF_WRITE_ONLY,
	OCL_BUF_READ_WRITE
} ocl_buf_type_t;

bool ocl_initialize(const ocl_core_t ocl);
bool ocl_compile(const ocl_core_t ocl);
void ocl_deinitialize(const ocl_core_t ocl);
cl_mem ocl_create_input_buffer_from_memory(const ocl_core_t ocl, uint8_t* data, size_t size_bytes);
cl_mem ocl_create_output_buffer(const ocl_core_t ocl, size_t size_bytes);
const char* ocl_get_sources();
bool ocl_load_kernels(const ocl_core_t ocl);
cl_mem ocl_create_buffer(const ocl_core_t ocl, ocl_buf_type_t type, size_t size_bytes, void *host_ptr);
bool ocl_enqueue_kernel(const ocl_core_t ocl, cl_kernel kernel, size_t global_work_size, size_t local_work_size);
cl_kernel ocl_get_kernel(const ocl_core_t ocl, ocl_kernel_t kernel);
void ocl_finish_frame(const ocl_core_t ocl);

void ocl_set_parameter_subtract_images(const cl_kernel kernel, const ocl_image_operation_t parameter, cl_mem b, cl_mem result);
void ocl_set_parameter_histogram(const cl_kernel kernel, const ocl_image_operation_t parameter, cl_mem result);
void ocl_set_parameter_brightest_spot(const cl_kernel kernel, const ocl_image_operation_t parameter, int cx, int cy, int rw, int rh, int sub_r, cl_mem result);
void ocl_set_parameter_matvec_bf16(const cl_kernel kernel, cl_mem y, cl_mem x, cl_mem W, int in_dim, int out_dim);
void ocl_set_parameter_iwt_update(const cl_kernel kernel, cl_mem nodes, float D, float l0, uint32_t num_nodes, float dt);
