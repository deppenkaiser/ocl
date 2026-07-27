#include "ocl/ocl.h"
#include "kernels.h"

#include <CL/cl.h>
#include <logging/logging.h>
#include <stdio.h>
#include <api/api.h>
#include <string.h>

const char *_ocl_kernel_names[OCL_KERNEL_COUNT] =
{
    "subtract_images",
    "histogram",
    "brightest_spot",
    "matvec_bf16",
    "matvec_bf16_fused",
    "matvec_f32",
    "sd_attention_f32",
    "sd_output_proj_f32",
    "sd_attention_out_f32",
    "sd_norm_qkv_f32",
	"iwt_flux",
	"iwt_q",
	"iwt_update_info",
	"iwt_apply_fluctuations",
	"iwt_mass_charge",
	"iwt_boundary_damping"
};

bool ocl_initialize(const ocl_core_t ocl)
{
    bool is_ok = false;

    if (clGetPlatformIDs(OCL_MAX_PLATFORMS, ocl->platforms.ids, &ocl->platforms.count) == CL_SUCCESS)
    {
        logging_log_message("OpenCL platforms found:");

        for (size_t i = 0; i < ocl->platforms.count; ++i)
        {
            if (clGetPlatformInfo(ocl->platforms.ids[0], CL_PLATFORM_NAME, sizeof(ocl->platforms.info[i].name), ocl->platforms.info[i].name, NULL) == CL_SUCCESS)
            {
                logging_log_message(ocl->platforms.info[i].name);
            }

            if (clGetPlatformInfo(ocl->platforms.ids[0], CL_PLATFORM_VENDOR, sizeof(ocl->platforms.info[i].vendor), ocl->platforms.info[i].vendor, NULL) == CL_SUCCESS)
            {
                logging_log_message(ocl->platforms.info[i].vendor);
            }

            if (clGetPlatformInfo(ocl->platforms.ids[0], CL_PLATFORM_VERSION, sizeof(ocl->platforms.info[i].version), ocl->platforms.info[i].version, NULL) == CL_SUCCESS)
            {
                logging_log_message(ocl->platforms.info[i].version);
            }
        }

        if (clGetDeviceIDs(ocl->platforms.ids[0], CL_DEVICE_TYPE_GPU, 1, ocl->devices.ids, &ocl->devices.count) == CL_SUCCESS)
        {
            cl_int error = CL_SUCCESS;

            logging_log_message("OpenCL devices found:");

            for (size_t i = 0; i < ocl->devices.count; ++i)
            {
                if (clGetDeviceInfo(ocl->devices.ids[i], CL_DEVICE_NAME, sizeof(ocl->devices.info[i].name), ocl->devices.info[i].name, NULL) == CL_SUCCESS)
                {
                    logging_log_message(ocl->devices.info[i].name);
                }
            }

            ocl->context = clCreateContext(NULL, 1, ocl->devices.ids, NULL, NULL, &error);
            if (error == CL_SUCCESS)
            {
                const cl_queue_properties properties[] = {0};
                ocl->queue = clCreateCommandQueueWithProperties(ocl->context, ocl->devices.ids[0], properties, &error);
                if (error == CL_SUCCESS)
                {
                    is_ok = true;
                }
            }
        }
    }
    
    return is_ok;
}

void ocl_deinitialize(const ocl_core_t ocl)
{
    for (size_t i = 0; i < OCL_MAX_KERNELS; ++i)
    {
        if (ocl->program.kernels[i] != NULL)
        {
            clReleaseKernel(ocl->program.kernels[i]);
            ocl->program.kernels[i] = NULL;
        }
    }

    if (ocl->program.binary != NULL)
    {
        clReleaseProgram(ocl->program.binary);
        ocl->program.binary = NULL;
    }

    if (ocl->queue != NULL)
    {
        clReleaseCommandQueue(ocl->queue);
        ocl->queue = NULL;
    }

    if (ocl->context != NULL)
    {
        clReleaseContext(ocl->context);
        ocl->context = NULL;
    }

    logging_log_message("OpenCL is deinitialized.");
}

cl_mem ocl_create_input_buffer_from_memory(const ocl_core_t ocl, uint8_t* data, size_t size_bytes)
{
    cl_int error = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_bytes, data, &error);
    if (error != CL_SUCCESS)
    {
        logging_log_message("Error: Buffer creation failed!");
    }
    return buffer;
}

cl_mem ocl_create_output_buffer(const ocl_core_t ocl, size_t size_bytes)
{
    cl_int error = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY, size_bytes, NULL, &error);
    if (error != CL_SUCCESS)
    {
        logging_log_message("Error: Buffer creation failed!");
    }
    return buffer;
}

cl_mem ocl_create_buffer(const ocl_core_t ocl, ocl_buf_type_t type, size_t size_bytes, void *host_ptr)
{
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    if (type == OCL_BUF_READ_ONLY) flags = CL_MEM_READ_ONLY | (host_ptr ? CL_MEM_COPY_HOST_PTR : 0);
    if (type == OCL_BUF_WRITE_ONLY) flags = CL_MEM_WRITE_ONLY;
    if (type == OCL_BUF_READ_WRITE) flags = CL_MEM_READ_WRITE | (host_ptr ? CL_MEM_COPY_HOST_PTR : 0);

    cl_int error = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(ocl->context, flags, size_bytes, host_ptr, &error);
    if (error != CL_SUCCESS) logging_log_message("ocl_create_buffer failed");
    return buffer;
}

bool ocl_enqueue_kernel(const ocl_core_t ocl, cl_kernel kernel, size_t global_work_size, size_t local_work_size)
{
    cl_int error = clEnqueueNDRangeKernel(ocl->queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    if (error != CL_SUCCESS) { logging_log_message("clEnqueueNDRangeKernel failed"); return false; }
    error = clFinish(ocl->queue);
    if (error != CL_SUCCESS) { logging_log_message("clFinish failed"); return false; }
    return true;
}

cl_kernel ocl_get_kernel(const ocl_core_t ocl, ocl_kernel_t kernel)
{
    if (kernel < OCL_KERNEL_COUNT) return ocl->program.kernels[kernel];
    return NULL;
}

bool ocl_load_kernels(const ocl_core_t ocl)
{
    bool is_ok = true;
    for (int i = 0; i < OCL_KERNEL_COUNT; ++i)
    {
        cl_int error = CL_SUCCESS;
        ocl->program.kernels[i] = clCreateKernel(ocl->program.binary, _ocl_kernel_names[i], &error);
        if (error != CL_SUCCESS)
        {
            fprintf(stderr, "Fehler: Kernel '%s' konnte nicht erstellt werden (Fehler %d)\n",
                    _ocl_kernel_names[i], error);
            is_ok = false;
            break;
        }
    }
    return is_ok;
}

static char *_ocl_build_source(void)
{
    const char *parts[] =
    {
        ocl_get_source_subtract_images(),
        ocl_get_source_histogram(),
        ocl_get_source_brightest_spot(),
        ocl_get_source_matvec_bf16(),
        ocl_get_source_matvec_bf16_fused(),
        ocl_get_source_matvec_f32(),
        ocl_get_source_sd_attention_f32(),
        ocl_get_source_sd_output_proj_f32(),
        ocl_get_source_sd_attention_out_f32(),
        ocl_get_source_sd_norm_qkv_f32(),
		ocl_get_source_iwt_flux(),
		ocl_get_source_iwt_q(),
		ocl_get_source_iwt_update_info(),
		ocl_get_source_iwt_apply_fluctuations(),
		ocl_get_source_iwt_mass_charge(),
		ocl_get_source_iwt_boundary_damping(),
        NULL
    };

    size_t total = 0;
    for (int i = 0; parts[i]; ++i) total += strlen(parts[i]);

    char *source = (char *)malloc(total + 1);
    if (!source) return NULL;
    source[0] = '\0';
    for (int i = 0; parts[i]; ++i) strcat(source, parts[i]);

    return source;
}

const char *ocl_get_sources(void)
{
    static char *cached = NULL;
    free(cached);
    cached = _ocl_build_source();
    return cached;
}

bool ocl_compile(const ocl_core_t ocl)
{
    bool is_ok = false;
    cl_int error = CL_SUCCESS;

    char *source = _ocl_build_source();
    if (!source) return false;

    ocl->program.binary = clCreateProgramWithSource(ocl->context, 1, (const char **)&source, NULL, &error);
    free(source);

    if (error == CL_SUCCESS)
    {
        error = clBuildProgram(ocl->program.binary, 1, ocl->devices.ids, NULL, NULL, NULL);
        if (error == CL_SUCCESS)
        {
            is_ok = true;
            logging_log_message("OpenCL is initialized.");
        }
        else
        {
            fprintf(stderr, "Fehler: Programm konnte nicht kompiliert werden (%d)\n", error);
            size_t log_size;
            clGetProgramBuildInfo(ocl->program.binary, ocl->devices.ids[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char* log = malloc(log_size);
            clGetProgramBuildInfo(ocl->program.binary, ocl->devices.ids[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Build-Log:\n%s\n", log);
            free(log);
        }
    }

    return is_ok;
}

void ocl_finish_frame(const ocl_core_t ocl)
{
    clFinish(ocl->queue);
}
