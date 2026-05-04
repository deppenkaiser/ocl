#include "ocl/ocl.h"
#include <CL/cl.h>
#include <logging/logging.h>
#include <stdio.h>

bool ocl_initialize(ocl_core_t ocl)
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

bool ocl_compile(ocl_core_t ocl, ocl_program_t program)
{
	bool is_ok = false;
    cl_int error = CL_SUCCESS;

	program->binary = clCreateProgramWithSource(ocl->context, 1, (const char**) &program->source, NULL, &error);
	if (error == CL_SUCCESS)
	{
		error = clBuildProgram(program->binary, 1, ocl->devices.ids, NULL, NULL, NULL);
		if (error == CL_SUCCESS)
		{
			is_ok = true;
		}
		else
		{
			fprintf(stderr, "Fehler: Programm konnte nicht kompiliert werden (%d)\n", error);
			size_t log_size;
			clGetProgramBuildInfo(program->binary, ocl->devices.ids[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			char* log = malloc(log_size);
			clGetProgramBuildInfo(program->binary, ocl->devices.ids[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
			fprintf(stderr, "Build-Log:\n%s\n", log);
			free(log);
		}
	}

	return is_ok;
}
