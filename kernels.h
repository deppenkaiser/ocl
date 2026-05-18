#pragma once

#include <api/api.h>

protected_import(const char*, ocl_get_source_subtract_images(void));
protected_import(const char*, ocl_get_source_histogram(void));
protected_import(const char*, ocl_get_source_brightest_spot(void));
protected_import(const char*, ocl_get_source_matvec_bf16(void));
protected_import(const char*, ocl_get_source_matvec_bf16_fused(void));
protected_import(const char*, ocl_get_source_matvec_f32(void));
