#pragma once

#include <api/api.h>

protected_import(const char*, ocl_get_source_subtract_images(void));
protected_import(const char*, ocl_get_source_histogram(void));
protected_import(const char*, ocl_get_source_brightest_spot(void));
protected_import(const char*, ocl_get_source_matvec_bf16(void));
protected_import(const char*, ocl_get_source_matvec_bf16_fused(void));
protected_import(const char*, ocl_get_source_matvec_f32(void));
protected_import(const char*, ocl_get_source_sd_attention_f32(void));
protected_import(const char*, ocl_get_source_sd_output_proj_f32(void));
protected_import(const char*, ocl_get_source_sd_attention_out_f32(void));
protected_import(const char*, ocl_get_source_sd_norm_qkv_f32(void));
protected_import(const char*, ocl_get_source_iwt_flux(void));
protected_import(const char*, ocl_get_source_iwt_q(void));
protected_import(const char*, ocl_get_source_iwt_update_info(void));
protected_import(const char*, ocl_get_source_iwt_apply_fluctuations(void));
protected_import(const char*, ocl_get_source_iwt_mass_charge(void));
protected_import(const char*, ocl_get_source_iwt_redshift_damping(void));
