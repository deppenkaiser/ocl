#pragma once

#include <api/api.h>

protected_import(const char*, ocl_get_source_iwt_flux(void));
protected_import(const char*, ocl_get_source_iwt_q(void));
protected_import(const char*, ocl_get_source_iwt_update_info(void));
protected_import(const char*, ocl_get_source_iwt_apply_fluctuations(void));
protected_import(const char*, ocl_get_source_iwt_mass_charge(void));
protected_import(const char*, ocl_get_source_iwt_redshift_damping(void));
protected_import(const char*, ocl_get_source_iwt_wave_count_points(void));
protected_import(const char*, ocl_get_source_iwt_wave_emit(void));
