# ocl – OpenCL-Wrapper

C-Wrapper um OpenCL: Plattform-/Device-Auswahl, Programm-Kompilierung,
Buffer-Erzeugung, Kernel-Enqueue und fachliche Kernels (Bildverarbeitung,
Matrix-Vektor, SD-Attention, IWT-Simulation).

## Umfang

- **Setup**: `ocl_initialize`, `ocl_compile`, `ocl_deinitialize`.
- **Buffer**: `ocl_create_input_buffer_from_memory`, `ocl_create_output_buffer`,
  `ocl_create_buffer` (Read/Write/Read-Write).
- **Kernel**: `ocl_load_kernels`, `ocl_get_kernel`, `ocl_enqueue_kernel`
  (global/local work size), `ocl_finish_frame`.
- **Parameter-Setter** je Kernel: `ocl_set_parameter_*`
  (subtract_images, histogram, brightest_spot, matvec_bf16, …).
- **Kernel-Satz** (`ocl_kernel_t`): u. a. `OCL_KERNEL_IWT_FLUX`,
  `OCL_KERNEL_IWT_Q`, `OCL_KERNEL_IWT_UPDATE_INFO`, `OCL_KERNEL_IWT_MASS_CHARGE`,
  `OCL_KERNEL_IWT_REDSHIFT_DAMPING`, `OCL_KERNEL_IWT_WAVE_EMIT`.

## Nutzung

```c
#include <ocl/ocl.h>

struct ocl_core core = {0};
if (ocl_initialize(&core) == false) return 1;
ocl_compile(&core);
/* ... Kernel-Setter + Enqueue ... */
ocl_deinitialize(&core);
```

## Abhängigkeiten

- `api`, `logging`, `threading`
- System: OpenCL-Header/Library (`CL_TARGET_OPENCL_VERSION` = 300)

## Build

```bash
cmake -S . -B build
cmake --build build
```

In ein Projekt einbinden: `add_subdirectory(../../libraries/ocl …)`,
Einbindung des Headers über den Include-Pfad `<ocl/ocl.h>`.