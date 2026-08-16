#include <api/api.h>

// ============================================================================
// IWT-KERNEL: FLUSS (Weber-Kern)
// ============================================================================

protected const char *ocl_get_source_iwt_flux(void)
{
    return
    "__kernel void iwt_flux(\n"
    "    __global const double* I_real,\n"
    "    __global const double* I_imag,\n"
    "    __global const double* Q,\n"
    "    __global const double* K,\n"
    "    __global double* sumJ,\n"
    "    int N,\n"
    "    double DT,\n"
    "    double gamma)           // NEU: fraktale Verstärkung\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    if (i >= N) return;\n"
    "\n"
    "    double Re_i = I_real[i];\n"
    "    double Im_i = I_imag[i];\n"
    "    double abs_i = sqrt(Re_i * Re_i + Im_i * Im_i + 1e-30);\n"
    "    double rho_i = abs_i * abs_i;\n"
    "    double Q_i = Q[i];\n"
    "\n"
    "    double sum = 0.0;\n"
    "\n"
    "    for (int j = 0; j < N; j++)\n"
    "    {\n"
    "        if (j == i) continue;\n"
    "\n"
    "        double Re_j = I_real[j];\n"
    "        double Im_j = I_imag[j];\n"
    "        double abs_j = sqrt(Re_j * Re_j + Im_j * Im_j + 1e-30);\n"
    "        double rho_j = abs_j * abs_j;\n"
    "        double Q_j = Q[j];\n"
    "\n"
    "        double Kij = K[i * N + j];\n"
    "\n"
    "        // ============================================================\n"
    "        // Diffusiver Fluss (bestehend)\n"
    "        // J_diffusive = K * (rho_i - rho_j) * (Q_i - Q_j)\n"
    "        // ============================================================\n"
    "        double J_diffusive = Kij * (rho_i - rho_j) * (Q_i - Q_j);\n"
    "\n"
    "        // ============================================================\n"
    "        // NEU: Fraktale Verstärkung (Kapitel 4, Gleichung 4.6)\n"
    "        // J_nonlinear = gamma * rho_i * (rho_i - rho_j)\n"
    "        // ============================================================\n"
    "        double J_nonlinear = gamma * rho_i * (rho_i - rho_j);\n"
    "\n"
    "        sum += J_diffusive + J_nonlinear;\n"
    "    }\n"
    "\n"
    "    sumJ[i] = sum;\n"
    "}\n";
}

// ============================================================================
// IWT-KERNEL: BOHM-POTENTIAL Q
// ============================================================================

protected const char* ocl_get_source_iwt_q(void)
{
    return
    "__kernel void iwt_q(\n"
    "    __global const double* I_real,\n"
    "    __global const double* I_imag,\n"
    "    __global double* Q,\n"
    "    int N,\n"
    "    double sum_abs_sq,\n"
    "    double hbar,\n"
    "    double m,\n"
    "    double beta,\n"           // NEU: Bohm-Kopplungsstärke
    "    double epsilon,\n"
    "    double Q_min)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    if (i >= N) return;\n"
    "\n"
    "    double Re_i = I_real[i];\n"
    "    double Im_i = I_imag[i];\n"
    "    double abs_i = sqrt(Re_i * Re_i + Im_i * Im_i + 1e-30);\n"
    "    double rho_i = abs_i * abs_i / (sum_abs_sq + 1e-30);\n"
    "\n"
    "    if (rho_i < 1e-30) {\n"
    "        Q[i] = Q_min;\n"
    "        return;\n"
    "    }\n"
    "\n"
    "    double sqrt_rho_i = sqrt(rho_i);\n"
    "\n"
    "    double laplace = 0.0;\n"
    "    for (int j = 0; j < N; j++)\n"
    "    {\n"
    "        if (j == i) continue;\n"
    "        double Re_j = I_real[j];\n"
    "        double Im_j = I_imag[j];\n"
    "        double abs_j = sqrt(Re_j * Re_j + Im_j * Im_j + 1e-30);\n"
    "        double rho_j = abs_j * abs_j / (sum_abs_sq + 1e-30);\n"
    "        if (rho_j < 1e-30) continue;\n"
    "        double sqrt_rho_j = sqrt(rho_j);\n"
    "        laplace += (sqrt_rho_j - sqrt_rho_i);\n"
    "    }\n"
    "\n"
    "    // Bohm-Potential mit beta als Vorfaktor\n"
    "    double prefactor = -beta * (hbar * hbar) / (2.0 * m);\n"
    "    Q[i] = prefactor * laplace / (sqrt_rho_i + epsilon) + Q_min;\n"
    "}\n";
}

// ============================================================================
// IWT-KERNEL: KONTINUITÄTS-UPDATE
// ============================================================================

protected const char* ocl_get_source_iwt_update_info(void)
{
    return
    "__kernel void iwt_update_info(\n"
    "    __global double* I_real,\n"
    "    __global double* I_imag,\n"
    "    __global double* I_phase,\n"
    "    __global const double* sumJ,\n"
    "    __global const double* Q,\n"
    "    int N,\n"
    "    double DT)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    if (i >= N) return;\n"
    "\n"
    "    double Re = I_real[i];\n"
    "    double Im = I_imag[i];\n"
    "    double abs_old = sqrt(Re * Re + Im * Im + 1e-30);\n"
    "    double rho_old = abs_old * abs_old;\n"
    "    double phase_old = I_phase[i];\n"
    "\n"
    "    // ============================================================\n"
    "    // LEAPFROG (VERLET) - SYMPLEKTISCHER INTEGRATOR\n"
    "    // ============================================================\n"
    "\n"
    "    // 1. Halber Schritt für die Phase (mit altem rho)\n"
    "    double phase_half = phase_old + 0.5 * DT * Q[i];\n"
    "\n"
    "    // 2. Vollständiger Schritt für rho (mit phase_half)\n"
    "    //    rho_new = rho_old - DT * flow (flow hängt von phase_half ab)\n"
    "    double flow = sumJ[i];\n"
    "    double rho_new = rho_old - DT * flow;\n"
    "\n"
    "    // 3. Halber Schritt für die Phase (mit rho_new)\n"
    "    //    Q wird mit rho_new neu berechnet (im Kernel muss Q dafür übergeben werden)\n"
    "    //    Hier verwenden wir einfach den alten Q-Wert, weil wir keinen zweiten Q-Kernel aufrufen wollen.\n"
    "    double phase_new = phase_half + 0.5 * DT * Q[i];\n"
    "\n"
    "    // Phasen-Faltung\n"
    "    double PI = 4.0 * atan(1.0);\n"
    "    double twoPI = 2.0 * PI;\n"
    "    phase_new = fmod(phase_new, twoPI);\n"
    "    if (phase_new > PI) phase_new = phase_new - twoPI;\n"
    "    if (phase_new < -PI) phase_new = phase_new + twoPI;\n"
    "\n"
    "    double abs_new = sqrt(fmax(rho_new, 0.0));\n"
    "\n"
    "    I_real[i] = abs_new * cos(phase_new);\n"
    "    I_imag[i] = abs_new * sin(phase_new);\n"
    "    I_phase[i] = phase_new;\n"
    "}\n";
}

// ============================================================================
// IWT-KERNEL: FLUKTUATIONEN ANWENDEN (Anhang P)
// ============================================================================

protected const char* ocl_get_source_iwt_apply_fluctuations(void)
{
    return
    "__kernel void iwt_apply_fluctuations(\n"
    "    __global double* I_real,\n"
    "    __global double* I_imag,\n"
    "    __global const double* xi_real,\n"
    "    __global const double* xi_imag,\n"
    "    int N)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    if (i >= N) return;\n"
	"    I_real[i] += xi_real[i];\n"
    "    I_imag[i] += xi_imag[i];\n"
    "}\n";
}

// ============================================================================
// IWT-KERNEL: MASSE UND LADUNG
// ============================================================================

protected const char* ocl_get_source_iwt_mass_charge(void)
{
    return
    "__kernel void iwt_mass_charge(\n"
    "    __global const double* I_real,\n"
    "    __global const double* I_imag,\n"
    "    __global const double* I_phase,\n"
    "    __global double* mass,\n"
    "    __global double* charge,\n"
    "    int N,\n"
    "    double delta)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    if (i >= N) return;\n"
    "\n"
    "    double Re_i = I_real[i];\n"
    "    double Im_i = I_imag[i];\n"
    "    double phase_i = I_phase[i];\n"
    "\n"
    "    // MASSE (Kapitel 3, Gleichung 3.8)\n"
    "    double mass_sum = 0.0;\n"
    "    double charge_sum = 0.0;\n"
    "\n"
    "    for (int j = -1; j <= 1; j++)\n"
    "    {\n"
    "        if (j == 0) continue;\n"
    "        int n = i + j;\n"
    "        if (n < 0 || n >= N) continue;\n"
    "\n"
    "        double Re_n = I_real[n];\n"
    "        double Im_n = I_imag[n];\n"
    "        double diff_re = Re_i - Re_n;\n"
    "        double diff_im = Im_i - Im_n;\n"
    "        mass_sum += diff_re * diff_re + diff_im * diff_im;\n"
    "\n"
    "        // LADUNG (Kapitel 3, Gleichung 3.4)\n"
    "        double phase_n = I_phase[n];\n"
    "        double diff_phase = phase_n - phase_i;\n"
    "        double PI = 4.0 * atan(1.0);\n"
    "        while (diff_phase > PI) diff_phase -= 2.0 * PI;\n"
    "        while (diff_phase < -PI) diff_phase += 2.0 * PI;\n"
    "        charge_sum += diff_phase;\n"
    "    }\n"
    "\n"
    "    mass[i] = delta * mass_sum;\n"
    "    charge[i] = charge_sum;\n"
    "}\n";
}

// ============================================================================
// IWT-KERNEL: ROTVERSCHIEBUNG ALS ENERGIESSENKE (Anhang Q)
// ============================================================================

protected const char* ocl_get_source_iwt_redshift_damping(void)
{
    return
    "__kernel void iwt_redshift_damping(\n"
    "    __global double* I_real,\n"
    "    __global double* I_imag,\n"
    "    __global double* I_phase,\n"
    "    int N,\n"
    "    double l0,\n"
    "    double D,\n"
    "    double L_Q0,\n"
    "    double delta)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    if (i >= N) return;\n"
    "    double Re_i = I_real[i];\n"
    "    double Im_i = I_imag[i];\n"
    "    double amplitude = sqrt(Re_i * Re_i + Im_i * Im_i + 1e-30);\n"
    "    double mass = 0.0;\n"
    "    if (i > 0) {\n"
    "        double diff_re = Re_i - I_real[i-1];\n"
    "        double diff_im = Im_i - I_imag[i-1];\n"
    "        mass += diff_re * diff_re + diff_im * diff_im;\n"
    "    }\n"
    "    if (i < N - 1) {\n"
    "        double diff_re = Re_i - I_real[i+1];\n"
    "        double diff_im = Im_i - I_imag[i+1];\n"
    "        mass += diff_re * diff_re + diff_im * diff_im;\n"
    "    }\n"
    "    mass *= delta;\n"
    "    double mass_factor = pow(l0 / L_Q0, D - 3.0);\n"
    "    double mass_out = mass * mass_factor;\n"
    "    double phase_in = I_phase[i];\n"
    "    double phase_out = phase_in * (mass / mass_out);\n"
    "    I_real[i] = amplitude * cos(phase_out);\n"
    "    I_imag[i] = amplitude * sin(phase_out);\n"
    "    I_phase[i] = phase_out;\n"
    "}\n";
}
