#include <api/api.h>

/**
 * iwt_kernels.c - OpenCL-Kernel für die IWT
 *
 * THEORIE: Kap. 4 "Diskretes Informations-Lagrange-Funktional"
 *          Anhang P "Vollständige Evolutionsgleichung der IWT"
 *          Anhang R "Phasenquantisierung"
 *
 * Diese Kernel implementieren die diskreten Operatoren der IWT auf der GPU.
 * Sie realisieren die vollständige Evolutionsgleichung (P.3):
 *
 * I_k^(n+1) = I_k^(n)
 *   + T * sum_l w_kl (I_l - I_k)                     // iwt_flux: diffusive + fraktale Verstärkung
 *   + T * lambda * Δ²I_k / I_k                       // iwt_q: Bohm-Potential
 *   + T * mu * I_k * ln(|I_k|/I_0)                   // iwt_flux: nichtlinearer Term
 *   + sqrt(ℏ/(2T)) * ξ_k^(n)                         // iwt_apply_fluctuations
 *
 * Die Kernel arbeiten auf den diskreten Knoten des fraktalen Netzwerks.
 */

/**
 * FLUSS-KERNEL: Berechnet sumJ = diffusive + nichtlineare Verstärkung.
 * THEORIE: Gleichung (P.3), Term 1+3:
 *   Term 1: sum_l w_kl (I_l - I_k)  (Lokaler Weber-Fluss)
 *   Term 3: mu * I_k * ln(|I_k|/I_0) (Nichtlineare Strukturbildung)
 *
 * Die Kopplungsgewichte w_kl sind aus K_kl normiert (Kap. 2, Axiom 4).
 * Der nichtlineare Term realisiert die fraktale Verstärkung (Kap. 4.6).
 */
protected const char *ocl_get_source_iwt_flux(void)
{
    return
    "__kernel void iwt_flux(\n"
    "    __global const double* I_real,\n"
    "    __global const double* I_imag,\n"
    "    __global const double* I_phase,\n"
    "    __global const double* Q,\n"
    "    __global const double* K,\n"
    "    __global double* sumJ,\n"
    "    int N,\n"
    "    double DT,\n"
    "    double gamma,\n"
    "    double kappa)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    if (i >= N) return;\n"
    "\n"
    "    double Re_i = I_real[i];\n"
    "    double Im_i = I_imag[i];\n"
    "    double abs_i = sqrt(Re_i*Re_i + Im_i*Im_i + 1e-30);\n"
    "    double rho_i = abs_i*abs_i;\n"
    "    double Q_i = Q[i];\n"
    "    double S_i = I_phase[i];\n"
    "\n"
    "    double PI = 4.0 * atan(1.0);\n"
    "    double twoPI = 2.0 * PI;\n"
    "\n"
    "    double sum = 0.0;\n"
    "\n"
    "    for (int j = 0; j < N; j++) {\n"
    "        if (j == i) continue;\n"
    "\n"
    "        double Re_j = I_real[j];\n"
    "        double Im_j = I_imag[j];\n"
    "        double abs_j = sqrt(Re_j*Re_j + Im_j*Im_j + 1e-30);\n"
    "        double rho_j = abs_j*abs_j;\n"
    "        double Q_j = Q[j];\n"
    "\n"
    "        double Kij = K[i*N + j];\n"
    "        if (Kij == 0.0) continue;\n"
    "\n"
    "        // Term 1: Diffusiver Fluss (Weber-Dynamik, Kap. 3.4)\n"
    "        double J_diffusive = Kij * (rho_i - rho_j) * (Q_i - Q_j);\n"
    "\n"
    "        // Term 3: Nichtlineare fraktale Verstärkung (Kap. 4.6)\n"
    "        double J_nonlinear = gamma * rho_i * (rho_i - rho_j);\n"
    "\n"
    "        sum += J_diffusive + J_nonlinear;\n"
    "\n"
    "        // ========================================================\n"
    "        // Advektion entlang des Phasengradienten (Fuehrung v = dS/m)\n"
    "        // IWT_NORM: Diskreter Transportterm der Fuehrungsgleichung.\n"
    "        // Theorie: WDBT+ Fuehrung v = grad(S)/m; Anhang P Term 1.\n"
    "        // Implementierung: Upwind-Fluss t = kappa*K*dS (gefaaltet auf\n"
    "        // [-pi,pi]); t>0: Abfluss t*rho_i, t<0: Zugang t*rho_j.\n"
    "        // Paarweise konservativ (Spiegelterm t_ji = -t_ij).\n"
    "        // ========================================================\n"
    "        double dS = I_phase[j] - S_i;\n"
    "        dS -= twoPI * rint(dS / twoPI);\n"
    "        if (fabs(dS) > 1e-12) {\n"
    "            double t = kappa * Kij * dS;\n"
    "            sum += (t > 0.0) ? (t * rho_i) : (t * rho_j);\n"
    "        }\n"
    "    }\n"
    "\n"
    "    sumJ[i] = sum;\n"
    "}\n";
}

/**
 * BOHM-POTENTIAL-KERNEL: Berechnet Q_k.
 * THEORIE: Gleichung (P.3), Term 2: T * lambda * Δ²I_k / I_k
 *
 * Q_k = -beta * hbar²/(2m) * Δ_d² sqrt(|I_k|) / sqrt(|I_k|)  (Kap. 3.3, Kap. 12)
 *
 * Das Bohm-Potential ist der globale, nicht-lokale Anteil der IWT-Dynamik.
 * Es ist die Quelle der Nichtlokalität der Quantenmechanik (Kap. 12).
 */
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
    "    double beta,\n"
    "    double epsilon,\n"
    "    double Q_min)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    if (i >= N) return;\n"
    "\n"
    "    double Re_i = I_real[i];\n"
    "    double Im_i = I_imag[i];\n"
    "    double abs_i = sqrt(Re_i*Re_i + Im_i*Im_i + 1e-30);\n"
    "    double rho_i = abs_i*abs_i / (sum_abs_sq + 1e-30);\n"
    "\n"
    "    if (rho_i < 1e-30) { Q[i] = Q_min; return; }\n"
    "\n"
    "    double sqrt_rho_i = sqrt(rho_i);\n"
    "\n"
    "    // Diskretes Laplace Δ_d² (Kap. 3.3)\n"
    "    double laplace = 0.0;\n"
    "    for (int j = 0; j < N; j++) {\n"
    "        if (j == i) continue;\n"
    "        double Re_j = I_real[j];\n"
    "        double Im_j = I_imag[j];\n"
    "        double abs_j = sqrt(Re_j*Re_j + Im_j*Im_j + 1e-30);\n"
    "        double rho_j = abs_j*abs_j / (sum_abs_sq + 1e-30);\n"
    "        if (rho_j < 1e-30) continue;\n"
    "        double sqrt_rho_j = sqrt(rho_j);\n"
    "        laplace += (sqrt_rho_j - sqrt_rho_i);\n"
    "    }\n"
    "\n"
    "    // Bohm-Potential (Kap. 12, Gleichung 12.1)\n"
    "    double prefactor = -beta * (hbar*hbar) / (2.0 * m);\n"
    "    Q[i] = prefactor * laplace / (sqrt_rho_i + epsilon) + Q_min;\n"
    "}\n";
}

/**
 * KONTINUITÄTS-UPDATE: Leapfrog-Integration.
 * THEORIE: Gleichung (P.3): I_k^(n+1) = I_k^(n) + T * Φ_k
 *
 * Verwendet einen symplektischen Leapfrog-Integrator:
 *   1. phase_half = phase_old + 0.5*DT*Q_i
 *   2. rho_new = rho_old - DT*flow
 *   3. phase_new = phase_half + 0.5*DT*Q_i
 *
 * Die Phase wird auf [-π, π] gefaltet (Anhang R).
 */
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
    "    double abs_old = sqrt(Re*Re + Im*Im + 1e-30);\n"
    "    double rho_old = abs_old*abs_old;\n"
    "    double phase_old = I_phase[i];\n"
    "\n"
    "    // 1. Halber Schritt für die Phase (mit altem rho)\n"
    "    double phase_half = phase_old + 0.5 * DT * Q[i];\n"
    "\n"
    "    // 2. Vollständiger Schritt für rho (mit phase_half)\n"
    "    double flow = sumJ[i];\n"
    "    double rho_new = rho_old - DT * flow;\n"
    "\n"
    "    // 3. Halber Schritt für die Phase (mit rho_new)\n"
    "    double phase_new = phase_half + 0.5 * DT * Q[i];\n"
    "\n"
    "    // Phasen-Faltung (Anhang R)\n"
    "    double PI = 4.0 * atan(1.0);\n"
    "    double twoPI = 2.0 * PI;\n"
    "    phase_new = fmod(phase_new, twoPI);\n"
    "    if (phase_new > PI) phase_new -= twoPI;\n"
    "    if (phase_new < -PI) phase_new += twoPI;\n"
    "\n"
    "    double abs_new = sqrt(fmax(rho_new, 0.0));\n"
    "\n"
    "    I_real[i] = abs_new * cos(phase_new);\n"
    "    I_imag[i] = abs_new * sin(phase_new);\n"
    "    I_phase[i] = phase_new;\n"
    "}\n";
}

/**
 * FLUKTUATIONEN ANWENDEN: Intrinsische Unschärfe.
 * THEORIE: Gleichung (P.3), Term 4: sqrt(ℏ/(2T)) * ξ_k^(n)
 *
 * Die Fluktuation ist KEINE externe Störung, sondern eine Eigenschaft der
 * diskreten Zeit T > 0 (Anhang O). Sie ist die mathematische Manifestation
 * des bandbegrenzten Frequenzspektrums.
 */
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

/**
 * MASSE UND LADUNG: Emergente physikalische Größen.
 * THEORIE: Kap. 3.3 "Information als Ursprung physikalischer Größen"
 *
 * Masse:   m_k = delta * sum_l |I_k - I_l|²   (Gleichung 3.8)
 * Ladung:  q_k = sum_l (phi_k - phi_l)        (Anhang R, Gleichung R.2)
 *
 * Die Ladungsquantisierung ist eine zwingende Konsequenz der diskreten
 * fraktalen Geometrie (Anhang R.5). Die Elementarladung e ist kein freier
 * Parameter, sondern emergiert aus der Raumstruktur (Anhang R.3).
 */
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
    "    int grid_size = (int)sqrt((double)N);\n"
    "    int ix = i % grid_size;\n"
    "    int iy = i / grid_size;\n"
    "\n"
    "    double Re_i = I_real[i];\n"
    "    double Im_i = I_imag[i];\n"
    "    double phase_i = I_phase[i];\n"
    "\n"
    "    double mass_sum = 0.0;\n"
    "    double charge_sum = 0.0;\n"
    "\n"
    "    // x-Richtung (links/rechts)\n"
    "    if (ix > 0) {\n"
    "        int j = i - 1;\n"
    "        double dr = Re_i - I_real[j];\n"
    "        double di = Im_i - I_imag[j];\n"
    "        mass_sum += dr*dr + di*di;\n"
    "        double dphi = I_phase[j] - phase_i;\n"
    "        charge_sum += dphi;\n"
    "    }\n"
    "    if (ix < grid_size - 1) {\n"
    "        int j = i + 1;\n"
    "        double dr = Re_i - I_real[j];\n"
    "        double di = Im_i - I_imag[j];\n"
    "        mass_sum += dr*dr + di*di;\n"
    "        double dphi = I_phase[j] - phase_i;\n"
    "        charge_sum += dphi;\n"
    "    }\n"
    "\n"
    "    // y-Richtung (oben/unten)\n"
    "    if (iy > 0) {\n"
    "        int j = i - grid_size;\n"
    "        double dr = Re_i - I_real[j];\n"
    "        double di = Im_i - I_imag[j];\n"
    "        mass_sum += dr*dr + di*di;\n"
    "        double dphi = I_phase[j] - phase_i;\n"
    "        charge_sum += dphi;\n"
    "    }\n"
    "    if (iy < grid_size - 1) {\n"
    "        int j = i + grid_size;\n"
    "        double dr = Re_i - I_real[j];\n"
    "        double di = Im_i - I_imag[j];\n"
    "        mass_sum += dr*dr + di*di;\n"
    "        double dphi = I_phase[j] - phase_i;\n"
    "        charge_sum += dphi;\n"
    "    }\n"
    "\n"
    "    mass[i] = delta * mass_sum;      // Gleichung 3.8\n"
    "    charge[i] = charge_sum;          // Anhang R, Gleichung R.2\n"
    "}\n";
}

/**
 * ROTVERSCHIEBUNG ALS ENERGIESSENKE (Anhang Q).
 * THEORIE: Gleichung (Q.9): Vakuum -> Fluktuationen -> Materie -> Rotverschiebung -> Vakuum
 *
 * Die Rotverschiebung entfernt Energie aus dem Materiesystem und gibt sie
 * an das Vakuum zurück. Dies schließt den Energiekreislauf und verhindert
 * den Wärmetod (Kap. 13).
 */
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
    "\n"
    "    double Re_i = I_real[i];\n"
    "    double Im_i = I_imag[i];\n"
    "    double amplitude = sqrt(Re_i*Re_i + Im_i*Im_i + 1e-30);\n"
    "\n"
    "    // Lokale Masse (Gleichung 3.8)\n"
    "    double mass = 0.0;\n"
    "    if (i > 0) {\n"
    "        double diff_re = Re_i - I_real[i-1];\n"
    "        double diff_im = Im_i - I_imag[i-1];\n"
    "        mass += diff_re*diff_re + diff_im*diff_im;\n"
    "    }\n"
    "    if (i < N - 1) {\n"
    "        double diff_re = Re_i - I_real[i+1];\n"
    "        double diff_im = Im_i - I_imag[i+1];\n"
    "        mass += diff_re*diff_re + diff_im*diff_im;\n"
    "    }\n"
    "    mass *= delta;\n"
    "\n"
    "    // Massenänderung durch Rotverschiebung (Anhang Q.3)\n"
    "    double mass_factor = pow(l0 / L_Q0, D - 3.0);\n"
    "    double mass_out = mass * mass_factor;\n"
    "\n"
    "    // Phasenänderung (Energieverlust)\n"
    "    double phase_in = I_phase[i];\n"
    "    double phase_out = phase_in * (mass / mass_out);\n"
    "\n"
    "    I_real[i] = amplitude * cos(phase_out);\n"
    "    I_imag[i] = amplitude * sin(phase_out);\n"
    "    I_phase[i] = phase_out;\n"
    "}\n";
}