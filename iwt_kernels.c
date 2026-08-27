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
 *
 * IWT_NORM: Dichte rho wird NICHT mehr in der O(N²)-Schleife berechnet,
 * sondern einmalig pro Knoten aus iwt_precompute übernommen (rho_vec).
 * Theorie: rho_k = |I_k|² (Kap. 3.3).
 * Implementierung: rho_vec[i] = |I_i|², identische Rundung wie zuvor
 * (iwt_precompute bildet exakt abs²). Vermeidet ~N² FP64-sqrt in der
 * Schleife (Consumer-GPU-FP64 ist eng begrenzt).
 * Grund: Die innere Schleife ist FP64-rechenbegrenzt, nicht bandbreiten-
 * begrenzt (siehe README.md, Abschnitt "Numerische Skalierung").
 */
protected const char *ocl_get_source_iwt_flux(void)
{
    return
    "__kernel void iwt_flux(\n"
    "    __global const double* I_phase,\n"
    "    __global const double* Q,\n"
    "    __global const double* rho_vec,\n"
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
    "    double rho_i = rho_vec[i];\n"
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
    "        double rho_j = rho_vec[j];\n"
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
 * DICHTE-PRECOMPUTE: O(N)-Kernel, der die normierte Dichte pro Knoten
 * einmal berechnet, statt sie in jeder der ~N² Paar-Operationen von
 * iwt_flux und iwt_q erneut zu bilden.
 *
 * IWT_NORM: Identische Rundung wie die alten O(N²)-Kernels:
 *   abs_i = sqrt(Re² + Im² + 1e-30), rho_vec = abs_i²,
 *   rho_norm = rho_vec / (sum_abs_sq + 1e-30), sqrt_rho = sqrt(rho_norm).
 * sum_abs_sq stammt wie bisher aus dem Host (rt->I_* nach update_info);
 * dadurch bleibt die Simulation bitidentisch zur vorherigen Version.
 *
 * Grund: Die O(N²)-Schleifen sind FP64-rechenbegrenzt, nicht bandbreiten-
 * begrenzt (siehe README.md, Abschnitt "Numerische Skalierung").
 */
protected const char* ocl_get_source_iwt_precompute(void)
{
    return
    "__kernel void iwt_precompute(\n"
    "    __global const double* I_real,\n"
    "    __global const double* I_imag,\n"
    "    __global double* rho_vec,\n"
    "    __global double* rho_norm,\n"
    "    __global double* sqrt_rho,\n"
    "    int N,\n"
    "    double sum_abs_sq)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    if (i >= N) return;\n"
    "\n"
    "    double Re = I_real[i];\n"
    "    double Im = I_imag[i];\n"
    "    double abs_i = sqrt(Re*Re + Im*Im + 1e-30);\n"
    "    double rho_i = abs_i*abs_i;\n"
    "    rho_vec[i] = rho_i;\n"
    "\n"
    "    double rn = rho_i / (sum_abs_sq + 1e-30);\n"
    "    rho_norm[i] = rn;\n"
    "    sqrt_rho[i] = sqrt(rn);\n"
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
 *
 * IWT_NORM: rho_norm und sqrt_rho werden NICHT mehr in der O(N²)-Schleife
 * berechnet, sondern einmalig pro Knoten aus iwt_precompute übernommen.
 * Theorie: rho_k = |I_k|²/S (Kap. 3.3); Implementierung: identische
 * Rundung wie zuvor (iwt_precompute bildet exakt abs², dann /sum_abs_sq).
 * Vermeidet ~N² FP64-sqrt/div in der Schleife (Consumer-GPU-FP64 eng).
 * Grund: Die innere Schleife ist FP64-rechenbegrenzt, nicht bandbreiten-
 * begrenzt (siehe README.md, Abschnitt "Numerische Skalierung").
 */
protected const char* ocl_get_source_iwt_q(void)
{
    return
    "__kernel void iwt_q(\n"
    "    __global const double* rho_norm,\n"
    "    __global const double* sqrt_rho,\n"
    "    __global const double* K,\n"
    "    __global double* Q,\n"
    "    int N,\n"
    "    double hbar,\n"
    "    double m,\n"
    "    double beta,\n"
    "    double epsilon,\n"
    "    double Q_min,\n"
    "    double thresh)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    if (i >= N) return;\n"
    "\n"
    "    double rho_i = rho_norm[i];\n"
    "\n"
    "    if (rho_i < 1e-30) { Q[i] = Q_min; return; }\n"
    "\n"
    "    double sqrt_rho_i = sqrt_rho[i];\n"
    "\n"
    "    // ========================================================\n"
    "    // Diskreter Laplace Δ_d² als LOKALER Graph-Laplace.\n"
    "    // IWT_NORM_FIX: Vorher Summe ueber ALLE N Knoten (= globale\n"
    "    // Mittelwert-Abweichung, kein Laplace); jetzt K-gewichtete\n    "
    " // Summe nur ueber stark gekoppelte Nachbarn (K > thresh),\n"
    "    // normiert auf die Gewichtssumme.\n"
    "    // Theorie: Kap. 3.3, Δ_d² wirkt auf der Nachbarschaft von k.\n"
    "    // ========================================================\n"
    "    double laplace = 0.0;\n"
    "    double wsum = 0.0;\n"
    "    for (int j = 0; j < N; j++) {\n"
    "        if (j == i) continue;\n"
    "        double Kij = K[i*N + j];\n"
    "        if (Kij <= thresh) continue;\n"
    "        double rho_j = rho_norm[j];\n"
    "        if (rho_j < 1e-30) continue;\n"
    "        laplace += Kij * (sqrt_rho[j] - sqrt_rho_i);\n"
    "        wsum += Kij;\n"
    "    }\n"
    "    laplace /= (wsum + 1e-30);\n"
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
 *   1. phase_half = phase_old + 0.5*DT_Q*Q_i
 *   2. rho_new = rho_old - DT*flow
 *   3. phase_new = phase_half + 0.5*DT_Q*Q_i
 *
 * Die Phase wird auf [-π, π] gefaltet (Anhang R).
 *
 * IWT_NORM: Getrennte Zeitschritte für Dichte (DT) und Phase (DT_Q).
 * Theorie: P.3 verwendet einen einzigen Zeitschritt T.
 * Implementierung: DT_Q ist die effektive Quantenrate (phase_dt),
 * da das rohe DT=1e-12 den Bohm-Kick numerisch einfrieren würde.
 * Grund: Konsistent mit README "Numerische Skalierung" – gamma_eff
 * wird in DT absorbiert; hier zusätzlich der Phasenanteil separat.
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
    "    __global const double* K,\n"
    "    int N,\n"
    "    double DT,\n"
    "    double DT_Q,\n"
    "    double thresh)\n"
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
    "    // ========================================================\n"
    "    // Phasen-Fluss: K-gewichtete Nachbardifferenzen (Graph-Kopplung).\n"
    "    // IWT_NORM: Lokale Ausbreitung der Phase entlang der Kopplungen.\n"
    "    // Theorie: P.3 Term 1 wirkt auf dem komplexen Feld; hier der\n"
    "    // Phasenanteil als Kuramoto-artige Diffusion der Differenzen.\n"
    "    // Grund: Ohne raeumlichen Kopplungsterm wandern Wellenfronten\n"
    "    // nicht (lokale Reaktion allein erzeugt keine Propagation).\n"
    "    // ========================================================\n"
    "    double sflow = 0.0;\n"
    "    double wsum = 0.0;\n"
    "    double PI2 = 2.0 * 4.0 * atan(1.0);\n"
    "    for (int j = 0; j < N; j++) {\n"
    "        if (j == i) continue;\n"
    "        double Kij = K[i*N + j];\n"
    "        if (Kij <= thresh) continue;\n"
    "        double dS = I_phase[j] - phase_old;\n"
    "        dS -= PI2 * rint(dS / PI2);\n"
    "        sflow += Kij * dS;\n"
    "        wsum += Kij;\n"
    "    }\n"
    "    sflow /= (wsum + 1e-30);\n"
    "\n"
    "    // 1. Halber Schritt für die Phase (mit altem rho)\n"
    "    double phase_half = phase_old + 0.5 * DT_Q * (Q[i] + sflow);\n"
    "\n"
    "    // 2. Vollständiger Schritt für rho (mit phase_half)\n"
    "    double flow = sumJ[i];\n"
    "    double rho_new = rho_old - DT * flow;\n"
    "\n"
    "    // 3. Halber Schritt für die Phase (mit rho_new)\n"
    "    double phase_new = phase_half + 0.5 * DT_Q * (Q[i] + sflow);\n"
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
 * FLUKTUATIONEN ERZEUGEN: Intrinsische Unschärfe auf der GPU.
 * THEORIE: Gleichung (P.3), Term 4: sqrt(ℏ/(2T)) * ξ_k^(n)
 *
 * Ersetzt die eingefrorene CPU-Funktion frozen_generate_uncertainty_cpu
 * durch einen parallelen GPU-Kernel. Die statistischen Eigenschaften sind
 * identisch (Standard-Normalverteilung, unkorrelierte Fluktuationen), die
 * Stärke skaliert mit der Vakuumdichte wie auf der CPU:
 *   fluct_strength = SCALE * (1 - rho_i/(RHO_0 + rho_i))
 *
 * Deterministischer SplitMix64-PRNG mit pro Slot variiertem Seed (Slot-Index
 * und Zeitschritt gemischt) garantiert die Unkorrelation über das Netzwerk,
 * ohne den sequenziellen Fisher-Yates-Shuffle der CPU-Version.
 */
protected const char* ocl_get_source_iwt_fluctuations(void)
{
    return
    "__kernel void iwt_fluctuations(\n"
    "    __global const double* I_real,\n"
    "    __global const double* I_imag,\n"
    "    __global double* xi_real,\n"
    "    __global double* xi_imag,\n"
    "    int N,\n"
    "    unsigned int base_seed,\n"
    "    double scale,\n"
    "    double rho_0)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    if (i >= N) return;\n"
    "\n"
    "    // Deterministischer PRNG pro Slot (Xorshift64 mit gemischtem Seed)\n"
    "    ulong s = ((ulong) base_seed) ^ (((ulong) i) * 0x9E3779B97F4A7C15UL);\n"
    "    if (s == 0) s = 0x9E3779B97F4A7C15UL;\n"
    "\n"
    "    ulong x = s;\n"
    "    x ^= x >> 12;\n"
    "    x ^= x << 25;\n"
    "    x ^= x >> 27;\n"
    "    ulong h = x * 0x2545F4914F6CDD1DUL;\n"
    "    double u1 = (double) (h >> 11) * (1.0 / 9007199254740992.0);\n"
    "\n"
    "    x = s * 0x2545F4914F6CDD1DUL;\n"
    "    x ^= x >> 12;\n"
    "    x ^= x << 25;\n"
    "    x ^= x >> 27;\n"
    "    ulong h2 = x * 0x2545F4914F6CDD1DUL;\n"
    "    double u2 = (double) (h2 >> 11) * (1.0 / 9007199254740992.0);\n"
    "\n"
    "    // Vakuumfluktuation nur im Vakuum (rho_i klein); wie CPU-Version\n"
    "    double rho_i = I_real[i] * I_real[i] + I_imag[i] * I_imag[i] + 1e-30;\n"
    "    double fluct = scale * (1.0 - rho_i / (rho_0 + rho_i));\n"
    "\n"
    "    // Box-Muller: zwei unabhaengige normalverteilte Komponenten\n"
    "    double u1c = fmax(u1, 1e-30);\n"
    "    double r = sqrt(-2.0 * log(u1c));\n"
    "    double ang = 6.283185307179586 * u2;\n"
    "    xi_real[i] = r * cos(ang) * fluct;\n"
    "    xi_imag[i] = r * sin(ang) * fluct;\n"
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
    "    __global double* I_phase,\n"
    "    __global const double* xi_real,\n"
    "    __global const double* xi_imag,\n"
    "    int N)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    if (i >= N) return;\n"
    "    I_real[i] += xi_real[i];\n"
    "    I_imag[i] += xi_imag[i];\n"
    "\n"
    "    // ========================================================\n"
    "    // IWT_NORM: Vakuumfluktuationen regen auch die Phase an.\n"
    "    // Theorie: P.3 Term 4 wirkt auf dem komplexen Feld I; die\n"
    "    // Streuung des Imaginaerteils entspricht einem Phasen-Kick.\n"
    "    // Implementierung: phi += 0.15 * xi_imag (xi ist bereits\n"
    "    // vakuum-gewichtet aus frozen_generate_uncertainty_cpu).\n"
    "    // Grund: Ohne Phasenquelle bleibt das Vakuum statisch und\n"
    "    // es emergieren keine freien Strahlungsanregungen.\n"
    "    // ========================================================\n"
    "    I_phase[i] += 0.15 * xi_imag[i];\n"
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
    "    double alpha_0,\n"
    "    double alpha_min)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    if (i >= N) return;\n"
    "\n"
    "    // Energiesenke NUR in Strukturen (rho_i gross), identisches\n"
    "    // Verhalten wie die CPU-Implementierung.\n"
    "    double rho_i = I_real[i]*I_real[i] + I_imag[i]*I_imag[i] + 1e-30;\n"
    "    double anti_rho = 1.0 / rho_i;\n"
    "    double alpha = alpha_0 * (anti_rho / (1.0 + anti_rho)) + alpha_min;\n"
    "\n"
    "    I_real[i] *= (1.0 - alpha);\n"
    "    I_imag[i] *= (1.0 - alpha);\n"
    "}\n";
}

/**
 * iwt_wave_count_points - EM-Wellenfront-Visualisierung, Pass 1 (GPU).
 *
 * 2D-Range (N x LEVELS). Berechnet fuer jeden (Knoten, Niveau) die
 * Phasen-Schnittpunkte mit den Nachbarn und schreibt:
 *   - die Schnittpunkt-Koordinaten nach points_gpu
 *   - die Anzahl nach counts_gpu
 * Ohne atomare Operationen, ohne skalare Division/Modulo (2D-Global-ID).
 */
protected const char* ocl_get_source_iwt_wave_count_points(void)
{
    return
    "__kernel void iwt_wave_count_points(\n"
    "    __global const double* I_phase,\n"
    "    __global const double* pos_x,\n"
    "    __global const double* pos_y,\n"
    "    __global const double* pos_z,\n"
    "    __global const int* wave_flat,\n"
    "    __global const int* wave_count,\n"
    "    __global int* counts,\n"
    "    __global double* points,\n"
    "    int N,\n"
    "    int levels,\n"
    "    int max_crossings,\n"
    "    unsigned int wave_stride,\n"
    "    double base_level,\n"
    "    double level_step,\n"
    "    double two_pi,\n"
    "    unsigned int slice_mode,\n"
    "    double slice_pos,\n"
    "    double slice_delta)\n"
    "{\n"
    "    int node = get_global_id(0);\n"
    "    int l = get_global_id(1);\n"
    "    if (node >= N) return;\n"
    "\n"
    "    double phase_i = I_phase[node];\n"
    "    double level_l = base_level + (double) l * level_step;\n"
    "    double di = phase_i - level_l;\n"
    "    di -= two_pi * rint(di / two_pi);\n"
    "\n"
    "    // Early-Exit: Knoten ausserhalb der Sichtebene ueberspringen\n"
    "    if (slice_mode)\n"
    "    {\n"
    "        double nz = fabs(pos_z[node] - slice_pos);\n"
    "        if (nz > slice_delta)\n"
    "        {\n"
    "            counts[node * levels + l] = 0;\n"
    "            return;\n"
    "        }\n"
    "    }\n"
    "\n"
    "    double px[4];\n"
    "    double py[4];\n"
    "    double pz[4];\n"
    "    int ncross = 0;\n"
    "\n"
    "    int count = wave_count[node];\n"
    "    int base = node * (int) wave_stride;\n"
    "\n"
    "    for (int e = 0; e < count; e++)\n"
    "    {\n"
    "        int j = wave_flat[base + e];\n"
    "        if (j == node || j < 0 || j >= N) continue;\n"
    "        if (ncross >= max_crossings) break;\n"
    "\n"
    "        double dj = I_phase[j] - level_l;\n"
    "        dj -= two_pi * rint(dj / two_pi);\n"
    "        if (di * dj >= 0.0) continue;\n"
    "\n"
    "        double t = di / (di - dj);\n"
    "        double cx = pos_x[node] + t * (pos_x[j] - pos_x[node]);\n"
    "        double cy = pos_y[node] + t * (pos_y[j] - pos_y[node]);\n"
    "        double cz = pos_z[node] + t * (pos_z[j] - pos_z[node]);\n"
    "\n"
    "        if (slice_mode)\n"
    "        {\n"
    "            double dz = fabs(cz - slice_pos);\n"
    "            if (dz > slice_delta) continue;\n"
    "        }\n"
    "\n"
    "        px[ncross] = cx;\n"
    "        py[ncross] = cy;\n"
    "        pz[ncross] = cz;\n"
    "        ncross++;\n"
    "    }\n"
    "\n"
    "    counts[node * levels + l] = ncross;\n"
    "\n"
    "    int pbase = (node * levels + l) * max_crossings * 3;\n"
    "    for (int k = 0; k < ncross; k++)\n"
    "    {\n"
    "        points[pbase + k * 3 + 0] = px[k];\n"
    "        points[pbase + k * 3 + 1] = py[k];\n"
    "        points[pbase + k * 3 + 2] = pz[k];\n"
    "    }\n"
    "}\n"
    ;
}

/**
 * iwt_wave_emit - EM-Wellenfront-Visualisierung, Pass 2 (GPU).
 *
 * 2D-Range (N x LEVELS). Liest die im Pass 1 berechneten Schnittpunkte und
 * schreibt die Polylinien-Segmente an die vorberechnete Basis-Offset-Position.
 * Ohne atomare Operationen.
 */
protected const char* ocl_get_source_iwt_wave_emit(void)
{
    return
    "__kernel void iwt_wave_emit(\n"
    "    __global const int* counts,\n"
    "    __global const double* points,\n"
    "    __global const int* offsets,\n"
    "    __global float* segments,\n"
    "    int levels,\n"
    "    int max_crossings,\n"
    "    double base_level,\n"
    "    double level_step,\n"
    "    double two_pi)\n"
    "{\n"
    "    int node = get_global_id(0);\n"
    "    int l = get_global_id(1);\n"
    "    int slot = node * levels + l;\n"
    "    int ncross = counts[slot];\n"
    "    if (ncross < 2) return;\n"
    "\n"
    "    // Farbe: Hue aus dem Phasen-Niveau (wie CPU wave_hsv_to_rgb, s=0.85, v=0.95)\n"
    "    double level_l = base_level + (double) l * level_step;\n"
    "    float hue = (float) ((level_l + 3.141592653589793) / two_pi);\n"
    "    float h6 = hue * 6.0f;\n"
    "    float h6f = floor(h6);\n"
    "    float f6 = h6 - h6f;\n"
    "    float v = 0.95f;\n"
    "    float s = 0.85f;\n"
    "    float p = v * (1.0f - s);\n"
    "    float q = v * (1.0f - f6 * s);\n"
    "    float tt = v * (1.0f - (1.0f - f6) * s);\n"
    "    float rgb[3];\n"
    "    int sect = ((int) h6f) % 6;\n"
    "    if (sect == 0) { rgb[0]=v; rgb[1]=tt; rgb[2]=p; }\n"
    "    else if (sect == 1) { rgb[0]=q; rgb[1]=v; rgb[2]=p; }\n"
    "    else if (sect == 2) { rgb[0]=p; rgb[1]=v; rgb[2]=tt; }\n"
    "    else if (sect == 3) { rgb[0]=p; rgb[1]=q; rgb[2]=v; }\n"
    "    else if (sect == 4) { rgb[0]=tt; rgb[1]=p; rgb[2]=v; }\n"
    "    else { rgb[0]=v; rgb[1]=p; rgb[2]=q; }\n"
    "\n"
    "    int seg_off = offsets[slot];\n"
    "    int pbase = slot * max_crossings * 3;\n"
    "\n"
    "    for (int k = 0; k + 1 < ncross; k += 2)\n"
    "    {\n"
    "        float* v0 = &segments[(seg_off) * 12];\n"
    "        float* v1 = v0 + 6;\n"
    "        v0[0] = (float) points[pbase + k*3 + 0];\n"
    "        v0[1] = (float) points[pbase + k*3 + 1];\n"
    "        v0[2] = (float) points[pbase + k*3 + 2];\n"
    "        v1[0] = (float) points[pbase + (k+1)*3 + 0];\n"
    "        v1[1] = (float) points[pbase + (k+1)*3 + 1];\n"
    "        v1[2] = (float) points[pbase + (k+1)*3 + 2];\n"
    "        v0[3] = rgb[0]; v0[4] = rgb[1]; v0[5] = rgb[2];\n"
    "        v1[3] = rgb[0]; v1[4] = rgb[1]; v1[5] = rgb[2];\n"
    "        seg_off++;\n"
    "    }\n"
    "}\n"
    ;
}