/* ═══════════════════════════════════════════════════════════════════════════
 * L104 NANO DAEMON — C Implementation v1.0.0
 * Atomized Fault Detection at Native Speed
 * GOD_CODE=527.5184818492612 | PHI=1.618033988749895 | PILOT: LONDEL
 *
 * Eight nano-probes detect sub-bit-level flaws invisible to higher layers:
 *   1. Sacred Constant ULP Drift       5. Entropy Source Health
 *   2. Memory Canary Integrity          6. Numerical Stability Audit
 *   3. FPU Environment Flags            7. Phase Drift Accumulation
 *   4. Cache Line Coherence             8. NDE Pipeline Integrity
 *
 * IPC: /tmp/l104_bridge/nano/c_outbox (JSON reports)
 * Heartbeat: /tmp/l104_bridge/nano/c_heartbeat
 * PID: /tmp/l104_bridge/nano/c_nano.pid
 *
 * Build: make nano-daemon   (in l104_core_c/)
 * Run:   ./build/l104_nano_daemon [--self-test | --tick <ms>]
 * ═══════════════════════════════════════════════════════════════════════════ */

#include "l104_nano_daemon.h"
#include "l104_sage_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <fenv.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <errno.h>
#include <fcntl.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#include <mach/mach.h>
#include <mach/host_info.h>
#include <sys/sysctl.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * INTERNAL — Sacred constants in volatile to prevent optimizer elimination
 * ═══════════════════════════════════════════════════════════════════════════ */

static volatile double g_god_code     = L104_NANO_GOD_CODE;
static volatile double g_phi          = L104_NANO_PHI;
static volatile double g_void_const   = L104_NANO_VOID_CONSTANT;

static volatile int g_shutdown_requested = 0;
static volatile int g_force_tick = 0;
static volatile int g_dump_status = 0;
static volatile int g_reload_requested = 0;

/* Pointer to daemon state for signal-driven status dump */
static l104_nano_daemon_state_t *g_daemon_state = NULL;

/* Cache line padding to detect false sharing */
typedef struct {
    double value;
    char _pad[56]; /* Total 64 bytes = typical cache line */
} __attribute__((aligned(64))) cache_line_t;

static cache_line_t g_cache_god   __attribute__((aligned(64)));
static cache_line_t g_cache_phi   __attribute__((aligned(64)));
static cache_line_t g_cache_void  __attribute__((aligned(64)));

/* IPC paths */
static const char *NANO_IPC_BASE     = "/tmp/l104_bridge/nano";
static const char *NANO_C_OUTBOX     = "/tmp/l104_bridge/nano/c_outbox";
static const char *NANO_C_HEARTBEAT  = "/tmp/l104_bridge/nano/c_heartbeat";
static const char *NANO_C_PID        = "/tmp/l104_bridge/nano/c_nano.pid";

/* ═══════════════════════════════════════════════════════════════════════════
 * UTILITY — Bit-level operations
 * ═══════════════════════════════════════════════════════════════════════════ */

uint64_t l104_double_to_bits(double v) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return bits;
}

static double bits_to_double(uint64_t bits) {
    double v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

int64_t l104_ulp_distance(double a, double b) {
    if (isnan(a) || isnan(b)) return INT64_MAX;
    if (a == b) return 0;

    uint64_t a_bits = l104_double_to_bits(a);
    uint64_t b_bits = l104_double_to_bits(b);

    /* Handle sign difference */
    if ((a_bits >> 63) != (b_bits >> 63)) {
        /* Different signs — distance is sum from each side of zero */
        return (int64_t)(a_bits & 0x7FFFFFFFFFFFFFFFULL) +
               (int64_t)(b_bits & 0x7FFFFFFFFFFFFFFFULL);
    }

    int64_t diff = (int64_t)a_bits - (int64_t)b_bits;
    return diff < 0 ? -diff : diff;
}

int l104_hamming_distance(uint64_t a, uint64_t b) {
    uint64_t x = a ^ b;
    int count = 0;
    while (x) {
        count += (int)(x & 1);
        x >>= 1;
    }
    return count;
}

uint64_t l104_nano_timestamp(void) {
#ifdef __APPLE__
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return mach_absolute_time() * tb.numer / tb.denom;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static void ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        mkdir(path, 0755);
    }
}

static void add_fault(l104_nano_fault_t *faults, int *count, int max,
                       l104_nano_fault_type_t type, l104_nano_severity_t sev,
                       double measured, double expected, double deviation,
                       int ulp_dist, const char *desc) {
    if (*count >= max) return;
    l104_nano_fault_t *f = &faults[*count];
    f->type = type;
    f->severity = sev;
    f->measured_value = measured;
    f->expected_value = expected;
    f->deviation = deviation;
    f->timestamp_ns = l104_nano_timestamp();
    f->ulp_distance = ulp_dist;
    strncpy(f->description, desc, sizeof(f->description) - 1);
    f->description[sizeof(f->description) - 1] = '\0';
    (*count)++;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PROBE 1: Sacred Constant ULP Drift Detection
 * Compares volatile runtime values against IEEE 754 bit-exact references.
 * Even 1 ULP drift indicates memory corruption or FPU misconfiguration.
 * ═══════════════════════════════════════════════════════════════════════════ */

int l104_nano_probe_constant_drift(l104_nano_fault_t *faults, int max_faults) {
    int count = 0;

    /* GOD_CODE check */
    double gc = g_god_code;
    uint64_t gc_bits = l104_double_to_bits(gc);
    int64_t gc_ulp = l104_ulp_distance(gc, L104_NANO_GOD_CODE);

    if (gc_bits != L104_GOD_CODE_BITS || gc_ulp != 0) {
        l104_nano_severity_t sev = gc_ulp <= 2 ? NANO_SEV_LOW :
                                    gc_ulp <= 8 ? NANO_SEV_MEDIUM : NANO_SEV_HIGH;
        add_fault(faults, &count, max_faults, NANO_FAULT_CONSTANT_DRIFT, sev,
                  gc, L104_NANO_GOD_CODE, gc - L104_NANO_GOD_CODE,
                  (int)gc_ulp, "GOD_CODE ULP drift detected");
    }

    /* PHI check */
    double phi = g_phi;
    int64_t phi_ulp = l104_ulp_distance(phi, L104_NANO_PHI);
    if (phi_ulp != 0) {
        l104_nano_severity_t sev = phi_ulp <= 2 ? NANO_SEV_LOW :
                                    phi_ulp <= 8 ? NANO_SEV_MEDIUM : NANO_SEV_HIGH;
        add_fault(faults, &count, max_faults, NANO_FAULT_CONSTANT_DRIFT, sev,
                  phi, L104_NANO_PHI, phi - L104_NANO_PHI,
                  (int)phi_ulp, "PHI ULP drift detected");
    }

    /* VOID_CONSTANT check */
    double vc = g_void_const;
    double expected_vc = 1.04 + L104_NANO_PHI / 1000.0;
    int64_t vc_ulp = l104_ulp_distance(vc, expected_vc);
    if (vc_ulp > 1) { /* Allow 1 ULP for computation */
        l104_nano_severity_t sev = vc_ulp <= 4 ? NANO_SEV_LOW : NANO_SEV_MEDIUM;
        add_fault(faults, &count, max_faults, NANO_FAULT_CONSTANT_DRIFT, sev,
                  vc, expected_vc, vc - expected_vc,
                  (int)vc_ulp, "VOID_CONSTANT ULP drift detected");
    }

    /* Cross-validation: GOD_CODE^(1/φ) should be close to known value */
    double gc_inv_phi = pow(gc, 1.0 / phi);
    double expected_inv = pow(L104_NANO_GOD_CODE, 1.0 / L104_NANO_PHI);
    int64_t cross_ulp = l104_ulp_distance(gc_inv_phi, expected_inv);
    if (cross_ulp > 4) {
        add_fault(faults, &count, max_faults, NANO_FAULT_CONSTANT_DRIFT, NANO_SEV_MEDIUM,
                  gc_inv_phi, expected_inv, gc_inv_phi - expected_inv,
                  (int)cross_ulp, "GOD_CODE^(1/PHI) cross-validation drift");
    }

    return count;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PROBE 2: Memory Canary Integrity
 * φ-scrambled sentinel values in dedicated memory regions.
 * Single-bit flips detected via Hamming distance check.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void init_canary(l104_nano_canary_t *canary) {
    /* Scramble sacred constants with XOR patterns derived from φ */
    uint64_t phi_bits = l104_double_to_bits(L104_NANO_PHI);
    uint64_t gc_bits  = l104_double_to_bits(L104_NANO_GOD_CODE);
    uint64_t vc_bits  = l104_double_to_bits(L104_NANO_VOID_CONSTANT);

    canary->canary_phi  = phi_bits ^ 0xA5A5A5A5A5A5A5A5ULL;
    canary->canary_god  = gc_bits  ^ 0x5A5A5A5A5A5A5A5AULL;
    canary->canary_void = vc_bits  ^ 0x1041041041041041ULL; /* 104-pattern */
    canary->canary_checksum = canary->canary_phi ^ canary->canary_god ^ canary->canary_void;

    /* Record Hamming weight baseline */
    int hw = 0;
    uint64_t v = canary->canary_checksum;
    while (v) { hw += (int)(v & 1); v >>= 1; }
    canary->hamming_baseline = (uint64_t)hw;
}

int l104_nano_probe_canary(l104_nano_canary_t *canary, l104_nano_fault_t *faults, int max) {
    int count = 0;

    /* Verify checksum coherence */
    uint64_t expected_cs = canary->canary_phi ^ canary->canary_god ^ canary->canary_void;
    if (expected_cs != canary->canary_checksum) {
        int hd = l104_hamming_distance(expected_cs, canary->canary_checksum);
        l104_nano_severity_t sev = hd <= 1 ? NANO_SEV_HIGH : NANO_SEV_CRITICAL;
        char desc[256];
        snprintf(desc, sizeof(desc), "Canary checksum corruption: %d-bit flip (Hamming=%d)",
                 hd, hd);
        add_fault(faults, &count, max, NANO_FAULT_MEMORY_CANARY, sev,
                  (double)expected_cs, (double)canary->canary_checksum,
                  (double)hd, hd, desc);
    }

    /* Verify individual canaries */
    uint64_t phi_bits = l104_double_to_bits(L104_NANO_PHI);
    uint64_t expected_phi = phi_bits ^ 0xA5A5A5A5A5A5A5A5ULL;
    if (canary->canary_phi != expected_phi) {
        int hd = l104_hamming_distance(canary->canary_phi, expected_phi);
        add_fault(faults, &count, max, NANO_FAULT_MEMORY_CANARY, NANO_SEV_CRITICAL,
                  0, 0, (double)hd, hd, "PHI canary sentinel bit-flip detected");
    }

    uint64_t gc_bits = l104_double_to_bits(L104_NANO_GOD_CODE);
    uint64_t expected_gc = gc_bits ^ 0x5A5A5A5A5A5A5A5AULL;
    if (canary->canary_god != expected_gc) {
        int hd = l104_hamming_distance(canary->canary_god, expected_gc);
        add_fault(faults, &count, max, NANO_FAULT_MEMORY_CANARY, NANO_SEV_CRITICAL,
                  0, 0, (double)hd, hd, "GOD_CODE canary sentinel bit-flip detected");
    }

    uint64_t vc_bits = l104_double_to_bits(L104_NANO_VOID_CONSTANT);
    uint64_t expected_vc = vc_bits ^ 0x1041041041041041ULL;
    if (canary->canary_void != expected_vc) {
        int hd = l104_hamming_distance(canary->canary_void, expected_vc);
        add_fault(faults, &count, max, NANO_FAULT_MEMORY_CANARY, NANO_SEV_CRITICAL,
                  0, 0, (double)hd, hd, "VOID_CONSTANT canary sentinel bit-flip detected");
    }

    return count;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PROBE 3: FPU Environment Flags
 * Detects silent FPU exceptions that corrupt quantum math silently.
 * ═══════════════════════════════════════════════════════════════════════════ */

l104_nano_fpu_state_t l104_nano_probe_fpu(l104_nano_fault_t *faults, int max_faults) {
    l104_nano_fpu_state_t state;
    int count = 0;

    int flags = fetestexcept(FE_ALL_EXCEPT);
    state.overflow  = (flags & FE_OVERFLOW)  != 0;
    state.underflow = (flags & FE_UNDERFLOW) != 0;
    state.divbyzero = (flags & FE_DIVBYZERO) != 0;
    state.inexact   = (flags & FE_INEXACT)   != 0;
    state.invalid   = (flags & FE_INVALID)   != 0;

    /* Denormal detection via computation */
    volatile double tiny = DBL_MIN;
    volatile double half = 0.5;
    volatile double subnormal = tiny * half;
    state.denormal = (subnormal != 0.0 && subnormal < DBL_MIN);

    /* Check rounding mode */
    state.rounding_mode = fegetround();
    if (state.rounding_mode != FE_TONEAREST) {
        add_fault(faults, &count, max_faults, NANO_FAULT_FPU_FLAGS, NANO_SEV_HIGH,
                  (double)state.rounding_mode, (double)FE_TONEAREST, 0, 0,
                  "FPU rounding mode not FE_TONEAREST — sacred math compromised");
    }

    if (state.overflow) {
        add_fault(faults, &count, max_faults, NANO_FAULT_FPU_FLAGS, NANO_SEV_MEDIUM,
                  0, 0, 0, 0, "FPU overflow flag set — possible silent overflow in quantum ops");
    }
    if (state.invalid) {
        add_fault(faults, &count, max_faults, NANO_FAULT_FPU_FLAGS, NANO_SEV_HIGH,
                  0, 0, 0, 0, "FPU invalid flag set — NaN produced in computation chain");
    }
    if (state.divbyzero) {
        add_fault(faults, &count, max_faults, NANO_FAULT_FPU_FLAGS, NANO_SEV_HIGH,
                  0, 0, 0, 0, "FPU divide-by-zero flag — potential singularity in god-code math");
    }

    /* Clear flags after reading */
    feclearexcept(FE_ALL_EXCEPT);

    return state;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PROBE 4: Cache Line Coherence
 * Timing side-channel detects false sharing / alignment anomalies.
 * ═══════════════════════════════════════════════════════════════════════════ */

int l104_nano_probe_cache_coherence(l104_nano_fault_t *faults, int max_faults) {
    int count = 0;
    const int ITERATIONS = 10000;

    g_cache_god.value  = L104_NANO_GOD_CODE;
    g_cache_phi.value  = L104_NANO_PHI;
    g_cache_void.value = L104_NANO_VOID_CONSTANT;

    /* Sequential access pattern timing */
    uint64_t t0 = l104_nano_timestamp();
    volatile double sum = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        sum += g_cache_god.value;
        sum += g_cache_phi.value;
        sum += g_cache_void.value;
    }
    uint64_t t1 = l104_nano_timestamp();
    double sequential_ns = (double)(t1 - t0);

    /* Strided (cache-hostile) access pattern */
    t0 = l104_nano_timestamp();
    sum = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        sum += g_cache_god.value + g_cache_void.value;
        sum += g_cache_phi.value + g_cache_god.value;
        sum += g_cache_void.value + g_cache_phi.value;
    }
    t1 = l104_nano_timestamp();
    double strided_ns = (double)(t1 - t0);

    /* If strided is >3x slower, cache coherence is compromised */
    double ratio = strided_ns / (sequential_ns + 1.0);
    if (ratio > 3.0) {
        char desc[256];
        snprintf(desc, sizeof(desc),
                 "Cache coherence anomaly: strided/sequential ratio=%.2f (threshold 3.0)", ratio);
        add_fault(faults, &count, max_faults, NANO_FAULT_CACHE_COHERENCE, NANO_SEV_LOW,
                  ratio, 1.0, ratio - 1.0, 0, desc);
    }

    /* Verify values weren't corrupted during timing */
    if (g_cache_god.value != L104_NANO_GOD_CODE ||
        g_cache_phi.value != L104_NANO_PHI ||
        g_cache_void.value != L104_NANO_VOID_CONSTANT) {
        add_fault(faults, &count, max_faults, NANO_FAULT_CACHE_COHERENCE, NANO_SEV_CRITICAL,
                  g_cache_god.value, L104_NANO_GOD_CODE, 0, 0,
                  "Cache line value corruption during coherence test");
    }

    (void)sum; /* Suppress unused warning */
    return count;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PROBE 5: Entropy Source Health
 * Validates OS RNG quality via poker test and serial correlation.
 * ═══════════════════════════════════════════════════════════════════════════ */

l104_nano_entropy_health_t l104_nano_probe_entropy(l104_nano_fault_t *faults, int max_faults) {
    l104_nano_entropy_health_t result;
    int count = 0;
    const int SAMPLE_SIZE = 2500;

    unsigned char buf[2500];
    FILE *fp = fopen("/dev/urandom", "rb");
    int bytes_read = 0;
    if (fp) {
        bytes_read = (int)fread(buf, 1, SAMPLE_SIZE, fp);
        fclose(fp);
    }
    result.bytes_tested = bytes_read;

    if (bytes_read < SAMPLE_SIZE) {
        result.healthy = false;
        result.chi_squared = 99999.0;
        result.poker_score = 0.0;
        result.serial_correlation = 1.0;
        add_fault(faults, &count, max_faults, NANO_FAULT_ENTROPY_HEALTH, NANO_SEV_HIGH,
                  (double)bytes_read, (double)SAMPLE_SIZE, 0, 0,
                  "Entropy source read failure — insufficient random bytes");
        return result;
    }

    /* Chi-squared test on byte distribution */
    int freq[256] = {0};
    for (int i = 0; i < bytes_read; i++) freq[buf[i]]++;
    double expected = (double)bytes_read / 256.0;
    double chi2 = 0.0;
    for (int i = 0; i < 256; i++) {
        double diff = (double)freq[i] - expected;
        chi2 += (diff * diff) / expected;
    }
    result.chi_squared = chi2;

    /* 4-bit poker test */
    int nibble_freq[16] = {0};
    for (int i = 0; i < bytes_read; i++) {
        nibble_freq[buf[i] >> 4]++;
        nibble_freq[buf[i] & 0x0F]++;
    }
    double poker = 0.0;
    int total_nibbles = bytes_read * 2;
    double nib_expected = (double)total_nibbles / 16.0;
    for (int i = 0; i < 16; i++) {
        double d = (double)nibble_freq[i] - nib_expected;
        poker += (d * d) / nib_expected;
    }
    result.poker_score = poker;

    /* Serial correlation coefficient */
    double sum_xy = 0, sum_x = 0, sum_y = 0, sum_x2 = 0, sum_y2 = 0;
    for (int i = 0; i < bytes_read - 1; i++) {
        double x = (double)buf[i], y = (double)buf[i + 1];
        sum_xy += x * y;
        sum_x += x; sum_y += y;
        sum_x2 += x * x; sum_y2 += y * y;
    }
    int n = bytes_read - 1;
    double denom = sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    result.serial_correlation = denom > 0 ? (n * sum_xy - sum_x * sum_y) / denom : 1.0;

    /* Health verdict */
    result.healthy = (chi2 < 350.0) && (poker < 35.0) && (fabs(result.serial_correlation) < 0.05);

    if (!result.healthy) {
        char desc[256];
        snprintf(desc, sizeof(desc),
                 "Entropy degradation: chi2=%.1f, poker=%.1f, serial_corr=%.4f",
                 chi2, poker, result.serial_correlation);
        add_fault(faults, &count, max_faults, NANO_FAULT_ENTROPY_HEALTH, NANO_SEV_MEDIUM,
                  chi2, 256.0, chi2 - 256.0, 0, desc);
    }

    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PROBE 6: Numerical Stability Audit
 * Runs φ-recursive sequences checking for catastrophic cancellation,
 * significance loss, and underflow spirals.
 * ═══════════════════════════════════════════════════════════════════════════ */

l104_nano_stability_t l104_nano_probe_stability(l104_nano_fault_t *faults, int max_faults) {
    l104_nano_stability_t result;
    int count = 0;

    /* PHI recurrence: f(n) = f(n-1) + f(n-2) with f(0)=1, f(1)=φ
     * Ratio f(n)/f(n-1) should converge to φ */
    double a = 1.0, b = L104_NANO_PHI;
    for (int i = 0; i < 100; i++) {
        double c = a + b;
        a = b;
        b = c;
    }
    double computed_phi = b / a;
    result.phi_recurrence_error = fabs(computed_phi - L104_NANO_PHI);

    if (result.phi_recurrence_error > 1e-12) {
        add_fault(faults, &count, max_faults, NANO_FAULT_NUMERICAL_STAB, NANO_SEV_MEDIUM,
                  computed_phi, L104_NANO_PHI, result.phi_recurrence_error,
                  (int)l104_ulp_distance(computed_phi, L104_NANO_PHI),
                  "PHI recurrence convergence failure — numerical instability");
    }

    /* GOD_CODE roundtrip: encode → decode → reencode */
    double gc = L104_NANO_GOD_CODE;
    double encoded = log(gc) / log(286.0) * L104_NANO_PHI;
    double decoded = pow(286.0, encoded / L104_NANO_PHI);
    result.god_code_roundtrip = decoded;

    double roundtrip_err = fabs(decoded - gc);
    if (roundtrip_err > 1e-10) {
        add_fault(faults, &count, max_faults, NANO_FAULT_NUMERICAL_STAB, NANO_SEV_MEDIUM,
                  decoded, gc, roundtrip_err, 0,
                  "GOD_CODE log-pow roundtrip precision loss");
    }

    /* Catastrophic cancellation test: (1 + ε) - 1 where ε is tiny */
    double epsilon = L104_NANO_PHI * 1e-15;
    double big_plus_eps = 1.0 + epsilon;
    double recovered = big_plus_eps - 1.0;
    result.cancellation_loss = -log2(fabs(recovered / epsilon));

    if (result.cancellation_loss > 4.0) { /* More than 4 significant bits lost */
        char desc[256];
        snprintf(desc, sizeof(desc),
                 "Catastrophic cancellation: %.1f significant bits lost", result.cancellation_loss);
        add_fault(faults, &count, max_faults, NANO_FAULT_NUMERICAL_STAB, NANO_SEV_LOW,
                  result.cancellation_loss, 0.0, result.cancellation_loss, 0, desc);
    }

    /* Underflow spiral: repeated division by φ */
    result.underflow_count = 0;
    double val = L104_NANO_GOD_CODE;
    for (int i = 0; i < 2000; i++) {
        val /= L104_NANO_PHI;
        if (val > 0 && val < DBL_MIN) result.underflow_count++;
        if (val == 0) break;
    }

    result.stable = (result.phi_recurrence_error < 1e-12) &&
                    (roundtrip_err < 1e-10) &&
                    (result.cancellation_loss < 8.0) &&
                    (result.underflow_count < 50);

    if (!result.stable) {
        add_fault(faults, &count, max_faults, NANO_FAULT_NUMERICAL_STAB, NANO_SEV_HIGH,
                  0, 0, 0, 0, "Numerical stability verdict: UNSTABLE");
    }

    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PROBE 7: Phase Drift Detection
 * Accumulated floating-point phase error in modular arithmetic chains.
 * ═══════════════════════════════════════════════════════════════════════════ */

l104_nano_phase_drift_t l104_nano_probe_phase_drift(l104_nano_fault_t *faults, int max_faults) {
    l104_nano_phase_drift_t result;
    int count = 0;

    /* Accumulate GOD_CODE phase modulo 2π for N iterations */
    const int N = 100000;
    double phase = 0.0;
    double god_phase = fmod(L104_NANO_GOD_CODE, 2.0 * M_PI);

    for (int i = 0; i < N; i++) {
        phase += god_phase;
        phase = fmod(phase, 2.0 * M_PI);
    }

    /* Analytical result: (N * GOD_CODE) mod 2π */
    double exact = fmod((double)N * L104_NANO_GOD_CODE, 2.0 * M_PI);

    result.accumulated_phase = phase;
    result.expected_phase = exact;
    result.drift_radians = fabs(phase - exact);
    result.drift_ulps = (double)l104_ulp_distance(phase, exact);
    result.iterations = N;

    if (result.drift_radians > 1e-8) {
        char desc[256];
        snprintf(desc, sizeof(desc),
                 "Phase drift after %d iterations: %.2e rad (%.0f ULPs)", N,
                 result.drift_radians, result.drift_ulps);
        l104_nano_severity_t sev = result.drift_radians < 1e-6 ? NANO_SEV_LOW :
                                    result.drift_radians < 1e-4 ? NANO_SEV_MEDIUM : NANO_SEV_HIGH;
        add_fault(faults, &count, max_faults, NANO_FAULT_PHASE_DRIFT, sev,
                  phase, exact, result.drift_radians, (int)result.drift_ulps, desc);
    }

    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PROBE 8: NDE Pipeline Integrity
 * Verifies NDE pipeline is a contraction mapping (fixed-point property).
 * ═══════════════════════════════════════════════════════════════════════════ */

int l104_nano_probe_nde_integrity(l104_nano_fault_t *faults, int max_faults) {
    int count = 0;

    /* Run NDE on a known input and verify bounds */
    double test_scores[3] = {0.75, 0.82, 0.91};
    double entropy = 0.3;
    double fid_std = 0.05;

    double result1 = l104_nde_full_pipeline(test_scores, 3, entropy, fid_std);

    /* Result must be in [0, 2] (reasonable score range) */
    if (result1 < 0.0 || result1 > 2.0 || isnan(result1) || isinf(result1)) {
        add_fault(faults, &count, max_faults, NANO_FAULT_NDE_INTEGRITY, NANO_SEV_HIGH,
                  result1, 1.0, fabs(result1 - 1.0), 0,
                  "NDE pipeline output out of bounds or NaN/Inf");
    }

    /* Fixed-point test: applying NDE twice should converge */
    double scores2[3] = {result1, result1, result1};
    double result2 = l104_nde_full_pipeline(scores2, 3, entropy, fid_std);
    double contraction = fabs(result2 - result1);

    if (contraction > 0.5) {
        char desc[256];
        snprintf(desc, sizeof(desc),
                 "NDE not contractive: |f(f(x))-f(x)|=%.4f (threshold 0.5)", contraction);
        add_fault(faults, &count, max_faults, NANO_FAULT_NDE_INTEGRITY, NANO_SEV_MEDIUM,
                  result2, result1, contraction, 0, desc);
    }

    /* Monotonicity: higher inputs should not produce lower outputs */
    double low_scores[3] = {0.3, 0.3, 0.3};
    double high_scores[3] = {0.9, 0.9, 0.9};
    double low_out = l104_nde_full_pipeline(low_scores, 3, entropy, fid_std);
    double high_out = l104_nde_full_pipeline(high_scores, 3, entropy, fid_std);

    if (high_out < low_out - 0.01) {
        add_fault(faults, &count, max_faults, NANO_FAULT_NDE_INTEGRITY, NANO_SEV_MEDIUM,
                  high_out, low_out, low_out - high_out, 0,
                  "NDE monotonicity violation: higher input → lower output");
    }

    return count;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * IPC — JSON Report Writer
 * ═══════════════════════════════════════════════════════════════════════════ */

int l104_nano_write_report(const l104_nano_tick_result_t *result) {
    ensure_dir("/tmp/l104_bridge");
    ensure_dir(NANO_IPC_BASE);
    ensure_dir(NANO_C_OUTBOX);

    /* Write report as JSON */
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/tick_%llu.json",
             NANO_C_OUTBOX, (unsigned long long)result->tick_number);

    FILE *fp = fopen(filename, "w");
    if (!fp) return -1;

    fprintf(fp, "{\n");
    fprintf(fp, "  \"daemon\": \"l104_nano_c\",\n");
    fprintf(fp, "  \"version\": \"%s\",\n", L104_NANO_VERSION);
    fprintf(fp, "  \"tick\": %llu,\n", (unsigned long long)result->tick_number);
    fprintf(fp, "  \"health\": %.6f,\n", result->health_score);
    fprintf(fp, "  \"fault_count\": %d,\n", result->fault_count);
    fprintf(fp, "  \"total_faults_ever\": %llu,\n", (unsigned long long)result->total_faults_ever);
    fprintf(fp, "  \"tick_duration_ns\": %llu,\n", (unsigned long long)result->tick_duration_ns);
    fprintf(fp, "  \"fpu\": {\n");
    fprintf(fp, "    \"overflow\": %s, \"underflow\": %s, \"invalid\": %s,\n",
            result->fpu.overflow ? "true" : "false",
            result->fpu.underflow ? "true" : "false",
            result->fpu.invalid ? "true" : "false");
    fprintf(fp, "    \"rounding_mode\": %d\n", result->fpu.rounding_mode);
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"entropy\": { \"healthy\": %s, \"chi2\": %.2f },\n",
            result->entropy.healthy ? "true" : "false", result->entropy.chi_squared);
    fprintf(fp, "  \"stability\": { \"stable\": %s, \"phi_error\": %.2e },\n",
            result->stability.stable ? "true" : "false", result->stability.phi_recurrence_error);
    fprintf(fp, "  \"phase_drift\": { \"drift_rad\": %.2e, \"iterations\": %d },\n",
            result->phase.drift_radians, result->phase.iterations);

    if (result->fault_count > 0) {
        fprintf(fp, "  \"faults\": [\n");
        for (int i = 0; i < result->fault_count; i++) {
            const l104_nano_fault_t *f = &result->faults[i];
            fprintf(fp, "    { \"type\": %d, \"severity\": %d, \"ulp\": %d, \"desc\": \"%s\" }%s\n",
                    f->type, f->severity, f->ulp_distance, f->description,
                    i < result->fault_count - 1 ? "," : "");
        }
        fprintf(fp, "  ]\n");
    } else {
        fprintf(fp, "  \"faults\": []\n");
    }
    fprintf(fp, "}\n");
    fclose(fp);
    return 0;
}

void l104_nano_heartbeat(void) {
    ensure_dir("/tmp/l104_bridge");
    ensure_dir(NANO_IPC_BASE);
    FILE *fp = fopen(NANO_C_HEARTBEAT, "w");
    if (fp) {
        fprintf(fp, "%llu\n", (unsigned long long)l104_nano_timestamp());
        fclose(fp);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * DAEMON LIFECYCLE — L104Daemon-grade assertions
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Validate configuration — mirrors L104Daemon's validateConfiguration().
 * Returns 1 on success, 0 on failure (caller should exit(1)). */
static int l104_nano_validate_configuration(int tick_interval_ms) {
    int valid = 1;

    /* Tick interval bounds (100ms – 60000ms) */
    if (tick_interval_ms < 100 || tick_interval_ms > 60000) {
        fprintf(stderr, "[L104 NanoDaemon/C] ERROR: Invalid tick interval: %dms (must be 100-60000)\n",
                tick_interval_ms);
        valid = 0;
    }

    /* Sacred constant bit-exact verification (dynamic roundtrip — compiler-safe) */
    /* Compare volatile copies against freshly-evaluated constants */
    double fresh_gc  = L104_NANO_GOD_CODE;
    double fresh_phi = L104_NANO_PHI;
    double fresh_v   = L104_NANO_VOID_CONSTANT;
    uint64_t gc_bits   = l104_double_to_bits(g_god_code);
    uint64_t phi_bits  = l104_double_to_bits(g_phi);
    uint64_t void_bits = l104_double_to_bits(g_void_const);
    uint64_t gc_ref    = l104_double_to_bits(fresh_gc);
    uint64_t phi_ref   = l104_double_to_bits(fresh_phi);
    uint64_t void_ref  = l104_double_to_bits(fresh_v);
    if (gc_bits != gc_ref) {
        fprintf(stderr, "[L104 NanoDaemon/C] ERROR: GOD_CODE volatile drift: 0x%llX != 0x%llX\n",
                (unsigned long long)gc_bits, (unsigned long long)gc_ref);
        valid = 0;
    }
    if (phi_bits != phi_ref) {
        fprintf(stderr, "[L104 NanoDaemon/C] ERROR: PHI volatile drift: 0x%llX != 0x%llX\n",
                (unsigned long long)phi_bits, (unsigned long long)phi_ref);
        valid = 0;
    }
    if (void_bits != void_ref) {
        fprintf(stderr, "[L104 NanoDaemon/C] ERROR: VOID_CONSTANT volatile drift: 0x%llX != 0x%llX\n",
                (unsigned long long)void_bits, (unsigned long long)void_ref);
        valid = 0;
    }

    /* Verify IPC directories exist (after creation) */
    struct stat st;
    const char *required_dirs[] = {"/tmp/l104_bridge", NANO_IPC_BASE, NANO_C_OUTBOX, NULL};
    for (int i = 0; required_dirs[i] != NULL; i++) {
        if (stat(required_dirs[i], &st) != 0 || !S_ISDIR(st.st_mode)) {
            fprintf(stderr, "[L104 NanoDaemon/C] ERROR: Required directory missing: %s\n",
                    required_dirs[i]);
            valid = 0;
        }
    }

    /* System resource check (macOS Mach API) */
#ifdef __APPLE__
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm_stat;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm_stat, &count) == KERN_SUCCESS) {
        uint64_t free_pages = (uint64_t)vm_stat.free_count;
        uint64_t free_mb = (free_pages * 4096) / (1024 * 1024);
        if (free_mb < 32) {
            fprintf(stderr, "[L104 NanoDaemon/C] WARNING: Low free memory: %lluMB (recommend ≥32MB)\n",
                    (unsigned long long)free_mb);
            /* Not fatal — just warn */
        }
    }
#endif

    if (valid) {
        printf("[L104 NanoDaemon/C] Configuration validated ✓\n");
    }
    return valid;
}

/* Kill any stale daemon instance from PID file — mirrors L104Daemon's killPreviousInstance() */
static void l104_nano_kill_previous_instance(void) {
    FILE *fp = fopen(NANO_C_PID, "r");
    if (!fp) return;

    int old_pid = 0;
    if (fscanf(fp, "%d", &old_pid) != 1) { fclose(fp); return; }
    fclose(fp);

    int my_pid = getpid();
    if (old_pid <= 0 || old_pid == my_pid) return;

    /* Check if old process is alive */
    if (kill(old_pid, 0) != 0) return;

    printf("[L104 NanoDaemon/C] Killing stale instance (PID %d)\n", old_pid);
    kill(old_pid, SIGTERM);

    /* Wait up to 2 seconds for graceful exit */
    for (int i = 0; i < 20; i++) {
        usleep(100000); /* 100ms */
        if (kill(old_pid, 0) != 0) break;
    }

    /* Force kill if still alive */
    if (kill(old_pid, 0) == 0) {
        printf("[L104 NanoDaemon/C] Stale PID %d did not exit — sending SIGKILL\n", old_pid);
        kill(old_pid, SIGKILL);
        usleep(100000);
    }
}

/* Dump full daemon status to stdout and /tmp/l104_bridge/nano/c_status.json — mirrors SIGUSR1 */
static void l104_nano_dump_status(const l104_nano_daemon_state_t *state) {
    printf("\n[L104 NanoDaemon/C] ═══ STATUS DUMP ═══\n");
    printf("  Version:      %s\n", L104_NANO_VERSION);
    printf("  PID:          %d\n", state->pid);
    printf("  Running:      %s\n", state->running ? "yes" : "no");
    printf("  Ticks:        %llu\n", (unsigned long long)state->tick_count);
    printf("  Total faults: %llu\n", (unsigned long long)state->total_faults);
    printf("  Health trend: %.6f\n", state->health_trend);
    printf("  Last health:  %.6f\n", state->last_tick.health_score);
    printf("  Last tick μs: %llu\n", (unsigned long long)(state->last_tick.tick_duration_ns / 1000));
    printf("  GOD_CODE:     %.13f (bits=0x%llX)\n", L104_NANO_GOD_CODE,
           (unsigned long long)l104_double_to_bits(L104_NANO_GOD_CODE));
    printf("  PHI:          %.15f\n", L104_NANO_PHI);
    printf("  VOID:         %.16f\n", L104_NANO_VOID_CONSTANT);

    /* System resources */
#ifdef __APPLE__
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm_stat;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm_stat, &count) == KERN_SUCCESS) {
        uint64_t free_mb = ((uint64_t)vm_stat.free_count * 4096) / (1024 * 1024);
        uint64_t active_mb = ((uint64_t)vm_stat.active_count * 4096) / (1024 * 1024);
        uint64_t wired_mb = ((uint64_t)vm_stat.wire_count * 4096) / (1024 * 1024);
        printf("  Memory:       free=%lluMB active=%lluMB wired=%lluMB\n",
               (unsigned long long)free_mb, (unsigned long long)active_mb,
               (unsigned long long)wired_mb);
    }
#endif
    printf("  ═══════════════════════════════\n\n");
    fflush(stdout);

    /* Write JSON status to file */
    char path[256];
    snprintf(path, sizeof(path), "%s/c_status.json", NANO_IPC_BASE);
    FILE *fp = fopen(path, "w");
    if (fp) {
        fprintf(fp, "{\n"
                "  \"daemon\": \"l104_nano_c\",\n"
                "  \"version\": \"%s\",\n"
                "  \"pid\": %d,\n"
                "  \"running\": %s,\n"
                "  \"tick_count\": %llu,\n"
                "  \"total_faults\": %llu,\n"
                "  \"health_trend\": %.6f,\n"
                "  \"last_health\": %.6f\n"
                "}\n",
                L104_NANO_VERSION, state->pid,
                state->running ? "true" : "false",
                (unsigned long long)state->tick_count,
                (unsigned long long)state->total_faults,
                state->health_trend,
                state->last_tick.health_score);
        fclose(fp);
    }
}

/* Reload: reinitialize canaries, clear FPU, re-verify constants — mirrors SIGHUP */
static void l104_nano_reload(l104_nano_daemon_state_t *state) {
    printf("[L104 NanoDaemon/C] SIGHUP — reloading (reinit canaries + FPU)\n");
    init_canary(&state->canary);
    g_cache_god.value  = L104_NANO_GOD_CODE;
    g_cache_phi.value  = L104_NANO_PHI;
    g_cache_void.value = L104_NANO_VOID_CONSTANT;
    feclearexcept(FE_ALL_EXCEPT);
    printf("[L104 NanoDaemon/C] Reload complete\n");
    fflush(stdout);
}

static void signal_handler(int sig) {
    if (sig == SIGTERM || sig == SIGINT) {
        g_shutdown_requested = 1;
    } else if (sig == SIGUSR1) {
        /* Force status dump on next tick */
        g_dump_status = 1;
    } else if (sig == SIGUSR2) {
        /* Force immediate tick */
        g_force_tick = 1;
    } else if (sig == SIGHUP) {
        /* Reload configuration */
        g_reload_requested = 1;
    }
}

int l104_nano_daemon_init(l104_nano_daemon_state_t *state) {
    memset(state, 0, sizeof(*state));

    /* Set up IPC directories */
    ensure_dir("/tmp/l104_bridge");
    ensure_dir(NANO_IPC_BASE);
    ensure_dir(NANO_C_OUTBOX);

    /* Kill stale instance — L104Daemon pattern */
    l104_nano_kill_previous_instance();

    /* Initialize canaries */
    init_canary(&state->canary);

    /* Write PID file */
    FILE *fp = fopen(NANO_C_PID, "w");
    if (fp) {
        fprintf(fp, "%d\n", getpid());
        fclose(fp);
    }
    state->pid = getpid();

    /* Initialize cache line sentinels */
    g_cache_god.value  = L104_NANO_GOD_CODE;
    g_cache_phi.value  = L104_NANO_PHI;
    g_cache_void.value = L104_NANO_VOID_CONSTANT;

    /* Clear FPU flags */
    feclearexcept(FE_ALL_EXCEPT);

    /* Set up signal handlers — full L104Daemon set */
    signal(SIGTERM, signal_handler);
    signal(SIGINT,  signal_handler);
    signal(SIGUSR1, signal_handler);
    signal(SIGUSR2, signal_handler);
    signal(SIGHUP,  signal_handler);

    /* Store global reference for signal-driven status dump */
    g_daemon_state = state;

    state->health_trend = 1.0;
    state->running = true;

    printf("[L104 NanoDaemon/C v%s] Initialized (PID=%d)\n", L104_NANO_VERSION, state->pid);
    printf("  Sacred constants: GOD_CODE=%.13f  PHI=%.15f  VOID=%.16f\n",
           L104_NANO_GOD_CODE, L104_NANO_PHI, L104_NANO_VOID_CONSTANT);
    printf("  IPC: %s\n", NANO_C_OUTBOX);
    printf("  Probes: 8 (constant_drift, canary, fpu, cache, entropy, stability, phase, nde)\n");

    return 1;
}

int l104_nano_daemon_tick(l104_nano_daemon_state_t *state) {
    uint64_t t0 = l104_nano_timestamp();
    l104_nano_tick_result_t *tick = &state->last_tick;
    memset(tick, 0, sizeof(*tick));
    tick->tick_number = state->tick_count;

    int total_faults = 0;
    int fi = 0; /* fault index into tick->faults */

    /* Probe 1: Sacred constant drift */
    fi += l104_nano_probe_constant_drift(&tick->faults[fi], 64 - fi);

    /* Probe 2: Memory canary integrity */
    fi += l104_nano_probe_canary(&state->canary, &tick->faults[fi], 64 - fi);

    /* Probe 3: FPU environment */
    tick->fpu = l104_nano_probe_fpu(&tick->faults[fi], 64 - fi);

    /* Probe 4: Cache coherence */
    fi += l104_nano_probe_cache_coherence(&tick->faults[fi], 64 - fi);

    /* Probe 5: Entropy health */
    tick->entropy = l104_nano_probe_entropy(&tick->faults[fi], 64 - fi);

    /* Probe 6: Numerical stability */
    tick->stability = l104_nano_probe_stability(&tick->faults[fi], 64 - fi);

    /* Probe 7: Phase drift */
    tick->phase = l104_nano_probe_phase_drift(&tick->faults[fi], 64 - fi);

    /* Probe 8: NDE integrity */
    fi += l104_nano_probe_nde_integrity(&tick->faults[fi], 64 - fi);

    /* Count total faults */
    for (int i = 0; i < 64; i++) {
        if (tick->faults[i].timestamp_ns > 0) total_faults++;
    }
    tick->fault_count = total_faults;
    state->total_faults += (uint64_t)total_faults;
    tick->total_faults_ever = state->total_faults;

    /* Compute health score (1.0 = perfect, degrade by severity) */
    double health = 1.0;
    for (int i = 0; i < total_faults; i++) {
        double penalty = 0.0;
        switch (tick->faults[i].severity) {
            case NANO_SEV_TRACE:    penalty = 0.001; break;
            case NANO_SEV_LOW:      penalty = 0.01;  break;
            case NANO_SEV_MEDIUM:   penalty = 0.05;  break;
            case NANO_SEV_HIGH:     penalty = 0.15;  break;
            case NANO_SEV_CRITICAL: penalty = 0.30;  break;
        }
        health -= penalty;
    }
    if (health < 0.0) health = 0.0;
    tick->health_score = health;

    /* Exponential moving average of health */
    state->health_trend = 0.9 * state->health_trend + 0.1 * health;

    uint64_t t1 = l104_nano_timestamp();
    tick->tick_duration_ns = t1 - t0;
    state->tick_count++;

    /* Write IPC report and heartbeat */
    l104_nano_write_report(tick);
    l104_nano_heartbeat();

    return total_faults;
}

void l104_nano_daemon_shutdown(l104_nano_daemon_state_t *state) {
    state->running = false;
    unlink(NANO_C_PID);
    printf("[L104 NanoDaemon/C] Shutdown after %llu ticks, %llu total faults, "
           "health_trend=%.4f\n",
           (unsigned long long)state->tick_count,
           (unsigned long long)state->total_faults,
           state->health_trend);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SELF-TEST
 * ═══════════════════════════════════════════════════════════════════════════ */

int l104_nano_self_test(void) {
    int failures = 0;
    printf("[L104 NanoDaemon/C] Self-test — 8 probes\n");

    l104_nano_fault_t faults[64];
    memset(faults, 0, sizeof(faults));

    /* Test 1: ULP distance */
    int64_t d = l104_ulp_distance(1.0, 1.0 + DBL_EPSILON);
    if (d != 1) { printf("  FAIL: ULP distance (got %lld, want 1)\n", (long long)d); failures++; }
    else printf("  PASS: ULP distance = 1\n");

    /* Test 2: Hamming distance */
    int hd = l104_hamming_distance(0xFF, 0x00);
    if (hd != 8) { printf("  FAIL: Hamming distance (got %d, want 8)\n", hd); failures++; }
    else printf("  PASS: Hamming distance = 8\n");

    /* Test 3: Constant drift probe (should find 0 faults) */
    int fc = l104_nano_probe_constant_drift(faults, 64);
    printf("  %s: Constant drift probe (%d faults)\n", fc == 0 ? "PASS" : "WARN", fc);

    /* Test 4: Canary probe */
    l104_nano_canary_t canary;
    init_canary(&canary);
    fc = l104_nano_probe_canary(&canary, faults, 64);
    if (fc != 0) { printf("  FAIL: Fresh canary has %d faults\n", fc); failures++; }
    else printf("  PASS: Canary integrity\n");

    /* Test 5: FPU flags (should be clean after feclearexcept) */
    feclearexcept(FE_ALL_EXCEPT);
    l104_nano_fpu_state_t fpu = l104_nano_probe_fpu(faults, 64);
    if (fpu.rounding_mode != FE_TONEAREST) {
        printf("  FAIL: FPU rounding not TONEAREST\n"); failures++;
    } else printf("  PASS: FPU environment clean\n");

    /* Test 6: Entropy health */
    l104_nano_entropy_health_t ent = l104_nano_probe_entropy(faults, 64);
    printf("  %s: Entropy health (chi2=%.1f, poker=%.1f, healthy=%s)\n",
           ent.healthy ? "PASS" : "WARN", ent.chi_squared, ent.poker_score,
           ent.healthy ? "yes" : "no");

    /* Test 7: Numerical stability */
    l104_nano_stability_t stab = l104_nano_probe_stability(faults, 64);
    printf("  %s: Numerical stability (phi_err=%.2e, stable=%s)\n",
           stab.stable ? "PASS" : "WARN", stab.phi_recurrence_error,
           stab.stable ? "yes" : "no");

    /* Test 8: Phase drift */
    l104_nano_phase_drift_t phase = l104_nano_probe_phase_drift(faults, 64);
    printf("  %s: Phase drift (%.2e rad over %d iterations)\n",
           phase.drift_radians < 1e-6 ? "PASS" : "WARN",
           phase.drift_radians, phase.iterations);

    printf("[L104 NanoDaemon/C] Self-test complete: %d failures\n", failures);
    return failures;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN DAEMON LOOP
 * ═══════════════════════════════════════════════════════════════════════════ */

int l104_nano_daemon_run(int tick_interval_ms) {
    l104_nano_daemon_state_t state;
    if (!l104_nano_daemon_init(&state)) return 1;

    struct timespec ts;
    ts.tv_sec  = tick_interval_ms / 1000;
    ts.tv_nsec = (tick_interval_ms % 1000) * 1000000L;

    printf("[L104 NanoDaemon/C] Running (tick=%dms)\n", tick_interval_ms);

    while (!g_shutdown_requested) {
        /* Handle SIGUSR1 → status dump */
        if (g_dump_status) {
            l104_nano_dump_status(&state);
            g_dump_status = 0;
        }

        /* Handle SIGHUP → reload */
        if (g_reload_requested) {
            l104_nano_reload(&state);
            g_reload_requested = 0;
        }

        int faults = l104_nano_daemon_tick(&state);

        if (faults > 0 || g_force_tick) {
            printf("[Tick %llu] health=%.4f faults=%d duration=%lluμs\n",
                   (unsigned long long)state.tick_count,
                   state.last_tick.health_score, faults,
                   (unsigned long long)(state.last_tick.tick_duration_ns / 1000));
            g_force_tick = 0;
        }

        nanosleep(&ts, NULL);
    }

    l104_nano_daemon_shutdown(&state);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * STANDALONE MAIN (when compiled as executable)
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifndef L104_NANO_AS_LIBRARY

int main(int argc, char *argv[]) {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  L104 NANO DAEMON — C Substrate v%s                       ║\n", L104_NANO_VERSION);
    printf("║  Atomized Fault Detection at Native Speed                      ║\n");
    printf("║  GOD_CODE=527.5184818492612 | PHI=1.618033988749895            ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    int tick_ms = 3000; /* Default: 3s tick */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--self-test") == 0) {
            return l104_nano_self_test();
        }
        if (strcmp(argv[i], "--validate") == 0) {
            /* Validate configuration only, exit 0/1 */
            ensure_dir("/tmp/l104_bridge");
            ensure_dir(NANO_IPC_BASE);
            ensure_dir(NANO_C_OUTBOX);
            return l104_nano_validate_configuration(3000) ? 0 : 1;
        }
        if (strcmp(argv[i], "--status") == 0) {
            /* Read PID file and send SIGUSR1 to running daemon */
            FILE *fp = fopen(NANO_C_PID, "r");
            if (!fp) { printf("No running daemon (PID file missing)\n"); return 1; }
            int pid = 0;
            if (fscanf(fp, "%d", &pid) != 1 || pid <= 0) { fclose(fp); printf("Invalid PID file\n"); return 1; }
            fclose(fp);
            if (kill(pid, 0) != 0) { printf("Daemon PID %d not running\n", pid); return 1; }
            kill(pid, SIGUSR1);
            printf("Sent SIGUSR1 to PID %d (status dump requested)\n", pid);
            return 0;
        }
        if (strcmp(argv[i], "--tick") == 0 && i + 1 < argc) {
            tick_ms = atoi(argv[i + 1]);
            if (tick_ms < 100) tick_ms = 100;
            if (tick_ms > 60000) tick_ms = 60000;
            i++;
        }
        if (strcmp(argv[i], "--once") == 0) {
            l104_nano_daemon_state_t state;
            l104_nano_daemon_init(&state);
            int faults = l104_nano_daemon_tick(&state);
            printf("[Single tick] health=%.4f  faults=%d  duration=%lluμs\n",
                   state.last_tick.health_score, faults,
                   (unsigned long long)(state.last_tick.tick_duration_ns / 1000));
            l104_nano_daemon_shutdown(&state);
            return faults > 0 ? 1 : 0;
        }
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: l104_nano_daemon [OPTIONS]\n\n");
            printf("Options:\n");
            printf("  --self-test    Run 8 nano probes and exit 0/1\n");
            printf("  --validate     Validate configuration and exit 0/1\n");
            printf("  --status       Send SIGUSR1 to running daemon for status dump\n");
            printf("  --once         Run single tick and exit\n");
            printf("  --tick <ms>    Tick interval in milliseconds (default: 3000)\n");
            printf("  --help         Show this help\n");
            printf("\nSignals:\n");
            printf("  SIGTERM/SIGINT  Graceful shutdown\n");
            printf("  SIGUSR1         Status dump to stdout + c_status.json\n");
            printf("  SIGUSR2         Force immediate tick\n");
            printf("  SIGHUP          Reload (reinit canaries + FPU)\n");
            return 0;
        }
    }

    /* ── L104Daemon-grade startup assertion gate ── */
    printf("[L104 NanoDaemon/C] Startup validation...\n");
    printf("  PID:      %d\n", getpid());
    printf("  Tick:     %dms\n", tick_ms);
    printf("  IPC:      %s\n", NANO_C_OUTBOX);

    /* Ensure directories exist BEFORE validation (L104Daemon pattern) */
    ensure_dir("/tmp/l104_bridge");
    ensure_dir(NANO_IPC_BASE);
    ensure_dir(NANO_C_OUTBOX);

    if (!l104_nano_validate_configuration(tick_ms)) {
        fprintf(stderr, "[L104 NanoDaemon/C] ERROR: Configuration validation failed — exiting\n");
        return 1;
    }

    return l104_nano_daemon_run(tick_ms);
}

#endif /* L104_NANO_AS_LIBRARY */
