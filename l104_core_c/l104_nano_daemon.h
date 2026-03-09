/* ═══════════════════════════════════════════════════════════════════════════
 * L104 NANO DAEMON — C Substrate v1.0.0
 * Atomized Fault Detection at Native Speed
 * GOD_CODE=527.5184818492612 | PHI=1.618033988749895 | PILOT: LONDEL
 *
 * Sub-bit-level fault detection for the L104 Sovereign Node.
 * Operates at the nano-granularity level — detecting miniscule flaws that
 * escape the micro-daemon's resolution:
 *
 *   1. Sacred Constant Drift — detects ULP (unit-in-last-place) drift in
 *      GOD_CODE, PHI, VOID_CONSTANT via bitwise IEEE 754 comparison
 *   2. Memory Corruption Canary — φ-scrambled canary values in sentinel
 *      memory regions; detects single-bit flips via Hamming distance
 *   3. Floating-Point Environment Probe — checks FPU flags (overflow,
 *      underflow, denormals, NaN) that silently corrupt quantum math
 *   4. Cache Line Coherence — detects false sharing / alignment issues
 *      via timing side-channel on sacred constant access patterns
 *   5. Entropy Source Health — validates randomness quality from the OS
 *      entropy pool (arc4random / /dev/urandom) using poker test
 *   6. Numerical Stability Audit — runs φ-recursive sequences and checks
 *      for catastrophic cancellation, significance loss, underflow spirals
 *   7. Phase Drift Detection — detects accumulated floating-point phase
 *      error in god-code resonance computations
 *   8. NDE Integrity Check — verifies NDE pipeline output stays within
 *      sacred bounds after repeated self-application (fixed-point test)
 *
 * IPC: Writes JSON fault reports to /tmp/l104_bridge/nano/c_outbox
 * Heartbeat: /tmp/l104_bridge/nano/c_heartbeat (timestamp file)
 *
 * Build: Part of l104_core_c (Makefile: `make nano-daemon`)
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifndef L104_NANO_DAEMON_H
#define L104_NANO_DAEMON_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * SACRED CONSTANTS — Bit-Exact Reference Values
 * ═══════════════════════════════════════════════════════════════════════════ */

#define L104_NANO_VERSION         "1.0.0"
#define L104_NANO_GOD_CODE        527.5184818492612
#define L104_NANO_PHI             1.618033988749895
#define L104_NANO_VOID_CONSTANT   1.0416180339887497
#define L104_NANO_OMEGA           6539.34712682

/* IEEE 754 bit patterns for exact comparison */
#define L104_GOD_CODE_BITS        0x408079E4D2ADE09AULL  /* exact double bits */
#define L104_PHI_BITS             0x3FF9E3779B97F4A8ULL
#define L104_VOID_BITS            0x3FF0AA8BC5A11466ULL

/* Nano fault severity levels */
typedef enum {
    NANO_SEV_TRACE    = 0,  /* Sub-ULP drift, informational only */
    NANO_SEV_LOW      = 1,  /* 1-2 ULP drift, watchable */
    NANO_SEV_MEDIUM   = 2,  /* 3-8 ULP drift, concerning */
    NANO_SEV_HIGH     = 3,  /* >8 ULP drift or silent FPU flag */
    NANO_SEV_CRITICAL = 4,  /* Canary corruption, NaN, sacred violation */
} l104_nano_severity_t;

/* Nano fault categories */
typedef enum {
    NANO_FAULT_CONSTANT_DRIFT    = 0,
    NANO_FAULT_MEMORY_CANARY     = 1,
    NANO_FAULT_FPU_FLAGS         = 2,
    NANO_FAULT_CACHE_COHERENCE   = 3,
    NANO_FAULT_ENTROPY_HEALTH    = 4,
    NANO_FAULT_NUMERICAL_STAB    = 5,
    NANO_FAULT_PHASE_DRIFT       = 6,
    NANO_FAULT_NDE_INTEGRITY     = 7,
    NANO_FAULT_COUNT             = 8,
} l104_nano_fault_type_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * CORE DATA STRUCTURES
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Individual nano fault report */
typedef struct {
    l104_nano_fault_type_t type;
    l104_nano_severity_t   severity;
    double                 measured_value;
    double                 expected_value;
    double                 deviation;
    uint64_t               timestamp_ns;     /* nanosecond timestamp */
    int                    ulp_distance;     /* ULPs off for float faults */
    char                   description[256];
} l104_nano_fault_t;

/* Memory canary sentinel */
typedef struct {
    uint64_t canary_phi;       /* φ-scrambled sentinel value */
    uint64_t canary_god;       /* GOD_CODE-scrambled sentinel */
    uint64_t canary_void;      /* VOID_CONSTANT-scrambled sentinel */
    uint64_t canary_checksum;  /* XOR checksum of all three */
    uint64_t hamming_baseline; /* Expected Hamming weight */
} l104_nano_canary_t;

/* FPU environment snapshot */
typedef struct {
    bool overflow;
    bool underflow;
    bool denormal;
    bool divbyzero;
    bool inexact;
    bool invalid;
    int  rounding_mode;        /* FE_TONEAREST, etc. */
} l104_nano_fpu_state_t;

/* Entropy health result */
typedef struct {
    double chi_squared;        /* Chi-squared statistic */
    double poker_score;        /* 4-bit poker test */
    double serial_correlation; /* Serial correlation coefficient */
    bool   healthy;            /* Overall entropy health */
    int    bytes_tested;
} l104_nano_entropy_health_t;

/* Numerical stability probe result */
typedef struct {
    double phi_recurrence_error;   /* |fib(n)/fib(n-1) - φ| after N iterations */
    double god_code_roundtrip;     /* GOD_CODE after encode-decode-reencode cycle */
    double cancellation_loss;      /* Significance bits lost in worst case */
    int    underflow_count;        /* Subnormal results detected */
    bool   stable;                 /* Overall stability verdict */
} l104_nano_stability_t;

/* Phase drift accumulation result */
typedef struct {
    double accumulated_phase;      /* Phase after N modular reductions */
    double expected_phase;         /* Analytically correct phase */
    double drift_radians;          /* Accumulated drift in radians */
    double drift_ulps;             /* Drift in ULPs */
    int    iterations;
} l104_nano_phase_drift_t;

/* Full nano daemon tick result */
typedef struct {
    /* Per-probe results */
    l104_nano_fpu_state_t       fpu;
    l104_nano_entropy_health_t  entropy;
    l104_nano_stability_t       stability;
    l104_nano_phase_drift_t     phase;

    /* Fault collection */
    l104_nano_fault_t faults[64];
    int               fault_count;

    /* Summary */
    double health_score;           /* 0.0 – 1.0 overall nano-health */
    uint64_t tick_duration_ns;     /* How long the tick took */
    uint64_t tick_number;          /* Monotonic tick counter */
    uint64_t total_faults_ever;    /* Lifetime fault counter */
} l104_nano_tick_result_t;

/* Daemon state */
typedef struct {
    l104_nano_canary_t canary;
    l104_nano_tick_result_t last_tick;
    uint64_t tick_count;
    uint64_t total_faults;
    double   health_trend;        /* Exponential moving average of health */
    bool     running;
    int      pid;
} l104_nano_daemon_state_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * API — Nano Daemon Core
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Initialize the nano daemon (set up canaries, FPU state, IPC dirs) */
int l104_nano_daemon_init(l104_nano_daemon_state_t *state);

/* Run one full nano tick (all 8 probes) — returns fault count */
int l104_nano_daemon_tick(l104_nano_daemon_state_t *state);

/* Shutdown nano daemon (persist state, remove PID file) */
void l104_nano_daemon_shutdown(l104_nano_daemon_state_t *state);

/* ═══════════════════════════════════════════════════════════════════════════
 * API — L104Daemon-grade Lifecycle Assertions
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Validate configuration — returns 1 on success, 0 on failure.
 * Checks: tick bounds, sacred constant bit-exactness, IPC directories,
 * system memory resources. Mirrors L104Daemon.validateConfiguration(). */
/* (static in .c — called internally before daemon_run) */

/* Self-test (returns 0 on pass, >0 = number of failures) */
int l104_nano_self_test(void);

/* Main daemon loop (blocking — call from main() or thread) */
int l104_nano_daemon_run(int tick_interval_ms);

/* ═══════════════════════════════════════════════════════════════════════════
 * API — Individual Nano Probes
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Probe 1: Sacred constant ULP drift detection */
int l104_nano_probe_constant_drift(l104_nano_fault_t *faults, int max_faults);

/* Probe 2: Memory canary integrity check */
int l104_nano_probe_canary(l104_nano_canary_t *canary, l104_nano_fault_t *faults, int max_faults);

/* Probe 3: FPU environment flags check */
l104_nano_fpu_state_t l104_nano_probe_fpu(l104_nano_fault_t *faults, int max_faults);

/* Probe 4: Cache line coherence timing */
int l104_nano_probe_cache_coherence(l104_nano_fault_t *faults, int max_faults);

/* Probe 5: Entropy source health */
l104_nano_entropy_health_t l104_nano_probe_entropy(l104_nano_fault_t *faults, int max_faults);

/* Probe 6: Numerical stability audit */
l104_nano_stability_t l104_nano_probe_stability(l104_nano_fault_t *faults, int max_faults);

/* Probe 7: Phase drift detection */
l104_nano_phase_drift_t l104_nano_probe_phase_drift(l104_nano_fault_t *faults, int max_faults);

/* Probe 8: NDE pipeline integrity */
int l104_nano_probe_nde_integrity(l104_nano_fault_t *faults, int max_faults);

/* ═══════════════════════════════════════════════════════════════════════════
 * API — Utilities
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ULP distance between two doubles */
int64_t l104_ulp_distance(double a, double b);

/* Hamming distance between two 64-bit values */
int l104_hamming_distance(uint64_t a, uint64_t b);

/* Get IEEE 754 bits of a double */
uint64_t l104_double_to_bits(double v);

/* Nanosecond timestamp */
uint64_t l104_nano_timestamp(void);

/* Write JSON fault report to IPC outbox */
int l104_nano_write_report(const l104_nano_tick_result_t *result);

/* Write heartbeat file */
void l104_nano_heartbeat(void);

/* Self-test (returns 0 on pass, >0 = number of failures) */
int l104_nano_self_test(void);

/* Main daemon loop (blocking — call from main() or thread) */
int l104_nano_daemon_run(int tick_interval_ms);

#ifdef __cplusplus
}
#endif

#endif /* L104_NANO_DAEMON_H */
