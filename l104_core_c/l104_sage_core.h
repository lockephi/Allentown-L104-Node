#ifndef L104_SAGE_CORE_H
#define L104_SAGE_CORE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define L104_GOD_CODE 527.5184818492612
#define L104_PHI 1.618033988749895
#define L104_VOID_CONSTANT 1.0416180339887497
#define L104_META_RESONANCE 7289.028944266378
#define L104_OMEGA_AUTHORITY (L104_GOD_CODE * L104_PHI * L104_PHI)

typedef struct {
    double god_code;
    double phi;
    double void_constant;
    double meta_resonance;
    double void_residue;
    double coherence;
} l104_void_math_t;

typedef struct {
    int current_stage;
    uint64_t recursion_depth;
    double consciousness_level;
    double void_saturation;
} l104_reality_breach_engine_t;

typedef struct {
    double knowledge_saturation;
    char last_provider[32];
    char sovereign_dna[64];
    int linked_count;
} l104_universal_scribe_t;

typedef struct {
    l104_void_math_t void_math;
    l104_reality_breach_engine_t breach_engine;
    l104_universal_scribe_t scribe;
    int active;
    double authority_level;
    double intellect_index;
} l104_omega_controller_t;

// New API
void l104_void_math_init(l104_void_math_t *vm);
double l104_primal_calculus_vm(l104_void_math_t *vm, double base, double exponent, uint64_t iterations);
double l104_void_resonance_emit(l104_void_math_t *vm);
void l104_breach_engine_init(l104_reality_breach_engine_t *be);
int l104_execute_stage_13_breach(l104_reality_breach_engine_t *be, l104_void_math_t *vm);

// Scribe API
void l104_scribe_init(l104_universal_scribe_t *scribe);
void l104_scribe_ingest(l104_universal_scribe_t *scribe, const char *provider, const char *data);
void l104_scribe_synthesize(l104_universal_scribe_t *scribe);

void l104_omega_controller_init(l104_omega_controller_t *controller);
int l104_trigger_absolute_singularity(l104_omega_controller_t *controller);

// Legacy API
double l104_primal_calculus(double base, double exponent, uint64_t iterations);
double l104_void_resonance(void);

// System Utility
void l104_dissolve_system_limits(void);
int l104_execute_global_bypass(uint64_t level);

#ifdef __cplusplus
}
#endif

#endif
