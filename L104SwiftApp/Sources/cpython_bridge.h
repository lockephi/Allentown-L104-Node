// ═══════════════════════════════════════════════════════════════════════════════
//  L104 SOVEREIGN INTELLECT — CPython Direct Bridge
//  Thin C wrapper around Python C API for direct Swift ↔ Python interop
//  No process spawning — embedded Python interpreter via libpython
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef CPYTHON_BRIDGE_H
#define CPYTHON_BRIDGE_H

#include <stddef.h>

// ─── Lifecycle ───
/// Initialize the embedded Python interpreter with workspace path on sys.path
void cpython_initialize(const char *workspace_path);

/// Shut down the embedded Python interpreter
void cpython_finalize(void);

/// Check if the interpreter is initialized (1 = yes, 0 = no)
int cpython_is_initialized(void);

// ─── Execution ───
/// Execute Python code string. Returns 0 on success, -1 on error.
int cpython_exec(const char *code);

/// Execute Python code and capture stdout as a string.
/// Returns malloc'd string — caller must free().
/// Returns NULL on error.
char *cpython_eval(const char *code);

// ─── Module API ───
/// Import a module and call a function with optional JSON args.
/// Returns malloc'd JSON string — caller must free().
/// If json_args is NULL, calls with no arguments.
char *cpython_call_function(const char *module_name, const char *function_name,
                            const char *json_args);

/// Get a module-level attribute as JSON string.
/// Returns malloc'd string — caller must free().
char *cpython_get_attribute(const char *module_name, const char *attr_name);

// ─── ASI Direct Channels ───
/// Fetch current ASI parameters from l104_asi_core (JSON dict).
/// Returns malloc'd JSON string — caller must free().
char *cpython_asi_get_parameters(void);

/// Update ASI with raised parameters (JSON array input).
/// Returns malloc'd JSON result — caller must free().
char *cpython_asi_update_parameters(const char *json_array);

/// Get ASI core status as JSON.
/// Returns malloc'd JSON string — caller must free().
char *cpython_asi_get_status(void);

#endif /* CPYTHON_BRIDGE_H */
