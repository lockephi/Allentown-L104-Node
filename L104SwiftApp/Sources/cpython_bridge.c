// ═══════════════════════════════════════════════════════════════════════════════
//  L104 SOVEREIGN INTELLECT — CPython Direct Bridge Implementation
//  Embeds the Python interpreter directly into the L104 process
//  Uses Python C API for zero-overhead Swift ↔ Python parameter exchange
// ═══════════════════════════════════════════════════════════════════════════════

#include "cpython_bridge.h"

// Python.h must come before any standard headers (Python requirement)
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ─── State ───
static int py_initialized = 0;

// ─── Internal Helpers ───

/// Capture Python stdout into a string via io.StringIO
static char *capture_stdout(const char *code) {
  if (!py_initialized)
    return NULL;

  // Redirect stdout to StringIO, execute code, capture result
  const char *template = "import io as _io, sys as _sys\n"
                         "_old_stdout = _sys.stdout\n"
                         "_sys.stdout = _io.StringIO()\n"
                         "try:\n"
                         "    exec(%s)\n"
                         "except Exception as _e:\n"
                         "    print(f'ERROR: {_e}')\n"
                         "_captured = _sys.stdout.getvalue()\n"
                         "_sys.stdout = _old_stdout\n";

  // Triple-quote the code for safe embedding
  size_t code_len = strlen(code);
  size_t buf_len = code_len + 4096;
  char *wrapper = (char *)malloc(buf_len);
  if (!wrapper)
    return NULL;

  // Build: exec("""<code>""")
  char *quoted_code = (char *)malloc(code_len + 16);
  if (!quoted_code) {
    free(wrapper);
    return NULL;
  }
  snprintf(quoted_code, code_len + 16, "\"\"\"%s\"\"\"", code);

  snprintf(wrapper, buf_len, template, quoted_code);
  free(quoted_code);

  if (PyRun_SimpleString(wrapper) != 0) {
    free(wrapper);
    return NULL;
  }
  free(wrapper);

  // Extract _captured from __main__
  PyObject *main_module = PyImport_AddModule("__main__");
  if (!main_module)
    return NULL;

  PyObject *main_dict = PyModule_GetDict(main_module);
  PyObject *captured = PyDict_GetItemString(main_dict, "_captured");
  if (!captured || !PyUnicode_Check(captured))
    return NULL;

  const char *result_str = PyUnicode_AsUTF8(captured);
  if (!result_str)
    return NULL;

  char *output = strdup(result_str);

  // Clean up temp variables
  PyDict_DelItemString(main_dict, "_captured");
  PyDict_DelItemString(main_dict, "_old_stdout");

  return output;
}

// ─── Lifecycle ───

void cpython_initialize(const char *workspace_path) {
  if (py_initialized)
    return;

  // Set PYTHONPATH before init
  if (workspace_path) {
    setenv("PYTHONPATH", workspace_path, 1);
    // Also ensure Python doesn't buffer output
    setenv("PYTHONUNBUFFERED", "1", 1);
  }

  Py_Initialize();
  py_initialized = 1;

  // Add workspace to sys.path and configure
  if (workspace_path) {
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
             "import sys, os\n"
             "workspace = '%s'\n"
             "if workspace not in sys.path:\n"
             "    sys.path.insert(0, workspace)\n"
             "# Also add .venv site-packages if present\n"
             "venv_site = os.path.join(workspace, '.venv', 'lib')\n"
             "if os.path.isdir(venv_site):\n"
             "    for d in os.listdir(venv_site):\n"
             "        sp = os.path.join(venv_site, d, 'site-packages')\n"
             "        if os.path.isdir(sp) and sp not in sys.path:\n"
             "            sys.path.insert(1, sp)\n"
             "os.chdir(workspace)\n",
             workspace_path);
    PyRun_SimpleString(cmd);
  }
}

void cpython_finalize(void) {
  if (py_initialized) {
    Py_Finalize();
    py_initialized = 0;
  }
}

int cpython_is_initialized(void) { return py_initialized; }

// ─── Execution ───

int cpython_exec(const char *code) {
  if (!py_initialized)
    return -1;
  return PyRun_SimpleString(code);
}

char *cpython_eval(const char *code) { return capture_stdout(code); }

// ─── Module API ───

char *cpython_call_function(const char *module_name, const char *function_name,
                            const char *json_args) {
  if (!py_initialized)
    return NULL;

  // Build Python code to import module, call function, return JSON
  size_t buf_len = 4096;
  if (json_args)
    buf_len += strlen(json_args);

  char *code = (char *)malloc(buf_len);
  if (!code)
    return NULL;

  if (json_args && strlen(json_args) > 0) {
    snprintf(code, buf_len,
             "import json\n"
             "from %s import %s\n"
             "args = json.loads('%s')\n"
             "if isinstance(args, list):\n"
             "    result = %s(*args)\n"
             "elif isinstance(args, dict):\n"
             "    result = %s(**args)\n"
             "else:\n"
             "    result = %s(args)\n"
             "print(json.dumps(result, default=str))\n",
             module_name, function_name, json_args, function_name,
             function_name, function_name);
  } else {
    snprintf(code, buf_len,
             "import json\n"
             "from %s import %s\n"
             "result = %s()\n"
             "print(json.dumps(result, default=str))\n",
             module_name, function_name, function_name);
  }

  char *result = capture_stdout(code);
  free(code);
  return result;
}

char *cpython_get_attribute(const char *module_name, const char *attr_name) {
  if (!py_initialized)
    return NULL;

  char code[2048];
  snprintf(
      code, sizeof(code),
      "import json\n"
      "import %s\n"
      "val = getattr(%s, '%s')\n"
      "if callable(val):\n"
      "    print(json.dumps({'type': 'callable', 'name': '%s'}, default=str))\n"
      "else:\n"
      "    print(json.dumps(val, default=str))\n",
      module_name, module_name, attr_name, attr_name);

  return capture_stdout(code);
}

// ─── ASI Direct Channels ───

char *cpython_asi_get_parameters(void) {
  return cpython_call_function("l104_asi_core", "get_current_parameters", NULL);
}

char *cpython_asi_update_parameters(const char *json_array) {
  if (!py_initialized || !json_array)
    return NULL;

  size_t buf_len = strlen(json_array) + 512;
  char *code = (char *)malloc(buf_len);
  if (!code)
    return NULL;

  snprintf(code, buf_len,
           "import json\n"
           "from l104_asi_core import update_parameters\n"
           "result = update_parameters(json.loads('%s'))\n"
           "print(json.dumps(result, default=str))\n",
           json_array);

  char *result = capture_stdout(code);
  free(code);
  return result;
}

char *cpython_asi_get_status(void) {
  return cpython_call_function("l104_asi_core", "asi_core.get_status", NULL);
}
