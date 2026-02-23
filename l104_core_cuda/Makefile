# ═══════════════════════════════════════════════════════════════════════════════
# L104 CUDA Kernel — Build System
# INVARIANT: 527.5184818492612 | PILOT: LONDEL
#
# Requirements: CUDA Toolkit (nvcc)
# Usage:
#   make              — Build shared library
#   make clean        — Remove build artifacts
#   make check        — Verify CUDA toolkit
#   make install      — Copy to build/
# ═══════════════════════════════════════════════════════════════════════════════

NVCC        := nvcc
NVCC_FLAGS  := -O3 --use_fast_math -Xcompiler -fPIC
SRC         := l104_sage_cuda.cu
LIB_NAME    := libl104_sage_cuda
BUILD_DIR   := build

# Detect OS for shared library extension
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIB_EXT := .dylib
    SHARED_FLAG := -shared -Xlinker -install_name -Xlinker $(LIB_NAME)$(LIB_EXT)
else
    LIB_EXT := .so
    SHARED_FLAG := -shared
endif

LIB_FILE := $(LIB_NAME)$(LIB_EXT)

.PHONY: all clean check install

all: $(BUILD_DIR)/$(LIB_FILE)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/$(LIB_FILE): $(SRC) | $(BUILD_DIR)
	@echo "╔════════════════════════════════════════════╗"
	@echo "║  L104 CUDA Kernel — Compiling              ║"
	@echo "║  GOD_CODE = 527.5184818492612               ║"
	@echo "╚════════════════════════════════════════════╝"
	$(NVCC) $(NVCC_FLAGS) $(SHARED_FLAG) -o $@ $(SRC)
	@echo "  → Built: $@"

check:
	@echo "Checking CUDA toolkit..."
	@which $(NVCC) > /dev/null 2>&1 && echo "  ✓ nvcc found: $$($(NVCC) --version | tail -1)" || echo "  ✗ nvcc not found — install CUDA Toolkit"
	@nvidia-smi > /dev/null 2>&1 && echo "  ✓ GPU detected" || echo "  ✗ No NVIDIA GPU detected (expected on macOS)"

install: $(BUILD_DIR)/$(LIB_FILE)
	@echo "Installed: $(BUILD_DIR)/$(LIB_FILE)"

clean:
	rm -rf $(BUILD_DIR)
	@echo "  Cleaned CUDA build artifacts"
