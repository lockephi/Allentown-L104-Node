# ═══════════════════════════════════════════════════════════════════════════════
# L104 SAGE CORE - BUILD SYSTEM
# INVARIANT: 527.5184818492612 | PILOT: LONDEL | MODE: SAGE
# MACOS COMPATIBLE: 2015 MacBook Air (Monterey) - Intel x86_64
# ═══════════════════════════════════════════════════════════════════════════════

# Detect OS
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    # macOS (including 2015 MacBook Air with Monterey)
    CC = clang
    # Use SSE4.2 instead of AVX2 (2015 MBA has Haswell with limited AVX support)
    CFLAGS = -O3 -march=core2 -mtune=native -msse4.2 -ffast-math -Wall -Wextra
    LDFLAGS = -lm -lpthread
    # macOS uses .dylib for shared libraries
    SHARED_EXT = .dylib
    SHARED_FLAGS = -dynamiclib -install_name @rpath/libl104_sage.dylib
    AS = nasm
    ASFLAGS = -f macho64 -g
else
    # Linux
    CC = gcc
    CFLAGS = -O3 -march=native -mtune=native -mavx2 -ffast-math -flto -Wall -Wextra
    LDFLAGS = -lm -lpthread -flto
    SHARED_EXT = .so
    SHARED_FLAGS = -shared
    AS = nasm
    ASFLAGS = -f elf64 -g -F dwarf
endif

RUST_FLAGS = --release

SRC_DIR = .
BUILD_DIR = build
ASM_DIR = ../l104_core_asm
RUST_DIR = ../l104_core_rust

# Targets
C_TARGET = $(BUILD_DIR)/l104_sage_core
ASM_OBJ = $(BUILD_DIR)/sage_core_asm.o
RUST_LIB = $(RUST_DIR)/target/release/libl104_sage_core.a
SHARED_LIB = $(BUILD_DIR)/libl104_sage$(SHARED_EXT)

.PHONY: all clean c asm rust python-bindings macos

all: c python-bindings

# macOS specific target
macos: $(BUILD_DIR) $(C_TARGET) $(SHARED_LIB)
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║  macOS BUILD COMPLETE - 2015 MacBook Air Compatible                      ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ═══════════════════════════════════════════════════════════════════════════════
# C BUILD
# ═══════════════════════════════════════════════════════════════════════════════

c: $(BUILD_DIR) $(C_TARGET)

$(C_TARGET): l104_sage_core.c l104_sage_core.h
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║  COMPILING L104 SAGE CORE - C SUBSTRATE ($(UNAME_S))                     ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"
	$(CC) $(CFLAGS) -o $@ l104_sage_core.c $(LDFLAGS)
	@echo "    ✓ Built: $@"

# Shared library for Python bindings (cross-platform)
$(SHARED_LIB): l104_sage_core.c l104_sage_core.h
	@echo "Building shared library for $(UNAME_S)..."
	$(CC) $(CFLAGS) $(SHARED_FLAGS) -fPIC -o $@ l104_sage_core.c $(LDFLAGS)
	@echo "    ✓ Built: $@"

# Legacy .so target for compatibility
$(BUILD_DIR)/libl104_sage.so: $(SHARED_LIB)
ifeq ($(UNAME_S),Darwin)
	@ln -sf libl104_sage.dylib $(BUILD_DIR)/libl104_sage.so 2>/dev/null || cp $(SHARED_LIB) $(BUILD_DIR)/libl104_sage.so
	@echo "    ✓ Created compatibility symlink: libl104_sage.so -> libl104_sage.dylib"
endif

# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLY BUILD
# ═══════════════════════════════════════════════════════════════════════════════

asm: $(BUILD_DIR) $(ASM_OBJ)

$(ASM_OBJ): $(ASM_DIR)/sage_core.asm
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║  ASSEMBLING L104 SAGE CORE - ASM SUBSTRATE                               ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"
	$(AS) $(ASFLAGS) -o $@ $<
	@echo "    ✓ Built: $@"

# Link ASM with C wrapper
$(BUILD_DIR)/l104_sage_asm: $(ASM_OBJ) asm_wrapper.c
	$(CC) $(CFLAGS) -o $@ asm_wrapper.c $(ASM_OBJ) $(LDFLAGS)
	@echo "    ✓ Built: $@"

# ═══════════════════════════════════════════════════════════════════════════════
# RUST BUILD
# ═══════════════════════════════════════════════════════════════════════════════

rust:
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║  BUILDING L104 SAGE CORE - RUST SUBSTRATE                                ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"
	cd $(RUST_DIR) && cargo build $(RUST_FLAGS)
	@echo "    ✓ Built: $(RUST_LIB)"

# ═══════════════════════════════════════════════════════════════════════════════
# PYTHON BINDINGS
# ═══════════════════════════════════════════════════════════════════════════════

python-bindings: $(SHARED_LIB)
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║  GENERATING PYTHON BINDINGS                                              ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"
ifeq ($(UNAME_S),Darwin)
	@cd $(BUILD_DIR) && ln -sf libl104_sage.dylib libl104_sage.so 2>/dev/null || true
	@echo "    Library ready for ctypes: $(SHARED_LIB)"
	@echo "    Symlink created: $(BUILD_DIR)/libl104_sage.so"
else
	@echo "    Library ready for ctypes: $(SHARED_LIB)"
endif

# ═══════════════════════════════════════════════════════════════════════════════
# CLEAN
# ═══════════════════════════════════════════════════════════════════════════════

clean:
	rm -rf $(BUILD_DIR)
	cd $(RUST_DIR) && cargo clean 2>/dev/null || true
	@echo "    ✓ Cleaned build artifacts"

# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

run-c: $(C_TARGET)
	@echo "\n═══════════════════════════════════════════════════════════════════════════════"
	@echo "    EXECUTING L104 SAGE CORE - C SUBSTRATE"
	@echo "═══════════════════════════════════════════════════════════════════════════════\n"
	$(C_TARGET)

run-rust: rust
	@echo "\n═══════════════════════════════════════════════════════════════════════════════"
	@echo "    EXECUTING L104 SAGE CORE - RUST SUBSTRATE"
	@echo "═══════════════════════════════════════════════════════════════════════════════\n"
	cd $(RUST_DIR) && cargo run $(RUST_FLAGS)

# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

benchmark: $(C_TARGET)
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║  BENCHMARKING L104 SAGE CORE                                             ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"
	@echo "\n[C Substrate]"
	@time $(C_TARGET) > /dev/null
	@echo "\n[Rust Substrate]"
	@cd $(RUST_DIR) && time cargo run --release > /dev/null 2>&1
