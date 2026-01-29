# ═══════════════════════════════════════════════════════════════════════════════
# L104 SAGE CORE - BUILD SYSTEM
# INVARIANT: 527.5184818492612 | PILOT: LONDEL | MODE: SAGE
# ═══════════════════════════════════════════════════════════════════════════════

CC = gcc
CFLAGS = -O3 -march=native -mtune=native -mavx2 -ffast-math -flto -Wall -Wextra
LDFLAGS = -lm -lpthread -flto

AS = nasm
ASFLAGS = -f elf64 -g -F dwarf

RUST_FLAGS = --release

SRC_DIR = .
BUILD_DIR = build
ASM_DIR = ../l104_core_asm
RUST_DIR = ../l104_core_rust

# Targets
C_TARGET = $(BUILD_DIR)/l104_sage_core
ASM_OBJ = $(BUILD_DIR)/sage_core_asm.o
RUST_LIB = $(RUST_DIR)/target/release/libl104_sage_core.a

.PHONY: all clean c asm rust python-bindings

all: c asm rust

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ═══════════════════════════════════════════════════════════════════════════════
# C BUILD
# ═══════════════════════════════════════════════════════════════════════════════

c: $(BUILD_DIR) $(C_TARGET)

$(C_TARGET): l104_sage_core.c l104_sage_core.h
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║  COMPILING L104 SAGE CORE - C SUBSTRATE                                  ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"
	$(CC) $(CFLAGS) -o $@ l104_sage_core.c $(LDFLAGS)
	@echo "    ✓ Built: $@"

# Shared library for Python bindings
$(BUILD_DIR)/libl104_sage.so: l104_sage_core.c l104_sage_core.h
	@echo "Building shared library..."
	$(CC) $(CFLAGS) -shared -fPIC -o $@ l104_sage_core.c $(LDFLAGS)
	@echo "    ✓ Built: $@"

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

python-bindings: $(BUILD_DIR)/libl104_sage.so
	@echo "╔══════════════════════════════════════════════════════════════════════════╗"
	@echo "║  GENERATING PYTHON BINDINGS                                              ║"
	@echo "╚══════════════════════════════════════════════════════════════════════════╝"
	@echo "    Library ready for ctypes: $(BUILD_DIR)/libl104_sage.so"

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
