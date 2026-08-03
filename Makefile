# stoke — one Makefile for Windows (mingw32-make, MSYS2 ucrt64 gcc) and
# WSL/Linux (make, gcc). Pattern mirrors ../loom/Makefile (a proven build on
# this exact box):
#   make                  build everything      -> build/win or build/linux
#   make test             build + run every tests/test_*.c binary
#   make SAN=asan test    ASan+UBSan (WSL/Linux) -> build/linux-asan
#   make SAN=tsan test    TSan (WSL/Linux)       -> build/linux-tsan
#   make clean            remove build/
#
# One BUILDDIR per (platform, sanitizer) combination, NEVER shared: object
# files built with different flags landing in the same directory is how you
# get a silent stale-.o link failure (AySen lesson -- see the plan's Global
# Constraints; `rm -rf build/<dir>` is the fix if a link ever looks wrong).

# Both mingw32-make (MSYS2 ucrt64) and WSL make provide sh; pin it so
# recipes are POSIX everywhere and never fall back to cmd.exe syntax.
SHELL := sh

ifeq ($(origin CC),default)
  CC := gcc
endif

# -O3 since 2026-08-03 (was -O1, inherited from ../loom and never defended).
# Rebuilding at -O3 made every matvec 3.1x faster with byte-identical output
# and a green matrix; the llama.cpp this is benchmarked against ships release
# builds, so -O1 was measuring the wrong binary. Ruling:
# docs/superpowers/memos/2026-08-03-m3-optimization-level-RULING-FABLE.md.
#
# -ffp-contract=off is PINNED, not inherited. Under -std=c11 both toolchains
# here already refuse to contract a*b+c (verified from codegen at -mfma:
# Windows gcc 15.2 and WSL gcc 13.3 both emit vmulss+vaddss, while the same
# source in gnu17 mode emits vfmadd132ss on both -- so the probe is sensitive
# and the agreement is real, not a blind check). It is pinned anyway because
# K3 compiles SIMD files with -mfma, where contraction becomes POSSIBLE, and
# a future toolchain flipping this ISO-mode default would break scalar<->SIMD
# f32 exactness (forward-note gate [1]) silently and cross-leg. Note that
# `gcc -Q --help=common` reports `fast` on BOTH legs and is therefore the
# wrong instrument here -- it prints the raw option default, not the
# language-adjusted value codegen actually uses.
CFLAGS  := -std=c11 -Wall -Wextra -Werror -g -O3 -ffp-contract=off -Isrc
LDFLAGS :=
LIBS    := -lm
SAN     ?=

ifeq ($(OS),Windows_NT)
  PLAT := win
  EXE  := .exe
else
  PLAT := linux
  EXE  :=
endif

# Sanitizer legs build at -O1, deliberately, and the trailing -O1 is what
# takes effect (gcc honours the LAST -O on the command line). They are NOT
# raised with the default: instrumentation gets LESS sensitive as the
# optimizer deletes or hoists the very accesses it exists to catch, and the
# diagnostics get harder to read. An instrumented binary is never the shipped
# code, so "what ships is what is sanitized" is not a requirement worth
# paying for -- what verifies the shipped artifact is the un-instrumented
# plain legs (Windows + WSL) running the same matrix at -O3.
SAN_SUFFIX :=
ifeq ($(SAN),asan)
  CFLAGS     += -O1 -fsanitize=address,undefined -fno-sanitize-recover=all
  LDFLAGS    += -fsanitize=address,undefined
  SAN_SUFFIX := -asan
endif
ifeq ($(SAN),tsan)
  CFLAGS     += -O1 -fsanitize=thread
  LDFLAGS    += -fsanitize=thread
  SAN_SUFFIX := -tsan
  # TSan reserves fixed shadow-memory regions at startup and dies with
  # "FATAL: ThreadSanitizer: unexpected memory mapping" when the kernel's
  # ASLR entropy places the binary where it wants to map. Newer kernels
  # (measured here: WSL2 6.6.87.2) default vm.mmap_rnd_bits high enough to
  # hit this, and the leg fails at PROCESS STARTUP -- 33/47 on 2026-08-03,
  # with a diagnostic that looks nothing like a test failure and reports
  # binaries that pass everywhere else.
  #
  # setarch -R disables randomization for the run and needs no privileges,
  # unlike the usual `sysctl vm.mmap_rnd_bits=28` advice. Disabling ASLR
  # under a sanitizer weakens nothing TSan checks -- it does not detect
  # races by address randomization.
  #
  # Resolved at USE time, not assigned here, so a box without setarch (or a
  # non-Linux one) simply runs the tests directly instead of failing on a
  # missing tool. THIS IS GATE CONDITION [4]'s LEG: K5 lands the thread
  # pool, and a tsan leg that cannot start is the one failure mode that
  # could be mistaken for a green matrix.
  TSAN_RUN = $(shell command -v setarch >/dev/null 2>&1 && \
                     echo "setarch $$(uname -m) -R")
endif

BUILDDIR := build/$(PLAT)$(SAN_SUFFIX)

TEST_SRC := $(wildcard tests/test_*.c)
TEST_BIN := $(patsubst tests/%.c,$(BUILDDIR)/%$(EXE),$(TEST_SRC))
HEADERS  := $(wildcard src/*.h) $(wildcard tests/*.h)

# src/main.c holds `stoke`'s own main() -- it must NOT be linked into test
# binaries (test.h's main() would collide with it), so it's built/linked
# only into the $(BUILDDIR)/stoke target below. Every other src/*.c is a
# library object shared by every test binary and by stoke itself.
LIB_SRC  := $(filter-out src/main.c,$(wildcard src/*.c)) $(wildcard src/kernels/*.c)
LIB_OBJ  := $(patsubst src/%.c,$(BUILDDIR)/%.o,$(LIB_SRC))
MAIN_OBJ := $(BUILDDIR)/main.o
STOKE_BIN := $(BUILDDIR)/stoke$(EXE)

.PHONY: all test clean

all: $(TEST_BIN) $(STOKE_BIN)

test: $(TEST_BIN) $(STOKE_BIN)
	$(TSAN_RUN) sh tests/run_tests.sh $(BUILDDIR) $(EXE)
	$(TSAN_RUN) sh tests/golden_info.sh $(BUILDDIR) $(EXE)
	$(TSAN_RUN) sh tests/tok_golden.sh $(BUILDDIR) $(EXE)
	$(TSAN_RUN) sh tests/tok_special_golden.sh $(BUILDDIR) $(EXE)
	$(TSAN_RUN) sh tests/unicode_sweep_golden.sh $(BUILDDIR) $(EXE)

# One BUILDDIR per (platform,sanitizer) combination (see header comment) --
# object files live here too, built with this invocation's exact CFLAGS, so
# a plain build's .o's are never accidentally linked into an ASan/TSan binary
# or vice versa.
#
# Makefile is a PREREQUISITE of every object: the BUILDDIR split covers the
# (platform, sanitizer) axes but NOT the flags themselves, so before this was
# added, editing CFLAGS left every stale .o in place and linked it into a
# binary claiming the new flags. The -O1 -> -O3 change that added this line is
# exactly that case.
$(BUILDDIR)/%.o: src/%.c $(HEADERS) Makefile | $(BUILDDIR)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

# bench.o alone is compiled with its own CFLAGS baked in as a string, so
# `bench --ops --jsonl` can report the flags the binary was actually built
# with and bench/history.jsonl rows stay comparable across a flag change.
# This is an EXPLICIT rule overriding the pattern rule above for one target;
# the define is passed on the command line rather than added to CFLAGS,
# because a target-specific `CFLAGS += ...$(CFLAGS)...` recurses in make.
# The recorded string therefore excludes this -D itself, which is correct --
# it is metadata about the build, not part of it.
$(BUILDDIR)/bench.o: src/bench.c $(HEADERS) Makefile | $(BUILDDIR)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -DSTOKE_BUILD_CFLAGS='"$(CFLAGS)"' -c -o $@ $<

# K3: the SIMD twins, and ONLY these, are compiled with the AVX2/FMA/F16C/
# AVX-VNNI ISA flags. Everything else stays at the baseline x86-64 the rest
# of the project targets.
#
# THE SCOPE IS THE POINT, not a tidiness preference. Putting these in the
# global CFLAGS would let gcc emit AVX2 anywhere -- including in the SCALAR
# kernels the twins are defined as agreeing with -- and the binary would then
# fault on any pre-Haswell CPU despite carrying a runtime probe and a
# complete scalar fallback. The fallback only means anything if the code
# reachable without the feature was compiled without it.
#
# -mfma is why -ffp-contract=off is pinned globally (see the CFLAGS note): in
# THIS file a*b+c may legally contract, and a contracted f32 op is a
# different result from the scalar reference, which would break gate [1]'s
# f32 leg silently and only on the legs whose toolchain default differs.
SIMD_CFLAGS := -mavx2 -mfma -mf16c -mavxvnni

$(BUILDDIR)/kernels/simd.o: src/kernels/simd.c $(HEADERS) Makefile | $(BUILDDIR)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(SIMD_CFLAGS) -c -o $@ $<

$(BUILDDIR)/%$(EXE): tests/%.c $(LIB_OBJ) $(HEADERS) Makefile | $(BUILDDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIB_OBJ) $(LIBS)

$(STOKE_BIN): $(MAIN_OBJ) $(LIB_OBJ) Makefile | $(BUILDDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(MAIN_OBJ) $(LIB_OBJ) $(LIBS)

$(BUILDDIR):
	mkdir -p $@

clean:
	rm -rf build
