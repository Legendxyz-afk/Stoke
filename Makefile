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
#
# EXTRA_CFLAGS is the A/B ARM HOOK, and it exists so an arm is never built by
# restating CFLAGS on the command line. A make command-line `CFLAGS=...`
# REPLACES the line above rather than extending it, so one omitted flag (-O3,
# -ffp-contract=off) silently builds an arm that differs from the shipping
# binary in more than the thing under test -- a confound that would read as a
# kernel result. Appended last so an arm's -D cannot be overridden by anything
# above it. Use:
#     mingw32-make BUILDDIR=build/win-mins-simd EXTRA_CFLAGS=-DSTOKE_MINS_SIMD=1
# Makefile is already a prerequisite of every object, so changing this forces
# the rebuild it should force.
EXTRA_CFLAGS ?=
CFLAGS  := -std=c11 -Wall -Wextra -Werror -g -O3 -ffp-contract=off -Isrc $(EXTRA_CFLAGS)
LDFLAGS :=
LIBS    := -lm -lpthread
SAN     ?=

ifeq ($(OS),Windows_NT)
  PLAT := win
  EXE  := .exe
  # advapi32: AdjustTokenPrivileges, to enable SE_INC_WORKING_SET_NAME before
  # asking for a hard minimum working set. See map_pin() in src/gguf.c -- the
  # privilege is held but DISABLED by default, and without it the request is
  # refused and the model gets trimmed out from under a running decode.
  #
  # ws2_32: the winsock 2 entry points src/net.c calls (WSAStartup, socket,
  # closesocket, ioctlsocket, getaddrinfo...). Added at M6 rung A2. On POSIX
  # these live in libc and there is nothing to link, which is one more thing
  # src/net.h exists to keep out of every other file.
  LIBS += -ladvapi32 -lws2_32
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
# src/kernels/*.h WAS MISSING HERE until 2026-08-14, and its absence is the
# stale-.o hazard this Makefile warns about elsewhere: editing simd.h or
# kernels.h left every object in place and relinked them unchanged, so `make`
# reported "up to date" for a tree that had changed. Comment-only edits got
# away with it; a struct layout, an inline function or a constant such as
# KERNELS_BATCH_TILE would have linked stale objects against a new header and
# the failure would have looked like a code bug, not a build one. Caught by
# noticing that a header edit produced no compilation.
HEADERS  := $(wildcard src/*.h) $(wildcard src/kernels/*.h) $(wildcard tests/*.h)

# src/main.c holds `stoke`'s own main() -- it must NOT be linked into test
# binaries (test.h's main() would collide with it), so it's built/linked
# only into the $(BUILDDIR)/stoke target below. Every other src/*.c is a
# library object shared by every test binary and by stoke itself.
LIB_SRC  := $(filter-out src/main.c,$(wildcard src/*.c)) $(wildcard src/kernels/*.c)
LIB_OBJ  := $(patsubst src/%.c,$(BUILDDIR)/%.o,$(LIB_SRC))
MAIN_OBJ := $(BUILDDIR)/main.o
STOKE_BIN := $(BUILDDIR)/stoke$(EXE)

# ⚠ DEFINED HERE, ABOVE `all`, NOT BESIDE ITS RULES FURTHER DOWN. A target's
# prerequisite list is expanded WHEN THE MAKEFILE IS READ, so a variable defined
# after `all:` expands to nothing there -- `make` then reports success having
# built neither binary, and `make test` fails with "No such file or directory"
# on a target it never attempted. Cost one build cycle on 2026-08-28.
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -Werror -g -O1 -Isrc

# ⚠ THE C++ TU IS OPTIONAL AT BUILD TIME AND LOUD ABOUT IT, and that is a
# deliberate change from "always required". It is gate item [5], so it must not
# vanish quietly -- but it was in `all`, which meant a leg with no C++ compiler
# could not build ANYTHING: on this box's WSL side, where g++ is not installed,
# `make` died on abi_tu_cxx and took the whole Linux leg of the matrix with it,
# including 17 test binaries, ASan and TSan that have nothing to do with C++.
# A missing optional toolchain should cost you that one box and no more. The
# test target below prints a SKIP line naming what to install, so the box is
# visibly unchecked rather than silently absent.
HAVE_CXX := $(shell command -v $(CXX) > /dev/null 2>&1 && echo yes)
ifeq ($(HAVE_CXX),yes)
  ABI_TU_CXX_BIN := $(BUILDDIR)/abi_tu_cxx$(EXE)
else
  ABI_TU_CXX_BIN :=
endif
ABI_TU_BIN := $(BUILDDIR)/abi_tu$(EXE) $(ABI_TU_CXX_BIN) $(BUILDDIR)/abi_smoke$(EXE) $(BUILDDIR)/abi_sample$(EXE)

# M6 rungs A3+A4: an HTTP client of our own, for the three behaviours curl
# cannot reach (pipelining, the SSE heartbeat, a client that stops reading).
# Built by
# the generic tests/%.c rule; listed here for the same reason abi_smoke is --
# it takes arguments and needs a server, so the TEST_SRC wildcard must not
# sweep it into run_tests.sh, and something has to name it or it never builds.
SERVE_BIN := $(BUILDDIR)/serve_probe$(EXE) $(BUILDDIR)/serve_fuzz$(EXE)

.PHONY: all test clean

all: $(TEST_BIN) $(STOKE_BIN) $(ABI_TU_BIN) $(SERVE_BIN)

test: $(TEST_BIN) $(STOKE_BIN) $(ABI_TU_BIN) $(SERVE_BIN)
	$(TSAN_RUN) sh tests/run_tests.sh $(BUILDDIR) $(EXE)
	$(BUILDDIR)/abi_tu$(EXE) && echo "abi_tu: stoke.h standalone C11 OK"
ifeq ($(HAVE_CXX),yes)
	$(BUILDDIR)/abi_tu_cxx$(EXE) && echo "abi_tu_cxx: stoke.h C++ compile+link OK"
else
	@echo "abi_tu_cxx: SKIP -- no $(CXX) on this leg, so M6 gate item [5]'s C++ translation unit is NOT checked here. It is a real box and this is a real gap: install it (apt install g++) to close it. The C leg below still runs."
endif
	sh tests/abi_isolation.sh
	sh tests/abi_checklist.sh
	sh tests/abi_checklist_mutants.sh
ifeq ($(SAN),tsan)
	@echo "abi_smoke / abi_sample / golden_info / tok_golden / tok_special_golden / unicode_sweep_golden / run_sampling / serve_golden / serve_raw / serve_openai: SKIP on the TSan leg -- MEASURED, not assumed: abi_smoke peaks at 7,364,440 kB RSS under TSan and this WSL has 7 GB total, so it fits when run alone and is OOM-killed inside make test. Two TSan runs died at exactly this point. TSan's value is the thread pool, and the 17 unit binaries above cover it with zero data races; every gate skipped here RUNS AND PASSES on the ASan leg. Raise WSL memory in .wslconfig to close this gap."
else
	$(TSAN_RUN) sh tests/abi_smoke.sh $(BUILDDIR) $(EXE)
	$(TSAN_RUN) sh tests/abi_sample.sh $(BUILDDIR) $(EXE)

	$(TSAN_RUN) sh tests/golden_info.sh $(BUILDDIR) $(EXE)
	$(TSAN_RUN) sh tests/tok_golden.sh $(BUILDDIR) $(EXE)
	$(TSAN_RUN) sh tests/tok_special_golden.sh $(BUILDDIR) $(EXE)
	$(TSAN_RUN) sh tests/unicode_sweep_golden.sh $(BUILDDIR) $(EXE)
	$(TSAN_RUN) sh tests/run_sampling.sh $(BUILDDIR) $(EXE)
	$(TSAN_RUN) sh tests/serve_golden.sh $(BUILDDIR) $(EXE)
	$(TSAN_RUN) sh tests/serve_raw.sh $(BUILDDIR) $(EXE)
	$(TSAN_RUN) sh tests/serve_openai.sh $(BUILDDIR) $(EXE)
endif

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

# ---- M6 gate item [5]: stoke.h compiles standalone, as C11 AND as C++ ----
#
# These two are NOT named tests/test_*.c on purpose, so the TEST_SRC wildcard
# does not sweep them into run_tests.sh: neither defines test.h's main(), and
# the C++ one cannot be compiled by $(CC) at all. They are built and RUN by the
# `test` target below.
#
# The C++ TU LINKS rather than merely compiling. A missing `extern "C"` guard
# compiles perfectly and fails only at link time, so a compile-only check would
# pass on the exact defect this exists to catch.
#
# -std=c++17 rather than the oldest C++ that works: a host on a modern standard
# is the common case, and the narrowing and enum rules that catch header
# mistakes are stricter there.
$(BUILDDIR)/abi_tu$(EXE): tests/abi_tu.c $(LIB_OBJ) $(HEADERS) Makefile | $(BUILDDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIB_OBJ) $(LIBS)

$(BUILDDIR)/abi_tu_cxx$(EXE): tests/abi_tu.cpp $(LIB_OBJ) $(HEADERS) Makefile | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $< $(LIB_OBJ) $(LIBS)

$(BUILDDIR)/abi_smoke$(EXE): tests/abi_smoke.c $(LIB_OBJ) $(HEADERS) Makefile | $(BUILDDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIB_OBJ) $(LIBS)

$(BUILDDIR)/abi_sample$(EXE): tests/abi_sample.c $(LIB_OBJ) $(HEADERS) Makefile | $(BUILDDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIB_OBJ) $(LIBS)

# tests/test_c_oracle.c is the ONE test that carries AVX2 intrinsics of its
# own: it holds instrumented copies of BOTH the m4 and the HEAD K-quant twins
# so rung C's integer totals can be compared old-against-new. It therefore
# needs simd.c's ISA flags, and this explicit rule overrides the pattern rule
# above for that single target.
#
# The scalar copies in the same file are then also compiled with -mavx2. That
# is safe for what this test asserts -- the integer totals are i32 adds and
# multiplies, exact and associative under any vectorization, and the bounds
# are three orders inside i32 -- and the f32 self-checks against the library
# kernels would catch it if anything else moved.
$(BUILDDIR)/test_c_oracle$(EXE): tests/test_c_oracle.c $(LIB_OBJ) $(HEADERS) Makefile | $(BUILDDIR)
	$(CC) $(CFLAGS) $(SIMD_CFLAGS) $(LDFLAGS) -o $@ $< $(LIB_OBJ) $(LIBS)

$(STOKE_BIN): $(MAIN_OBJ) $(LIB_OBJ) Makefile | $(BUILDDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(MAIN_OBJ) $(LIB_OBJ) $(LIBS)

$(BUILDDIR):
	mkdir -p $@

clean:
	rm -rf build
