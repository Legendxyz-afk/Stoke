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

CFLAGS  := -std=c11 -Wall -Wextra -Werror -g -O1 -Isrc
LDFLAGS :=
LIBS    :=
SAN     ?=

ifeq ($(OS),Windows_NT)
  PLAT := win
  EXE  := .exe
else
  PLAT := linux
  EXE  :=
endif

SAN_SUFFIX :=
ifeq ($(SAN),asan)
  CFLAGS     += -fsanitize=address,undefined -fno-sanitize-recover=all
  LDFLAGS    += -fsanitize=address,undefined
  SAN_SUFFIX := -asan
endif
ifeq ($(SAN),tsan)
  CFLAGS     += -fsanitize=thread
  LDFLAGS    += -fsanitize=thread
  SAN_SUFFIX := -tsan
endif

BUILDDIR := build/$(PLAT)$(SAN_SUFFIX)

TEST_SRC := $(wildcard tests/test_*.c)
TEST_BIN := $(patsubst tests/%.c,$(BUILDDIR)/%$(EXE),$(TEST_SRC))
HEADERS  := $(wildcard src/*.h) $(wildcard tests/*.h)

# src/main.c holds `stoke`'s own main() -- it must NOT be linked into test
# binaries (test.h's main() would collide with it), so it's built/linked
# only into the $(BUILDDIR)/stoke target below. Every other src/*.c is a
# library object shared by every test binary and by stoke itself.
LIB_SRC  := $(filter-out src/main.c,$(wildcard src/*.c))
LIB_OBJ  := $(patsubst src/%.c,$(BUILDDIR)/%.o,$(LIB_SRC))
MAIN_OBJ := $(BUILDDIR)/main.o
STOKE_BIN := $(BUILDDIR)/stoke$(EXE)

.PHONY: all test clean

all: $(TEST_BIN) $(STOKE_BIN)

test: $(TEST_BIN) $(STOKE_BIN)
	sh tests/run_tests.sh $(BUILDDIR) $(EXE)
	sh tests/golden_info.sh $(BUILDDIR) $(EXE)
	sh tests/tok_golden.sh $(BUILDDIR) $(EXE)
	sh tests/tok_special_golden.sh $(BUILDDIR) $(EXE)

# One BUILDDIR per (platform,sanitizer) combination (see header comment) --
# object files live here too, built with this invocation's exact CFLAGS, so
# a plain build's .o's are never accidentally linked into an ASan/TSan binary
# or vice versa.
$(BUILDDIR)/%.o: src/%.c $(HEADERS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILDDIR)/%$(EXE): tests/%.c $(LIB_OBJ) $(HEADERS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIB_OBJ) $(LIBS)

$(STOKE_BIN): $(MAIN_OBJ) $(LIB_OBJ) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(MAIN_OBJ) $(LIB_OBJ) $(LIBS)

$(BUILDDIR):
	mkdir -p $@

clean:
	rm -rf build
