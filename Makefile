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

.PHONY: all test clean

all: $(TEST_BIN)

test: $(TEST_BIN)
	sh tests/run_tests.sh $(BUILDDIR) $(EXE)

$(BUILDDIR)/%$(EXE): tests/%.c $(HEADERS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

$(BUILDDIR):
	mkdir -p $@

clean:
	rm -rf build
