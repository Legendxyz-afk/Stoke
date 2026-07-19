@echo off
rem Windows-native leg of the check matrix:  scripts\win-check.cmd
rem Counterpart of scripts/wsl-check.sh (which covers the WSL plain/asan/tsan
rem legs). Exists to close the half-gap where a bare `mingw32-make test` on
rem the reference box silently SKIPs all golden gates if the model file is
rem missing/moved -- STOKE_REQUIRE_GOLDEN=1 turns that into a hard FAILURE
rem here, exactly as wsl-check.sh does on the Linux side.
setlocal
cd /d "%~dp0.."
set STOKE_REQUIRE_GOLDEN=1
set "PATH=C:\msys64\ucrt64\bin;C:\msys64\usr\bin;%PATH%"
echo === [1/1] mingw32-make test (Windows native, STOKE_REQUIRE_GOLDEN=1) ===
mingw32-make test
if errorlevel 1 (
  echo WIN CHECK: FAILED
  exit /b 1
)
echo WIN CHECK: ALL GREEN
exit /b 0
