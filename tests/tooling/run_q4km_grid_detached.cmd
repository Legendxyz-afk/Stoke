@echo off
rem RUN [2v2-iii]'s Q4_K_M REFERENCE GRID -- ruling section 5, the last evidence
rem the M3 stamp is waiting on.
rem
rem ▶ DETACHED because a window is hours against a 10 min agent-tool ceiling.
rem Task Scheduler is this tree's established pattern for that and is the only
rem one measured to survive; every prior multi-hour gate leg used it.
rem
rem ▶ THE TASK MUST BE ONE-TIME AND MUST BE DELETED WHEN THE GRID CLOSES.
rem 2026-08-02 found `stoke-m2-ref-gate` still carrying an ENABLED AtLogOn
rem trigger, relaunching a finished gate one minute after every sign-in, and a
rem watchdog popping a console every 15 minutes long after its job was done.
rem A spent helper outliving its job is a defect, not a leftover.
rem
rem ▶ DO NOT RUN THIS WHILE ANYTHING HOLDS THE Q8_0 FILE. Alternating two large
rem mmapped models is the recorded 2026-08-04 page-cache churn.
rem
rem ▶ TWO CONCURRENT INSTANCES ARE NOW ALLOWED, AND ONLY UNDER THE GUARD.
rem This header used to say flatly "do not run two windows at once", which was
rem the right rule on 2026-08-07 at 5.82 GB free and is not a general one:
rem commit c0aea9f measured that gguf_reader mmaps the model, so a second
rem process on the SAME file shares those pages and costs only its ~1.9 GB
rem private working set. Pass `EXTRA` as the 4th argument for every instance
rem after the first -- q4km_grid.py then charges the reduced bar, but ONLY after
rem verifying by COMMAND LINE that a ref_forward.py process is really running.
rem Do not pass it to the first instance and do not launch instances
rem simultaneously: the guard reads free RAM at launch, so two launches racing
rem each other both see the memory neither has taken yet. Stagger them.
rem
rem ▶ DISJOINTNESS IS THE CALLER'S. Give each instance a disjoint `ONLY` set.
rem The probe files are per leg and the manifest name carries `--only`, so an
rem accidental overlap is loud rather than silent -- but it is still wasted
rem hours. `p00-probe` may not be selected without `p00-bare`; the tool refuses.
rem
rem ▶ RESUME IS THE OUTPUT DIRECTORY'S. A leg is done iff its probe file exists
rem AND revalidates. Re-running this launcher after a stop continues from the
rem first leg that does not, and spends nothing on the ones that do. Resume
rem granularity is the LEG, not the position: ref_forward.py has no intra-leg
rem checkpoint, so an interrupted leg costs the whole leg (~1.5 h at the
rem 2026-08-07 prior). Plan interruptions around leg boundaries.
rem
rem USAGE:  run_q4km_grid_detached.cmd A|B|CONT <rate_s_per_pos> ["p20,p45"] [EXTRA]
cd /d "%~dp0..\.."
set WHICH=%~1
set RATE=%~2
set ONLY=%~3
set EXTRA=%~4
if "%WHICH%"=="" set WHICH=A
if "%RATE%"=="" set RATE=18.42

set OUT=tests\tooling\.gate_cache\q4km_grid
if not exist "%OUT%" mkdir "%OUT%"

rem The log name carries ONLY for the same reason the manifest name does: it is
rem otherwise the one shared artifact, and two concurrent instances appending to
rem it would interleave their lines into a record neither can be read out of.
set SLUG=
set ONLYARG=
if not "%ONLY%"=="" (
  set SLUG=-%ONLY:,=_%
  set ONLYARG=--only "%ONLY%"
)
set EXTRAARG=
if /I "%EXTRA%"=="EXTRA" set EXTRAARG=--additional-instance

set LOG=%OUT%\window%SLUG%.log

echo. >> "%LOG%"
echo ===== Q4KM GRID %WHICH% %DATE% %TIME% (rate %RATE% s/pos, only '%ONLY%' %EXTRAARG%) ===== >> "%LOG%"

if /I "%WHICH%"=="CONT" (
  python -u tests\tooling\q4km_grid.py --continuations --out "%OUT%" --rate-s-per-pos %RATE% %ONLYARG% %EXTRAARG% >> "%LOG%" 2>&1
) else (
  python -u tests\tooling\q4km_grid.py --run %WHICH% --out "%OUT%" --rate-s-per-pos %RATE% %ONLYARG% %EXTRAARG% >> "%LOG%" 2>&1
)

echo ===== Q4KM GRID %WHICH% EXIT %ERRORLEVEL% %DATE% %TIME% ===== >> "%LOG%"
exit /b 0
