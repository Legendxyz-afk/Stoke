@echo off
rem Detached launcher for the M0 baseline matrix -- run via Task Scheduler (or a double-click)
rem so the multi-hour run survives terminal/session restarts. Resumable: run_matrix.py skips
rem cells whose raw JSON already exists. Console output appends to matrix-console-4.log.
cd /d "%~dp0.."
python bench\run_matrix.py >> "bench\results\raw\matrix-console-4.log" 2>&1
