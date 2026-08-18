@echo off
REM Start the RAG instance with no setup prompts: provisions each HPC cluster
REM only if its deployed source is stale (or the SIF is missing), then launches
REM the web server.
REM
REM Run a full interactive setup (SSH keys, config, prompts) with setup.cmd
REM instead. To force a redeploy of the remote clusters, run:
REM   start.cmd --provision-hpc
setlocal
cd /d "%~dp0"
set "ARGS=--non-interactive --start --provision-if-needed %*"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 scripts\setup_instance.py %ARGS%
) else (
  python scripts\setup_instance.py %ARGS%
)
set "START_EXIT=%errorlevel%"
if not "%START_EXIT%"=="0" pause
exit /b %START_EXIT%
