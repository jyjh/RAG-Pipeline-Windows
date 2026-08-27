@echo off
REM Start the RAG instance with no setup prompts: provisions each HPC cluster
REM only if its deployed source is stale (or the SIF is missing), then launches
REM the web server. Missing Python dependencies are installed automatically.
REM
REM Run a full interactive setup (SSH keys, config, prompts) with setup.cmd
REM instead. To force a redeploy of the remote clusters, run:
REM   start.cmd --provision-hpc
REM
REM To deploy an initial PDF corpus (parsed on the HPC cluster, indexed
REM locally, nested directories inside the zip are fine), run:
REM   start.cmd --initial-corpus corpus.zip
setlocal
cd /d "%~dp0"
set "ARGS=--non-interactive --start --provision-if-needed %*"
where py >nul 2>nul
if %errorlevel%==0 (
  set "RAG_PY=py -3"
  goto :run
)
where python >nul 2>nul
if %errorlevel%==0 (
  set "RAG_PY=python"
  goto :run
)
echo Python 3.11+ was not found on PATH. Install it from
echo https://www.python.org/downloads/ and enable "Add python.exe to PATH"
echo in the installer, then run start again.
pause
exit /b 1
:run
%RAG_PY% scripts\setup_instance.py %ARGS%
set "START_EXIT=%errorlevel%"
if not "%START_EXIT%"=="0" pause
exit /b %START_EXIT%
