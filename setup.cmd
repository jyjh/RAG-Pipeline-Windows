@echo off
REM One-click guided setup: configures the web server (and optionally the HPC
REM cluster), creates .venv, installs requirements.txt, runs preflight checks,
REM and starts the server. All flags pass through to scripts\setup_instance.py.
setlocal
cd /d "%~dp0"
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
echo in the installer, then run setup again.
pause
exit /b 1
:run
%RAG_PY% scripts\setup_instance.py %*
set "RAG_SETUP_EXIT=%errorlevel%"
if not "%RAG_SETUP_EXIT%"=="0" pause
exit /b %RAG_SETUP_EXIT%
