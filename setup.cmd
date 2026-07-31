@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 scripts\setup_instance.py %*
) else (
  python scripts\setup_instance.py %*
)
set "RAG_SETUP_EXIT=%errorlevel%"
if not "%RAG_SETUP_EXIT%"=="0" pause
exit /b %RAG_SETUP_EXIT%
