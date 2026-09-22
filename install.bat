@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>nul

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto nopython

%PY% "%~dp0setup_glados.py"
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%

:nopython
echo.
echo [ERROR] Python not found in PATH.
echo Install Python 3.10 - 3.12 from https://www.python.org/downloads/
echo and enable the "Add Python to PATH" checkbox during setup.
echo.
pause
exit /b 1
