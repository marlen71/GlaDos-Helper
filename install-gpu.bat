@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>nul
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto nopython
%PY% "%~dp0setup_glados.py" --gpu %*
echo.
pause
exit /b 0
:nopython
echo [ERROR] Python not found in PATH.
pause
exit /b 1
