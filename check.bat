@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>nul

set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" set "VPY=python"

"%VPY%" "%~dp0check.py"
echo.
pause
