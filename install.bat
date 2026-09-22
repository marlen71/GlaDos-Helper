@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo === Установка GLaDOS Helper ===
echo Папка: %CD%
echo.

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY (
  echo [ОШИБКА] Python не найден. Установите Python 3.10-3.12 с python.org
  echo и отметьте галочку "Add Python to PATH".
  pause & exit /b 1
)

echo Использую: %PY%
%PY% --version
echo.

if exist ".venv\Scripts\python.exe" (
  echo Окружение .venv уже есть, использую его.
) else (
  echo Создаю виртуальное окружение .venv ...
  %PY% -m venv ".venv"
  if errorlevel 1 (
    echo [ОШИБКА] Не удалось создать .venv
    pause & exit /b 1
  )
)

set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo [ОШИБКА] Не найден "%VPY%"
  pause & exit /b 1
)

echo.
echo --- Обновляю pip ---
"%VPY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto failed

echo.
echo --- Ставлю PyTorch (CPU, ~200 МБ) ---
"%VPY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
  echo [ВНИМАНИЕ] CPU-сборка torch не установилась, пробую обычную с PyPI...
  "%VPY%" -m pip install torch torchaudio
)

echo.
echo --- Ставлю остальные зависимости ---
"%VPY%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto failed

echo.
echo --- Проверка ---
"%VPY%" "%~dp0check.py"
if errorlevel 1 goto failed

echo.
echo ============================================
echo  Готово! Запускайте run.bat
echo ============================================
pause
exit /b 0

:failed
echo.
echo [ОШИБКА] Установка прервана. Скопируйте сообщение выше.
pause
exit /b 1
