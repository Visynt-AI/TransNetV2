@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "WORKER_COUNT=%~1"
if not defined WORKER_COUNT set "WORKER_COUNT=1"

for /f "delims=0123456789" %%A in ("%WORKER_COUNT%") do set "WORKER_COUNT="
if not defined WORKER_COUNT (
  echo Invalid worker count: %~1
  echo Usage: start_workers.bat [worker_count]
  exit /b 1
)

if %WORKER_COUNT% LSS 1 (
  echo worker_count must be at least 1.
  exit /b 1
)

if not exist ".env" (
  echo Missing .env file. Copy .env.example to .env and fill in your settings first.
  exit /b 1
)

set "UV_CACHE_DIR=%CD%\.uv-cache"
set "PYTHONUNBUFFERED=1"
set "PYTHONUTF8=1"

if not exist "logs" mkdir logs
if not exist "%UV_CACHE_DIR%" mkdir "%UV_CACHE_DIR%"

if not exist ".venv\Scripts\python.exe" (
  echo Project virtual environment not found. Running uv sync...
  uv sync
  if errorlevel 1 (
    echo uv sync failed.
    exit /b 1
  )
)

set "PYTHON=.venv\Scripts\python.exe"

%PYTHON% -c "import boto3, ffmpeg, numpy, pika, PIL, torch, torchvision" >nul 2>nul
if errorlevel 1 (
  echo Python dependencies are incomplete. Running uv sync...
  uv sync
  if errorlevel 1 (
    echo uv sync failed.
    exit /b 1
  )
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg.exe was not found in PATH.
  echo Install ffmpeg and make sure the ffmpeg command is available before starting workers.
  exit /b 1
)

for /f "usebackq tokens=1* delims==" %%A in (`findstr /B /C:"WEIGHTS_PATH=" ".env"`) do set "WEIGHTS_PATH=%%B"
if not defined WEIGHTS_PATH set "WEIGHTS_PATH=.\weights\transnetv2-pytorch-weights.pth"
set "WEIGHTS_PATH=%WEIGHTS_PATH:/=\%"

if not exist "%WEIGHTS_PATH%" (
  echo Weights file not found: %WEIGHTS_PATH%
  echo Put the model file there or update WEIGHTS_PATH in .env.
  exit /b 1
)

%PYTHON% -c "import socket, sys; from dotenv import dotenv_values; from urllib.parse import urlparse; url = dotenv_values('.env').get('S3_ENDPOINT_URL', ''); parsed = urlparse(url); host = parsed.hostname; port = parsed.port or (443 if parsed.scheme == 'https' else 80); sock = socket.socket(); sock.settimeout(2); ok = bool(host) and sock.connect_ex((host, port)) == 0; sock.close(); sys.exit(0 if ok else 4)" >nul 2>nul
if errorlevel 1 (
  echo Warning: the configured S3 endpoint is not reachable right now.
  echo Workers will start, but tasks that access S3 will fail until the endpoint is available.
)

for /L %%I in (1,1,%WORKER_COUNT%) do (
  set "LOG_FILE=%CD%\logs\worker-%%I.log"
  echo Starting worker %%I of %WORKER_COUNT%...
  start "TransNet Worker %%I" /min cmd /c "cd /d ""%CD%"" && set UV_CACHE_DIR=%UV_CACHE_DIR%&& set PYTHONUNBUFFERED=1&& set PYTHONUTF8=1&& ""%PYTHON%"" main.py >> ""!LOG_FILE!"" 2>&1"
)

echo Started %WORKER_COUNT% worker^(s^). Logs: %CD%\logs
exit /b 0
