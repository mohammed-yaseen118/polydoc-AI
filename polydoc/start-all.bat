@echo off
cls
color 0A
title PolyDoc - Full Stack Launcher

echo.
echo ================================================================
echo   PolyDoc AI - Complete System Launcher
echo   Backend + Frontend + MongoDB
echo ================================================================
echo.

REM Check Python
echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python.
    pause
    exit /b 1
) else (
    echo ✅ Python found
)

REM Check Node.js
echo [2/4] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found! Please install Node.js.
    pause
    exit /b 1
) else (
    echo ✅ Node.js found
)

REM Check MongoDB (optional)
echo [3/4] Checking MongoDB...
mongod --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  MongoDB not found - will skip MongoDB startup
    set START_MONGO=false
) else (
    echo ✅ MongoDB found
    set START_MONGO=true
)

echo.
echo [4/4] Starting all services...
echo.

REM Optimize cache for faster model loading
echo 🛠️ Optimizing AI model cache...
python optimize_cache.py
echo.

REM Start MongoDB first if available
if "%START_MONGO%"==\"true" (
    echo 🍃 Starting MongoDB...
    start "MongoDB Server" cmd /k "mongod --dbpath ./data/db"
    timeout /t 3 /nobreak >nul
    echo    MongoDB starting in background...
) else (
    echo 🍃 Skipping MongoDB (not installed)
)

REM Start Optimized Full AI Backend
echo 🚀 Starting PolyDoc AI Backend (port 8000)...
start "PolyDoc Backend" cmd /k "python main.py"
echo    Full AI backend with optimized loading (Hindi/Kannada + AI models)...

REM Wait for backend to be ready
echo ⏳ Waiting for backend to be ready...
set BACKEND_READY=false
for /L %%i in (1,1,30) do (
    REM Try curl first, then PowerShell as fallback
    curl -s http://localhost:8000/health >nul 2>&1
    if not errorlevel 1 (
        set BACKEND_READY=true
        goto :backend_ready
    ) else (
        REM Fallback to PowerShell
        powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:8000/health' -TimeoutSec 2 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
        if not errorlevel 1 (
            set BACKEND_READY=true
            goto :backend_ready
        )
    )
    timeout /t 2 /nobreak >nul
)

:backend_ready
if "%BACKEND_READY%"=="true" (
    echo ✅ Backend is ready!
) else (
    echo ⚠️  Backend may still be starting (timeout reached)
)

REM Start Frontend
echo 🌐 Starting Frontend (port 3003)...
start "PolyDoc Frontend" cmd /k "npm install && npm run dev"
echo    Frontend starting...

REM Wait a bit more
timeout /t 3 /nobreak >nul

echo.
echo ================================================================
echo ✅ All Services Started!
echo.
echo 🚀 Backend:  http://localhost:8000
echo 🌐 Frontend: http://localhost:3003 (starting...)
if "%START_MONGO%"=="true" (
    echo 🍃 MongoDB:  Running in background
)
echo.
echo Services are running in separate windows.
echo Your main frontend will be at: http://localhost:3003
echo ================================================================
echo.

REM Open frontend after delay
echo ⏳ Waiting for frontend to be ready...
timeout /t 10 /nobreak >nul

echo 🌐 Opening frontend...
start http://localhost:3003

echo.
echo All services launched! You can close this window.
pause
