@echo off
setlocal EnableDelayedExpansion

if "%~1"=="" (
  echo Usage: build_k10_config.bat CONFIG_INDEX
  exit /b 1
)

set "CONFIG_INDEX=%~1"
set "COMPARE_MODE=%~2"
set "SCRIPT_DIR=%~dp0"
for %%P in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fP"
set "CONFIG_FILE=%SCRIPT_DIR%k10_sampled_configs.csv"
set "BUILD_DIR=%PROJECT_ROOT%\build-k10-autotune"

for /f "usebackq skip=%CONFIG_INDEX% tokens=1-9 delims=," %%A in ("%CONFIG_FILE%") do (
  if not defined NUM_THREADS (
    set "NUM_THREADS=%%A"
    set "BM=%%B"
    set "BN=%%C"
    set "BK=%%D"
    set "WM=%%E"
    set "WN=%%F"
    set "WNITER=%%G"
    set "TM=%%H"
    set "TN=%%I"
  )
)

if not defined NUM_THREADS (
  echo Invalid configuration index: %CONFIG_INDEX%
  exit /b 1
)

set "CONDA_ENV=D:\anaconda3\envs\cuda-lab"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "PATH=%CONDA_ENV%\Library\bin;%CONDA_ENV%\Scripts;%PATH%"

echo Config %CONFIG_INDEX%: threads=%NUM_THREADS% BM=%BM% BN=%BN% BK=%BK% WM=%WM% WN=%WN% WNITER=%WNITER% TM=%TM% TN=%TN%

set "K11_ARGUMENTS="
if /i "%COMPARE_MODE%"=="compare" (
  echo Compare mode: Kernel 10 and Kernel 11 use identical parameters.
  set "K11_ARGUMENTS=-DSGEMM_K11_NUM_THREADS=%NUM_THREADS% -DSGEMM_K11_BM=%BM% -DSGEMM_K11_BN=%BN% -DSGEMM_K11_BK=%BK% -DSGEMM_K11_WM=%WM% -DSGEMM_K11_WN=%WN% -DSGEMM_K11_WNITER=%WNITER% -DSGEMM_K11_TM=%TM% -DSGEMM_K11_TN=%TN%"
)

call "%VCVARS%"
if errorlevel 1 exit /b 1

for /f "delims=" %%C in ('where cl.exe') do (
  if not defined CXX_COMPILER set "CXX_COMPILER=%%C"
)
if not defined CXX_COMPILER (
  echo Cannot find cl.exe after loading vcvars64.bat
  exit /b 1
)

"%CMAKE%" -S "%PROJECT_ROOT%" -B "%BUILD_DIR%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_MAKE_PROGRAM=%CONDA_ENV%\Scripts\ninja.exe" ^
  "-DCMAKE_CXX_COMPILER=%CXX_COMPILER%" ^
  "-DCMAKE_CUDA_COMPILER=%CONDA_ENV%\Library\bin\nvcc.exe" ^
  "-DCUDAToolkit_ROOT=%CONDA_ENV%\Library" ^
  -DSGEMM_K10_AUTOTUNE_MODE=ON ^
  -DSGEMM_K10_NUM_THREADS=%NUM_THREADS% ^
  -DSGEMM_K10_BM=%BM% ^
  -DSGEMM_K10_BN=%BN% ^
  -DSGEMM_K10_BK=%BK% ^
  -DSGEMM_K10_WM=%WM% ^
  -DSGEMM_K10_WN=%WN% ^
  -DSGEMM_K10_WNITER=%WNITER% ^
  -DSGEMM_K10_TM=%TM% ^
  -DSGEMM_K10_TN=%TN% ^
  !K11_ARGUMENTS!
if errorlevel 1 exit /b 1

"%CMAKE%" --build "%BUILD_DIR%" --target sgemm
if errorlevel 1 exit /b 1

echo Built: %BUILD_DIR%\sgemm.exe
