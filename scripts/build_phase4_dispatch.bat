@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%P in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fP"
set "BUILD_DIR=%PROJECT_ROOT%\build-phase4-dispatch"
set "CONDA_ENV=D:\anaconda3\envs\cuda-lab"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "PATH=%CONDA_ENV%\Library\bin;%CONDA_ENV%\Scripts;%PATH%"

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
  -DSGEMM_K10_AUTOTUNE_MODE=OFF
if errorlevel 1 exit /b 1

"%CMAKE%" --build "%BUILD_DIR%" --target sgemm
if errorlevel 1 exit /b 1

echo Built phase-four dispatcher: %BUILD_DIR%\sgemm.exe
