@echo off
REM Compile script for matrix multiplication algorithms
REM This works without MPI headers - compiles with OpenMP only

echo ======================================
echo Matrix Multiplication Compiler
echo ======================================
echo.

REM Use the OpenMP version that compiles without MPI
if exist main_openmp.exe (
    echo main_openmp.exe already compiled.
) else (
    echo Compiling OpenMP version (no MPI required)...
    g++ -o main_openmp.exe main_openmp.cpp -fopenmp -O2 -std=c++11
    if %ERRORLEVEL% EQU 0 (
        echo Compilation successful!
    ) else (
        echo Compilation failed!
        exit /b 1
    )
)

echo.
echo ======================================
echo Running: Naive vs Strassen Algorithms
echo ======================================
echo.

main_openmp.exe

pause
