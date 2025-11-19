@echo off
REM Compile the distributed-memory matrix multiplication code
REM This script compiles and runs on 2 CPU cluster

echo Compiling main2.cpp for distributed-memory execution...
mpicc -o main2.exe main2.cpp -fopenmp -std=c++11

if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
    exit /b 1
)

echo Compilation successful!
echo.
echo Running on 2 processes (distributed-memory cluster):
echo.

REM Run with 2 processes
mpirun -np 2 main2.exe

pause
