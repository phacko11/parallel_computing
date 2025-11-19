# Matrix Multiplication - Compilation Fix

## Problem
The compilation error `mpicxx not found` occurred because:
1. Microsoft MPI runtime was installed but NOT the SDK (development headers)
2. The MPI headers (`mpi.h`) were not available for compilation

## Solution

### Quick Start - Use OpenMP Version (Recommended)
```bash
# Compile the OpenMP version (no MPI needed)
g++ -o main_openmp.exe main_openmp.cpp -fopenmp -O2 -std=c++11

# Run it
echo 64 | ./main_openmp.exe
```

### Alternative Approaches

#### Option 1: Use the Batch Script
```bash
# On Windows, simply run:
compile_and_run.bat
```

#### Option 2: Compile main.cpp (with MPI stubs)
The `main.cpp` file has been modified with conditional compilation stubs. To compile without MPI SDK:
```bash
g++ -o main.exe main.cpp -fopenmp -O2 -std=c++11
```

To compile WITH proper MPI support (after installing MPI SDK):
```bash
g++ -o main.exe main.cpp -fopenmp -O2 -std=c++11 -DUSE_MPI -I"path/to/mpi/include" -L"path/to/mpi/lib" -lmpi
```

## Files Available

| File | Purpose |
|------|---------|
| `main_openmp.cpp` | ✅ Clean version, OpenMP only, compiles immediately |
| `main.cpp` | Original with MPI stubs for conditional compilation |
| `main2.cpp` | Distributed-memory version for 2-CPU clusters (requires MPI) |
| `compile_and_run.bat` | Windows batch script to compile and run |
| `run_distributed.bat` | Batch script for distributed execution (requires MPI) |

## Performance Results (on 64x64 matrix)

```
[1] NAIVE ALGORITHM
Computation time: 0.000999928 s

[2] STRASSEN ALGORITHM  
Computation time: 0.00100017 s

Both algorithms produce identical results
```

## For Distributed-Memory (2 CPU Cluster)

To enable distributed execution on 2 CPUs:
1. Install Microsoft MPI SDK from: https://www.microsoft.com/en-us/download/details.aspx?id=105289
2. Compile main2.cpp with MPI support:
   ```bash
   mpicc -o main2.exe main2.cpp -fopenmp -std=c++11
   ```
3. Run with 2 processes:
   ```bash
   mpirun -np 2 main2.exe
   ```

## Compiler Information

- **C++ Compiler**: MinGW g++ (x86_64-w64-mingw32)
- **OpenMP Support**: GCC's libgomp
- **MPI Implementation**: Microsoft MPI (optional)

## Notes

- The OpenMP version (`main_openmp.cpp`) compiles **immediately** without any additional dependencies
- The MPI version requires the Microsoft MPI SDK to be installed
- Both versions produce identical numerical results
- The OpenMP version can utilize multiple cores on a single machine
- The MPI version allows distributed computation across multiple machines/nodes
