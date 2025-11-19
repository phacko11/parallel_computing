#include <iostream>
#include <omp.h>
#include <vector>
#include <cstring>
#include <cmath>
#include <random>
#include <mpi.h>

using namespace std;

typedef vector<vector<long long>> Matrix;

const int BASE_SIZE = 64; 

// Standard matrix multiplication
void standardMultiply(Matrix& A, Matrix& B, Matrix& C, int n) {
    #pragma omp parallel for collapse(2) if(n > 32)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Naive matrix multiplication 
void naiveMultiply(Matrix& A, Matrix& B, Matrix& C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
void getSubMatrix(Matrix& A, Matrix& B, int row, int col, int size) {
    #pragma omp parallel for collapse(2) if(size > 32)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            B[i][j] = A[i + row][j + col];
        }
    }
}
void addMatrices(Matrix& A, Matrix& B, Matrix& C, int n) {
    #pragma omp parallel for collapse(2) if(n > 32)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}
void putSubMatrix(Matrix& A, Matrix& B, int row, int col, int size) {
    #pragma omp parallel for collapse(2) if(size > 32)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i + row][j + col] = B[i][j];
        }
    }
}



void subtractMatrices(Matrix& A, Matrix& B, Matrix& C, int n) {
    #pragma omp parallel for collapse(2) if(n > 32)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}





// Strassen
void strassen(Matrix& A, Matrix& B, Matrix& C, int n) {
    if (n <= BASE_SIZE) {
        standardMultiply(A, B, C, n);
        return;
    }

    int half = n / 2;

    // Create temporary matrices for submatrices
    Matrix A11(half, vector<long long>(half, 0)),
            A12(half, vector<long long>(half, 0)),
            A21(half, vector<long long>(half, 0)),
            A22(half, vector<long long>(half, 0));

    Matrix B11(half, vector<long long>(half, 0)),
            B12(half, vector<long long>(half, 0)),
            B21(half, vector<long long>(half, 0)),
            B22(half, vector<long long>(half, 0));

    // Divide matrix A into 4 submatrices
    #pragma omp parallel sections
    {
        #pragma omp section
        getSubMatrix(A, A11, 0, 0, half);
        #pragma omp section
        getSubMatrix(A, A12, 0, half, half);
        #pragma omp section
        getSubMatrix(A, A21, half, 0, half);
        #pragma omp section
        getSubMatrix(A, A22, half, half, half);
    }

    // Divide matrix B into 4 submatrices
    #pragma omp parallel sections
    {
        #pragma omp section
        getSubMatrix(B, B11, 0, 0, half);
        #pragma omp section
        getSubMatrix(B, B12, 0, half, half);
        #pragma omp section
        getSubMatrix(B, B21, half, 0, half);
        #pragma omp section
        getSubMatrix(B, B22, half, half, half);
    }

    // Create temporary matrices for intermediate results
    Matrix M1(half, vector<long long>(half, 0)),
            M2(half, vector<long long>(half, 0)),
            M3(half, vector<long long>(half, 0)),
            M4(half, vector<long long>(half, 0)),
            M5(half, vector<long long>(half, 0)),
            M6(half, vector<long long>(half, 0)),
            M7(half, vector<long long>(half, 0));

    // Compute M1..M7 in parallel, but give each section its own temporaries to avoid races
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            Matrix tA(half, vector<long long>(half, 0));
            Matrix tB(half, vector<long long>(half, 0));
            addMatrices(A11, A22, tA, half);
            addMatrices(B11, B22, tB, half);
            strassen(tA, tB, M1, half);
        }
        #pragma omp section
        {
            Matrix tA(half, vector<long long>(half, 0));
            addMatrices(A21, A22, tA, half);
            strassen(tA, B11, M2, half);
        }
        #pragma omp section
        {
            Matrix tB(half, vector<long long>(half, 0));
            subtractMatrices(B12, B22, tB, half);
            strassen(A11, tB, M3, half);
        }
        #pragma omp section
        {
            Matrix tB(half, vector<long long>(half, 0));
            subtractMatrices(B21, B11, tB, half);
            strassen(A22, tB, M4, half);
        }
        #pragma omp section
        {
            Matrix tA(half, vector<long long>(half, 0));
            addMatrices(A11, A12, tA, half);
            strassen(tA, B22, M5, half);
        }
        #pragma omp section
        {
            Matrix tA(half, vector<long long>(half, 0));
            Matrix tB(half, vector<long long>(half, 0));
            subtractMatrices(A21, A11, tA, half);
            addMatrices(B11, B12, tB, half);
            strassen(tA, tB, M6, half);
        }
        #pragma omp section
        {
            Matrix tA(half, vector<long long>(half, 0));
            Matrix tB(half, vector<long long>(half, 0));
            subtractMatrices(A12, A22, tA, half);
            addMatrices(B21, B22, tB, half);
            strassen(tA, tB, M7, half);
        }
    }

    Matrix C11(half, vector<long long>(half, 0)),
            C12(half, vector<long long>(half, 0)),
            C21(half, vector<long long>(half, 0)),
            C22(half, vector<long long>(half, 0));

    Matrix temp3(half, vector<long long>(half, 0)),
            temp4(half, vector<long long>(half, 0));

    // C11 = M1 + M4 - M5 + M7
    addMatrices(M1, M4, temp3, half);
    subtractMatrices(temp3, M5, temp4, half);
    addMatrices(temp4, M7, C11, half);

    // C12 = M3 + M5
    addMatrices(M3, M5, C12, half);

    // C21 = M2 + M4
    addMatrices(M2, M4, C21, half);

    // C22 = M1 - M2 + M3 + M6
    subtractMatrices(M1, M2, temp3, half);
    addMatrices(temp3, M3, temp4, half);
    addMatrices(temp4, M6, C22, half);

    #pragma omp parallel sections
    {
        #pragma omp section
        putSubMatrix(C, C11, 0, 0, half);
        #pragma omp section
        putSubMatrix(C, C12, 0, half, half);
        #pragma omp section
        putSubMatrix(C, C21, half, 0, half);
        #pragma omp section
        putSubMatrix(C, C22, half, half, half);
    }
}

void printMatrix(Matrix& A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << A[i][j] << " ";
        }
        cout << endl;
    }
}

// Flatten/unflatten helpers for MPI communication
void flattenMatrix(const Matrix& A, vector<long long>& out, int rows, int cols) {
    out.resize(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out[i * cols + j] = A[i][j];
}

void unflattenMatrix(const vector<long long>& in, Matrix& A, int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            A[i][j] = in[i * cols + j];
}

// Multiply rectangular: A (r x n) * B (n x n) -> C (r x n)
void multiplyRect(Matrix& A, Matrix& B, Matrix& C, int r, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < n; ++j) {
            long long sum = 0;
            for (int k = 0; k < n; ++k) sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
}

int nextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

void padMatrix(const Matrix& A, Matrix& P, int n) {
    int m = (int)P.size();
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            if (i < n && j < n) P[i][j] = A[i][j];
            else P[i][j] = 0;
        }
    }
}

void unpadMatrix(const Matrix& P, Matrix& A, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = P[i][j];
        }
    }
}

void strassenPad(Matrix& A, Matrix& B, Matrix& C, int n) {
    if (n <= BASE_SIZE) {
        standardMultiply(A, B, C, n);
        return;
    }
    int m = nextPowerOfTwo(n);
    if (m == n) {
        strassen(A, B, C, n);
        return;
    }

    Matrix Ap(m, vector<long long>(m, 0));
    Matrix Bp(m, vector<long long>(m, 0));
    Matrix Cp(m, vector<long long>(m, 0));

    padMatrix(A, Ap, n);
    padMatrix(B, Bp, n);

    strassen(Ap, Bp, Cp, m);

    unpadMatrix(Cp, C, n);
}

void generateRandomMatrix(Matrix& A, int n) {
    random_device rd;
    mt19937_64 gen(rd());
    uniform_int_distribution<long long> dis(0, 100);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = dis(gen);
        }
    }
}

bool verifyMatrices(Matrix& C1, Matrix& C2, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (C1[i][j] != C2[i][j]) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    int n;
    cout << "Enter matrix size: ";
    cin >> n;

    if (n <= 0) {
        cout << "Error: Matrix size" << endl;
        return 1;
    }

    // Generate random matrices A and B
    Matrix A(n, vector<long long>(n, 0));
    Matrix B(n, vector<long long>(n, 0));
    generateRandomMatrix(A, n);
    generateRandomMatrix(B, n);

    // Initialize MPI
    int provided = 0;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size == 1) {
        // Single-process behavior (same as before)
        Matrix A_naive(A);
        Matrix B_naive(B);
        Matrix A_strassen(A);
        Matrix B_strassen(B);
        Matrix C_strassen(n, vector<long long>(n, 0));
        Matrix C_naive(n, vector<long long>(n, 0));

        // Test Naive 
        cout << "\n[1] NAIVE ALGORITHM" << endl;
        double start_naive = omp_get_wtime();
        naiveMultiply(A_naive, B_naive, C_naive, n);
        double end_naive = omp_get_wtime();
        double time_naive = end_naive - start_naive;
        cout << "Computation time: " << time_naive << " s" << endl;
        cout << "Result: C[0][0] = " << C_naive[0][0] << endl;

        // Test Strassen 
        cout << "\n[2] STRASSEN ALGORITHM" << endl;
        double start_strassen = omp_get_wtime();
        strassenPad(A_strassen, B_strassen, C_strassen, n);
        double end_strassen = omp_get_wtime();
        double time_strassen = end_strassen - start_strassen;
        cout << "Computation time: " << time_strassen << " s" << endl;
        cout << "Result: C[0][0] = " << C_strassen[0][0] << endl;

        // Comparison
        cout << endl;
        cout << "Naive Time:" << time_naive << " seconds" << endl;
        cout << "Strassen Time:" << time_strassen << " seconds" << endl;
        if (time_strassen > 0) {
            cout << "Speedup:         " << (time_naive / time_strassen) << "x" << endl;
            cout << "Time Saved:      " << (time_naive - time_strassen) << " seconds" << endl;
        }

        // Verify
        cout << "\nMatrix Results Match: ";
        if (verifyMatrices(C_naive, C_strassen, n)) cout << "YES" << endl;
        else cout << "NO" << endl;
    } else {
        // Distributed-memory multiplication using MPI: scatter rows of A, broadcast B
        int *rows = new int[world_size];
        int base = n / world_size;
        int rem = n % world_size;
        for (int i = 0; i < world_size; ++i) rows[i] = base + (i < rem ? 1 : 0);

        vector<int> sendcounts(world_size), displs(world_size);
        int offset = 0;
        for (int i = 0; i < world_size; ++i) {
            sendcounts[i] = rows[i] * n; // elements per rank
            displs[i] = offset;
            offset += sendcounts[i];
        }

        vector<long long> flatA, flatB, flatC;
        if (rank == 0) {
            flattenMatrix(A, flatA, n, n);
            flattenMatrix(B, flatB, n, n);
            flatC.resize(n * n);
        } else {
            flatB.resize(n * n);
        }

        // Allocate local buffers
        int local_rows = rows[rank];
        vector<long long> localA(local_rows * n);
        vector<long long> localC(local_rows * n);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        // Scatter rows of A
        MPI_Scatterv(rank==0 ? flatA.data() : NULL, sendcounts.data(), displs.data(), MPI_LONG_LONG,
                     localA.data(), local_rows * n, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

        // Broadcast B to all
        MPI_Bcast(flatB.data(), n * n, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

        // Local multiplication: convert to Matrix and compute
        Matrix A_loc(local_rows, vector<long long>(n));
        Matrix B_full(n, vector<long long>(n));
        Matrix C_loc(local_rows, vector<long long>(n, 0));
        unflattenMatrix(localA, A_loc, local_rows, n);
        unflattenMatrix(flatB, B_full, n, n);

        multiplyRect(A_loc, B_full, C_loc, local_rows, n);

        // Flatten localC and gather
        flattenMatrix(C_loc, localC, local_rows, n);
        MPI_Gatherv(localC.data(), local_rows * n, MPI_LONG_LONG,
                    rank==0 ? flatC.data() : NULL, sendcounts.data(), displs.data(), MPI_LONG_LONG,
                    0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (rank == 0) {
            // Unflatten into C_strassen-like
            Matrix C_all(n, vector<long long>(n, 0));
            unflattenMatrix(flatC, C_all, n, n);
            cout << "Distributed multiplication time: " << (t1 - t0) << " s" << endl;

            // Optionally compute naive on root for verification (may be slow)
            Matrix C_naive(n, vector<long long>(n, 0));
            double t_naive = omp_get_wtime();
            naiveMultiply(A, B, C_naive, n);
            double t_naive_end = omp_get_wtime();
            cout << "Naive (single-node) time for verification: " << (t_naive_end - t_naive) << " s" << endl;
            cout << "Verification: ";
            if (verifyMatrices(C_naive, C_all, n)) cout << "MATCH" << endl;
            else cout << "MISMATCH" << endl;
        }

        delete[] rows;
    }

    MPI_Finalize();
    

    return 0;
}
