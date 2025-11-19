#include <iostream>
#include <omp.h>
#include <vector>
#include <cstring>
#include <cmath>
#include <random>

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

// Strassen algorithm
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
        cout << "Error: Matrix size must be positive" << endl;
        return 1;
    }

    // Generate random matrices A and B
    Matrix A(n, vector<long long>(n, 0));
    Matrix B(n, vector<long long>(n, 0));
    generateRandomMatrix(A, n);
    generateRandomMatrix(B, n);

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
    cout << "Naive Time:     " << time_naive << " seconds" << endl;
    cout << "Strassen Time:  " << time_strassen << " seconds" << endl;
    if (time_strassen > 0) {
        cout << "Speedup:        " << (time_naive / time_strassen) << "x" << endl;
        cout << "Time Saved:     " << (time_naive - time_strassen) << " seconds" << endl;
    }

    // Verify
    cout << "\nMatrix Results Match: ";
    if (verifyMatrices(C_naive, C_strassen, n)) cout << "YES" << endl;
    else cout << "NO" << endl;

    return 0;
}
