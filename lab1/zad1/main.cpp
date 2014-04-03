//============================================================================
// Name        : laboratoria1.cpp
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using std::cout;
using std::cin;
using std::endl;
using std::sqrt;

typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;
typedef std::chrono::duration<double> Duration;
typedef std::vector<double> DoubleMatrix;

/*!
 * \brief initializeRandomValue Initializes
 * the pseudo-random number generator.
 */
void initializeRandomValue()
{
    srand(time(NULL));
}

/*!
 * \brief getRandomValue Gets a random value in range <0, 1)
 * from the basic pseudo-random number generator.
 * \return A random double-precision value in range <0, 1).
 */
double getRandomValue()
{
    return ((double)rand()) / RAND_MAX;
}

/*!
 * \brief fillMatrixWithZeros Fills matrix
 * with pseudo-random double precision floating-point values.
 * \param matrix Matrix to fill.
 * \param size Requested matrices size after the initialization
 * (i.e. how many values to insert).
 */
void fillMatrixWithRandomValues(DoubleMatrix &matrix, unsigned int size)
{
    for (unsigned int i = 0; i < size; ++i)
    {
        matrix.push_back(getRandomValue());
    }
}

/*!
 * \brief fillMatrixWithZeros Fills matrix
 * with double-precision zeros.
 * \param matrix Matrix to fill.
 * \param size Requested matrices size after the initialization
 * (i.e. how many values to insert).
 */
void fillMatrixWithZeros(DoubleMatrix &matrix, unsigned int size)
{
    for (unsigned int i = 0; i < size; ++i)
    {
        matrix.push_back(0.0);
    }
}

/*!
 * \brief initializeMatrices Initializes the two matrices given as parameters
 * \param A First matrix.
 * \param B Second matrix.
 * \param size Requested matrices size after the initialization
 * (i.e. how many values to insert).
 */
void initializeMatrices(DoubleMatrix &A, DoubleMatrix &B, unsigned int size)
{
    initializeRandomValue();
    fillMatrixWithRandomValues(A, size);
    fillMatrixWithRandomValues(B, size);
}

/*!
 * \brief assertSizeEqual Asserts that the two matrices
 * given as parameters are of equal size.
 * Throws an invalid argument exception if that is not the case.
 * \param A First matrix.
 * \param B Second matrix.
 */
void assertSizeEqual(const DoubleMatrix &A, const DoubleMatrix &B)
{
    if(A.size() != B.size())
    {
        throw std::invalid_argument("Matrices are not of equal size!");
    }
}

/*!
 * \brief sumMatrices Sums the two matrices given as parameters in parallel.
 * Spawns threadCount threads for this operation.
 * \param A First matrix - a vector of double precision floating-point values.
 * \param B Second matrix.
 * \param threadCount Number of threads to spawn in the parallel OpenMP block.
 * \return Third matrix - the result of addition of A and B matrices.
 */
const DoubleMatrix *sumMatrices(const DoubleMatrix &A, const DoubleMatrix &B, unsigned int threadCount)
{
    assertSizeEqual(A, B);

    DoubleMatrix *C = new DoubleMatrix();
    fillMatrixWithZeros(*C, A.size());

    #pragma omp parallel for default(none) shared(A, B, C) num_threads(threadCount)
    for (unsigned int i = 0; i < A.size(); i++)
    {
        C->at(i) = A.at(i) + B.at(i);
    }

    return C;
}

/*!
 * \brief performSumOperations Performs matrix addition of A and B matrices 10000 times.
 * Spawns threadCount threads for these operations.
 * \param A First matrix - a vector of double precision floating-point values.
 * \param B Second matrix.
 * \param threadCount Number of threads to spawn in the parallel OpenMP block.
 */
void performSumOperations(const DoubleMatrix &A, const DoubleMatrix &B, unsigned int threadCount)
{
    const DoubleMatrix *C;
    TimePoint start = std::chrono::system_clock::now();
    for (unsigned int i = 0; i < 10000; ++i)
    {
        C = sumMatrices(A, B, threadCount);
        delete(C);
    }
    TimePoint end = std::chrono::system_clock::now();

    cout << ((Duration)(end - start)).count();
}

/*!
 * \brief multiplyMatrices Multiplies the two matrices given as parameters in parallel.
 * Spawns threadCount threads for this operation.
 * \param A First matrix - a vector of double precision floating-point values.
 * \param B Second matrix.
 * \param threadCount Number of threads to spawn in the parallel OpenMP block.
 * \return Third matrix - the result of multiplication of A and B matrices.
 */
const DoubleMatrix *multiplyMatrices(const DoubleMatrix &A, const DoubleMatrix &B, unsigned int threadCount)
{
    assertSizeEqual(A, B);

    DoubleMatrix *C = new DoubleMatrix();
    fillMatrixWithZeros(*C, A.size());
    unsigned int rowSize = sqrt(A.size());

    #pragma omp parallel for default(none) shared(A, B, C) firstprivate(rowSize) num_threads(threadCount)
    for (unsigned int i = 0; i < rowSize; ++i)
    {
        //double local = 0.0;
        for (unsigned int j = 0 ; j < rowSize; ++j)
        {
            C->at(i * rowSize + j) = A.at(i * rowSize + j) * B.at(i + j * rowSize);
            //local += A.at(i * rowSize + j) * B.at(i + j * rowSize);
        }
    }

    return C;
}

/*!
 * \brief performMultiplyOperations Performs matrix multiplication of A and B matrices 10000 times.
 * Spawns threadCount threads for these operations.
 * \param A First matrix - a vector of double precision floating-point values.
 * \param B Second matrix.
 * \param threadCount Number of threads to spawn in the parallel OpenMP block.
 */
void performMultiplyOperations(const DoubleMatrix &A, const DoubleMatrix &B, unsigned int threadCount)
{
    const DoubleMatrix *C;
    TimePoint start = std::chrono::system_clock::now();
    for (unsigned int i = 0; i < 10000; ++i)
    {
        C = multiplyMatrices(A, B, threadCount);
        delete(C);
    }
    TimePoint end = std::chrono::system_clock::now();

    cout << ((Duration)(end - start)).count();
}

/*!
 * \brief main Entry point for the program.
 * Initializes and performs matrix operations.
 * \param argc Number of arguments given to the program.
 * \param argv Arguments given to the program.
 * \return C-standard return code: 0 if success,
 * other value if errors occurred during the execution.
 */
int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        cout << "Usage: ./macierz_omp <threadCount> <matrixSize>" << endl;
        return -1;
    }

    unsigned int threadCount = std::stoi(argv[1]);
    unsigned int matrixSize = std::stoi(argv[2]);

    DoubleMatrix A, B;
    initializeMatrices(A, B, matrixSize);

    performSumOperations(A, B, threadCount);
    cout << "\t";
    performMultiplyOperations(A, B, threadCount);

    return 0;
}
