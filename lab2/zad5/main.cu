#include <iostream>
#include <cstdio>

using std::cout;
using std::endl;

__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int N)
{
	int threadId = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * 1024 * 1024;
	int row = threadId / N;
	int col = threadId % N;

	float result = 0.f;

	for (int i = 0; i < N; ++i)
	{
        if (row < N && col < N)
        {
	        result += A[(row * N) + i] * B[(i * N) + col];
        }
	}

	C[(row * N) + col] = result;
}

float *initializeMatrix(unsigned int size)
{
	float *matrix = new float[size * size];
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size; ++j)
		{
	    	matrix[(i * size) + j] = rand() % 100;
	    }
	}
	return matrix;
}

float *allocateDeviceMemory(int bufferSize)
{
    float *device;
	cudaMalloc(&device, bufferSize);
	return device;
}

void copyHostMemoryToDevice(float *host, float *device, int bufferSize)
{
	cudaMemcpy(device, host, bufferSize, cudaMemcpyHostToDevice);
}

void createTimerEvents(cudaEvent_t &start, cudaEvent_t &stop)
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

void destroyTimerEvents(cudaEvent_t &start, cudaEvent_t &stop)
{
	cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void freeMemory(float *devA, float *hostA, float *devB, float *hostB, float *devC)
{
    cudaFree(devA);
    free(hostA);

    cudaFree(devB);
    free(hostB);

    cudaFree(devC);
}

void startTimer(cudaEvent_t &start)
{
    cudaEventRecord(start, 0);
}

void stopTimer(cudaEvent_t &stop)
{
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
}

float readExecutionTime(cudaEvent_t &start, cudaEvent_t &stop)
{
    float time;
    cudaEventElapsedTime(&time, start, stop);
    return time;
}

int main(int argc, char *argv[])
{
	if (argc != 3)
	{
		std::cerr << "Usage: ./macierz_cuda threadCount matrixSize" << endl;
		return -1;
	}

	unsigned int matrixSize = atoi(argv[1]);
	unsigned int threadCount = atoi(argv[2]);

	dim3 dimBlock;
	dimBlock.x = matrixSize;
	dimBlock.y = matrixSize;

    float *hostA = initializeMatrix(matrixSize);
    float *hostB = initializeMatrix(matrixSize);

	int allocBuffer = matrixSize * matrixSize * sizeof(float);
	float *devA = allocateDeviceMemory(allocBuffer);
	float *devB = allocateDeviceMemory(allocBuffer);
	float *devC = allocateDeviceMemory(allocBuffer);

	copyHostMemoryToDevice(hostA, devA, allocBuffer);
	copyHostMemoryToDevice(hostB, devB, allocBuffer);

    cudaEvent_t start, stop;
	createTimerEvents(start, stop);

    startTimer(start);

    matrixMultiplyKernel<<<dimBlock, threadCount>>>(devA, devB, devC, matrixSize);

    stopTimer(stop);
	
    cout << readExecutionTime(start, stop) << endl;

    destroyTimerEvents(start, stop);
    freeMemory(devA, hostA, devB, hostB, devC);
    
    return 0;
}

