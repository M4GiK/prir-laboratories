//============================================================================
// Name        : gauss_gpu
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================
#include "kernel_gpu.h"
#include "VideoOperations.h"

#include <iomanip>
#include <iostream>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/timeb.h>

using std::cout;
using std::cin;
using std::endl;

typedef struct timeb TimePoint;

/** Grid variables **/
static int threadsOnX = 1;

/** Grid variables **/
static int threadsOnY = 1;

/** Blocks assigned per kernel **/
static int blocksPerKernel = 1;

/** Time stamp for time calculation **/
TimePoint startTime, stopTime;

/**
 * This method based on run parameters. Calculates the amount of threads/blocks for the application use.
 *
 * @param threadCount The amount of threads.
 */
void prepareGrid(unsigned int threadCount)
{
	// Round the number of threads
	if ((threadCount % 2) == 1)
	{
		++threadCount;
	}

	// Divide into blocks
	// TODO to refactoring!
	for (int i = 512; i > 0; i--)
	{
		double blocks = (double) threadCount / i;

		if (blocks == (int) (blocks))
		{
			blocksPerKernel = int(blocks);

			double divThreads = sqrt(i);
			if (divThreads == int(divThreads))
			{
				threadsOnX = int(divThreads);
				threadsOnY = int(divThreads);
			}
			else
			{
				threadsOnX = i;
				threadsOnY = 1;
			}

			break;
		}
	}
}

/**
 * TODO add comments.
 *
 * @param input
 * @param output
 */
void performKernelCalculation(cv::Mat& input, cv::Mat& output)
{
	unsigned char *inputPixel, *outputPixel;

	int inputBytes = input.rows * input.step;
	int outputBytes = output.rows * output.step;

	// Prepare grid for kernel
	dim3 dimBlock(threadsOnX, threadsOnY);
	dim3 dimGrid(ceil((double) output.cols / dimBlock.x),
			ceil((double) output.rows / dimBlock.y));

	// Allocate memory for frame calculation
	cudaMalloc((void**) &inputPixel, inputBytes);
	cudaMalloc((void**) &outputPixel, outputBytes);

	// Reset output memory to 0
	cudaMemset(outputPixel, 0, outputBytes);

	// Copy input frame to GPU memory
	cudaMemcpy(inputPixel, input.ptr(), inputBytes, cudaMemcpyHostToDevice);
	unsigned int hostCounter = 0;
	cudaMemcpyToSymbol(blockCounter, &hostCounter, sizeof(unsigned int), 0,
			cudaMemcpyHostToDevice);

	int *devKernel;
	int hostKernel[5][5] =
	{
	{ 0, 1, 2, 1, 0 },
	{ 1, 4, 8, 4, 1 },
	{ 2, 8, 16, 8, 2 },
	{ 1, 4, 8, 4, 1 },
	{ 0, 1, 2, 1, 0 } };

	// Prepare memory for calculation
	int memorySize = 5 * 5 * sizeof(int);
	cudaMalloc((void**) &devKernel, memorySize);
	cudaMemcpy(devKernel, hostKernel, memorySize, cudaMemcpyHostToDevice);

	// Apply filter
	cudaGauss(blocksPerKernel, dimBlock, inputPixel, outputPixel, input.cols,
			input.rows, input.step, devKernel, 5, dimGrid.x,
			dimGrid.x * dimGrid.y);

	// Synchronize the device
	cudaDeviceSynchronize();

	// Copy result to host memory
	cudaMemcpy(output.ptr(), outputPixel, outputBytes, cudaMemcpyDeviceToHost);

	// Free the memory
	cudaFree(inputPixel);
	cudaFree(outputPixel);
}

/**
 * TODO add comments.
 *
 * @param threadCount
 * @param videoInput
 * @param videoOutput
 */
void performGaussianBlur(unsigned int threadCount, std::string videoInput,
		std::string videoOutput)
{
	VideoOperations videoOperations = new VideoOperations(videoInput,
			videoOutput);
	cv::Mat input;

	ftime(&startTime);

	while (videoOperations.readFrames(input))
	{
		cv::Mat output(videoOperations.outHeight, videoOperations.outWidth,
				cv::CV_8UC3);
		performKernelCalculation(input, output);
	}

	ftime(&stopTime);

	videoOperations.release();
}

/**
 * This method calculates time for CUDA operations.
 *
 * @return time in seconds.
 */
double getTime()
{
	return ((double) stopTime.time + ((double) stopTime.millitm * 0.001))
			- ((double) startTime.time + ((double) startTime.millitm * 0.001));
}

/**
 * TODO add comments.
 *
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		std::cerr << "Usage: ./gauss_gpu threadCount videoInput videoOutput"
				<< endl;
		return -1;
	}

	unsigned int threadCount = argv[1];
	std::string videoInput = argv[2];
	std::string videoOutput = argv[3];

	prepareGrid(threadCount);
	performGaussianBlur(threadCount, videoInput, videoOutput);
	cout << getTime() << endl;

	return 0;
}

