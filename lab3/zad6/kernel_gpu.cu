//============================================================================
// Name        : gauss_gpu
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================
#include "kernel_gpu.h"

/** Grid variables **/
static int threadsOnX = 1;

/** Grid variables **/
static int threadsOnY = 1;

/** Blocks assigned per kernel **/
static int blocksPerKernel = 1;

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

/** Global number of blocks **/
__device__ unsigned int blockCounter;

/**
 * CUDA implementation for Gaussian blur.
 * TODO add comments.
 *
 * @param input
 * @param output
 * @param width
 * @param height
 * @param inputWidthStep
 * @param kernel
 * @param kernelSize
 * @param gridWidth
 * @param numBlocks
 */
__global__ void gaussBlur(unsigned char* input, unsigned char* output,
		int width, int height, int inputWidthStep, int *kernel, int kernelSize,
		int gridWidth, int numBlocks)
{

	__shared__ unsigned int index; // index for block calculation
	__shared__ unsigned int xBlock, yBlock; // x and y of block

	// Neverending loop, for index calculation
	for (;;)
	{
		if ((threadIdx.x == 0) && (threadIdx.y == 0))
		{
			index = atomicAdd(&blockCounter, 1);
			xBlock = index % gridWidth;
			yBlock = index / gridWidth;
		}

		// Synchronize the threads
		__syncthreads();

		// Break the loop if we've exceeded the number of loops
		if (index >= numBlocks)
		{
			break;
		}
		// Calculate rows and columns of output part of frame
		int row = yBlock * blockDim.y + threadIdx.y;
		int col = xBlock * blockDim.x + threadIdx.x;

		// Make sure we don't go out of range
		if ((row >= height) || (col >= width))
		{
			continue;
		}

		// Get middle coords of kernel
		int xKernel = kernelSize / 2;
		int yKernel = kernelSize / 2;

		// Prepare pixel value variables
		double pixelVal = 0;
		double b = 0, g = 0, r = 0;
		int pixelIndex = row * inputWidthStep + 3 * col;

		// Calculate new values for pixels
		for (int i = 0; i < kernelSize; ++i)
		{
			int iRow = kernelSize - 1 - i;

			for (int j = 0; j < kernelSize; ++j)
			{
				int iCol = kernelSize - 1 - j;
				int jRow = row + (i - yKernel);
				int jCol = col + (j - xKernel);

				if ((jRow >= 0) && (jRow < height) && (jCol >= 0)
						&& (jCol < width))
				{
					double iPixelVal = kernel[iRow * kernelSize + iCol];
					int kernelIndex = jRow * inputWidthStep + (3 * jCol);

					b += input[kernelIndex] * iPixelVal;
					g += input[kernelIndex + 1] * iPixelVal;
					r += input[kernelIndex + 2] * iPixelVal;

					pixelVal += iPixelVal;
				}
			}
		}

		// First check if the values fit in pixel range...
		double bPixelVal = b / pixelVal;
		if (bPixelVal < 0)
		{
			bPixelVal = 0;
		}
		else if (bPixelVal > 255)
		{
			bPixelVal = 255;
		}

		double gPixelVal = g / pixelVal;
		if (gPixelVal < 0)
		{
			gPixelVal = 0;
		}
		else if (gPixelVal > 255)
		{
			gPixelVal = 255;
		}

		double rPixelVal = r / pixelVal;
		if (rPixelVal < 0)
		{
			rPixelVal = 0;
		}
		else if (rPixelVal > 255)
		{
			rPixelVal = 255;
		}

		// ... then write the output
		output[pixelIndex] = (unsigned char) (bPixelVal);
		output[pixelIndex + 1] = (unsigned char) (gPixelVal);
		output[pixelIndex + 2] = (unsigned char) (rPixelVal);
	}
}

/**
 * This method invoke CUDA implementation from "C" code.
 * TODO add comments.
 *
 * @param blocks
 * @param block_size
 * @param input
 * @param output
 * @param width
 * @param height
 * @param inputWidthStep
 * @param kernel
 * @param kernelSize
 * @param gridWidth
 * @param numBlocks
 */
extern "C" void cudaGauss(dim3 blocks, dim3 block_size, unsigned char* input,
		unsigned char* output, int width, int height, int inputWidthStep,
		int *kernel, int kernelSize, int gridWidth, int numBlocks)
{
	gaussBlur<<< blocks, block_size >>> (input, output, width , height, inputWidthStep, kernel, kernelSize, gridWidth, numBlocks);
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

	// Prepare grid for kernel.
	dim3 dimBlock(threadsOnX, threadsOnY);
	dim3 dimGrid(ceil((double) output.cols / dimBlock.x),
			ceil((double) output.rows / dimBlock.y));

	// Allocate memory for frame calculation.
	cudaMalloc((void**) &inputPixel, inputBytes);
	cudaMalloc((void**) &outputPixel, outputBytes);

	// Reset output memory to 0.
	cudaMemset(outputPixel, 0, outputBytes);

	// Copy input frame to GPU memory.
	cudaMemcpy(inputPixel, input.ptr(), inputBytes, cudaMemcpyHostToDevice);
	unsigned int hostCounter = 0;
	cudaMemcpyToSymbol(blockCounter, &hostCounter, sizeof(unsigned int), 0,
			cudaMemcpyHostToDevice);

	// Prepare table to gauss conversion.
	int *devKernel;
	int hostKernel[5][5];
	memcpy(hostKernel, GAUSS, sizeof(GAUSS));

	// Prepare memory for calculation
	int memorySize = 5 * 5 * sizeof(int);
	cudaMalloc((void**) &devKernel, memorySize);
	cudaMemcpy(devKernel, hostKernel, memorySize, cudaMemcpyHostToDevice);

	// Apply filter Gauss blur.
	cudaGauss(blocksPerKernel, dimBlock, inputPixel, outputPixel, input.cols,
			input.rows, input.step, devKernel, 5, dimGrid.x,
			dimGrid.x * dimGrid.y);

	// Synchronize the device.
	cudaDeviceSynchronize();

	// Copy result to host memory.
	cudaMemcpy(output.ptr(), outputPixel, outputBytes, cudaMemcpyDeviceToHost);

	// Free the memory.
	cudaFree(inputPixel);
	cudaFree(outputPixel);
}
