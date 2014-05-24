//============================================================================
// Name        : gauss_gpu
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================
#include "kernel_gpu.h"

/** Pointer to Gauss kernel **/
int **devKernel;

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
	for (int i = 512; i > 0; i--)
	{
		double blocks = (double) threadCount / i;

		if (blocks == (int) (blocks))
		{
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
 * Prepares table to gauss conversion.
 */
void prepareFilter()
{
	cudaMalloc((void**) &devKernel, sizeof(KERNEL));
	cudaMemcpy(devKernel, KERNEL, sizeof(KERNEL), cudaMemcpyHostToDevice);
}

/**
 * This method returns sum of given array.
 *
 * @param array 		The array with data to calculate the sum.
 * @return 				The sum of given array.
 */
int sumArray(const int array[KERNEL_SIZE][KERNEL_SIZE])
{
	int sum = 0;
	for (int i = 0; i < KERNEL_SIZE; i++)
	{
		for (int j = 0; j < KERNEL_SIZE; j++)
		{
			sum += array[i][j];
		}
	}

	return sum;
}

/**
 * This method gets current thread id.
 *
 * @return The proper thread id.
 */__device__ int getThreadId()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;

	return threadId;
}

/**
 * CUDA implementation for Gaussian blur.
 *
 * @param imageIn 		The pixel to perform.
 * @param imageOut 		The result of pixel conversion.
 * @param width			The width of frame.
 * @param height 		The height of frame.
 * @param channels 		The channels color for pixel.
 * @param kernel		The kernel for gaussian blur.
 * @param kernelSize	The size of kernel.
 * @param adjustedSize	The adjusted size of kernel (usually the half of kernel size rounded down).
 * @param sum			The sum of all kernel values.
 */__global__ void gaussBlur(unsigned char *imageIn, unsigned char *imageOut,
		int width, int height, int channels, int **kernel, int kernelSize,
		int adjustedSize, int sum)
{
	const int index = getThreadId();

	if (index < width * height)
	{
		if (!(index % width < adjustedSize
				|| index % width >= width - adjustedSize
				|| index / width < adjustedSize
				|| index / width >= height - adjustedSize))

		{
			float x = 0.0f;
			float y = 0.0f;
			float z = 0.0f;

			for (int j = 0; j < kernelSize; ++j)
			{
				for (int i = 0; i < kernelSize; ++i)
				{
					// Compute index shift to neighboring cords.
					int shift = ((j + i) / kernelSize - adjustedSize) * width
							+ (j + i) % kernelSize - adjustedSize;

					x += imageIn[(index + shift) * channels] * kernel[j][i];
					y += imageIn[(index + shift) * channels + 1] * kernel[j][i];
					z += imageIn[(index + shift) * channels + 2] * kernel[j][i];
				}

			}

			// Apply to output image and save result.
			imageOut[index * channels] = (unsigned char) (x / sum);
			imageOut[index * channels + 1] = (unsigned char) (y / sum);
			imageOut[index * channels + 2] = (unsigned char) (z / sum);
		}
	}
}

/**
 * This method invoke CUDA implementation from "C" code.
 *
 * @param inputPixel 	The pixel to perform.
 * @param outputPixel 	The result of pixel conversion.
 * @param width			The width of frame.
 * @param height 		The height of frame.
 * @param channels 		The channels color for pixel.
 */
extern "C" void cudaGauss(unsigned char* inputPixel, unsigned char* outputPixel,
		int width, int height, int channels)
{
	unsigned char *imageIn, *imageOut;
	int size = width * height * channels;

	cudaMalloc((void**) &imageIn, sizeof(unsigned char) * size);
	cudaMalloc((void**) &imageOut, sizeof(unsigned char) * size);
	cudaMemcpy(imageIn, inputPixel, sizeof(unsigned char) * size,
			cudaMemcpyHostToDevice);
	cudaMemcpy(imageOut, outputPixel, sizeof(unsigned char) * size,
			cudaMemcpyHostToDevice);

	// Prepare grid for kernel.
	dim3 dimBlock(threadsOnX, threadsOnY);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
			(height + dimBlock.y - 1) / dimBlock.y);

	gaussBlur<<< dimGrid, dimBlock >>> (imageIn, imageOut, width , height, channels, devKernel, KERNEL_SIZE, std::floor(KERNEL_SIZE / 2), sumArray(KERNEL));

	cudaMemcpy(inputPixel, imageIn, size, cudaMemcpyDeviceToHost);
	cudaFree(inputPixel);
	cudaFree(outputPixel);

}

/**
 * Performs kernel operations.
 *
 * @param input The matrix data to perform.
 * @return The performed frame.
 */
cv::Mat performKernelCalculation(cv::Mat& input)
{
	cv::Mat output = input.clone();

	int width = input.cols;
	int height = input.rows;
	int channels = input.channels();
	unsigned char *inputPixel = input.data;
	unsigned char *outputPixel = output.data;

	// Apply filter Gauss blur.
	cudaGauss(inputPixel, outputPixel, width, height, channels);

	return output;
}
