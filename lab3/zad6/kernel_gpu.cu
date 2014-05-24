//============================================================================
// Name        : gauss_gpu
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================
#include "kernel_gpu.h"

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
 * @param array The array with data to calculate the sum.
 * @return The sum of given array.
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

/** Global number of blocks **/
__device__ unsigned int blockCounter;


__device__ int getGlobalId()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	return threadId;
}

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
__global__ void gaussBlur(unsigned char *imageIn, unsigned char *imageOut,
		int width, int height, int channels,
		const int KERNEL[KERNEL_SIZE][KERNEL_SIZE], int kernelSize, int adjustedSize, int sum)
{
	const int index = getGlobalId();

	if (index < width * height)
	{
		if (index % width < adjustedSize
				|| index % width >= width - adjustedSize
				|| index / width < adjustedSize
				|| index / width >= height - adjustedSize)
		{
		}
		else
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

					x += imageIn[(index + shift) * channels] * KERNEL[j][i];
					y += imageIn[(index + shift) * channels + 1] * KERNEL[j][i];
					z += imageIn[(index + shift) * channels + 2] * KERNEL[j][i];
				}

			}
			// Apply to output image and save result.
			imageOut[index * channels] = (unsigned char) (x / sum);
			imageOut[index * channels + 1] = (unsigned char) (y / sum);
			imageOut[index * channels + 2] = (unsigned char) (z / sum);
		}
	}
//	__shared__ unsigned int index; // index for block calculation
//	__shared__ unsigned int xBlock, yBlock; // x and y of block
//
//	// Neverending loop, for index calculation
//	for (;;)
//	{
//		if ((threadIdx.x == 0) && (threadIdx.y == 0))
//		{
//			index = atomicAdd(&blockCounter, 1);
//			xBlock = index % gridWidth;
//			yBlock = index / gridWidth;
//		}
//
//		// Synchronize the threads
//		__syncthreads();
//
//		// Break the loop if we've exceeded the number of loops
//		if (index >= numBlocks)
//		{
//			break;
//		}
//		// Calculate rows and columns of output part of frame
//		int row = yBlock * blockDim.y + threadIdx.y;
//		int col = xBlock * blockDim.x + threadIdx.x;
//
//		// Make sure we don't go out of range
//		if ((row >= height) || (col >= width))
//		{
//			continue;
//		}
//
//		// Get middle coords of kernel
//		int xKernel = kernelSize / 2;
//		int yKernel = kernelSize / 2;
//
//		// Prepare pixel value variables
//		double pixelVal = 0;
//		double b = 0, g = 0, r = 0;
//		int pixelIndex = row * inputWidthStep + 3 * col;
//
//		// Calculate new values for pixels
//		for (int i = 0; i < kernelSize; ++i)
//		{
//			int iRow = kernelSize - 1 - i;
//
//			for (int j = 0; j < kernelSize; ++j)
//			{
//				int iCol = kernelSize - 1 - j;
//				int jRow = row + (i - yKernel);
//				int jCol = col + (j - xKernel);
//
//				if ((jRow >= 0) && (jRow < height) && (jCol >= 0)
//						&& (jCol < width))
//				{
//					double iPixelVal = kernel[iRow * kernelSize + iCol];
//					int kernelIndex = jRow * inputWidthStep + (3 * jCol);
//
//					b += input[kernelIndex] * iPixelVal;
//					g += input[kernelIndex + 1] * iPixelVal;
//					r += input[kernelIndex + 2] * iPixelVal;
//
//					pixelVal += iPixelVal;
//				}
//			}
//		}
//
//		// First check if the values fit in pixel range...
//		double bPixelVal = b / pixelVal;
//		if (bPixelVal < 0)
//		{
//			bPixelVal = 0;
//		}
//		else if (bPixelVal > 255)
//		{
//			bPixelVal = 255;
//		}
//
//		double gPixelVal = g / pixelVal;
//		if (gPixelVal < 0)
//		{
//			gPixelVal = 0;
//		}
//		else if (gPixelVal > 255)
//		{
//			gPixelVal = 255;
//		}
//
//		double rPixelVal = r / pixelVal;
//		if (rPixelVal < 0)
//		{
//			rPixelVal = 0;
//		}
//		else if (rPixelVal > 255)
//		{
//			rPixelVal = 255;
//		}
//
//		// ... then write the output
//		output[pixelIndex] = (unsigned char) (bPixelVal);
//		output[pixelIndex + 1] = (unsigned char) (gPixelVal);
//		output[pixelIndex + 2] = (unsigned char) (rPixelVal);
//	}
}

/**
 * This method invoke CUDA implementation from "C" code.
 *
 * @param inputPixel
 * @param outputPixel
 * @param width
 * @param height
 * @param channelscd
 * @param blocksPerKernel
 */
extern "C" void cudaGauss(unsigned char* inputPixel, unsigned char* outputPixel,
		int width, int height, int channels, int blocksPerKernel)
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

	gaussBlur<<< dimGrid, dimBlock >>> (imageIn, imageOut, width , height, channels, KERNEL, KERNEL_SIZE, std::floor(KERNEL_SIZE / 2), sumArray(KERNEL));

	cudaMemcpy(inputPixel, imageIn, size, cudaMemcpyDeviceToHost);
	cudaFree(inputPixel);
	cudaFree(outputPixel);

}

/**
 * Performs kernel operations.
 *
 * @param input The matrix data to perform.
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
	cudaGauss(inputPixel, outputPixel, width, height, channels,
			blocksPerKernel);

	return output;
}
