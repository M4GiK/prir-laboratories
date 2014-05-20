//============================================================================
// Name        : gauss_gpu
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================
#include <cuda_runtime_api.h>
#include <cuda.h>

/**
 * Header for the method of doing a gaussian blur.
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
		int *kernel, int kernelSize, int gridWidth, int numBlocks);

/** Global number of blocks **/
__device__ unsigned int blockCounter;

/**  Filter table for Gauss blur **/
static int GAUSS[5][5] = 	{
		{ 0, 1, 2, 1, 0 },
		{ 1, 4, 8, 4, 1 },
		{ 2, 8, 16, 8, 2 },
		{ 1, 4, 8, 4, 1 },
		{ 0, 1, 2, 1, 0 } };
