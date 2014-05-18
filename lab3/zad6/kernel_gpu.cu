//============================================================================
// Name        : gauss_gpu
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================

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

	//--------------------------------------------------------------------------------------------
	// TODO Make implementation for Gaussian blur. This below one is for grayscale.
//	const int i = blockIdx.x * (blockDim.x * blockDim.y)
//			+ blockDim.x * threadIdx.y + threadIdx.x;
//
//	if (i < width * height)
//	{
//		float v = 0.3 * image[i].x + 0.6 * image[i].y + 0.1 * image[i].z;
//		image[i] = make_float4(v, v, v, 0);
//	}
	//--------------------------------------------------------------------------------------------
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
