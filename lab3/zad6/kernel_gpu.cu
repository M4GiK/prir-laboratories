//============================================================================
// Name        : gauss_gpu
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================

/**
 * CUDA implementation for Gaussian blur.
 *
 * @param image
 * @param width
 * @param height
 */
__global__ void gaussBlur(float4* image, int width, int height)
{

	//--------------------------------------------------------------------------------------------
	// TODO Make implementation for Gaussian blur. This below one is for grayscale.
	const int i = blockIdx.x * (blockDim.x * blockDim.y)
			+ blockDim.x * threadIdx.y + threadIdx.x;

	if (i < width * height)
	{
		float v = 0.3 * image[i].x + 0.6 * image[i].y + 0.1 * image[i].z;
		image[i] = make_float4(v, v, v, 0);
	}
	//--------------------------------------------------------------------------------------------
}

/**
 * This method invoke CUDA implementation from "C" code.
 *
 * @param image
 * @param width
 * @param height
 * @param blocks
 * @param block_size
 */
extern "C" void cudaGauss(float* image, int width, int height, dim3 blocks,
		dim3 block_size)
{
	gaussBlur<<< blocks, block_size >>> ((float4*)image, width, height);
}
