//============================================================================
// Name        : gauss_gpu
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================
#include <cuda_runtime_api.h>
#include <cuda.h>

/**
 * Header for the method of doing a gaussian blur.
 *
 * @param image
 * @param width
 * @param height
 * @param blocks
 * @param block_size
 */
extern "C" void cudaGauss(float* image, int width, int height, dim3 blocks,
		dim3 block_size);
