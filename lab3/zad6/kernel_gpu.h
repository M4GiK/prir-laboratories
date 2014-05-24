//============================================================================
// Name        : gauss_gpu
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "VideoOperations.h"

/**
 * Header for the method of doing a gaussian blur.
 *
 * @param inputPixel
 * @param outputPixel
 * @param width
 * @param height
 * @param channels
 * @param blocksPerKernel
 */
extern "C" void cudaGauss(unsigned char* inputPixel, unsigned char* outputPixel,
		int width, int height, int channels, int blocksPerKernel);

/** Grid variables **/
static int threadsOnX = 1;

/** Grid variables **/
static int threadsOnY = 1;

/** Blocks assigned per kernel **/
static int blocksPerKernel = 1;

/** Size for Gauss kernel **/
const static int KERNEL_SIZE = 5;

/** Kernel for Gauss blur, if the gausse is more then picture is more blurred **/
const static int KERNEL[KERNEL_SIZE][KERNEL_SIZE] = {
		{ 0, 1, 2, 1, 0 },
		{ 1, 4, 8, 4, 1 },
		{ 2, 8, 16, 8, 2 },
		{ 1, 4, 8, 4, 1 },
		{ 0, 1, 2, 1, 0 } };

/** Pointer to kernel **/
int *devKernel;

/**
 * This method based on run parameters. Calculates the amount of threads/blocks for the application use.
 *
 * @param threadCount The amount of threads
 */
void prepareGrid(unsigned int threadCount);

/**
 * Prepares table to gauss conversion.
 */
void prepareFilter();

/**
 * This method returns sum of given array.
 *
 * @param array The array with data to calculate the sum.
 * @return The sum of given array.
 */
int sumArray(int array[KERNEL_SIZE][KERNEL_SIZE]);

/**
 * Performs kernel operations.
 */
cv::Mat performKernelCalculation(cv::Mat& input);
