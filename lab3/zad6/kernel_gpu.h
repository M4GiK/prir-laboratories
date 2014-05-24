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
 * @param inputPixel The pixel to perform.
 * @param outputPixel The result of pixel conversion.
 * @param width	The width of frame.
 * @param height The height of frame.
 * @param channels The channels color for pixel.
 */
extern "C" void cudaGauss(unsigned char* inputPixel, unsigned char* outputPixel,
		int width, int height, int channels);

/** Grid variables **/
static int threadsOnX = 1;

/** Grid variables **/
static int threadsOnY = 1;

/** Size for Gauss kernel **/
const static int KERNEL_SIZE = 3;

/** Kernel for Gauss blur, if the gausse is more then picture is more blurred **/
const static float KERNEL[9] = {1.0f, 2.0f, 1.0f, 2.0f, 4.0f, 2.0f, 1.0f, 2.0f, 1.0f};

/**
 * Gets time in seconds for CUDA kernel operations.
 *
 * @return Time in seconds for CUDA operations.
 */
float getGPUTime();

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
 * Starts event time calculation.
 *
 * @param start The cuda event time instance.
 * @param stop The cuda event time instance.
 */
void cudaEventTimer_start(cudaEvent_t *start, cudaEvent_t *stop);

/**
 * Stops event time calculation.
 *
 * @param start The cuda event time instance.
 * @param stop The cuda event time instance.
 * @return The event elapsed time.
 */
float cudaEventTimer_stop(cudaEvent_t start, cudaEvent_t stop);

/**
 * This method returns sum of given array.
 *
 * @param array The array with data to calculate the sum.
 * @return The sum of given array.
 */
int sumArray(const float array[KERNEL_SIZE*KERNEL_SIZE]);

/**
 * Performs kernel operations.
 */
cv::Mat performKernelCalculation(cv::Mat& input);
