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

/** Time stamp for time calculation **/
TimePoint startTime, stopTime;

/**
 * TODO add comments.
 *
 * @param threadCount
 * @param videoInput
 * @param videoOutput
 */
void performGaussianBlur(std::string videoInput, std::string videoOutput)
{
	VideoOperations *videoOperations = new VideoOperations(videoInput,
			videoOutput);
	cv::Mat input;

	ftime(&startTime);
	cudaEvent_t start, stop;

	while (videoOperations->readFrames(input))
	{
		cv::Mat currentFrame, finalFrame;
		videoOperations->inputVideo>>currentFrame;
		finalFrame = performKernelCalculation(currentFrame);
		videoOperations->saveFrames(finalFrame);
	}

	ftime(&stopTime);

	videoOperations->release();
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
 * The main method of application which converts given video to blurred video.
 *
 * @param argc	Number of arguments given to the program.
 * 						This value should be equal to 3.
 *
 * @param argv			The first parameter:  <threadCount>
 * 						The second parameter: <videoInput>
 * 						The third parameter:  <videoOutput>
 *
 * @return C-standard return code: 0 if success,
 * 						other value if errors occurred during the execution.
 */
int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		std::cerr << "Usage: ./gauss_gpu threadCount videoInput videoOutput"
				<< endl;
		return -1;
	}

	unsigned int threadCount = atoi(argv[1]);
	std::string videoInput = argv[2];
	std::string videoOutput = argv[3];

	prepareGrid(threadCount);
	prepareFilter();
	performGaussianBlur(videoInput, videoOutput);
	cout << getTime() << endl;

	return 0;
}

