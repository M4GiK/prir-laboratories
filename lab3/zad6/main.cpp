//============================================================================
// Name        : gauss_gpu
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================
#include "kernel_gpu.h"

#include <cv.h>
#include <highgui.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using std::cout;
using std::cin;
using std::endl;
using std::string;

typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;
typedef std::chrono::duration<double> Duration;
typedef std::vector<double> DoubleMatrix;

void assertFileExist(const std::ios &input)
{
	if (!input.good())
	{
		throw new std::invalid_argument("Input file was not found.");
	}
}

void loadFile(string filePath)
{
	std::ifstream fileStream (filePath);
	assertFileExist (fileStream);
}

void performGaussianBlur(unsigned int threadCount, string videoInput,
		string videoOutput)
{
	loadFile(videoInput);

}

int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		std::cerr << "Usage: ./gauss_gpu threadCount videoInput videoOutput"
				<< endl;
		return -1;
	}

	unsigned int threadCount = std::stoi(argv[1]);
	string videoInput = argv[2];
	string videoOutput = argv[3];

	performGaussianBlur(threadCount, videoInput, videoOutput);

	return 0;
}

