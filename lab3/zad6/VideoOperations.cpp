/*
 * VideoOperations.cpp
 *
 *  Created on: 17-05-2014
 *      Author: Michał Szczygieł <michal.szczygiel@wp.pl>
 */

#include "VideoOperations.h"

/**
 *
 */
VideoOperations::VideoOperations(std::string inputFile, std::string outputFile)
{
	VideoOperations::openVideo(inputFile);

}

/**
 *
 */
VideoOperations::~VideoOperations()
{
	// TODO Auto-generated destructor stub
}

/**
 *  Opens video file.
 *
 * @param inputFile The path to file.
 */
void openVideo(std::string inputFile)
{
	VideoOperations::inputVideo = cv::VideoCapture(inputFile);
	if (!VideoOperations::inputVideo.isOpened())
	{
		std::cerr << "Could not open the input video: " << inputFile
				<< std::endl;
	}
}

/**
 * Opens video file.
 *
 * @param inputFile The path to file.
 * @return The descriptor for video file capture.
 */
cv::VideoCapture openVideo(std::string inputFile)
{
	VideoOperations::inputVideo = cv::VideoCapture(inputFile);
	if (VideoOperations::inputVideo.isOpened())
	{
		return VideoOperations::inputVideo;
	}
	else
	{
		std::cerr << "Could not open the input video: " << inputFile
				<< std::endl;
		return -1;
	}
}

/**
 * Prepares output file.
 *
 * @param outputFile The path for output file.
 * @return The descriptor for video writer.
 */
cv::VideoWriter prepareOutputVideo(std::string outputFile)
{
	int outWidth = (int) (VideoOperations::inputVideo.get(
			CV_CAP_PROP_FRAME_WIDTH));
	int outHeight = (int) (VideoOperations::inputVideo.get(
			CV_CAP_PROP_FRAME_HEIGHT));

	VideoOperations::outputVideo = cv::VideoWriter(outputFile,
			(int) VideoOperations::inputVideo.get(CV_CAP_PROP_FOURCC),
			(int) VideoOperations::inputVideo.get(CV_CAP_PROP_FPS),
			cv::Size(outWidth, outHeight));
	if (VideoOperations::outputVideo.isOpened())
	{
		return VideoOperations::outputVideo;
	}
	else
	{
		std::cerr << "Failed opening file: " << outputFile << std::endl;
		return -1;
	}
}

