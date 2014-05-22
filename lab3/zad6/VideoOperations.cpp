/*
 * VideoOperations.cpp
 *
 *  Created on: 17-05-2014
 *      Author: Michał Szczygieł <michal.szczygiel@wp.pl>
 */

#include "VideoOperations.h"

/**
 * The constructor for VideoOperations.
 */
VideoOperations::VideoOperations(std::string inputFile, std::string outputFile)
{
	VideoOperations::openVideo(inputFile);
	VideoOperations::prepareOutputVideo(outputFile);
}

/**
 * The destructor for VideoOperations.
 */
VideoOperations::~VideoOperations()
{
	VideoOperations::inputVideo.~VideoCapture();
	VideoOperations::outputVideo.~VideoWriter();
}

/**
 * This method closes video file or capturing device.
 */
void release()
{
	VideoOperations::inputVideo.release();
}

/**
 *  Opens video file.
 *
 * @param inputFile The path to file.
 */
void openVideo(std::string inputFile)
{
	assertFileExist(inputFile);

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
cv::VideoCapture getOpenVideo(std::string inputFile)
{
	assertFileExist(inputFile);

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
	VideoOperations::outWidth = (int) (VideoOperations::inputVideo.get(
			CV_CAP_PROP_FRAME_WIDTH));
	VideoOperations::outHeight = (int) (VideoOperations::inputVideo.get(
			CV_CAP_PROP_FRAME_HEIGHT));

	VideoOperations::outputVideo = cv::VideoWriter(outputFile,
			(int) VideoOperations::inputVideo.get(CV_CAP_PROP_FOURCC),
			(int) VideoOperations::inputVideo.get(CV_CAP_PROP_FPS),
			cv::Size(VideoOperations::outWidth, VideoOperations::outHeight));
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

/**
 * Reads frames from given matrix.
 *
 * @param input The matrix to read.
 * @return True if can read input stream, false if can't or end given input stream.
 */
bool readFrames(cv::Mat input)
{
	return VideoOperations::inputVideo.read(input);
}

/**
 * This method saves frames from given matrix.
 *
 * @param output The matrix with data.
 */
void saveFrames(cv::Mat output)
{
	VideoOperations::outputVideo.write(output);
}

/**
 * This method checks if given file path is existing.
 *
 * @param filePath The file path to check.
 */
void assertFileExist(const std::string filePath)
{
	std::ifstream fileStream(filePath);
	if (!fileStream.good())
	{
		throw new std::invalid_argument("Input file was not found.");
	}
	else
	{
		fileStream.close();
	}
}
