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
 * This method checks if given file path is existing.
 *
 * @param filePath The file path to check.
 */
void VideoOperations::assertFileExist(const std::string filePath)
{
	std::ifstream fileStream(filePath.c_str());
	if (!fileStream.good())
	{
		std::cerr << "Input file was not found: " << filePath << std::endl;
		throw(errno);
	}
	else
	{
		fileStream.close();
	}
}

/**
 * This method closes video file or capturing device.
 */
void VideoOperations::release()
{
	VideoOperations::inputVideo.release();
}

/**
 *  Opens video file.
 *
 * @param inputFile The path to file.
 */
void VideoOperations::openVideo(std::string inputFile)
{
	assertFileExist(inputFile);

	VideoOperations::inputVideo = cv::VideoCapture(inputFile);
	if (!VideoOperations::inputVideo.isOpened())
	{
		std::cerr << "Could not open the input video: " << inputFile
				<< std::endl;
		throw(errno);
	}
}

/**
 * Opens video file.
 *
 * @param inputFile The path to file.
 * @return The descriptor for video file capture.
 */
cv::VideoCapture VideoOperations::getOpenVideo(std::string inputFile)
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
		throw(errno);
	}
}

/**
 * Prepares output file.
 *
 * @param outputFile The path for output file.
 * @return The descriptor for video writer.
 */
cv::VideoWriter VideoOperations::prepareOutputVideo(std::string outputFile)
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
		throw(errno);
	}
}

/**
 * Reads frames from given matrix.
 *
 * @param input The matrix to read.
 * @return True if can read input stream, false if can't or end given input stream.
 */
bool VideoOperations::readFrames(cv::Mat input)
{
	return VideoOperations::inputVideo.read(input);
}

/**
 * This method saves frames from given matrix.
 *
 * @param output The matrix with data.
 */
void VideoOperations::saveFrames(cv::Mat output)
{
	VideoOperations::outputVideo.write(output);
}

