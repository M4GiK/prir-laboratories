/*
 * VideoOperations.h
 *
 *  Created on: 17-05-2014
 *      Author: Michał Szczygieł <michal.szczygiel@wp.pl>
 */

// Basic OpenCV structures (cv::Mat)
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cerrno>
#include <sstream>
#include <fstream>
#include <iostream>

#ifndef VIDEOOPERATIONS_H_
#define VIDEOOPERATIONS_H_

class VideoOperations
{
public:
	int outWidth;
	int outHeight;
	cv::VideoCapture inputVideo;
	cv::VideoWriter outputVideo;

	VideoOperations(std::string inputFile, std::string outputFile);
	virtual ~VideoOperations();
	void assertFileExist(const std::string filePath);
	bool readFrames(cv::Mat input);
	void saveFrames(cv::Mat output);
	void release();
	void openVideo(std::string inputFile);
	cv::VideoCapture getOpenVideo(std::string inputFile);
	cv::VideoWriter prepareOutputVideo(std::string outputFile);
};

#endif /* VIDEOOPERATIONS_H_ */
