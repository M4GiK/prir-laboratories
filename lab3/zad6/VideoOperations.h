/*
 * VideoOperations.h
 *
 *  Created on: 17-05-2014
 *      Author: Michał Szczygieł <michal.szczygiel@wp.pl>
 */

#include <opencv2/core/core.hpp>		// Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>

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
	bool readFrames(cv::Mat input);
	void saveFrames(cv::Mat output);
	void assertFileExist(const std::string filePath);
	void release();
	void openVideo(std::string inputFile);
	cv::VideoCapture getOpenVideo(std::string inputFile);
	cv::VideoWriter prepareOutputVideo(std::string outputFile);
};

#endif /* VIDEOOPERATIONS_H_ */
