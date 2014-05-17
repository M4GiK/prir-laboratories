/*
 * VideoOperations.h
 *
 *  Created on: 17-05-2014
 *      Author: Michał Szczygieł <michal.szczygiel@wp.pl>
 */

#include "opencv2/opencv.hpp"			// Basic OpenCV structures (cv::Mat)
#include <iostream>
#include <sstream>

#ifndef VIDEOOPERATIONS_H_
#define VIDEOOPERATIONS_H_

class VideoOperations
{
public:
	cv::VideoCapture inputVideo;
	cv::VideoWriter outputVideo;

	VideoOperations(std::string inputFile, std::string outputFile);
	virtual ~VideoOperations();
	void openVideo(std::string inputFile);
	cv::VideoCapture openVideo(std::string inputFile);
	cv::VideoWriter prepareOutputVideo(std::string outputFile);
};

#endif /* VIDEOOPERATIONS_H_ */
