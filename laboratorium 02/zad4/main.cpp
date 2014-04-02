//============================================================================
// Name        : wykrywanie_omp
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================

#define _CRT_SECURE_NO_WARNINGS
#include <cerrno>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <map>
#include <math.h>
#include <string.h>
#include <stdexcept>
#include <stdio.h>
#include <stdexcept>

using std::cout;
using std::endl;

typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;
typedef std::chrono::duration<double> Duration;



/**
 * The main method of application which analyze of document based on trigram.
 *
 * @param argc  Number of arguments given to the program.
 * 				This value should be equal to 2.
 * @param argv
 * 				The first parameter:  <threadCount>
 * 				The second parameter:  <file>
 * @return  C-standard return code: 0 if success,
 * 			other value if errors occurred during the execution.
 */
int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		cout << "Usage: ./wykrywanie_omp <threadCount> <file> " << endl;
		return -1;
	}

	unsigned int threadCount = std::stoi(argv[1]);
	char *filePath = argv[2];

	TimePoint start = std::chrono::system_clock::now();

	TimePoint end = std::chrono::system_clock::now();

	Duration elapsedMillis = end - start;

	return 0;
}
