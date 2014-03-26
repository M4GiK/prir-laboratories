//============================================================================
// Name        : analiza_omp
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using std::cout;
using std::cin;
using std::endl;

typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;
typedef std::chrono::duration<double> Duration;

/**
 *
 * @param threadCount
 * @param fileName
 */
void analyzeDocument(unsigned int threadCount, std::string fileName)
{

}

/**
 * The main method of application which analyze of document based on trigram.
 *
 * @param argc  Number of arguments given to the program.
 * 				This value should be equal to 3.
 * @param argv
 * 				The first parameter: <threadCount>
 * 				The second parameter: <file>
 * @return  C-standard return code: 0 if success,
 * 			other value if errors occurred during the execution.
 */
int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		cout << "Usage: ./analiza_omp <threadCount> <file>" << endl;
		return -1;
	}

	unsigned int threadCount = std::stoi(argv[1]);
	std::string fileName = argv[2];

	TimePoint start = std::chrono::system_clock::now();
	analyzeDocument(threadCount, fileName);
	TimePoint end = std::chrono::system_clock::now();

	Duration elapsedMillis = end - start;
	cout << elapsedMillis.count() << endl;

	return 0;
}
