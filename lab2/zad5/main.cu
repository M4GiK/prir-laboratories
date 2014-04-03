//============================================================================
// Name        : macierz_cuda
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================
#include <chrono>
#include <iostream>
#include <string>

using std::cout;
using std::endl;

typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;
typedef std::chrono::duration<double> Duration;


/**
 * The main method of application which  performs matrix multiplication of A and B matrices.
 *
 * @param argc  Number of arguments given to the program.
 * 				This value should be equal to 2.
 * @param argv
 * 				The first parameter:  <threadCount>
 * 				The second parameter: <size>
 * @return  C-standard return code: 0 if success,
 * 			other value if errors occurred during the execution.
 */
int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		cout << "Usage: ./macierz_cuda <threadCount> <size>" << endl;
		return -1;
	}

	unsigned int threadCount = std::stoi(argv[1]);
	unsigned int size = std::stoi(argv[2]);

	TimePoint start = std::chrono::system_clock::now();

	TimePoint end = std::chrono::system_clock::now();

	Duration elapsedMillis = end - start;

	return 0;
}
