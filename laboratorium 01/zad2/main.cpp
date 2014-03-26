//============================================================================
// Name        : pi_omp
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
using std::sqrt;

typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;
typedef std::chrono::duration<double> Duration;

/**
 * This method initialize seed for erand48 method.
 *
 * @param seed The array to fill.
 * @param threadCount The amount of threads.
 */
void initializeSeed(unsigned short seed[3], unsigned int threadCount)
{
	seed[0] = 11;
	seed[1] = 22;
	seed[2] = threadCount;
}

/**
 * Generates a groups of pseudo-random points.
 * The generation process is splitted by using OpenMP.
 *
 * @param threadCount The amount of threads.
 * @param pointCount The amount of points to generate.
 * @return The value of hit points into wheel.
 */
unsigned int generatePoints(unsigned int threadCount, int pointCount)
{
	unsigned int wheelHits = 0;
	unsigned short seed[3];
	double x;
	double y;

	#pragma omp parallel num_threads(threadCount)
	{
		wheelHits = 0;
		initializeSeed(seed, threadCount);
		#pragma omp for firstprivate(seed) private(x,y) reduction(+:wheelHits)
		for (int i = 0; i < pointCount; i++)
		{
			x = erand48(seed);
			y = erand48(seed);
			if (std::pow(x, 2) + std::pow(y, 2) <= 1)
			{
				++wheelHits;
			}
		}
	}

	return wheelHits;
}

/**
 * Calculates the value of PI from points which hit into wheel.
 *
 * @param wheelHits The amount of hit points into wheel.
 * @param pointCount The amount of generated points.
 * @return The approximate value of PI number.
 */
double calculatePi(unsigned int wheelHits, unsigned int pointCount)
{
	return 4.0 * ((double) wheelHits / (double) pointCount);
}

/**
 * Calculates the approximate value of PI number using Monte Carlo method.
 * This method generates points into square,
 * and after this process calculate the PI value.
 *
 * @param threadCount The amount of threads.
 * @param pointCount The amount of points to generate.
 * @return The approximate value of PI number.
 */
double approximatePiMonteCarlo(unsigned int threadCount,
		unsigned int pointCount)
{
	unsigned int wheelHits = generatePoints(threadCount, pointCount);

	return calculatePi(wheelHits, pointCount);
}

/**
 * The main method of application which calculate approximate value of PI number.
 *
 * @param argc This value should be equal to 3.
 * @param argv
 * 				The first parameter: <threadCount>
 * 				The second parameter: <pointCount>
 * @return 0 as a correct value of exit of the program.
 */
int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		cout << "Usage: ./pi_omp <threadCount> <pointCount>" << endl;
		return -1;
	}

	unsigned int threadCount = std::stoi(argv[1]);
	int pointCount = std::stoi(argv[2]);

	TimePoint start = std::chrono::system_clock::now();
	approximatePiMonteCarlo(threadCount, pointCount);
	TimePoint end = std::chrono::system_clock::now();

	Duration elapsedMillis = end - start;
	cout << elapsedMillis.count() << endl;

	return 0;
}
