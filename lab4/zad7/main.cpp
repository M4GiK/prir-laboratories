#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include <mpi.h>

using namespace std;

using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::string;

typedef std::vector<long> LongVector;
typedef std::multimap<long, long> LongMultimap;

int main(int argc, char * * argv);

/**
 * @brief Searches for primes located in inputPath file, performing
 * Rabin-Miller tests repeatCount times on every number.
 * @param inputPath Location of the file containg numbers.
 * @param repeatCount Number of repeats to be performed in Rabin-Miller test.
 */
void searchForPrimes(const string &inputPath, const unsigned int repeatCount);

/**
 * @brief Reads numbers from a file located in inputPath.
 * @param inputPath File location.
 * @return Numbers read from the file.
 */
const LongVector *readNumbersFromFile(const string &inputPath);

/**
 * @brief Searches for primes in the collection given as a parameter.
 * @param numbers Numbers to be checked.
 * @param results Collection to save the results to.
 * @param repeatCount Number of repeats to be performed in Rabin-Miller test.
 */
void searchForPrimesUsingMPI(const LongVector *numbers, LongMultimap &results, const unsigned int repeatCount);

/**
 * @brief Sends the number to computing node for a Rabin-Miller test.
 * @param input Number to be sent.
 * @param destRank Rank of the computing node.
 * @param sendRequest Send request of the operation.
 */
void sendNumberToTest(long input, const int destRank, MPI_Request &sendRequest);

/**
 * @brief Performs Rabin-Miller test on the received number.
 * @param input Received number.
 * @param receiveRequest Receive request of the operation.
 * @param repeatCount Number of repeats for Rabin-Miller test.
 */
void performTest(long input, MPI_Request &receiveRequest, const unsigned int repeatCount);

/**
 * @brief Saves responses from normal nodes to the leader node.
 * @param results Results of the computation done so far.
 * @param destRank Rank of the computing node.
 * @param status Status of the operation.
 */
void saveResponses(LongMultimap &results, const int destRank, MPI_Status &status);

/**
 * @brief Prints results in a manner described
 * in the assignment guidelines.
 * @param results Results of the computation.
 */
void printResults(const LongMultimap &results);

/**
 * @brief Prints computation time in a manner described
 * in the assignment guidelines.
 * @param start Timestamp for start of computation.
 * @param end Timestamp for end of computation.
 */
void printTime(double start, double end);

/**
 * @brief Performs a Rabin-Miller test
 * to check whether value is a prime number.
 * @param value Value to check.
 * @param repeatCount Number of repeats.
 * @return Whether the value given has passed
 * repeatCount rounds of Rabin-Miller test or not.
 */
bool rabinMillerTest(long value, unsigned int repeatCount);

/**
 * @brief Performs a modular exponentiation: a^exponent mod m.
 * @param a Base for modular exponentiation.
 * @param exponent Exponent for modular exponentiation.
 * @param m Modulus for modular exponentiation.
 * @return  a^exponent mod m.
 */
long modPow(long a, long exponent, long m);

/**
 * @brief Returns 2^exponent.
 * @param exponent Exponent for f(x) = 2^x.
 * @return 2^exponent.
 */
long powerOfTwo(long exponent);

int main(int argc, char * * argv)
{
	if (argc != 3)
	{
		cerr << "Usage: " << argv[0] << " inputPath repeatCount" << endl;
		return 1;
	}

	const string inputPath = argv[1];
	const unsigned int repeatCount = atoi(argv[2]);

	searchForPrimes(inputPath, repeatCount);
}

void searchForPrimes(const string &inputPath, const unsigned int repeatCount)
{
	LongMultimap results;
	const LongVector *numbers = readNumbersFromFile(inputPath);

	searchForPrimesUsingMPI(numbers, results, repeatCount);
	
	delete numbers;
}

const LongVector *readNumbersFromFile(const string &inputPath)
{
	std::ifstream file(inputPath.c_str());
	return new LongVector(std::istream_iterator<long>(file), (std::istream_iterator<long>()));
}

void searchForPrimesUsingMPI(const LongVector *numbers, LongMultimap &results, const unsigned int repeatCount)
{
	MPI_Request sendRequest, receiveRequest;
	MPI_Status status;
	MPI::Init();
	
	long rank = MPI::COMM_WORLD.Get_rank();
	long size = MPI::COMM_WORLD.Get_size();
	
	if (size == 1)
	{
		cerr << "Parameter -n should be greater then 1" << endl;
		exit (EXIT_FAILURE);
	}

	double start = MPI_Wtime();
	for (LongVector::const_iterator it = numbers->begin(); it != numbers->end(); ++it)
	{
		long input = *it;
		int pos = it - numbers->begin();
		int destRank = (pos % (size - 1)) + 1;
		
		if (rank == 0) sendNumberToTest(input, destRank, sendRequest);
		else if (rank == destRank) performTest(input, receiveRequest, repeatCount);
		
		if (rank == 0) saveResponses(results, destRank, status);
	}
	double end = MPI_Wtime();

	MPI::Finalize();

	if (rank == 0) 
	{
		printResults(results);
		printTime(start, end);
	}
}

void sendNumberToTest(long input, const int destRank, MPI_Request &sendRequest)
{
	MPI_Isend(&input, 1, MPI_LONG, destRank, 0, MPI_COMM_WORLD, &sendRequest);
}

void performTest(long input, MPI_Request &receiveRequest, const unsigned int repeatCount)
{
	MPI_Irecv(&input, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD, &receiveRequest);
	
	// MPI_BOOLs are cumbersome; 0 - not prime, 1 - prime
	long result[2];
	result[0] = input;
	result[1] = (rabinMillerTest(input, repeatCount) ? 1 : 0);
	
	// Count = 2 is important here
	MPI_Send(result, 2, MPI_LONG, 0, 1, MPI_COMM_WORLD);
}

void saveResponses(LongMultimap &results, const int destRank, MPI_Status &status)
{
	long result[2];
	MPI_Recv(result, 2, MPI_LONG, destRank, 1, MPI_COMM_WORLD, &status);

	results.insert(pair<long, long>(result[0], result[1]));
}

void printResults(const LongMultimap &results)
{
	for (LongMultimap::const_iterator it = results.begin(); it != results.end(); ++it)
	{
		cout << (*it).first << ": " << ((*it).second == 1 ? "pierwsza" : "zlozona") << "\n";
	}
}

void printTime(double start, double end)
{
	double time = (end - start) * 1000.0;
	cout << "Czas: " << time << "ms\n";
}

bool rabinMillerTest(long value, unsigned int repeatCount)
{
	srand(time(NULL));

	if (value < 4)
	{
		return true;
	}

	if (value % 2 == 0)
	{
		return false;
	}

	long s = 0;
	long sEnd = 1;
	while ((sEnd & (value - 1)) == 0)
	{
		++s;
		sEnd <<= 1;
	}
	long d = value / sEnd;

	for (unsigned int i = 0; i < repeatCount; ++i)
	{
		long a = 1 + ( rand() % (value - 1));
		
		if (modPow(a, d, value) != 1)
		{
			bool isPrime = false;
			
			for (long r = 0; r <= s - 1; ++r)
			{
				if (modPow(a, powerOfTwo(r) * d, value) == value - 1)
				{
					isPrime = true;
					break;
				}
			}
			
			if (!isPrime)
			{
				return false;
			}
		}
	}
	return true;
}

long modPow(long a, long exponent, long m)
{
	long result = 1;
	long long x = a % m;

	for (long i = 1; i <= exponent; i <<= 1)
	{
		x %= m;
		if ((exponent & i) != 0)
		{
			result *= x;
			result %= m;
		}
		x *= x;
	}

	return result;
}

long powerOfTwo(long exponent)
{
	return 1 << exponent;
}
