//============================================================================
// Name        : wykrywanie_omp
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================

#define _CRT_SECURE_NO_WARNINGS
#include <cerrno>
#include <chrono>
#include <fstream>
#include <iterator>
#include <string>

using std::cout;
using std::endl;

typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;
typedef std::chrono::duration<double> Duration;


/**
 * Gets contents of given file.
 *
 * @param filename The name of file to read.
 * @return Contents file as a string.
 */
std::string getFileContents(const char* filename)
{
	std::ifstream in(filename, std::ios::in);

	if (in)
	{
		std::string contents((std::istreambuf_iterator<char>(in)),
				std::istreambuf_iterator<char>());
		in.close();

		return (contents);
	}

	throw(errno);
}

/**
 * Prepares data to analyze process. Cuts the size of result to fit modulo 3.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 */
void prepareData(std::string &contents)
{
	int limit = contents.size() - (contents.size() % 3);
	contents = contents.substr(0, limit);
}

/**
 * Calculates data portion to fit modulo 3 for threads.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 * @param contents The contents to analyze.
 * @return The size of portion data for one thread.
 */
int getPortionforThread(unsigned int threadCount, std::string contents)
{
	int portion = ceil((contents.size()) / threadCount);
	portion += 3 - (portion % 3);

	return portion;
}

/**
 * Analyzes trigrams for given string using OpenMP.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 * @param contents The contents to analyze.
 * @param portion The portion of data for one thread.
 * @return
 */
std::map<std::string, int> analyzeProcess(unsigned int threadCount,
		std::string contents, int portion)
{
	std::map<std::string, int> trigram;

	#pragma omp parallel num_threads(threadCount)
	{
		std::string threeLetters;
		unsigned int endPosition = portion * (omp_get_thread_num() + 1);

		if (endPosition > contents.size())
		{
			endPosition = contents.size();
		}

		#pragma for default(none) shared(contents, trigram) firstprivate(portion) private(threeLetters)
		for (int i = portion * omp_get_thread_num();
				i != portion * (omp_get_thread_num() + 1); i += 3)
		{
			threeLetters = std::string(contents.substr(i, 3));
			trigram[threeLetters]++;
		}
	}

	return trigram;
}

/**
 * Collects trigrams from given data.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 * @param contents The contents to analyze.
 * @return The map with trigrams.
 */
std::map<std::string, int> collectTrigram(unsigned int threadCount,
		std::string contents)
{
	prepareData(contents);
	int portion = getPortionforThread(threadCount, contents);
	std::map<std::string, int> trigram = analyzeProcess(threadCount, contents,
			portion);

	return trigram;
}

/**
 * This method analyzes the given document. Compare with another file to estimate the best adhesion.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 * @param contents The contents to analyze.
 */
void analyzeDocument(unsigned int threadCount, std::string contents)
{

	std::map<std::string, int> trigrams = collectTrigram(threadCount, contents);

	TimePoint start = std::chrono::system_clock::now();
	// TODO make compare
	TimePoint end = std::chrono::system_clock::now();

	Duration elapsedMillis = end - start;
	cout << elapsedMillis.count() << endl;

}


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

	analyzeDocument(threadCount, getFileContents(filePath));

	return 0;
}
