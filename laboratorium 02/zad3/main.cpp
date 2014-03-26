//============================================================================
// Name        : analiza_omp
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================

#include <cerrno>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

using std::cout;
using std::cin;
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
	std::ifstream in(filename, std::ios::in | std::ios::binary);

	if (in)
	{
		std::string contents;
		in.seekg(0, std::ios::end);
		contents.resize(in.tellg());
		in.seekg(0, std::ios::beg);
		in.read(&contents[0], contents.size());
		in.close();

		return (contents);
	}
	throw(errno);
}

/**
 * This method analyzes  the given document. Splits contents to trigrams which are analyze.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 * @param contents The contents to analyze The name of file to analyze.
 */
void analyzeDocument(unsigned int threadCount, std::string contents)
{

}

/**
 *
 * @param numberOfFiles
 * @param nameFiles
 * @return
 */
std::string getFilesContents(int numberOfFiles, char* nameFiles[]) {
	std::string contentsBuffor;

	for (int i = 0; i < numberOfFiles; ++i) {
		contentsBuffor += getFileContents(nameFiles[i]);
	}

	return contentsBuffor;
}

/**
 * The main method of application which analyze of document based on trigram.
 *
 * @param argc  Number of arguments given to the program.
 * 				This value should be equal to 3.
 * @param argv
 * 				The first parameter:  <threadCount>
 * 				The second parameter: <langCode>
 * 				The third parameter:  <file>
 * @return  C-standard return code: 0 if success,
 * 			other value if errors occurred during the execution.
 */
int main(int argc, char* argv[])
{
	if (argc < 4)
	{
		cout << "Usage: ./analiza_omp <threadCount> <langCode> <file>" << endl;
		return -1;
	}

	unsigned int threadCount = std::stoi(argv[1]);
	char *langCode = argv[2];

	TimePoint start = std::chrono::system_clock::now();
	analyzeDocument(threadCount, getFilesContents(argc - 3 , argv));
	TimePoint end = std::chrono::system_clock::now();

	Duration elapsedMillis = end - start;
	cout << elapsedMillis.count() << endl;

	return 0;
}
