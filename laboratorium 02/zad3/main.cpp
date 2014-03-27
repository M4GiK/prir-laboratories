//============================================================================
// Name        : analiza_omp
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================

#define _CRT_SECURE_NO_WARNINGS
#include <cerrno>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <string.h>
#include <stdexcept>
#include <stdio.h>
#include <stdexcept>

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
 *
 * @param contents
 * @return
 */
std::string collectTrigram(std::string contents)
{
	return NULL;
}

/**
 * This method analyzes  the given document. Splits contents to trigrams which are analyze.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 * @param contents The contents to analyze The name of file to analyze.
 * @return Data with information about processed analyze.
 */
std::string analyzeDocument(unsigned int threadCount, std::string contents)
{
	TimePoint start = std::chrono::system_clock::now();
	std::string trigrams = collectTrigram(contents);
	TimePoint end = std::chrono::system_clock::now();

	Duration elapsedMillis = end - start;
	cout << elapsedMillis.count() << endl;

	return trigrams;
}

/**
 * Gets contents data from given files.
 *
 * @param numberOfFiles The amount of files to read.
 * @param nameFiles The array of name files.
 * @return The contents of read files in single string.
 */
std::string getFilesContents(int numberOfFiles, char* nameFiles[])
{
	std::string contentsBuffor;

	for (int i = 0; i < numberOfFiles; ++i)
	{
		contentsBuffor += getFileContents(nameFiles[i]);
	}

	return contentsBuffor;
}

/**
 * Saves given data to file.
 * This method also appends extension "dat" for name which is langCode.
 *
 * @param dataToSave The data to save.
 * @param langCode The name of file which will be save.
 */
void saveToFile(std::string dataToSave, char *langCode)
{
	std::ofstream outfile(strcat(langCode, ".dat"));
	try
	{
		if (!outfile.is_open())
		{
			std::cerr << "Couldn't open '" << langCode << endl;
		}
		else
		{
			outfile << dataToSave << endl;
			outfile.close();
		}
	} catch (const std::exception& e)
	{
		std::cerr << "\n*** ERROR: " << e.what() << endl;
	}
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

	std::string dataToSave = analyzeDocument(threadCount,
			getFilesContents(argc - 3, argv));
	saveToFile(dataToSave, langCode);

	return 0;
}
