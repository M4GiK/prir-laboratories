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
 * Prepares data to analyze process. Cuts the size of result to fit modulo 3.
 * Calculates also data portion to fit modulo 3 for threads.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 * @param contents The contents to analyze.
 * @return The size of portion data for one thread.
 */
int prepareData(unsigned int threadCount, std::string &contents)
{
	int limit = contents.size() - (contents.size() % 3);
	contents = contents.substr(0, limit);

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
	return NULL;
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
	int portion = prepareData(threadCount, contents);
	std::map<std::string, int> trigram = analyzeProcess(threadCount, contents,
			portion);

	return trigram;
}

/**
 * Converts map to string. This method prepares data to save into string.
 *
 * @param trigrams The map with data to convert.
 * @return Converted map into string.
 */
std::string mapToString(std::map<std::string, int> trigrams)
{
	std::string convertedMap;

	for (std::map<std::string, int>::iterator iterator = trigrams.begin();
			iterator != trigrams.end(); ++iterator)
	{
		convertedMap += (*iterator).first + " "
				+ std::to_string((*iterator).second);
		convertedMap += "\n";
	}

	return convertedMap;
}

/**
 * This method analyzes  the given document. Splits contents to trigrams which are analyze.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 * @param contents The contents to analyze.
 * @return Data with information about processed analyze.
 */
std::string analyzeDocument(unsigned int threadCount, std::string contents)
{
	TimePoint start = std::chrono::system_clock::now();
	std::map<std::string, int> trigrams = collectTrigram(threadCount, contents);
	TimePoint end = std::chrono::system_clock::now();

	Duration elapsedMillis = end - start;
	cout << elapsedMillis.count() << endl;

	return mapToString(trigrams);
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
