//============================================================================
// Name        : wykrywanie_omp
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================

#include <dirent.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <omp.h>
#include <string>

using std::cout;
using std::endl;
using std::string;

typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;
typedef std::chrono::duration<double> Duration;
typedef std::map<string, int> Histogram;

/**
 * Replaces line character codes to spaces.
 *
 * @param contentsBuffor The contexts for replaces line character code to spaces.
 * @return The contexts with replaced characters.
 */
string removeLineEndings(string contentsBuffor)
{
	replace(contentsBuffor.begin(), contentsBuffor.end(), '\n', ' ');
	return contentsBuffor;
}

/**
 * Gets contents of given file.
 *
 * @param filename The name of file to read.
 * @return Contents file as a string.
 */
string readTextFromFile(const string &filename)
{
	std::ifstream in(filename, std::ios::in);

	if (in)
	{
		string contents((std::istreambuf_iterator<char>(in)),
				std::istreambuf_iterator<char>());
		in.close();

		return removeLineEndings(contents);
	}

	throw(errno);
}

/**
 * Prepares data to analyze process. Cuts the size of result to fit modulo threadCount.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 */
void prepareData(string &contents, unsigned int threadCount)
{
    int limit = contents.size() - (contents.size() % threadCount);
	contents = contents.substr(0, limit);
}

/**
 * Calculates data portion to fit modulo 3 for threads.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 * @param contents The contents to analyze.
 * @return The size of portion data for one thread.
 */
int calculateChunkSize(unsigned int threadCount, string contents)
{
	int portion = ceil((contents.size()) / threadCount);
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
Histogram analyzeInput(unsigned int threadCount, string contents, int portion)
{
	Histogram trigrams;
	string threeLetters;
	omp_lock_t lock;
	omp_init_lock(&lock);

	#pragma omp parallel num_threads(threadCount) shared(trigrams)
	{
		#pragma omp for private(threeLetters) schedule(static, portion)
		for (unsigned int i = 0; i < contents.size(); i += 3)
		{
			omp_set_lock(&lock);
			threeLetters = string(contents.substr(i, 3));
			trigrams[threeLetters]++;
			omp_unset_lock(&lock);
		}
	}

	return trigrams;
}

/**
 * Collects trigrams from given data.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 * @param contents The contents to analyze.
 * @return The map with trigrams.
 */
Histogram collectTrigrams(unsigned int threadCount, string contents)
{
    prepareData(contents, threadCount);
	int portion = calculateChunkSize(threadCount, contents);
	Histogram trigrams = analyzeInput(threadCount, contents, portion);

	return trigrams;
}

/**
 * Gets list of all files with extension 'dat'.
 * This method list files in directory.
 *
 * @return The container with all files with extension 'dat'.
 */
std::deque<string> *getFiles()
{
	std::deque<string> *files = new std::deque<string>();
	DIR *dir;
	struct dirent *ent;

	if ((dir = opendir(".")) != NULL)
	{
		while ((ent = readdir(dir)) != NULL)
		{
			string fileName = ent->d_name;
			if (fileName.find(".dat") != string::npos)
			{
				files->push_back(fileName);
			}
		}
		closedir(dir);

		return files;
	}

	throw(errno);
}

/**
 * Gets trigrams and frequency from given file.
 * Gets each one line and pack it into map.
 *
 * @param filename The name of file with trigrams.
 * @return The histogram with information from file.
 */
Histogram getTrigramsFromFile(string filename)
{
	Histogram trigrams;
	std::ifstream in(filename, std::ios::in);

	if (in.is_open())
	{
		string contents;
		while (!in.eof())
		{
			getline(in, contents);
			if (contents.size() > 3)
			{
				string trigram = contents.substr(0, 3);
				int frequency = atoi(
						(contents.substr(3, contents.length() - 1).c_str()));
				trigrams[trigram] = frequency;
			}
		}
		in.close();

		return trigrams;
	}

	throw(errno);
}

/**
 * Gets contexts of all files with extension 'dat'.
 *
 * @param files The list of all files with extension 'dat'.
 * @return The contexts of all files.
 */
std::map<string, Histogram> getContextsFiles(std::deque<string> *files)
{
	std::map<string, Histogram> filesContexs;

	for (std::deque<string>::iterator it = files->begin(); it != files->end();
			++it)
	{
		filesContexs[*it] = getTrigramsFromFile(*it);
	}

	return filesContexs;
}

/**
 * Compares trigrams, and collects coverage.
 *
 * @param trigrams The trigrams analyzed as execution parameter.
 * @param trigramsToCompare Dataset of trigrams to compare.
 * @return The coverage in percent.
 */
double compareTrigrams(Histogram trigrams, Histogram trigramsToCompare)
{
	int maxCoverage = trigrams.size();
	int currentCoverage = 0;

	for (Histogram::iterator iti = trigrams.begin(); iti != trigrams.end();
			++iti)
	{
		for (Histogram::iterator itj = trigramsToCompare.begin();
				itj != trigramsToCompare.end(); ++itj)
		{
			if((*iti).first == (*itj).first)
			{
				currentCoverage++;
			}
		}
	}

	return (100 * currentCoverage) / (double) maxCoverage;
}

/**
 * Prints information about analyze and compare process.
 *
 * @param elapsedMillis Time to display.
 * @param result The result to display.
 */
void printInformations(Duration elapsedMillis, std::map<string, double> result)
{
	for (std::map<string, double>::iterator it = result.begin();
			it != result.end(); ++it)
	{
		cout << (*it).first << " " << (*it).second << "%" << endl;
	}

	cout << elapsedMillis.count() << endl;
}

/**
 * This method analyzes the given document. Compare with another file to estimate the best adhesion.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 * @param contents The contents to analyze.
 */
void analyzeDocument(unsigned int threadCount, string contents)
{
	Histogram trigrams = collectTrigrams(threadCount, contents);
	std::deque<string> *files = getFiles();
	std::map<string, Histogram> trigramsToCompare = getContextsFiles(files);
	std::map<string, double> result;

    TimePoint start = std::chrono::system_clock::now();
    #pragma omp parallel for num_threads(threadCount) shared(trigramsToCompare, result) firstprivate(files) schedule(dynamic)
	for (unsigned int i = 0; i < files->size(); ++i)
	{
		result[files->at(i)] = compareTrigrams(trigrams, trigramsToCompare[files->at(i)]);
	}
	TimePoint end = std::chrono::system_clock::now();

	Duration elapsedMillis = (end - start) * 1000.0;

	printInformations(elapsedMillis, result);
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
	string filePath = argv[2];

	analyzeDocument(threadCount, readTextFromFile(filePath));

	return 0;
}
