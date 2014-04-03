//============================================================================
// Name        : wykrywanie_omp
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================

#include <dirent.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <deque>
#include <iostream>
#include <fstream>
#include <iterator>
#include <map>
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
string replaceLines(string contentsBuffor)
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
string getFileContents(const string &filename)
{
	std::ifstream in(filename, std::ios::in);

	if (in)
	{
		string contents((std::istreambuf_iterator<char>(in)),
				std::istreambuf_iterator<char>());
		in.close();

		return replaceLines(contents);
	}

	throw(errno);
}

/**
 * Prepares data to analyze process. Cuts the size of result to fit modulo 3.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 */
void prepareData(string &contents)
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
int getPortionforThread(unsigned int threadCount, string contents)
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
Histogram analyzeProcess(unsigned int threadCount, string contents, int portion)
{
	Histogram trigrams;
	string threeLetters;

	#pragma omp parallel num_threads(threadCount) shared(trigrams)
	{
		#pragma omp for private(threeLetters) schedule(static, portion)
		for (unsigned int i = 0; i < contents.size(); i += 3)
		{
			threeLetters = string(contents.substr(i, 3));
			trigrams[threeLetters]++;
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
	prepareData(contents);
	int portion = getPortionforThread(threadCount, contents);
	Histogram trigrams = analyzeProcess(threadCount, contents, portion);

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
 * Gets contexts of all files with extension 'dat'.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 * @param files The list of all files with extension 'dat'.
 * @return The contexts of all files.
 */
std::map<string, std::map<string, int>> getContextsFiles(
		unsigned int threadCount, std::deque<string> *files)
{
	std::map<string, std::map<string, int>> filesContexs;

	for (std::deque<string>::iterator it = files->begin(); it != files->end();
			++it)
	{
		filesContexs[*it] = collectTrigrams(threadCount, getFileContents(*it));
	}

	return filesContexs;
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
	std::map<string, std::map<string, int>> filesContexs = getContextsFiles(
			threadCount, files);

	TimePoint start = std::chrono::system_clock::now();
	#pragma omp parallel for schedule(static) private(files)
	for (int i = 0; i < files->size(); ++i)
	{

	}
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
	string filePath = argv[2];

	analyzeDocument(threadCount, getFileContents(filePath));

	return 0;
}
