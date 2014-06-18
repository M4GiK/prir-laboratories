//============================================================================
// Name        : analiza_omp
// Author      : Michał Szczygieł & Aleksander Śmierciak
//============================================================================

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <omp.h>
#include <string>
#include <utility>

using std::cout;
using std::endl;
using std::string;

typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;
typedef std::chrono::duration<double> Duration;
typedef std::map<string, int> Histogram;

/**
 * Gets contents of given file.
 *
 * @param filename The name of file to read.
 * @return Contents file as a string.
 */
string readTextFromFile(const string fileName)
{
    std::ifstream in(fileName, std::ios::in);

	if (in)
	{
        string contents((std::istreambuf_iterator<char>(in)),
				std::istreambuf_iterator<char>());
		in.close();

		return (contents);
	}

	throw(errno);
}

/**
 * Saves given data to file.
 * This method also appends extension "dat" for name which is langCode.
 *
 * @param dataToSave The data to save.
 * @param langCode The name of file which will be save.
 */
void saveToFile(string dataToSave, string langCode)
{
    std::ofstream outfile(langCode + ".dat");
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
 * Calculates data portion to fit modulo 3 for threads.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 * @param contents The contents to analyze.
 * @return The size of portion data for one thread.
 */
int calculateChunkSize(string contents, unsigned int threadCount)
{
    return ceil(((double)contents.size()) / threadCount);
}

/**
 * Analyzes trigrams for given string using OpenMP.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 * @param contents The contents to analyze.
 * @param portion The portion of data for one thread.
 * @return
 */
Histogram analyzeInput(string contents, unsigned int threadCount)
{
    Histogram trigramDistribution;

    unsigned int chunkSize = calculateChunkSize(contents, threadCount);
    // Uncomment if you want use thread safe operations.
	//omp_lock_t lock;
	//omp_init_lock(&lock);

    #pragma omp parallel
    {
        unsigned int threadNumber = omp_get_thread_num();

        unsigned int startPos = chunkSize * threadNumber;
        unsigned int endPos = chunkSize * (threadNumber + 1);
        endPos = std::min(endPos, (unsigned int)contents.size());

        // Alternative for thread safe operations by use a critical section.
        //#pragma omp critical
        for (unsigned int i = startPos; i < endPos; i += 3)
        {
        	// Uncomment if you want use thread safe operations.
        	//omp_set_lock(&lock);
            string trigram = contents.substr(i, 3);
            ++trigramDistribution[trigram];
            // Uncomment if you want use thread safe operations.
            //omp_unset_lock(&lock);
        }
    }
    return trigramDistribution;
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
 * Collects trigrams from given data.
 *
 * @param threadCount Number of threads to spawn in the parallel OpenMP block.
 * @param contents The contents to analyze.
 * @return The map with trigrams.
 */
Histogram collectTrigrams(string contents, unsigned int threadCount)
{
    prepareData(contents, threadCount);
    Histogram trigrams = analyzeInput(contents, threadCount);

    return trigrams;
}

/**
 * Converts map to string. This method prepares data to save into string.
 *
 * @param trigrams The map with data to convert.
 * @return Converted map into string.
 */
string mapToString(Histogram trigrams)
{
    string convertedMap;

    for (Histogram::iterator iterator = trigrams.begin();
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
string analyzeDocument(string contents, unsigned int threadCount)
{
	TimePoint start = std::chrono::system_clock::now();
    Histogram trigrams = collectTrigrams(contents, threadCount);
	TimePoint end = std::chrono::system_clock::now();

	Duration elapsedMillis = (end - start) * 1000.0;
	cout << elapsedMillis.count() << endl;

	return mapToString(trigrams);
}

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
 * Gets contents data from given files.
 *
 * @param numberOfFiles The amount of files to read.
 * @param nameFiles The array of name files.
 * @return The contents of read files in single string.
 */
string readTextFromFiles(std::deque<string> fileNames)
{
    string filesContent;

    for (unsigned int i = 0; i < fileNames.size(); ++i)
	{
        filesContent += readTextFromFile(fileNames.at(i));
	}

    return filesContent;
}

std::deque<string> retrieveFileNames(char *fileNameChains[], unsigned int size)
{
    std::deque<string> fileNames;
    // ignore first three parameters;
    // they are executable name, thread count and language code
    for (unsigned int i = 3; i < size; ++i)
    {
        fileNames.push_back(fileNameChains[i]);
    }
    return fileNames;
}

/**
 * The main method of application which analyze of document based on trigram.
 *
 * @param argc  Number of arguments given to the program.
 * 				This value should be equal to 3.
 * @param argv
 * 				The first parameter:  <threadCount>
 * 				The second parameter: <langCode>
 * 				The third parameter:  <file> [files...]
 * @return  C-standard return code: 0 if success,
 * 			other value if errors occurred during the execution.
 */
int main(int argc, char *argv[])
{
	if (argc < 4)
	{
        std::cerr << "Usage: ./analiza_omp <threadCount> <langCode> <file> [files...]" << endl;
		return -1;
	}

	unsigned int threadCount = std::stoi(argv[1]);
    string langCode = argv[2];

    std::deque<string> fileNames = retrieveFileNames(argv, argc);
    string filesContent = readTextFromFiles(fileNames);

    string textFromFiles = removeLineEndings(filesContent);

    string dataToSave = analyzeDocument(textFromFiles, threadCount);


	saveToFile(dataToSave, langCode);

	return 0;
}
