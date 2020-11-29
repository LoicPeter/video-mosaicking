
#include <string>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include "utilities.hpp"

using namespace std;

void drawUniqueNumbers(const int& N, const int& maxNumber, std::vector<int>& drawnNumbers, std::vector<int>& remainingNumbers)
{
    if (N > (maxNumber+1))
        cout << "drawUniqueNumbers tried to draw more numbers than possible. Check that the number of features per node is node higher than the total number of available features" << endl;
    int sizeList(maxNumber+1);
    vector<int> list(sizeList,0);
    for (int k=1; k<sizeList; ++k)
        list[k] = k;
    random_shuffle(list.begin(),list.end());
	drawnNumbers = std::vector<int>(N);
	remainingNumbers = std::vector<int>(sizeList-N);
	for (int k=0; k<N; ++k)
		drawnNumbers[k] = list[k];
	for (int k=0; k<(sizeList-N); ++k)
		remainingNumbers[k] = list[k+N];
}

void drawUniqueNumbers(const int& N, const int& maxNumber, std::vector<int>& drawnNumbers) // maxNumber inclusive
{
    if (N > (maxNumber+1))
        cout << "drawUniqueNumbers tried to draw more numbers than possible. Check that the number of features per node is node higher than the total number of available features" << endl;
    int sizeList(maxNumber+1);
    vector<int> list(sizeList,0);
    for (int k=1; k<sizeList; ++k)
        list[k] = k;
    random_shuffle(list.begin(),list.end());
	drawnNumbers = std::vector<int>(N);
	for (int k=0; k<N; ++k)
		drawnNumbers[k] = list[k];
}

float computeEntropy(const vector<float>& hist)
{
    float res(0),sum(0);
    int sizeHist((int)hist.size());
    for (int k=0; k<sizeHist; ++k)
    {
        sum += hist[k];
    }


    for (int k=0; k<sizeHist; ++k)
    {
        if (hist[k]>0)
            res -= (hist[k]/sum)*(log(hist[k]/sum));
    }
    return res;
}

std::string addZeros(int n, int nbDigits)
{
	int nbDigitsOfn(1);
	int threshold(10);
	while (n>=threshold)
	{
		++nbDigitsOfn;
		threshold = threshold*10;
	}
	ostringstream res;
	for (int k=0; k<(nbDigits - nbDigitsOfn); ++k)
		res << 0;
	res << n;
	return res.str();
}

void generateRandomPartition(std::vector<int>& list1, std::vector<int>& list2, int N)
{
	vector<int> list(N);
	for (int k=0; k<N; ++k)
		list[k] = k+1;
	random_shuffle(list.begin(),list.end());
	list1 = std::vector<int>(N/2);
	list2 = std::vector<int>(N/2);
	for (int k=0; k<(N/2); ++k)
	{
		list1[k] = list[k];
		list2[k] = list[k+N/2];
	}
}


void generateKfoldPartition(std::vector<std::vector<int>>& lists, int N, int K) // generate a partition of the set {0,...,N-1} made of K equal parts
{
	if ((N % K)!=0)
		cout << "The space cannot be partitioned in " << K << " equal parts" << endl;
	std::vector<int> listFoldMinIndexes(K), listFoldMaxIndexes(K), nbPointsPerFold(K);
	for (int k=0; k<K; ++k)
	{
		listFoldMinIndexes[k] = (k*N)/K;
		listFoldMaxIndexes[k] = ((k+1)*N)/K;
		nbPointsPerFold[k] = listFoldMaxIndexes[k] - listFoldMinIndexes[k];
	}
	lists.resize(K);
	vector<int> originalSet(N);
	for (int k=0; k<N; ++k)
		originalSet[k] = k;
	random_shuffle(originalSet.begin(),originalSet.end());
	for (int k=0; k<K; ++k)
	{
		lists[k] = std::vector<int>(nbPointsPerFold[k]);
		for (int n=listFoldMinIndexes[k]; n<listFoldMaxIndexes[k]; ++n)
			lists[k][n-listFoldMinIndexes[k]] = originalSet[n];
	}
}

std::string removeExtension(const std::string& inputString)
{
	std::string res;
	int length = (int)inputString.size();
	res = inputString.substr(0,length-4);
	return res;
}

std::string replaceSlashes(const std::string& inputString)
{
	std::string res = inputString;
	int length = (int)inputString.size();
	for (int k=0; k<length; ++k)
	{
		if ((res[k]=='/') || (res[k]=='\\'))
			res[k] = '-';
	}
	return res;
}

std::string getPathFromFilename(const std::string& inputString)
{
	std::string res = inputString;
	int length = (int)inputString.size();
	for (int k=(length-1); k>0; --k)
	{
		if ((inputString[k]!='/') && (inputString[k]!='\\'))
			res.pop_back();
		else
			break;
	}
	return res;
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}
