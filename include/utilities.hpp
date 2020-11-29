
#ifndef UTILITIES_H_INCLUDED
#define UTILITIES_H_INCLUDED

#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <iostream>
#include <istream>
#include <sstream>


float computeEntropy(const std::vector<float>& hist);

void drawUniqueNumbers(const int& N, const int& maxNumber, std::vector<int>& drawnNumbers);

void drawUniqueNumbers(const int& N, const int& maxNumber, std::vector<int>& drawnNumbers, std::vector<int>& remainingNumbers);

template <typename ArrayType>
double computeMean(const std::vector<ArrayType>& inputArray)
{
	int N = (int)inputArray.size();
	double m(0);
	for (int k=0; k<N; ++k)
		m += inputArray[k];
	m = m/((double)N);
	return m;
}

template <typename ArrayType>
double computeSum(const std::vector<ArrayType>& inputArray)
{
	int N = (int)inputArray.size();
	double m(0);
	for (int k=0; k<N; ++k)
		m += inputArray[k];
	return m;
}

template <typename ArrayType>
double computeMean(const std::vector<ArrayType>& inputArray, const std::vector<double>& weights)
{
	int N = (int)inputArray.size();
	double m(0);
	double sumWeights(0);
	for (int k=0; k<N; ++k)
	{
		m += weights[k]*inputArray[k];
		sumWeights += weights[k];
	}
	m = m/sumWeights;
	return m;
}

template <typename ArrayType>
double computeStd(const std::vector<ArrayType>& inputArray)
{
	int N = (int)inputArray.size();
	double m = computeMean<ArrayType>(inputArray);
	std::vector<ArrayType> squaredArray(N);
	for (int k=0; k<N; ++k)
		squaredArray[k] = inputArray[k]*inputArray[k];
	double m2 = computeMean<ArrayType>(squaredArray);
	double res = sqrt(m2 - m*m);
	return res;
}

template <typename ArrayType>
ArrayType computeMax(const std::vector<ArrayType>& inputArray, int& indMax)
{
	int N = (int)inputArray.size();
	ArrayType m(inputArray[0]);
	indMax = 0;
	for (int k=1; k<N; ++k)
	{
		if (m<inputArray[k])
		{
			m = inputArray[k];
			indMax = k;
		}
	}
	return m;
}

template <typename ArrayType>
ArrayType computeMin(const std::vector<ArrayType>& inputArray, int& indMin)
{
	int N = (int)inputArray.size();
	ArrayType m(inputArray[0]);
	indMin = 0;
	for (int k=1; k<N; ++k)
	{
		if (m>inputArray[k])
		{
			m = inputArray[k];
			indMin = k;
		}
	}
	return m;
}

template <typename ArrayType>
void computeMinAndMax(const std::vector<ArrayType>& inputArray, ArrayType& minValue, ArrayType& maxValue, int& indMin, int& indMax)
{
	int N = (int)inputArray.size();
	indMin = 0;
	indMax = 0;
	minValue = inputArray[0];
	maxValue = inputArray[0];
	for (int k=1; k<N; ++k)
	{
		if (minValue>inputArray[k])
		{
			minValue = inputArray[k];
			indMin = k;
		}
		else
		{
			if (maxValue<inputArray[k])
			{
				maxValue = inputArray[k];
				indMax = k;
			}
		}
	}
}

template <typename ArrayType>
std::vector<ArrayType> createTruncatedVector(const std::vector<ArrayType> vec, int startIndex, int nbSamples)
{
	std::vector<ArrayType> vec_trunc(nbSamples);
	for (int ind=0; ind<nbSamples; ++ind)
		vec_trunc[ind] = vec[startIndex + ind];
	return vec_trunc;
}

template <typename ArrayType>
void computeMinAndMax(const std::vector<ArrayType>& inputArray, ArrayType& minValue, ArrayType& maxValue)
{
	int dummyIndMin, dummyIndMax;
	computeMinAndMax<ArrayType>(inputArray,minValue,maxValue,dummyIndMin,dummyIndMax);
}

template <typename ArrayType>
void perform_fast_quick_sort(std::vector<ArrayType>& input_array, int start_idx, int end_idx)
{

 // we first perform the partitioning
 int bound,running_idx;
 ArrayType pivot_value = input_array[start_idx];
 bound = start_idx;
 running_idx = start_idx+1;

 while(running_idx<=end_idx)
 {
  ArrayType current_value = input_array[running_idx];
  if(current_value<pivot_value)
  {
   // here we need to swap
   bound++;
   ArrayType swap_value = input_array[bound];
   input_array[running_idx] = swap_value;
   input_array[bound] = current_value;
  }
  running_idx++;
 }

 // last swap with pivot
 ArrayType swap_value = input_array[bound];
 input_array[bound] = pivot_value;
 input_array[start_idx] = swap_value;

 // now we perform the recursive calls
 if(bound>start_idx)
 {
  perform_fast_quick_sort(input_array,start_idx,bound);
 }
 if(end_idx>bound)
 {
  perform_fast_quick_sort(input_array,bound+1,end_idx);
 }

}

template <typename ArrayType, typename LabelType>
void perform_fast_quick_sort(std::vector<ArrayType>& input_array, std::vector<LabelType>& label_array, int start_idx, int end_idx)
{

 // we first perform the partitioning
 int bound,running_idx;
 ArrayType pivot_value = input_array[start_idx];
 bound = start_idx;
 running_idx = start_idx+1;

 while(running_idx<=end_idx)
 {
  ArrayType current_value = input_array[running_idx];
  if(current_value<pivot_value)
  {
   // here we need to swap
   bound++;
   ArrayType swap_value = input_array[bound];
   input_array[running_idx] = swap_value;
   input_array[bound] = current_value;
   // we swap the labels similarly
   LabelType swap_label = label_array[bound];
   label_array[bound] = label_array[running_idx];
   label_array[running_idx] = swap_label;
  }
  running_idx++;
 }

 // last swap with pivot
 ArrayType swap_value = input_array[bound];
 input_array[bound] = pivot_value;
 input_array[start_idx] = swap_value;
 LabelType swap_label = label_array[bound];
 label_array[bound] = label_array[start_idx];
 label_array[start_idx] = swap_label;

 // now we perform the recursive calls
 if(bound>start_idx)
 {
  perform_fast_quick_sort(input_array,label_array,start_idx,bound);
 }
 if(end_idx>bound)
 {
  perform_fast_quick_sort(input_array,label_array,bound+1,end_idx);
 }

}

std::string addZeros(int n, int nbDigits);

template <typename ArrayType>
void saveVector(const std::vector<ArrayType>& vect, std::string fileName)
{
	std::ofstream infile;
	infile.open(fileName.c_str());
	int N = (int)vect.size();
	infile << N << " ";
	for (int k=0; k<N; ++k)
		infile << vect[k] << " ";
	infile.close();
}

template <typename ArrayType>
void saveVectorOfVectors(const std::vector<std::vector<ArrayType>>& vect, std::string fileName)
{
	std::ofstream infile;
	infile.open(fileName.c_str());
	int N = (int)vect.size();
	infile << N << std::endl;
	for (int k=0; k<N; ++k)
	{
		int N2 = (int)vect[k].size();
		infile << N2 << " ";
		for (int l=0; l<N2; ++l)
			infile << vect[k][l] << " ";
		infile << std::endl;
	}
	infile.close();
}

template <typename ArrayType>
void loadVectorOfVectors(std::vector<std::vector<ArrayType>>& vect, std::string fileName)
{
	std::ifstream infile;
	infile.open(fileName.c_str());
	int N(0);
	infile >> N;
	vect = std::vector<std::vector<ArrayType>>(N);
	for (int k=0; k<N; ++k)
	{
		int N2;
		infile >> N2;
		vect[k] = std::vector<ArrayType>(N2);
		for (int l=0; l<N2; ++l)
			infile >> vect[k][l];
	}
	infile.close();
}

template <typename InputType, typename OutputType>
void saveUnorderedMap(const std::unordered_map<InputType,OutputType>& myMap, std::string fileName)
{
	std::ofstream infile(fileName);
	for (auto it = myMap.begin(); it!=myMap.end(); ++it)
	{
		InputType entry = it->first;
		infile << entry << " " << myMap[entry] << std::endl;
	}
	infile.close();
}

template <>
inline void saveUnorderedMap<int,std::vector<int>>(const std::unordered_map<int,std::vector<int>>& myMap, std::string fileName)
{
	std::ofstream infile(fileName);
	for (auto it = myMap.begin(); it!=myMap.end(); ++it)
	{
		int entry = it->first;
		infile << entry << " ";
		std::vector<int> currentVec = it->second;
		int N = (int)(currentVec.size());
		infile << N << " ";
		for (int k=0; k<N; ++k)
			infile << currentVec[k] << " ";
		infile << std::endl;
	}
	infile.close();
}

template <typename InputType, typename OutputType>
void loadUnorderedMap(std::unordered_map<InputType,OutputType>& myMap, std::string fileName)
{
	std::ifstream infile(fileName);
	InputType entry;
	OutputType value;
	while (infile >> entry >> value)
	{
		myMap[entry] = value;
	}
	infile.close();
}

template <>
inline void loadUnorderedMap<int,std::vector<int>>(std::unordered_map<int,std::vector<int>>& myMap, std::string fileName)
{
	std::ifstream infile(fileName);
	int entry;
	while (infile >> entry)
	{
		int sizeVec;
		infile >> sizeVec;
		std::vector<int> currentVec(sizeVec);
		for (int k=0; k<sizeVec; ++k)
			infile >> currentVec[k];
		myMap[entry] = currentVec;
	}
	infile.close();
}


template <>
inline void saveVector<unsigned char>(const std::vector<unsigned char>& vect, std::string fileName)
{
	std::ofstream infile;
	infile.open(fileName.c_str());
	int N = (int)vect.size();
	infile << N << " ";
	for (int k=0; k<N; ++k)
		infile << (int)vect[k] << " ";
	infile.close();
}

template <typename ArrayType>
void loadVector(std::vector<ArrayType>& vect, std::string fileName)
{
	//cout << "Warning: We shorten the vector loading" << endl;
	std::ifstream infile;
	infile.open(fileName.c_str());
	int N(0);
	infile >> N;
	//N = 30000;
	vect = std::vector<ArrayType>(N);
	for (int k=0; k<N; ++k)
		infile >> vect[k];
	infile.close();
}

template <>
inline void loadVector<bool>(std::vector<bool>& vect, std::string fileName)
{
	//cout << "Warning: We shorten the vector loading" << endl;
	std::ifstream infile;
	infile.open(fileName.c_str());
	int N(0);
	infile >> N;
	//N = 30000;
	vect = std::vector<bool>(N);
	for (int k=0; k<N; ++k)
	{
		int temp;
		infile >> temp;
		bool b = (temp==1);
		vect[k] = b;
	}
	infile.close();
}

template <>
inline void loadVector<unsigned char>(std::vector<unsigned char>& vect, std::string fileName)
{
	//cout << "Warning: We shorten the vector loading" << endl;
	std::ifstream infile;
	infile.open(fileName.c_str());
	int N(0);
	infile >> N;
	//N = 30000;
	vect = std::vector<unsigned char>(N);
	for (int k=0; k<N; ++k)
	{
		int temp;
		infile >> temp;
		unsigned char b = (unsigned char)temp;
		if (temp > 255)
			std::cout << "Warning: Try to load an unsigned char vector, but values are above 255" << std::endl;
		vect[k] = b;
	}
	infile.close();
}

void generateRandomPartition(std::vector<int>& list1, std::vector<int>& list2, int N);

void generateKfoldPartition(std::vector<std::vector<int>>& lists, int N, int K);


std::string removeExtension(const std::string& inputString);

std::string replaceSlashes(const std::string& inputString);

std::string getPathFromFilename(const std::string& inputString);


// Iterative
template <typename ArrayType>
int binaryIntervalSearch(const std::vector<ArrayType>& inputArray, const ArrayType& x) // inputArray = [a0,a1,...,a(N-1)] with ai < a(i+1). Returns the greatest i such that ai <= x. Hypothesis: a0 <= x < a(N-1). Initially indMin = 0, indMax = N-1
{
	int ind(0);
	while (x>=inputArray[ind+1])
		++ind;
	return ind;
}


// Recursive
//template <typename ArrayType>
//int binaryIntervalSearch(const std::vector<ArrayType>& inputArray, const ArrayType& x) // inputArray = [a0,a1,...,a(N-1)] with ai < a(i+1). Returns the greatest i such that ai <= x. Hypothesis: a0 <= x < a(N-1). Initially indMin = 0, indMax = N-1
//{
//	int N = (int)inputArray.size();
//	if (N>=2)
//		return binaryIntervalSearch<ArrayType>(inputArray,x,0,N-1,0);
//	else
//	{
//		std::cout << "Error! binaryIntervalSearch not defined for a vector of size <= 1" << std::endl;
//		return -1;
//	}
//}

template <typename ArrayType>
int binaryIntervalSearch(const std::vector<ArrayType>& inputArray, const ArrayType& x, int indMin, int indMax, int currentRes) // inputArray = [a0,a1,...,a(N-1)] with ai < a(i+1). Returns the greatest i such that ai <= x. Hypothesis: a0 <= x < a(N-1). Initially indMin = 0, indMax = N-1
{
	int N = indMax - indMin + 1;
	//if (N<=1)
	//	std::cout << "Error! binaryIntervalSearch not defined for a vector of size <= 1" << std::endl;
	if (N==2)
		return currentRes;
	else // N >= 3
	{
		if (inputArray[indMin + N/2]<=x)
			return binaryIntervalSearch<ArrayType>(inputArray,x,indMin + N/2,indMax,N/2 + currentRes);
		else
			return binaryIntervalSearch<ArrayType>(inputArray,x,indMin,indMin + N/2,currentRes);
	}
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

std::vector<std::string> split(const std::string &s, char delim);





#endif
