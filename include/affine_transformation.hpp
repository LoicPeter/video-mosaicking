#ifndef AFFINE_TRANSFORMATION_INCLUDED
#define AFFINE_TRANSFORMATION_INCLUDED

#include <opencv2/core.hpp>
#include <vector>
#include <iostream>

template <typename T>
void multiply_affine_and_affine(T* result, const T* const H1, const T* const H2) // case where H1[6] = 0, H1[7] = 0, H1[8] = 1, H2[6] = 0, H2[7] = 0, H2[8] = 1
{
    result[0] = H1[0]*H2[0] + H1[1]*H2[3];
    result[1] = H1[0]*H2[1] + H1[1]*H2[4];
    result[2] = H1[0]*H2[2] + H1[1]*H2[5] + H1[2];
    result[3] = H1[3]*H2[0] + H1[4]*H2[3];
    result[4] = H1[3]*H2[1] + H1[4]*H2[4];
    result[5] = H1[3]*H2[2] + H1[4]*H2[5] + H1[5];
}

template <typename T>
void inverse_affine_matrix(T* inv_p, const T* const p) // returns a closed form for the inverse of a affine matrix (p[6] = 0, p[7] = 0, p[8] = 1)
{
    T det = p[0]*p[4] - p[1]*p[3];
    if (det!=T(0))
    {
        T comatrix[9];
        comatrix[0] = p[4];
        comatrix[1] = -p[3];
       // comatrix[2] = 0;
        comatrix[3] = -p[1];
        comatrix[4] = p[0];
       // comatrix[5] = 0;
        comatrix[6] = p[1]*p[5] - p[2]*p[4];
        comatrix[7] = -(p[0]*p[5] - p[2]*p[3]);
       // comatrix[8] = p[0]*p[4] - p[1]*p[3];

        for (int i=0; i<2; ++i)
        {
            for (int j=0; j<3; ++j)
                inv_p[3*i+j] = comatrix[3*j+i]/det;
        }
    }
    else
        std::cout << "In inverse_affine_matrix, the input matrix was found to be singular and thus cannot be inversed" << std::endl;
}

void getDisplacedPointsAffine(std::vector<cv::Point2f>& output_points, double *h_ij, const std::vector<cv::Point2f>& input_points);

#endif
