#include "affine_transformation.hpp"

void getDisplacedPointsAffine(std::vector<cv::Point2f>& output_points, double *h_ij, const std::vector<cv::Point2f>& input_points)
{
    int nb_input_points = (int)input_points.size();
    if (output_points.size()!=nb_input_points)
        output_points.resize(nb_input_points);
    for (int p=0; p<nb_input_points; ++p)
    {
        output_points[p].x = h_ij[0]*input_points[p].x +  h_ij[1]*input_points[p].y +  h_ij[2];
        output_points[p].y = h_ij[3]*input_points[p].x +  h_ij[4]*input_points[p].y +  h_ij[5];
    }
}


