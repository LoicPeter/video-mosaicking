#include "linear_algebra.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>


void compute_pseudo_inverse(Eigen::MatrixXd& output, const Eigen::MatrixXd& input)
{
    Eigen::MatrixXd I(input.rows(),input.rows());
    I.setIdentity();
    output = (input.transpose() * input).ldlt().solve(input.transpose() * I);
}

void convertMatToSparse(Eigen::SparseMatrix<double,Eigen::ColMajor> sparse_A, const cv::Mat& A)
{
    CV_Assert(A.depth()==CV_64F);
    int rows = A.rows;
    int cols = A.cols;
    sparse_A = Eigen::SparseMatrix<double,Eigen::ColMajor>(rows,cols);
    std::vector<Eigen::Triplet<double>> triplet_list_A;
    for (int i=0; i<rows; ++i)
    {
        for (int j=0; j<cols; ++j)
        {
            double val = A.at<double>(i,j);
            if (val!=0)
                triplet_list_A.push_back(Eigen::Triplet<double>(i,j,val));
        }
    }

    sparse_A.setFromTriplets(triplet_list_A.begin(),triplet_list_A.end());
}



void computeKroneckerProduct(Eigen::MatrixXd& product, const Eigen::MatrixXd& A, const Eigen::MatrixXd& B)
{
    int nA = A.rows();
    int pA = A.cols();
    int nB = B.rows();
    int pB = B.cols();

    product = Eigen::MatrixXd(nA*nB,pA*pB);
    for (int i_a=0; i_a<nA; ++i_a)
    {
        for (int j_a=0; j_a<pA; ++j_a)
        {
            for (int i_b=0; i_b<nB; ++i_b)
            {
                for (int j_b=0; j_b<pB; ++j_b)
                {
                    product(i_a*nB + i_b,j_a*pB + j_b) = A(i_a,j_a)*B(i_b,j_b);
                }
            }
        }
    }
}



void computeKroneckerProduct(Eigen::SparseMatrix<double>& res, const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& B)
{
    int nb_rows_A = A.rows();
    int nb_cols_A = A.cols();
    int nb_rows_B = B.rows();
    int nb_cols_B = B.cols();
    res = Eigen::SparseMatrix<double>(nb_rows_A*nb_rows_B,nb_cols_A*nb_cols_B);
    
    std::vector<Eigen::Triplet<double>> triplet_list;   
    int nb_non_zeros_A = A.nonZeros();
    triplet_list.reserve(nb_non_zeros_A*nb_rows_B*nb_cols_B);
    
    for (int k=0; k<A.outerSize(); ++k) // iterate over non zero elements of A
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A,k); it; ++it)
        {
            double val_A = it.value();
            int i_A = it.row();   // row index
            int j_A = it.col();   // col index (here it is equal to k)
            
            for (int j_B=0; j_B<nb_cols_B; ++j_B)
            {
                for (int i_B=0; i_B<nb_rows_B; ++i_B)
                {
                    double val_B = B(i_B,j_B);
                    if (val_B!=0)
                    {
                        int i_res = i_A*nb_rows_B + i_B;
                        int j_res = j_A*nb_cols_B + j_B;
                        triplet_list.push_back(Eigen::Triplet<double>(i_res,j_res,val_A*val_B));
                    }
                }
            }
        }
    }
    
    res.setFromTriplets(triplet_list.begin(),triplet_list.end());
}

void computeSparseKroneckerProduct(Eigen::SparseMatrix<double>& res, const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B)
{
    int nb_rows_A = A.rows();
    int nb_cols_A = A.cols();
    int nb_rows_B = B.rows();
    int nb_cols_B = B.cols();
    res = Eigen::SparseMatrix<double>(nb_rows_A*nb_rows_B,nb_cols_A*nb_cols_B);
    
    std::vector<Eigen::Triplet<double>> triplet_list;   
    int nb_non_zeros_A = A.nonZeros();
    int nb_non_zeros_B = B.nonZeros();
    triplet_list.reserve(nb_non_zeros_A*nb_non_zeros_B);
    
    for (int k=0; k<A.outerSize(); ++k) // iterate over non zero elements of A
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it_A(A,k); it_A; ++it_A)
        {
            double val_A = it_A.value();
            int i_A = it_A.row();   // row index
            int j_A = it_A.col();   // col index (here it is equal to k)
            
            for (int l=0; l<B.outerSize(); ++l) // iterate over non zero elements of A
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it_B(B,l); it_B; ++it_B)
                {
                    double val_B = it_B.value();
                    int i_B = it_B.row();   // row index
                    int j_B = it_B.col();   // col index
                    int i_res = i_A*nb_rows_B + i_B;
                    int j_res = j_A*nb_cols_B + j_B;
                    triplet_list.push_back(Eigen::Triplet<double>(i_res,j_res,val_A*val_B));
                }
            }
        }
    }
    
    res.setFromTriplets(triplet_list.begin(),triplet_list.end());
}




void product_dense_blockdiagonal_with_modes(std::vector<Eigen::MatrixXd>& res, const Eigen::MatrixXd& A, const std::vector<Eigen::MatrixXd>& blocks, const std::vector<int>& block_modes, int nb_modes)
{
    int nb_rows_A = A.rows();
    int nb_cols_A = A.cols();
    int nb_blocks = (int)blocks.size();
    int nb_rows_S(0), nb_cols_S(0);
    for (int b=0; b<nb_blocks; ++b)
    {
        nb_rows_S += blocks[b].rows();
        nb_cols_S += blocks[b].cols();
    }
    if (nb_cols_A != nb_rows_S)
        std::cout << "Error: In product_dense_blockdiagonal_with_modes, the dimensions of the input matrices are not compatible: Cols A = " << nb_cols_A << " and Rows S = " << nb_rows_S << std::endl;
    else
    {
        res.resize(nb_modes);
        for (int m=0; m<nb_modes; ++m)
            res[m] = Eigen::MatrixXd::Zero(nb_rows_A,nb_cols_S);
        int j_R(0), j_A(0);
        for (int b=0; b<nb_blocks; ++b)
        {
            int m = block_modes[b];
            int nb_rows_block = blocks[b].rows();
            int nb_cols_block = blocks[b].cols();
            res[m].block(0,j_R,nb_rows_A,nb_cols_block) = A.block(0,j_A,nb_rows_A,nb_rows_block)*blocks[b];            
            j_R+=nb_cols_block;
            j_A+=nb_rows_block;
        }
    }
}

void product_dense_blockdiagonal_m_th_mode(Eigen::MatrixXd& res, int input_mode, const Eigen::MatrixXd& A, const std::vector<Eigen::MatrixXd>& blocks, const std::vector<int>& block_modes)
{
    int nb_rows_A = A.rows();
    int nb_cols_A = A.cols();
    int nb_blocks = (int)blocks.size();
    int nb_rows_S(0), nb_cols_S(0);
    for (int b=0; b<nb_blocks; ++b)
    {
        nb_rows_S += blocks[b].rows();
        nb_cols_S += blocks[b].cols();
    }
    if (nb_cols_A != nb_rows_S)
        std::cout << "Error: In product_dense_blockdiagonal_m_th_mode, the dimensions of the input matrices are not compatible: Cols A = " << nb_cols_A << " and Rows S = " << nb_rows_S << std::endl;
    else
    {
        res = Eigen::MatrixXd::Zero(nb_rows_A,nb_cols_S);
        int j_R(0), j_A(0);
        for (int b=0; b<nb_blocks; ++b)
        {
            int m = block_modes[b];
            int nb_rows_block = blocks[b].rows();
            int nb_cols_block = blocks[b].cols();
            if (m==input_mode)
                res.block(0,j_R,nb_rows_A,nb_cols_block) = A.block(0,j_A,nb_rows_A,nb_rows_block)*blocks[b];            
            j_R+=nb_cols_block;
            j_A+=nb_rows_block;
        }
    }
}



void affine_matrix_to_vector(Eigen::VectorXd& T_vec, const Eigen::Matrix3d& T)
{
    T_vec = Eigen::VectorXd(6);
    int k(0);
    for (int i=0; i<2; ++i)
    {
        for (int j=0; j<3; ++j)
        {
           T_vec(k) = T(i,j);
            ++k;
        }
    }
}


void affine_vector_to_matrix(Eigen::Matrix3d& T, const Eigen::VectorXd& T_vec)
{
    T.setIdentity();
    int k(0);
    for (int i=0; i<2; ++i)
    {
        for (int j=0; j<3; ++j)
        {
            T(i,j) = T_vec(k);
            ++k;
        }
    }
}

void affine_vector_to_matrix(cv::Mat& T, const Eigen::VectorXd& T_vec)
{
    T = cv::Mat::eye(3,3,CV_64F);
    int k(0);
    for (int i=0; i<2; ++i)
    {
        for (int j=0; j<3; ++j)
        {
            T.at<double>(i,j) = T_vec(k);
            ++k;
        }
    }
}

void vectorize_2_times_2(Eigen::VectorXd& T_vec, const Eigen::Matrix2d& T)
{
    T_vec = Eigen::VectorXd(4);
    T_vec(0) = T(0,0);
    T_vec(1) = T(0,1);
    T_vec(2) = T(1,0);
    T_vec(3) = T(1,1);
}




void opencv_affine_to_eigen_affine(Eigen::Matrix3d& h_eigen, const cv::Mat& h_open_cv)
{
    double* h_data = (double*)h_open_cv.data;
    h_eigen(0,0) = h_data[0];
    h_eigen(0,1) = h_data[1];
    h_eigen(0,2) = h_data[2];
    h_eigen(1,0) = h_data[3];
    h_eigen(1,1) = h_data[4];
    h_eigen(1,2) = h_data[5];
    h_eigen(2,0) = 0;
    h_eigen(2,1) = 0;
    h_eigen(2,2) = 1;
}


void opencv_mat_to_eigen_mat(Eigen::MatrixXd& mat_eigen, const cv::Mat& mat_opencv)
{
    int nb_rows = mat_opencv.rows;
    int nb_cols = mat_opencv.cols;
    mat_eigen = Eigen::MatrixXd(nb_rows,nb_cols);
    for (int j=0; j<nb_cols; ++j)
    {
        for (int i=0; i<nb_rows; ++i)
            mat_eigen(i,j) = mat_opencv.at<double>(i,j);
    }
    
}

void opencv_point_to_eigen_homogeneous(Eigen::Vector3d& eigen_tilde, const cv::Point2f& p)
{
    eigen_tilde(0) = p.x;
    eigen_tilde(1) = p.y;
    eigen_tilde(2) = 1;
}

void add_sparse_matrix_to_dense_matrix(Eigen::MatrixXd& D, const Eigen::SparseMatrix<double>& S)
{
    for (int l=0; l<S.outerSize(); ++l) // iterate over non zero elements of S
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(S,l); it; ++it)
        {
            double current_val = it.value();
            int current_i = it.row();   // row index
            int current_j = it.col();   // col index
            D(current_i,current_j) += current_val;
        }
    }
}

void create_sparse_matrix_from_dense_matrix(Eigen::SparseMatrix<double>& S, const Eigen::MatrixXd& D)
{
    int nb_rows = D.rows();
    int nb_cols = D.cols();
    S = Eigen::SparseMatrix<double>(nb_rows,nb_cols);
    std::vector<Eigen::Triplet<double>> triplet_list;
    for (int j=0; j<nb_cols; ++j)
    {
        for (int i=0; i<nb_rows; ++i)
        {
            double val = D(i,j);
            if (val!=0)
                triplet_list.push_back(Eigen::Triplet<double>(i,j,val));
        }
    }
    S.setFromTriplets(triplet_list.begin(),triplet_list.end());
}

void create_sparse_matrix_with_one_nonzero_element(Eigen::SparseMatrix<double>& S, int nb_rows, int nb_cols, int i_el, int j_el, double val_el)
{
    S.resize(nb_rows,nb_cols);
    std::vector<Eigen::Triplet<double>> t_vec(1,Eigen::Triplet<double>(i_el,j_el,val_el)); 
    S.setFromTriplets(t_vec.begin(),t_vec.end());
}


bool is_point_inside_parallelogram(const cv::Point2f& point_of_interest, const std::vector<cv::Point2f>& polygon)
{
    bool res(false);
    int nb_vertices = (int)polygon.size();
    if (nb_vertices!=4)
        std::cout << "Calling is_point_inside_parallelogram on a polygon that does not have 4 points" << std::endl;
   // for (int k=0; k<4; ++k)
   //     std::cout << "(" << polygon[k].x << " " << polygon[k].y << ") ";
   // std::cout << std::endl;
    // Test if the points are in the correct order, i.e. make sure that AB || CD
    cv::Point2f AB(polygon[1].x - polygon[0].x, polygon[1].y - polygon[0].y);
    cv::Point2f AC(polygon[2].x - polygon[0].x, polygon[2].y - polygon[0].y);
    cv::Point2f CD(polygon[3].x - polygon[2].x, polygon[3].y - polygon[2].y);
    cv::Point2f AM(point_of_interest.x - polygon[0].x,point_of_interest.y - polygon[0].y);
   // double det_AB_CD = (AB.x)*(CD.y) - (AB.y)*(CD.x);
   // if (det_AB_CD!=0)
   //     std::cout << "Points seem to be in the wrong order: Determinant of AB and CD = " << det_AB_CD << std::endl;
    double det_AB_AC = (AB.x)*(AC.y) - (AB.y)*(AC.x);
    double beta = ((AB.x)*(AM.y) - (AB.y)*(AM.x))/det_AB_AC;
    double alpha = ((AC.y)*(AM.x) - (AM.y)*(AC.x))/det_AB_AC;
    if ((alpha>=0) && (alpha <=1) && (beta>=0) && (beta<=1))
        res = true;
    return res;
}
