#ifndef LINEAR_ALGEBRA_INCLUDED
#define LINEAR_ALGEBRA_INCLUDED

#include <iostream>
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/core.hpp>


template <typename SparseMatrixType>
void compute_sparse_inverse(SparseMatrixType& output, const SparseMatrixType& input)
{
    Eigen::SparseQR<SparseMatrixType,Eigen::COLAMDOrdering<int>> solver;
    solver.compute(input);
    int nb_rows = input.rows();
    Eigen::SparseMatrix<double> I(nb_rows,nb_rows);
    I.setIdentity();
    output = solver.solve(I);
}

template <typename SparseMatrixType>
void fast_sparse_dense_product(SparseMatrixType& res, const SparseMatrixType& S, const Eigen::MatrixXd& D, int i_block_top_corner, int j_block_top_corner, int block_size_i, int block_size_j)
{
    int p = D.cols();
    res = SparseMatrixType(S.rows(),p);
    Eigen::MatrixXd non_zero_row_block = (S.block(i_block_top_corner,j_block_top_corner,block_size_i,block_size_j))*D.block(j_block_top_corner,0,block_size_j,p); // it would be better if we preallocate this

    std::vector<Eigen::Triplet<double>> triplet_list;
    triplet_list.reserve(non_zero_row_block.rows()*non_zero_row_block.cols());
    for (int j=0; j<p; ++j)
    {
        for (int i=0; i<block_size_i; ++i)
        {
            triplet_list.push_back(Eigen::Triplet<double>(i+i_block_top_corner,j,non_zero_row_block(i,j)));
        }

    }
    res.setFromTriplets(triplet_list.begin(),triplet_list.end());
}


template <typename LeftMatrixType>
void product_dense_blockdiagonal(Eigen::MatrixXd& res, const LeftMatrixType& A, const std::vector<Eigen::MatrixXd>& blocks)
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
         std::cout << "Error: In product_dense_blockdiagonal, the dimensions of the input matrices are not compatible: Cols A = " << nb_cols_A << " and Rows S = " << nb_rows_S << std::endl;
    else
    {
        res = Eigen::MatrixXd(nb_rows_A,nb_cols_S);
        int j_R(0), j_A(0);
        for (int b=0; b<nb_blocks; ++b)
        {
            int nb_rows_block = blocks[b].rows();
            int nb_cols_block = blocks[b].cols();
            res.block(0,j_R,nb_rows_A,nb_cols_block) = A.block(0,j_A,nb_rows_A,nb_rows_block)*blocks[b];            
            j_R+=nb_cols_block;
            j_A+=nb_rows_block;
        }
    }
}

void compute_pseudo_inverse(Eigen::MatrixXd& output, const Eigen::MatrixXd& input);

void convertMatToSparse(Eigen::SparseMatrix<double,Eigen::ColMajor> sparse_A, const cv::Mat& A);

void computeKroneckerProduct(Eigen::MatrixXd& product, const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);

void computeKroneckerProduct(Eigen::SparseMatrix<double>& res, const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& B);

void computeSparseKroneckerProduct(Eigen::SparseMatrix<double>& res, const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B);

void product_dense_blockdiagonal_with_modes(std::vector<Eigen::MatrixXd>& res, const Eigen::MatrixXd& A, const std::vector<Eigen::MatrixXd>& blocks, const std::vector<int>& block_modes, int nb_modes);

void product_dense_blockdiagonal_m_th_mode(Eigen::MatrixXd& res, int input_mode, const Eigen::MatrixXd& A, const std::vector<Eigen::MatrixXd>& blocks, const std::vector<int>& block_modes);

void affine_matrix_to_vector(Eigen::VectorXd& T_vec, const Eigen::Matrix3d& T);

void affine_vector_to_matrix(Eigen::Matrix3d& T, const Eigen::VectorXd& T_vec);

void affine_vector_to_matrix(cv::Mat& T, const Eigen::VectorXd& T_vec);

void vectorize_2_times_2(Eigen::VectorXd& T_vec, const Eigen::Matrix2d& T);

void opencv_affine_to_eigen_affine(Eigen::Matrix3d& h_eigen, const cv::Mat& h_open_cv);

void opencv_mat_to_eigen_mat(Eigen::MatrixXd& mat_eigen, const cv::Mat& mat_open_cv);

void opencv_point_to_eigen_homogeneous(Eigen::Vector3d& eigen_tilde, const cv::Point2f& p);

void add_sparse_matrix_to_dense_matrix(Eigen::MatrixXd& D, const Eigen::SparseMatrix<double>& S);

void create_sparse_matrix_from_dense_matrix(Eigen::SparseMatrix<double>& S, const Eigen::MatrixXd& D);

void create_sparse_matrix_with_one_nonzero_element(Eigen::SparseMatrix<double>& S, int nb_rows, int nb_cols, int i_el, int j_el, double val_el);

bool is_point_inside_parallelogram(const cv::Point2f& point_of_interest, const std::vector<cv::Point2f>& polygon);

#endif // LINEAR_ALGEBRA_INCLUDED
