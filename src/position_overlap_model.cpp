#include "position_overlap_model.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
//#include "stdafx.h"
#include "optimization.h"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

// PositionOverlapModel::PositionOverlapModel(const std::vector<MeasuredAffineRegistration>& pairwise_measurements, const std::vector<int>& measurement_modes, const std::vector<double>& variance_parameters, int ind_reference_frame, int nb_frames, int size_square_interior, int size_square_exterior, const std::vector<cv::Point2f>& overlap_corners)
// {
//     m_ind_reference_frame = ind_reference_frame;
//     m_nb_frames = nb_frames;
//     m_variance_parameters = variance_parameters;
//   //  m_size_square_interior = size_square_interior;
//   //  m_size_square_exterior = size_square_exterior;
//    // m_overlap_corners = overlap_corners;
//     m_gamma(0) = 0.25*(overlap_corners[0].x + overlap_corners[1].x + overlap_corners[2].x + overlap_corners[3].x);
//     m_gamma(1) = 0.25*(overlap_corners[0].y + overlap_corners[1].y + overlap_corners[2].y + overlap_corners[3].y);
//     this->set_pairwise_measurements(pairwise_measurements,measurement_modes);
// }

PositionOverlapModel::PositionOverlapModel()
{}

PositionOverlapModel::~PositionOverlapModel()
{}

PositionOverlapModel::PositionOverlapModel(const std::vector<MeasuredAffineRegistration>& pairwise_measurements, const std::vector<int>& measurement_modes, const std::vector<double>& variance_parameters, int ind_reference_frame, int nb_frames, const Eigen::Vector2d& gamma)
{
    m_ind_reference_frame = ind_reference_frame;
    m_nb_frames = nb_frames;
    m_variance_parameters = variance_parameters;
    m_gamma = gamma;
    this->estimate_size_overlap_area(pairwise_measurements);
    this->set_pairwise_measurements(pairwise_measurements,measurement_modes);
}

PositionOverlapModel::PositionOverlapModel(const std::vector<MeasuredAffineRegistration>& pairwise_measurements, const std::vector<int>& measurement_modes, const std::vector<double>& variance_parameters, int ind_reference_frame, int nb_frames, const Eigen::Vector2d& gamma, double size_overlap_area)
{
    std::cout << "Create model with fixed overlap corners" << std::endl;
    m_ind_reference_frame = ind_reference_frame;
    m_nb_frames = nb_frames;
    m_variance_parameters = variance_parameters;
    m_gamma = gamma;
    m_size_overlap_area = size_overlap_area;
    this->set_pairwise_measurements(pairwise_measurements,measurement_modes);
}

void create_overlap_corners(std::vector<cv::Point2f>& overlap_corners, const Eigen::Vector2d& gamma, double size_overlap_area)
{
    double half_size_overlap_area = size_overlap_area/2;
    overlap_corners.resize(4);
    overlap_corners[0] = cv::Point2f(gamma(0) - half_size_overlap_area,gamma(1) - half_size_overlap_area);
    overlap_corners[1] = cv::Point2f(gamma(0) - half_size_overlap_area,gamma(1) + half_size_overlap_area);
    overlap_corners[2] = cv::Point2f(gamma(0) + half_size_overlap_area,gamma(1) - half_size_overlap_area);
    overlap_corners[3] = cv::Point2f(gamma(0) + half_size_overlap_area,gamma(1) + half_size_overlap_area);
}

void PositionOverlapModel::set_pairwise_measurements(const std::vector<MeasuredAffineRegistration>& pairwise_measurements, const std::vector<int>& measurement_modes)
{
    m_pairwise_measurements = pairwise_measurements;
    m_measurement_modes = measurement_modes;
    
    // We compute useful matrices for bundle adjustment and covariance propagation
    compute_matrices_closed_form_bundle_adjustment_on_correspondences(A,b,linear_factor_derivatives,constant_factor_derivatives, triplet_list_standard_basis,pairwise_measurements,m_nb_frames,m_ind_reference_frame);
    
    // Compute the Cholesky decomposition of A and the inverse of A
    A_cholesky.compute(A);
    Eigen::MatrixXd I(A.rows(),A.rows());
    I.setIdentity();
    A_inv = A_cholesky.solve(I);
    
    // Compute the mean estimated positions
    this->compute_mean_estimated_positions();
    
    // Compute the scaled covariance matrices correpsonding to each measurement
    createDiagonalCovariancesCorrespondences(scaled_factors_covariance_matrices_pairwise_measurements,pairwise_measurements);
    
    // Compute the covariance matrix of the estimated positions
    this->compute_covariance_estimated_positions();
    
}

void PositionOverlapModel::compute_mean_estimated_positions()
{
    // We solve without the reference frame
    estimated_absolute_positions_inv_without_ref = A_cholesky.solve(b);
    
    // We add the reference frame 
    Eigen::VectorXd estimated_absolute_positions_inv;
    add_reference_frame_to_global_positions_affine(estimated_absolute_positions_inv,estimated_absolute_positions_inv_without_ref,m_ind_reference_frame);
    
    // We take the inverse of each position matrix (bundle adjustment on correspondences optimises over the inverses)
    inverse_vector_of_estimated_affine_global_positions(mean_estimated_absolute_positions,estimated_absolute_positions_inv);
}

void PositionOverlapModel::compute_covariance_estimated_positions()
{
    // Compute the left factor in covariance propagation
    Eigen::MatrixXd F_uv, left_cov_propagation_global_cov_matrix;
    Eigen::SparseMatrix<double> F_uv_sparse;
    fill_F_uv(F_uv,linear_factor_derivatives,constant_factor_derivatives,estimated_absolute_positions_inv_without_ref);
    create_sparse_matrix_from_dense_matrix(F_uv_sparse,F_uv);  // could be probably done directly
    left_cov_propagation_global_cov_matrix = A_inv*F_uv_sparse;
  //  left_cov_propagation_global_cov_matrix = A_cholesky.solve(F_uv);
    
    // Compute the left factor of the covariance matrix of the measurements
    int nb_measurements = (int)m_pairwise_measurements.size();
    std::vector<Eigen::MatrixXd> factors_covariance_matrices_pairwise_measurements(nb_measurements);
    for (int k=0; k<nb_measurements; ++k)
        factors_covariance_matrices_pairwise_measurements[k] = std::sqrt(m_variance_parameters[m_measurement_modes[k]])*scaled_factors_covariance_matrices_pairwise_measurements[k];
    
    // We make the product of the two left factors
    Eigen::MatrixXd covariance_estimated_absolute_positions_inv_without_ref_factor;
    product_dense_blockdiagonal(covariance_estimated_absolute_positions_inv_without_ref_factor,left_cov_propagation_global_cov_matrix,factors_covariance_matrices_pairwise_measurements);
    
    // Transpose and multiply to obtain the final covariance matrix without reference frame
    std::cout << "The multiplication - matrix size = " << covariance_estimated_absolute_positions_inv_without_ref_factor.rows() << " x " << covariance_estimated_absolute_positions_inv_without_ref_factor.cols() << std::endl;
    Eigen::MatrixXd covariance_estimated_absolute_positions_inv_without_ref = covariance_estimated_absolute_positions_inv_without_ref_factor*(covariance_estimated_absolute_positions_inv_without_ref_factor.transpose());
    std::cout << "Done" << std::endl;
     
    // We add the reference frame to obtain the final
    add_reference_frame_to_global_covariance_matrix_affine(covariance_estimated_absolute_positions_inv,covariance_estimated_absolute_positions_inv_without_ref,m_ind_reference_frame);
    
//     int N = A_inv.rows();
//     int P = A_inv.cols();
//     Eigen::MatrixXd temp = A.inverse();
//     cv::Mat vis_cov(N,P,CV_64F,cv::Scalar::all(0)), vis_cov_norm;
//     for (int i=0; i<N; ++i)
//     {
//         for (int j=0; j<P; ++j)
//         {
//             vis_cov.at<double>(i,j) = 10*std::abs(A_inv(i,j));
//         }
//     }
//     cv::normalize(vis_cov,vis_cov_norm,1,0,cv::NORM_MINMAX);
//     cv::imshow("Covariance BA",vis_cov_norm);
//     cv::waitKey(-1);
    
    
    
    
    
    
    // Compute modes for faster update
//     int nb_modes = (int)m_variance_parameters.size();
//     scaled_covariance_matrix_modes.resize(nb_modes);
//     for (int m=0; m<nb_modes; ++m)
//     {
//         Eigen::MatrixXd factor_scaled_covariance_matrix_mode = Eigen::MatrixXd::Zero(covariance_estimated_absolute_positions_inv_without_ref_factor.rows(),covariance_estimated_absolute_positions_inv_without_ref_factor.cols());
//         product_dense_blockdiagonal_m_th_mode(factor_scaled_covariance_matrix_mode,m,left_cov_propagation_global_cov_matrix,scaled_factors_covariance_matrices_pairwise_measurements,m_measurement_modes);
//         Eigen::MatrixXd scaled_covariance_matrix_mode_without_ref = factor_scaled_covariance_matrix_mode*(factor_scaled_covariance_matrix_mode.transpose());
//          add_reference_frame_to_global_covariance_matrix_affine(scaled_covariance_matrix_modes[m],scaled_covariance_matrix_mode_without_ref,m_ind_reference_frame);
//     }
//     
//     std::cout << "Sanity check covariance matrix and modes: ";
//     Eigen::MatrixXd cov_matrix_2 = m_variance_parameters[0]*scaled_covariance_matrix_modes[0];
//     for (int m=1; m<nb_modes; ++m)
//         cov_matrix_2 = cov_matrix_2 + m_variance_parameters[m]*scaled_covariance_matrix_modes[m];
//     Eigen::MatrixXd diff = cov_matrix_2 - covariance_estimated_absolute_positions_inv;
//     std::cout << diff.array().abs().maxCoeff() << std::endl;
        
    
}


void PositionOverlapModel::add_pairwise_measurement(const MeasuredAffineRegistration& new_measurement, int measurement_mode)
{
    // We update the matrices
    update_bundle_adjustment_on_correspondences(A,b,A_cholesky,A_inv,linear_factor_derivatives,constant_factor_derivatives,triplet_list_standard_basis,m_ind_reference_frame,new_measurement);
    
    //Create scaled covariance matrix on correspondences
    int nb_added_landmarks = new_measurement.landmarks_input.size();
    Eigen::MatrixXd factorised_cov_matrix_measurements(2*nb_added_landmarks,2*nb_added_landmarks);
    factorised_cov_matrix_measurements.setIdentity();
    
    // Add it to the list
    scaled_factors_covariance_matrices_pairwise_measurements.push_back(factorised_cov_matrix_measurements);
    
    // Add the new measurement
    m_pairwise_measurements.push_back(new_measurement);
    
    // Add the mode of the new measurement
    m_measurement_modes.push_back(measurement_mode);
    
    // Update the mean positions and their covariance matrix
    std::cout <<"Update mean..." << std::endl;
    this->compute_mean_estimated_positions();
    std::cout <<"Update covariance..." << std::endl;
    this->compute_covariance_estimated_positions();
    std::cout << "Done" << std::endl;
}


// // This function performs the linearisation to compute the distribution of T_{j,i}(input_point)
void PositionOverlapModel::compute_asymmetric_relative_displacement_distribution(Eigen::Vector2d& mean_coordinates, Eigen::Matrix2d& covariance_coordinates, int i, int j, const Eigen::Vector2d& input_point) const
{
    // We switch to homogeneous coordinates 
    Eigen::Vector3d input_point_tilde, mean_coordinates_tilde; 
    input_point_tilde(0) = input_point(0);
    input_point_tilde(1) = input_point(1);
    input_point_tilde(2) = 1;
    
    Eigen::Matrix3d T_i, T_j, T_i_inv; // note that it is more convenient to consider T instead of Theta
    affine_vector_to_matrix(T_i,mean_estimated_absolute_positions.segment<6>(6*i));
    affine_vector_to_matrix(T_j,mean_estimated_absolute_positions.segment<6>(6*j));
    T_i_inv = T_i.inverse();
    
    // Mean
    mean_coordinates_tilde = T_j*T_i_inv*input_point_tilde;
    mean_coordinates = mean_coordinates_tilde.segment<2>(0);
    
    // Covariance
    Eigen::MatrixXd Gamma_ij(2,12), temp_block(2,6), relative_cov_matrix(12,12);
    computeKroneckerProduct(temp_block,T_j.block<2,2>(0,0),input_point_tilde.transpose());
    Gamma_ij.block<2,6>(0,0) = temp_block;
    computeKroneckerProduct(temp_block,(-1)*T_j.block<2,2>(0,0),mean_coordinates_tilde.transpose());
    Gamma_ij.block<2,6>(0,6) = temp_block;
    retain_blocks_i_and_j_in_covariance_matrix(relative_cov_matrix,covariance_estimated_absolute_positions_inv,i,j);
    covariance_coordinates = Gamma_ij*relative_cov_matrix*(Gamma_ij.transpose());     
    
    
//    std::cout << "Only translation terms linear approximation" << std::endl;
//     Eigen::MatrixXd Gamma_ij_translation(2,4), relative_cov_matrix_translation(4,4);
//     for (int d=0; d<4; ++d)
//         Gamma_ij_translation.block<2,1>(0,d) = Gamma_ij.block<2,1>(0,3*d+2);
//     for (int k=0; k<4; ++k)
//     {
//         for (int l=0; l<4; ++l)
//         {
//             relative_cov_matrix_translation(k,l) = relative_cov_matrix(3*k+2,3*l+2);
//         }    
//     }
//     covariance_coordinates = Gamma_ij_translation*relative_cov_matrix_translation*(Gamma_ij_translation.transpose());
    
 //   std::cout << "Only i moves" << std::endl;
 //   covariance_coordinates = Gamma_ij.block<2,6>(0,0)*relative_cov_matrix.block<6,6>(0,0)*( Gamma_ij.block<2,6>(0,0).transpose());   
    
 //   std::cout << "Only j moves" << std::endl;
 //   covariance_coordinates = Gamma_ij.block<2,6>(0,6)*relative_cov_matrix.block<6,6>(6,6)*( Gamma_ij.block<2,6>(0,6).transpose());  
    
}


void PositionOverlapModel::compute_relative_displacement_distribution(Eigen::Vector2d& mean_coordinates, Eigen::Matrix2d& covariance_coordinates, int i, int j, const Eigen::Vector2d& input_point) const
{
    this->compute_asymmetric_relative_displacement_distribution(mean_coordinates,covariance_coordinates,j,i,input_point);
}

// This function performs the linearisation to compute the distribution of T_{j,i}(input_point)
// void PositionOverlapModel::compute_relative_displacement_distribution(Eigen::Vector2d& mean_coordinates, Eigen::Matrix2d& covariance_coordinates, int i, int j, const Eigen::Vector2d& input_point) const
// {
//     // We switch to homogeneous coordinates 
//     Eigen::Vector3d input_point_tilde, mean_coordinates_tilde; 
//     input_point_tilde(0) = input_point(0);
//     input_point_tilde(1) = input_point(1);
//     input_point_tilde(2) = 1;
//     
//     // Identity
//     Eigen::Matrix2d I_2;
//     I_2.setIdentity();
//     
//     Eigen::Matrix3d T_i, T_j, T_j_inv, Theta_j_Theta_i_inv; // note that it is more convenient to consider T instead of Theta
//     affine_vector_to_matrix(T_i,mean_estimated_absolute_positions.segment<6>(6*i));
//     affine_vector_to_matrix(T_j,mean_estimated_absolute_positions.segment<6>(6*j));
//     T_j_inv = T_j.inverse();
//     Theta_j_Theta_i_inv = T_j_inv*T_i;
//     
//     // Mean
//     mean_coordinates_tilde = Theta_j_Theta_i_inv*input_point_tilde;
//     mean_coordinates = mean_coordinates_tilde.segment<2>(0);
//     
//     Eigen::Vector3d Theta_i_inv_gamma = T_i*input_point_tilde;
//     
//     // Covariance
//     Eigen::MatrixXd Gamma_ij(2,12), temp_block(2,6), relative_cov_matrix(12,12);
//     computeKroneckerProduct(temp_block,(-1)*Theta_j_Theta_i_inv.block<2,2>(0,0),Theta_i_inv_gamma.transpose());
//     Gamma_ij.block<2,6>(0,0) = temp_block;
//     computeKroneckerProduct(temp_block,I_2,Theta_i_inv_gamma.transpose());
//     Gamma_ij.block<2,6>(0,6) = temp_block;
//     retain_blocks_i_and_j_in_covariance_matrix(relative_cov_matrix,covariance_estimated_absolute_positions_inv,i,j);
//     covariance_coordinates = Gamma_ij*relative_cov_matrix*(Gamma_ij.transpose());     
//     
//     
// //     std::cout << "Only translation terms linear approximation" << std::endl;
//     Eigen::MatrixXd Gamma_ij_translation(2,4), relative_cov_matrix_translation(4,4);
//     for (int d=0; d<4; ++d)
//         Gamma_ij_translation.block<2,1>(0,d) = Gamma_ij.block<2,1>(0,3*d+2);
//     for (int k=0; k<4; ++k)
//     {
//         for (int l=0; l<4; ++l)
//         {
//             relative_cov_matrix_translation(k,l) = relative_cov_matrix(3*k+2,3*l+2);
//         }    
//     }
//     covariance_coordinates = Gamma_ij_translation*relative_cov_matrix_translation*(Gamma_ij_translation.transpose());
//     
//     
// }


void PositionOverlapModel::estimate_size_overlap_area(const std::vector<MeasuredAffineRegistration>& pairwise_measurements)
{
    int nb_measurements = (int)pairwise_measurements.size();
    
    double max_displacement(0);
    
    Eigen::Vector3d gamma_tilde;
    gamma_tilde(0) = m_gamma(0);
    gamma_tilde(1) = m_gamma(1);
    gamma_tilde(2) = 1;
    
    Eigen::Vector3d displacement_vector;
    double displacement_ij, displacement_ji, displacement;
    
    for (int m=0; m<nb_measurements; ++m)
    {
        
        Eigen::Matrix3d affine_matrix, affine_matrix_inv;
        get_affine_matrix_from_parameters(affine_matrix,(double*)pairwise_measurements[m].mean_affine_parameters.data);
        
        displacement_vector = affine_matrix*gamma_tilde - gamma_tilde;
        displacement_ij = displacement_vector.norm();
        
        // Npw we do it fot he inverse
        displacement_vector = (affine_matrix.inverse())*gamma_tilde - gamma_tilde;
        displacement_ji = displacement_vector.norm();
        
        
        displacement = 0.5*(displacement_ij + displacement_ji);
        if (displacement>max_displacement)
            max_displacement = displacement;
        
    }
    
    m_size_overlap_area = 2*max_displacement;
    std::cout << "Estimated size overlap area: " <<  m_size_overlap_area << std::endl;
    
}

void get_affine_matrix_from_parameters(Eigen::Matrix3d& affine_matrix, const double* parameters)
{
    affine_matrix(0,0) = parameters[0];
    affine_matrix(0,1) = parameters[1];
    affine_matrix(0,2) = parameters[2];
    affine_matrix(1,0) = parameters[3];
    affine_matrix(1,1) = parameters[4];
    affine_matrix(1,2) = parameters[5];
    affine_matrix(2,0) = 0;
    affine_matrix(2,1) = 0;
    affine_matrix(2,2) = 1;
}

void PositionOverlapModel::compute_overlap_bounds_and_reward(double& lower_bound, double& upper_bound, double& reward, int i, int j, RewardType chosen_reward) const
{
    Eigen::Vector2d mean_gamma; 
    Eigen::Matrix2d covariance_gamma;
    this->compute_relative_displacement_distribution(mean_gamma,covariance_gamma,i,j,m_gamma);
    
    Eigen::Vector2d mean_displacement = mean_gamma - m_gamma;
    compute_closed_form_overlap_probability_and_bounds(lower_bound,upper_bound,reward,covariance_gamma,m_size_overlap_area,m_size_overlap_area,mean_displacement,chosen_reward);
}


void PositionOverlapModel::compute_overlap_probability_with_sampling(double& probability_overlap,  int i, int j, int nb_samples, boost::random::mt19937& rng) const
{
    Eigen::Vector2d mean_gamma, mean_displacement; 
    Eigen::Matrix2d covariance_gamma;
    this->compute_relative_displacement_distribution(mean_gamma,covariance_gamma,i,j,m_gamma);
    mean_displacement = mean_gamma - m_gamma;
    

 //   std::cout << "Mean: " << mean_displacement << std::endl;
 //   std::cout << "Covariance: " << covariance_gamma << std::endl;
    std::vector<cv::Point2f> overlap_corners;
    create_overlap_corners(overlap_corners,m_gamma,m_size_overlap_area);
  //  estimateOverlapProbabilitiesWithSampling(probability_overlap,mean_displacement,covariance_gamma,nb_samples,rng,overlap_corners);
    estimateOverlapProbabilitiesWithSampling(probability_overlap,mean_displacement,covariance_gamma,nb_samples,rng,0.5*m_size_overlap_area);
//    std::cout << "Proba: " << probability_overlap << std::endl;
//    std::cout << "----" << std::endl;
}

void PositionOverlapModel::compute_symmetrical_overlap_bounds_and_reward(double& lower_bound, double& upper_bound, double& reward, int i, int j, RewardType chosen_reward) const
{
    double lower_bound_ij,upper_bound_ij, reward_ij, lower_bound_ji,upper_bound_ji, reward_ji;
    
    this->compute_overlap_bounds_and_reward(lower_bound_ij,upper_bound_ij,reward_ij,i,j,chosen_reward);
    this->compute_overlap_bounds_and_reward(lower_bound_ji,upper_bound_ji,reward_ji,j,i,chosen_reward);
    
    lower_bound = 0.5*(lower_bound_ij + lower_bound_ji);
    upper_bound = 0.5*(upper_bound_ij + upper_bound_ji);
    reward = 0.5*(reward_ij + reward_ji);
}

void PositionOverlapModel::compute_symmetrical_overlap_probability_with_sampling(double& probability_overlap,  int i, int j, int nb_samples, boost::random::mt19937& rng) const
{
    double probability_overlap_ij, probability_overlap_ji;
    this->compute_overlap_probability_with_sampling(probability_overlap_ij,i,j,nb_samples,rng);
    this->compute_overlap_probability_with_sampling(probability_overlap_ji,j,i,nb_samples,rng);
    probability_overlap = 0.5*(probability_overlap_ij + probability_overlap_ji);
}

void PositionOverlapModel::update_variance(double variance, int mode)
{
    if (mode>=m_variance_parameters.size())
        std::cout << "Tried to update a mode higher than the number of modes" << std::endl;
    else
    {
        double old_variance = m_variance_parameters[mode];
        m_variance_parameters[mode] = variance;
        covariance_estimated_absolute_positions_inv += (variance - old_variance)*scaled_covariance_matrix_modes[mode];
    }
}


// This reestimates the variance parameter of the initial chain
void PositionOverlapModel::reestimate_initial_variance(const std::vector<LabeledImagePair>& labeled_pairs, double annotation_variance)
{
    int nb_labeled_pairs = labeled_pairs.size();
    
    // Sanity check 
    int nb_measurements = this->m_measurement_modes.size();
    for (int k=0; k<nb_measurements; ++k)
    {
        if (m_measurement_modes[k]!=0)
            std::cout << "Error! In PositionOverlapModel::reestimate_initial_variance, there are some measurements that are not from the initial chain. The rest of the optimization will not give the desired output. Store separately the initial position overlap model and train this one instead." << std::endl;
    }        
    
    // Normalisation factor: we normalise the distance to avoid numerical divergence
    double normalisation_factor = 10000;
    
    
    double current_variance = m_variance_parameters[0];
    
    std::cout << "Current variance: " << current_variance << std::endl;
    

    // Count labels
    std::vector<int> label_count(2);
    perform_label_count(label_count,labeled_pairs);
    std::vector<double> instance_weights(2);
    for (int l=0; l<2; ++l)
        instance_weights[l] = 1/((double)2*label_count[l]); 
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es;
    std::map<std::pair<int,int>,Eigen::Vector2d> scaled_eigenvalues, dot_products;
    
    double max_dot_product_overlap(0), max_dot_product_non_overlap(0);
    
    for (int ind_pair=0; ind_pair<nb_labeled_pairs; ++ind_pair)
    {
        int i = labeled_pairs[ind_pair].image_indices.first;
        int j = labeled_pairs[ind_pair].image_indices.second;
        
        Eigen::Vector2d mean_coordinates;
        Eigen::Matrix2d covariance_coordinates;
        this->compute_relative_displacement_distribution(mean_coordinates,covariance_coordinates,i,j,this->m_gamma);
    
        // precompute the scaled eigenvalues and the dot products
        es.computeDirect(covariance_coordinates);
        Eigen::Vector2d eigenvalues = es.eigenvalues();
        Eigen::Matrix2d cov_eigenvectors = es.eigenvectors();
        Eigen::Matrix2d cov_eigenvectors_tr = cov_eigenvectors.transpose();            
        scaled_eigenvalues[std::pair<int,int>(i,j)] = (1/current_variance)*eigenvalues;
        dot_products[std::pair<int,int>(i,j)] = (1/normalisation_factor)*cov_eigenvectors_tr*(mean_coordinates - this->m_gamma);
        
    }
    
    VarianceOptimisation var_optimisation;
    var_optimisation._labeled_pairs = labeled_pairs;
    var_optimisation._scaled_eigenvalues_ptr = &scaled_eigenvalues;
    var_optimisation._dot_products_ptr = &dot_products;
    
    var_optimisation._half_size_interior = (1/normalisation_factor)*(m_size_overlap_area/2);
    var_optimisation._half_size_exterior =  (1/normalisation_factor)*(m_size_overlap_area/2);
    
  //  var_optimisation._half_size_interior = max_dot_product_overlap;
  //  var_optimisation._half_size_exterior = max_dot_product_non_overlap;
  
    
    //  var_optimisation._nb_modes = 1;
  //  var_optimisation._nb_optimised_variances = 1;
    var_optimisation._instance_weights = instance_weights;
    void *var_optimisation_ptr;
    var_optimisation_ptr = &var_optimisation;
    
    alglib::real_1d_array x;
    double min_standard_deviation = std::sqrt(annotation_variance);
    std::vector<double> inverse_std(1,1.0/(std::sqrt(current_variance)));
    x.setcontent(1,&(inverse_std[0]));

    double epsg = 0.00001;
    double epsf = 0.00001;
    double epsx = 0;
    double diffstep = 1.0e-6;  
    alglib::ae_int_t maxits = 0;

    // BLEIC
//     std::vector<double> min_x(1,0.01), max_x(1,1.0/min_standard_deviation);
//     alglib::real_1d_array bnd_l, bnd_u;
//     bnd_l.setcontent(1,&(min_x[0]));
//     bnd_u.setcontent(1,&(max_x[0]));
//     alglib::minbleicstate state;
//     alglib::minbleicreport rep;
//     alglib::minbleiccreate(x, state);
//     alglib::minbleicsetbc(state, bnd_l, bnd_u);
//     alglib::minbleicsetcond(state, epsg, epsf, epsx, maxits);
//     alglib::minbleicoptimize(state,compute_hyperbolic_loss_one_variable,NULL,var_optimisation_ptr);
//     alglib::minbleicresults(state, x, rep);
   
    alglib::minlbfgsstate state;
    alglib::minlbfgsreport rep;
    alglib::minlbfgscreate(1, x, state);
    alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
    alglib::minlbfgsoptimize(state,compute_hyperbolic_loss_one_variable,NULL,var_optimisation_ptr);
    alglib::minlbfgsresults(state, x, rep);
    
    
    this->update_variance(std::pow<double>(normalisation_factor,2)/(x[0]*x[0]),0);
    std::cout << "Reestimated variance: " << m_variance_parameters[0] << std::endl;           
}

double PositionOverlapModel::compute_Sawhney_probability(double& sawhney_score, int i, int j, const std::vector<cv::Point2f>& image_corners) const
{
    Eigen::Matrix3d T_ir, T_ri, T_jr, T_rj;
    affine_vector_to_matrix(T_ir,this->mean_estimated_absolute_positions.segment<6>(6*i));
    T_ri = T_ir.inverse();
    affine_vector_to_matrix(T_jr,this->mean_estimated_absolute_positions.segment<6>(6*j));
    T_rj = T_jr.inverse();
    
    cv::Point2f centre = 0.25*(image_corners[0] + image_corners[1] + image_corners[2] + image_corners[3]);
    Eigen::Vector3d centre_eigen, displaced_centre_i, displaced_centre_j;
    opencv_point_to_eigen_homogeneous(centre_eigen,centre);
    displaced_centre_i = T_ri*centre_eigen;
    displaced_centre_j = T_rj*centre_eigen;
    
    // We compute the diameter of the warepd frame
    double diameter_i(0), diameter_j(0);
    std::vector<Eigen::Vector3d> corners_eigen(4), displaced_corners_i(4), displaced_corners_j(4);
    for (int c=0; c<4; ++c)
    {
        opencv_point_to_eigen_homogeneous(corners_eigen[c],image_corners[c]);
        displaced_corners_i[c] = T_ri*corners_eigen[c];
        displaced_corners_j[c] = T_rj*corners_eigen[c];
        
        Eigen::Vector3d diff_i = displaced_corners_i[c] - displaced_centre_i;
        Eigen::Vector3d diff_j = displaced_corners_j[c] - displaced_centre_j;
        
        diameter_i += 0.5*diff_i.norm();
        diameter_j += 0.5*diff_j.norm();
    }

    // Displacemnt between the warped centres
    Eigen::Vector3d d_i_minus_d_j = displaced_centre_i - displaced_centre_j;
    double norm_displacement = d_i_minus_d_j.norm();
    
    sawhney_score = std::max<double>(0,norm_displacement - 0.5*std::abs(diameter_i - diameter_j))/std::min<double>(diameter_i,diameter_j); // as defined in the paper
    
    // we return something between 0 and 1 which matcheso ur own score
    return std::max<double>(0,1 - sawhney_score);
    
}


// Bundle adjustment directly on the input points! (21.10.2019)
 void compute_matrices_closed_form_bundle_adjustment_on_correspondences(Eigen::MatrixXd& A, Eigen::VectorXd& b, std::vector<Eigen::SparseMatrix<double>>& linear_factor_derivatives, std::vector<Eigen::SparseMatrix<double>>& constant_factor_derivatives, std::vector<Eigen::Triplet<double>>& triplet_list_standard_basis, const std::vector<MeasuredAffineRegistration>& measured_registrations, int nb_frames, int ind_reference_frame)
{
    int nb_parameters_per_frame = 6;
    int nb_measurements = (int)measured_registrations.size();
    
    // Compute the total number of correspondences
    int nb_correspondences_total(0);
    for (int k=0; k<nb_measurements; ++k)
        nb_correspondences_total += (int)(measured_registrations[k].landmarks_input.size()); // number of correspondences for this pair
    
    // Initialisation matrices
    linear_factor_derivatives.resize(2*nb_correspondences_total);
    constant_factor_derivatives.resize(2*nb_correspondences_total);
    A = Eigen::MatrixXd::Zero(6*(nb_frames-1),6*(nb_frames-1));
    b = Eigen::VectorXd::Zero(6*(nb_frames-1));
    Eigen::MatrixXd X_i_tilde_tr, X_j_tilde_tr;
   
    Eigen::SparseMatrix<double> e_i_prime_tr(1,nb_frames-1), e_j_prime_tr(1,nb_frames-1);
    
    // Identity matrix
    Eigen::MatrixXd I_2(2,2);
    I_2.setIdentity();
    
    // List of triplets to compute quickly the (transpose of) the standard vector basis in the dimensions nb_frames - 1
    triplet_list_standard_basis.resize(nb_frames-1);
    for (int k=0; k<(nb_frames-1); ++k)
        triplet_list_standard_basis[k] = Eigen::Triplet<double>(0,k,1);
    
    int m(0);
    for (int k=0; k<nb_measurements; ++k)
    {
        
        // i and j are inversed in comparison to the paper, i.e. frame_indices.first<frame_indices.second (in the paper, i>j since j is the index of the fixed image).
        // Now, we redefine i and j as in the paper by picking i as second index and j as first
        int i = measured_registrations[k].frame_indices.second;  
        int j = measured_registrations[k].frame_indices.first;
        
        int L = (int)measured_registrations[k].landmarks_input.size(); // number of correspondences for this pair
        X_i_tilde_tr.resize(L,3);
        X_j_tilde_tr.resize(L,3);
        for (int l=0; l<L; ++l)
        {
            X_j_tilde_tr(l,0) = measured_registrations[k].landmarks_input[l].x;
            X_j_tilde_tr(l,1) = measured_registrations[k].landmarks_input[l].y;
            X_j_tilde_tr(l,2) = 1;
        }
        for (int l=0; l<L; ++l)
        {
            X_i_tilde_tr(l,0) = measured_registrations[k].landmarks_output[l].x;
            X_i_tilde_tr(l,1) = measured_registrations[k].landmarks_output[l].y;
            X_i_tilde_tr(l,2) = 1;
        }
        
        
    //    std::cout << "(j,i) = (" << j << "," << i << ")" << std::endl;
    //    std::cout << "X_j = " << std::endl << X_j_tilde_tr.transpose() << std::endl;
    //    std::cout << "X_i = " << std::endl << X_i_tilde_tr.transpose() << std::endl;
        
        
        Eigen::SparseMatrix<double> A_ij_term_i(2*L,6*(nb_frames-1)), A_ij_term_j(2*L,6*(nb_frames-1)), A_ij, A_ij_tr, deriv_A_ij(2*L,6*(nb_frames-1)), deriv_A_ij_tr, e_i_prime_tr_kron_I_2, e_j_prime_tr_kron_I_2, deriv_b_ij(2*L,1);
        Eigen::VectorXd b_ij = Eigen::VectorXd::Zero(2*L);
        
        // ---------
        // A_ij
        // ---------
            
        // Compute and add first term of A_ij
        A_ij_term_i.setZero();
        A_ij_term_j.setZero();
        if (j!=ind_reference_frame)
        {
            // Compute the standard basis vector
            int j_prime;
            if (j>ind_reference_frame)
                j_prime = j - 1;
            else
                j_prime = j;
            e_j_prime_tr.setFromTriplets(triplet_list_standard_basis.begin() + j_prime, triplet_list_standard_basis.begin() + j_prime + 1);
                
            computeKroneckerProduct(e_j_prime_tr_kron_I_2,e_j_prime_tr,I_2);
            computeKroneckerProduct(A_ij_term_j,e_j_prime_tr_kron_I_2,X_j_tilde_tr);
        }
            
        // Compute and add second term of A_ij
        if (i!=ind_reference_frame)
        {
            // Compute the standard basis vector
            int i_prime;
            if (i>ind_reference_frame)
                i_prime = i - 1;
            else
                i_prime = i;
            e_i_prime_tr.setFromTriplets(triplet_list_standard_basis.begin() + i_prime, triplet_list_standard_basis.begin() + i_prime + 1);
                
            computeKroneckerProduct(e_i_prime_tr_kron_I_2,e_i_prime_tr,I_2);
            computeKroneckerProduct(A_ij_term_i,e_i_prime_tr_kron_I_2,X_i_tilde_tr);
        }
            
        A_ij = A_ij_term_j - A_ij_term_i;
        
            
        // ----------
        // b_ij 
        // ----------
            
        if (j==ind_reference_frame)
        {
            Eigen::Map<Eigen::VectorXd> vec_X_j(X_j_tilde_tr.data(),2*L);
            b_ij = -vec_X_j;
        //    std::cout << b_ij << std::endl;
        }
            
        if (i==ind_reference_frame)
        {
            Eigen::Map<Eigen::VectorXd> vec_X_i(X_i_tilde_tr.data(),2*L);
            b_ij = vec_X_i;
            std::cout << b_ij << std::endl;
        }
            
            
        // ----------------
        // Update A and b
        // ----------------
        
        A_ij_tr = A_ij.transpose();
        Eigen::SparseMatrix<double> A_ij_tr_A_ij = A_ij_tr*A_ij;
        Eigen::VectorXd A_ij_tr_b_ij = A_ij_tr*b_ij;
        add_sparse_matrix_to_dense_matrix(A,A_ij_tr_A_ij);
        Eigen::VectorXd delta_b = A_ij_tr*b_ij;  
        b = b + delta_b; 
            
        // ---------------
        // Derivatives for covariance propagation
        // ---------------
 
// Before trying to fix the weird patterns
//         Eigen::SparseMatrix<double> X_j_tilde_tr_deriv;
//         for (int l=0; l<L; ++l)
//         {
//             for (int d=0; d<2; ++d)
//             {
//                 int current_index = 2*l + d;
//                 if (j==ind_reference_frame) // the derivative of A_ij is 0, we only compute the derivative of b_ij
//                 {
//                     deriv_A_ij.setZero();
//                     create_sparse_matrix_with_one_nonzero_element(deriv_b_ij,2*L,1,current_index,0,-1);
//                 }
//                 else
//                 {
//                     create_sparse_matrix_with_one_nonzero_element(X_j_tilde_tr_deriv,L,3,l,d,1);
//                     computeSparseKroneckerProduct(deriv_A_ij,e_j_prime_tr_kron_I_2,X_j_tilde_tr_deriv);
//                     deriv_b_ij.setZero();
//                 }
//                 
//                 deriv_A_ij_tr = deriv_A_ij.transpose();
//                 linear_factor_derivatives[m] = deriv_A_ij_tr*A_ij + A_ij_tr*deriv_A_ij;
//                 constant_factor_derivatives[m] = deriv_A_ij_tr*b_ij + A_ij_tr*deriv_b_ij;
//                 
//                 ++m;
//                 
//             }
//             
//         }
        
        
        Eigen::SparseMatrix<double> X_i_tilde_tr_deriv;
        for (int l=0; l<L; ++l)
        {
            for (int d=0; d<2; ++d)
            {
                int current_index = 2*l + d;
                if (i==ind_reference_frame) // the derivative of A_ij is 0, we only compute the derivative of b_ij
                {
                    deriv_A_ij.setZero();
                    create_sparse_matrix_with_one_nonzero_element(deriv_b_ij,2*L,1,d*L + l,0,1);
                }
                else
                {
                    create_sparse_matrix_with_one_nonzero_element(X_i_tilde_tr_deriv,L,3,l,d,-1);
                    computeSparseKroneckerProduct(deriv_A_ij,e_i_prime_tr_kron_I_2,X_i_tilde_tr_deriv);
                    deriv_b_ij.setZero();
                }
                
                deriv_A_ij_tr = deriv_A_ij.transpose();
                linear_factor_derivatives[m] = deriv_A_ij_tr*A_ij + A_ij_tr*deriv_A_ij;
                constant_factor_derivatives[m] = -(deriv_A_ij_tr*b_ij + A_ij_tr*deriv_b_ij);
                
                ++m;
                
            }
            
        }
        
        
        

    }
    
}



// Lots of copy-pasted code from the bundle adjustment function above here...
void update_bundle_adjustment_on_correspondences(Eigen::MatrixXd& A, Eigen::VectorXd& b, Eigen::LLT<Eigen::MatrixXd>& A_cholesky, Eigen::MatrixXd& A_inv, std::vector<Eigen::SparseMatrix<double>>& linear_factor_derivatives, std::vector<Eigen::SparseMatrix<double>>& constant_factor_derivatives, const std::vector<Eigen::Triplet<double>>& triplet_list_standard_basis, int ind_reference_frame, const MeasuredAffineRegistration& new_measured_registration)
{
    int nb_parameters_per_frame = 6;
    int nb_cols_A = A.cols(); // = 6*(nb_frames-1)

    int nb_frames = nb_cols_A/6 + 1;
    
    Eigen::SparseMatrix<double> e_i_prime_tr(1,nb_frames-1), e_j_prime_tr(1,nb_frames-1);
    
    // Identity matrix
    Eigen::MatrixXd I_2(2,2);
    I_2.setIdentity();

    // i and j are inversed in comparison to the paper, i.e. frame_indices.first<frame_indices.second (in the paper, i>j).
    // Now, we redefine i and j as in the paper by picking i as second index and j as first
    int i = new_measured_registration.frame_indices.second;  
    int j = new_measured_registration.frame_indices.first;
    int L = (int)new_measured_registration.landmarks_input.size();

    Eigen::MatrixXd X_i_tilde_tr(L,3), X_j_tilde_tr(L,3);
    for (int l=0; l<L; ++l)
    {
        X_j_tilde_tr(l,0) = new_measured_registration.landmarks_input[l].x;
        X_j_tilde_tr(l,1) = new_measured_registration.landmarks_input[l].y;
        X_j_tilde_tr(l,2) = 1;
    }
    for (int l=0; l<L; ++l)
    {
        X_i_tilde_tr(l,0) = new_measured_registration.landmarks_output[l].x;
        X_i_tilde_tr(l,1) = new_measured_registration.landmarks_output[l].y;
        X_i_tilde_tr(l,2) = 1;
    }
        
    Eigen::SparseMatrix<double> A_ij_term_i(2*L,6*(nb_frames-1)), A_ij_term_j(2*L,6*(nb_frames-1)), A_ij, A_ij_tr, deriv_A_ij(2*L,6*(nb_frames-1)), deriv_A_ij_tr, e_i_prime_tr_kron_I_2, e_j_prime_tr_kron_I_2, deriv_b_ij(2*L,1);
    Eigen::VectorXd b_ij(2*L);
        
    // ---------
    // A_ij
    // ---------
            
    // Compute and add first term of A_ij
    A_ij_term_i.setZero();
    A_ij_term_j.setZero();
    if (j!=ind_reference_frame)
    {
        // Compute the standard basis vector
        int j_prime;
        if (j>ind_reference_frame)
            j_prime = j - 1;
        else
            j_prime = j;
        e_j_prime_tr.setFromTriplets(triplet_list_standard_basis.begin() + j_prime, triplet_list_standard_basis.begin() + j_prime + 1);
                
        computeKroneckerProduct(e_j_prime_tr_kron_I_2,e_j_prime_tr,I_2);
        computeKroneckerProduct(A_ij_term_j,e_j_prime_tr_kron_I_2,X_j_tilde_tr);
    }
            
    // Compute and add second term of A_ij
    if (i!=ind_reference_frame)
    {
        // Compute the standard basis vector
        int i_prime;
        if (i>ind_reference_frame)
            i_prime = i - 1;
        else
            i_prime = i;
        e_i_prime_tr.setFromTriplets(triplet_list_standard_basis.begin() + i_prime, triplet_list_standard_basis.begin() + i_prime + 1);
                
        computeKroneckerProduct(e_i_prime_tr_kron_I_2,e_i_prime_tr,I_2);
        computeKroneckerProduct(A_ij_term_i,e_i_prime_tr_kron_I_2,X_i_tilde_tr);
    }
            
    A_ij = A_ij_term_j - A_ij_term_i;
            
            
    // ----------
    // b_ij 
    // ----------
            
    b_ij.setZero();
    if (j==ind_reference_frame)
    {
        Eigen::Map<Eigen::VectorXd> vec_X_j(X_j_tilde_tr.block(0,0,L,2).data(),2*L);
        b_ij = -vec_X_j;
    }
            
    if (i==ind_reference_frame)
    {
        Eigen::Map<Eigen::VectorXd> vec_X_i(X_i_tilde_tr.block(0,0,L,2).data(),2*L);
        b_ij = vec_X_i;
    }
            
            
    // ----------------
    // Update A and b
    // ----------------
        
        A_ij_tr = A_ij.transpose();
        add_sparse_matrix_to_dense_matrix(A,A_ij_tr*A_ij);
        b += A_ij_tr*b_ij;  
            
    // ---------------
    // Derivatives for covariance propagation
    // ---------------
   
//     Eigen::SparseMatrix<double> X_j_tilde_tr_deriv;
//     for (int l=0; l<L; ++l)
//     {
//         for (int d=0; d<2; ++d)
//         {
//             int current_index = 2*l + d;
//             if (j==ind_reference_frame) // the derivative of A_ij is 0, we only compute the derivative of b_ij
//             {
//                 deriv_A_ij.setZero();
//                 create_sparse_matrix_with_one_nonzero_element(deriv_b_ij,2*L,1,current_index,0,-1);
//             }
//             else
//             {
//                 create_sparse_matrix_with_one_nonzero_element(X_j_tilde_tr_deriv,L,3,l,d,1);
//                 computeSparseKroneckerProduct(deriv_A_ij,e_j_prime_tr_kron_I_2,X_j_tilde_tr_deriv);
//                 deriv_b_ij.setZero();
//             }
//                 
//             deriv_A_ij_tr = deriv_A_ij.transpose();
//             linear_factor_derivatives.push_back(deriv_A_ij_tr*A_ij + A_ij_tr*deriv_A_ij);
//             constant_factor_derivatives.push_back(deriv_A_ij_tr*b_ij + A_ij_tr*deriv_b_ij);
//                 
//         }
//             
//     }
        Eigen::SparseMatrix<double> X_i_tilde_tr_deriv;
        for (int l=0; l<L; ++l)
        {
            for (int d=0; d<2; ++d)
            {
                int current_index = 2*l + d;
                if (i==ind_reference_frame) // the derivative of A_ij is 0, we only compute the derivative of b_ij
                {
                    deriv_A_ij.setZero();
                    create_sparse_matrix_with_one_nonzero_element(deriv_b_ij,2*L,1,d*L + l,0,1);
                }
                else
                {
                    create_sparse_matrix_with_one_nonzero_element(X_i_tilde_tr_deriv,L,3,l,d,-1);
                    computeSparseKroneckerProduct(deriv_A_ij,e_i_prime_tr_kron_I_2,X_i_tilde_tr_deriv);
                    deriv_b_ij.setZero();
                }
                
                deriv_A_ij_tr = deriv_A_ij.transpose();
                linear_factor_derivatives.push_back(deriv_A_ij_tr*A_ij + A_ij_tr*deriv_A_ij);
                constant_factor_derivatives.push_back(-(deriv_A_ij_tr*b_ij + A_ij_tr*deriv_b_ij));                
            }
            
        }
        
    
    // ---------------------
    // Low-rank update of the Cholesky decompositon of A
    // ---------------------
    Eigen::SparseQR<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<int>> A_ij_QR(A_ij_tr);
    int rank = A_ij_QR.rank();
    Eigen::MatrixXd Q = Eigen::MatrixXd(A_ij_QR.matrixQ());
    Eigen::MatrixXd R = Eigen::MatrixXd(A_ij_QR.matrixR());
    Eigen::MatrixXd Q_truncated =  Q.block(0,0,A_ij_tr.rows(),rank);
    Eigen::MatrixXd R_truncated =  R.block(0,0,rank,R.cols());
   // std::cout << R << std::endl;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(R_truncated*(R_truncated.transpose()));
    Eigen::MatrixXd V = Q_truncated*es.eigenvectors();
    Eigen::VectorXd D = es.eigenvalues();
    Eigen::VectorXd A_inv_v;
    for (int r=0; r<rank; ++r)
    {
        // Update Cholesky decomposition
        A_cholesky.rankUpdate(V.col(r),D(r));
        
        // Update inverse (Sherman-Morrison formula)
        A_inv_v = A_inv*(V.col(r));
        double denom = 1 + D(r)*(V.col(r).transpose())*A_inv_v;
        A_inv -= (D(r)/denom)*A_inv_v*(A_inv_v.transpose());
    }
}



void createDiagonalCovariancesCorrespondences(std::vector<Eigen::MatrixXd>& factors_covariance_matrices, const std::vector<MeasuredAffineRegistration>& measured_registrations)
{
    int nb_measurements = (int)measured_registrations.size();
    factors_covariance_matrices.resize(nb_measurements);
    for (int ind_measurement=0; ind_measurement<nb_measurements; ++ind_measurement)
    {
        int nb_landmarks = (int)measured_registrations[ind_measurement].landmarks_input.size();
        factors_covariance_matrices[ind_measurement] = Eigen::MatrixXd(2*nb_landmarks,2*nb_landmarks);
        factors_covariance_matrices[ind_measurement].setIdentity();
    }
}


void fill_F_uv(Eigen::MatrixXd& F_uv, const std::vector<Eigen::SparseMatrix<double>>& linear_factors_derivatives, const std::vector<Eigen::SparseMatrix<double>>& constant_factors_derivatives, const Eigen::VectorXd& estimated_global_positions_without_ref)
{
    int nb_rows = estimated_global_positions_without_ref.rows();
    int nb_cols = linear_factors_derivatives.size();
    F_uv.resize(nb_rows,nb_cols);
    for (int j=0; j<nb_cols; ++j)
        F_uv.block(0,j,nb_rows,1) = linear_factors_derivatives[j]*estimated_global_positions_without_ref + constant_factors_derivatives[j];
}


// Switches parametrisation from 6(N-1) to 6N by adding the identity
void add_reference_frame_to_global_positions_affine(Eigen::VectorXd& global_positions_with_ref, const Eigen::VectorXd& global_positions_without_ref, int ind_reference_frame)
{
    int nb_parameters_per_frame = 6;
    int nb_frames = global_positions_without_ref.rows()/nb_parameters_per_frame + 1;
    global_positions_with_ref = Eigen::VectorXd::Zero(nb_frames*nb_parameters_per_frame);
    int i_without_ref(0);
    for (int i_with_ref=0; i_with_ref<nb_frames; ++i_with_ref)
    {
        if (i_with_ref==ind_reference_frame)
        {
            global_positions_with_ref(6*i_with_ref) = 1;
            global_positions_with_ref(6*i_with_ref+4) = 1;
        }
        else
        {
            global_positions_with_ref.segment<6>(6*i_with_ref) = global_positions_without_ref.segment<6>(6*i_without_ref);
            ++i_without_ref;
        }
            
    }
}

void add_reference_frame_to_global_covariance_matrix_affine(Eigen::MatrixXd& global_cov_matrix_with_ref, const Eigen::MatrixXd& global_cov_matrix_without_ref, int ind_reference_frame)
{
    double eps = 0.0000000000000001;
    int nb_parameters_per_frame = 6;
    Eigen::MatrixXd I(nb_parameters_per_frame,nb_parameters_per_frame);
    I.setIdentity();
    int nb_frames = global_cov_matrix_without_ref.rows()/nb_parameters_per_frame + 1;
    global_cov_matrix_with_ref = Eigen::MatrixXd::Zero(nb_frames*nb_parameters_per_frame,nb_frames*nb_parameters_per_frame);
    
    global_cov_matrix_with_ref.block(nb_parameters_per_frame*ind_reference_frame,nb_parameters_per_frame*ind_reference_frame,nb_parameters_per_frame,nb_parameters_per_frame) = eps*I;
    if (ind_reference_frame>0) // top left corner
        global_cov_matrix_with_ref.block(0,0,ind_reference_frame*nb_parameters_per_frame,ind_reference_frame*nb_parameters_per_frame) = global_cov_matrix_without_ref.block(0,0,ind_reference_frame*nb_parameters_per_frame,ind_reference_frame*nb_parameters_per_frame);
    if (ind_reference_frame<(nb_frames-1)) // bottom right corner
        global_cov_matrix_with_ref.block((ind_reference_frame+1)*nb_parameters_per_frame,(ind_reference_frame+1)*nb_parameters_per_frame,(nb_frames-ind_reference_frame-1)*nb_parameters_per_frame,(nb_frames-ind_reference_frame-1)*nb_parameters_per_frame) = global_cov_matrix_without_ref.block(ind_reference_frame,ind_reference_frame,(nb_frames-ind_reference_frame-1)*nb_parameters_per_frame,(nb_frames-ind_reference_frame-1)*nb_parameters_per_frame);
    if ((ind_reference_frame>0) && (ind_reference_frame<(nb_frames-1)))
        std::cout << "In add_reference_frame_to_global_covariance_matrix_affine: we still need to implement the case of non extremal reference frames (add bottom left and top right corner" << std::endl;
    
}


void inverse_vector_of_estimated_affine_global_positions(Eigen::VectorXd& input_vec_inv, const Eigen::VectorXd& input_vec)
{
    int size_vec = input_vec.rows();
    input_vec_inv.resize(size_vec);
    int nb_frames = input_vec.rows()/6;
    Eigen::Matrix3d T =  Eigen::Matrix3d::Zero();
    Eigen::Matrix3d T_inv;
    T(2,2) = 1;
    for (int n=0; n<nb_frames; ++n)
    {
        Eigen::Map<const Eigen::MatrixXd> T_block(input_vec.segment<6>(6*n).data(),3,2);
        T.block<3,2>(0,0) = T_block;
        T_inv = T.inverse();
        Eigen::Map<const Eigen::VectorXd> param_inv(T_inv.data(),6);
        input_vec_inv.segment<6>(6*n) = param_inv;
    }
}



void retain_blocks_i_and_j_in_covariance_matrix(Eigen::MatrixXd& covariance_matrix_ij, const Eigen::MatrixXd& full_covariance_matrix, int i, int j)
{
    covariance_matrix_ij = Eigen::MatrixXd(12,12);
    covariance_matrix_ij.block<6,6>(0,0) = full_covariance_matrix.block<6,6>(6*i,6*i);
    covariance_matrix_ij.block<6,6>(6,0) = full_covariance_matrix.block<6,6>(6*j,6*i);
    covariance_matrix_ij.block<6,6>(0,6) = full_covariance_matrix.block<6,6>(6*i,6*j);
    covariance_matrix_ij.block<6,6>(6,6) = full_covariance_matrix.block<6,6>(6*j,6*j);
}

void compute_closed_form_overlap_probability_and_bounds(double& closed_form_overlap_probability_low, double& closed_form_overlap_probability_up, double& uncertainty_reward, const Eigen::Matrix2d& covariance_ij, double size_square_interior, double size_square_exterior, const Eigen::Vector2d& mean_vector_ij, RewardType chosen_reward)
{
    // Compute the lower and upper bound using the eigenvectors, i.e better bounds but probably more complex to compute and optimise
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es;
    es.computeDirect(covariance_ij);
    Eigen::Matrix2d cov_eigenvectors = es.eigenvectors();
    Eigen::Matrix2d cov_eigenvectors_tr = cov_eigenvectors.transpose();
    Eigen::Vector2d dot_products = cov_eigenvectors_tr*mean_vector_ij;
    Eigen::Vector2d eigenvalues = es.eigenvalues();
 //   std::cout << "Eigenvalues: " << eigenvalues << std::endl;
 //   std::cout << "Dot products: " << dot_products << std::endl;
    closed_form_overlap_probability_low = 1;
    closed_form_overlap_probability_up = 1;

    double half_size_square_interior = 0.5*size_square_interior;
    double half_size_square_exterior = 0.5*size_square_exterior;

    
    for (int d=0; d<2; ++d)
    {
        if (eigenvalues[d]<=0)
        {
            std::cout << "A supposedly positive eigenvalue is equal to " << eigenvalues[d] << std::endl;
            std::cout << "Corresponding matrix: " << covariance_ij << std::endl;
        }
        

        closed_form_overlap_probability_low  = closed_form_overlap_probability_low*0.5*(erf((dot_products[d] + half_size_square_interior/std::sqrt(2))/(sqrt(2*eigenvalues[d]))) - erf((dot_products[d] - half_size_square_interior/std::sqrt(2))/(sqrt(2*eigenvalues[d]))));
        closed_form_overlap_probability_up  = closed_form_overlap_probability_up*0.5*(erf((dot_products[d] + half_size_square_exterior)/(sqrt(2*eigenvalues[d]))) - erf((dot_products[d] - half_size_square_exterior)/(sqrt(2*eigenvalues[d]))));
    }
    

    if (chosen_reward == our_reward)
        uncertainty_reward = sqrt(covariance_ij.determinant());
        //uncertainty_reward = eigenvalues(0) + eigenvalues(1);
    else
    {
        if (chosen_reward == entropy_reward)
            uncertainty_reward = std::log(covariance_ij.determinant());
    }
}

void estimateOverlapProbabilitiesWithSampling(double& probability_overlap, const Eigen::Vector2d& mean_vector_ij,
                const Eigen::Matrix2d& covariance_ij, int nb_samples, boost::random::mt19937& rng, const std::vector<cv::Point2f>& corners)
{
    // Cholesky decomposition
    Eigen::Matrix2d L = covariance_ij.llt().matrixL();

    // Boost normal distribution sampler
    boost::normal_distribution<double> normal_dist(0.0,1.0);

    // Initialise geometric corners
    cv::Point2f centre, sample_cv;
    centre = (0.25)*(corners[0] + corners[1] + corners[2] + corners[3]);

    // Sampling
    Eigen::MatrixXd normal_samples(2,1), sample;
    int count_j_in_i(0);
    for (int ind=0; ind<nb_samples; ++ind)
    {
        // Sample 12 normal random variables
        for (int s=0; s<2; ++s)
            normal_samples(s,0) = normal_dist(rng);

        // This is the sample for the desired 2x2 distrbution
        sample = mean_vector_ij + L*normal_samples;
        sample_cv = cv::Point2f(sample(0),sample(1));
        
        if (is_point_inside_parallelogram(centre + sample_cv,corners))
            ++count_j_in_i;
    }

    probability_overlap = ((double)count_j_in_i)/((double)nb_samples);
}


void estimateOverlapProbabilitiesWithSampling(double& probability_overlap, const Eigen::Vector2d& mean_vector_ij,
                const Eigen::Matrix2d& covariance_ij, int nb_samples, boost::random::mt19937& rng, double radius_overlap)
{
    // Cholesky decomposition
    Eigen::Matrix2d L = covariance_ij.llt().matrixL();

    // Boost normal distribution sampler
    boost::normal_distribution<double> normal_dist(0.0,1.0);

    // Sampling
    Eigen::Vector2d normal_sample, sample;
    int count_j_in_i(0);
    for (int ind=0; ind<nb_samples; ++ind)
    {
        // Sample 12 normal random variables
        for (int s=0; s<2; ++s)
            normal_sample(s) = normal_dist(rng);

        // This is the sample for the desired 2x2 distrbution
        sample = mean_vector_ij + L*normal_sample;
        if (sample.norm()<=radius_overlap)
            ++count_j_in_i;
    }

    probability_overlap = ((double)count_j_in_i)/((double)nb_samples);
}




// void PositionOverlapModel::reestimate_variances(const std::vector<LabeledImagePair>& labeled_pairs, const std::vector<Eigen::MatrixXd>& initial_scaled_factors_covariance_matrices_pairwise_measurements, const std::vector<Eigen::MatrixXd>& scaled_factors_covariance_matrices_pairwise_measurements, const Eigen::MatrixXd& initial_left_cov_propagation_global_cov_matrix, const Eigen::MatrixXd& left_cov_propagation_global_cov_matrix, const std::vector<int>& measurement_modes, const std::map<std::pair<int,int>,Eigen::MatrixXd>& left_cov_propagations_relative_cov_matrix, std::map<std::pair<int,int>,double>& displacement_ij, std::map<std::pair<int,int>,Eigen::Vector2d>& mean_vector_ij, InputData& input_data, Settings& settings)
// {
//     int nb_modes = (int)estimated_variances.size();
//     double half_size_interior = 0.5*(1/settings.factor_overlap_area)*std::min<int>(input_data.nb_rows,input_data.nb_cols);
//     double half_size_exterior = 0.5*settings.factor_overlap_area*std::max<int>(input_data.nb_rows,input_data.nb_cols);
//     if (settings.reestimate_from_initial_measurements_only)
//     {
//         int nb_initial_modes(1), nb_optimised_variances(1);
//         if (nb_modes>1)
//         {
//             nb_initial_modes = nb_modes - 1;
//             nb_optimised_variances = nb_modes-1;
//         }
//         reestimate_variances_generic(estimated_variances,labeled_pairs,initial_scaled_factors_covariance_matrices_pairwise_measurements,initial_left_cov_propagation_global_cov_matrix,measurement_modes,nb_initial_modes,nb_optimised_variances,left_cov_propagations_relative_cov_matrix,displacement_ij,mean_vector_ij,half_size_interior,half_size_exterior,settings.ind_reference_frame,settings.annotation_variance);
//     }
//     else
//     {
//         int nb_optimised_variances = std::max<int>(1,nb_modes-1);
//         reestimate_variances_generic(estimated_variances,labeled_pairs,scaled_factors_covariance_matrices_pairwise_measurements,left_cov_propagation_global_cov_matrix,measurement_modes,nb_modes,nb_optimised_variances,left_cov_propagations_relative_cov_matrix,displacement_ij,mean_vector_ij,half_size_interior,half_size_exterior,settings.ind_reference_frame,settings.annotation_variance);
//     }
//     
// }
// 




// x is the inverse of the standard deviation
void compute_hyperbolic_loss_one_variable(const alglib::real_1d_array &x, double &func, alglib::real_1d_array &grad, void *ptr)
{
    bool do_we_want_to_check_gradients = false; // set to true to debug and check the gradients numerically, false for best performance
    bool do_concave_approximation = false;
    
    bool check_gradients;
    if (do_we_want_to_check_gradients)
        check_gradients = grad[0]!=(-1);
    else
        check_gradients = false;
    
    
    // inverse of standard deviation
    double inv_std = x[0];
    
    VarianceOptimisation* struct_inf_var = static_cast<VarianceOptimisation*>(ptr);

    std::vector<LabeledImagePair> labeled_pairs = struct_inf_var->_labeled_pairs;
    std::map<std::pair<int,int>,Eigen::Vector2d> *scaled_eigenvalues_ptr = struct_inf_var->_scaled_eigenvalues_ptr;
    std::map<std::pair<int,int>,Eigen::Vector2d> *dot_products_ptr = struct_inf_var->_dot_products_ptr;
   
    
    double half_size_interior = struct_inf_var->_half_size_interior;
    double half_size_exterior = struct_inf_var->_half_size_exterior;
    int nb_labeled_pairs = labeled_pairs.size();
    func = 0;
    grad[0] = 0;
    double instance_weight_non_overlap = struct_inf_var->_instance_weights[0];
    double instance_weight_overlap = struct_inf_var->_instance_weights[1];
    
    for (int ind_pair=0; ind_pair<nb_labeled_pairs; ++ind_pair)
    {
        bool overlap = labeled_pairs[ind_pair].this_pair_overlaps;
        std::pair<int,int> image_indices = labeled_pairs[ind_pair].image_indices;
        double instance_weight;
        if (overlap)
            instance_weight = instance_weight_overlap;
        else
            instance_weight = instance_weight_non_overlap;
        
        double added_log, added_grad, dummy_1, dummy_2;
        if (overlap)
        {
            compute_log_overlap_bound_hyperbolic_one_variance(added_log,added_grad,inv_std,half_size_interior,scaled_eigenvalues_ptr->at(image_indices),dot_products_ptr->at(image_indices));
  //          std::cout << "Overlap: Added grad = " << added_grad << std::endl;
        }
        else
        {
            if (do_concave_approximation)
                compute_log_non_overlap_bound_hyperbolic_one_variance(dummy_1,dummy_2,added_log,added_grad,inv_std,half_size_exterior,scaled_eigenvalues_ptr->at(image_indices),dot_products_ptr->at(image_indices));
            else
                compute_log_non_overlap_bound_hyperbolic_one_variance(added_log,added_grad,dummy_1,dummy_2,inv_std,half_size_exterior,scaled_eigenvalues_ptr->at(image_indices),dot_products_ptr->at(image_indices));
  //          std::cout << "No overlap: Added grad = " << added_grad << std::endl;
        }
        
//         if (overlap)
//             std::cout << "Pair " << ind_pair << " overlaps: Log = " << added_log << " ; Grad = " << added_grad << std::endl;
//         else
//             std::cout << "Pair " << ind_pair << " does not overlap: Log = " << added_log << " ; Grad = " << added_grad << std::endl;
        func -= instance_weight*added_log;
       // grad[0] -= instance_weight_overlap*added_grad*inv_std;
        grad[0] -= instance_weight*added_grad;
    }
    
    
 //   std::cout << "Inverse standard deviation: " << x[0] << " - Loss: " << func << std::endl;
    
     // Check gradients empirically
    if (check_gradients)
    {
        int dim_total=1;
        double eps = 0.000000001;
        for (int p=0; p<dim_total; ++p)
        {
            alglib::real_1d_array x_eps, dummy_grad;
            double func_eps, flag_grad(-1);
            x_eps.setcontent(dim_total,&(x[0]));
            dummy_grad.setcontent(dim_total,&flag_grad);
            x_eps[p] = x[p] + eps;
            compute_hyperbolic_loss_one_variable(x_eps,func_eps,dummy_grad,ptr);
            double empirical_grad_p = (func_eps - func)/eps;
            std::cout << "Parameter " << p << ": Empirical grad: " << empirical_grad_p << " ; Computed grad: " << grad[p] << std::endl;
        }    
    }

}

void compute_log_overlap_bound_hyperbolic_one_variance(double& log_bound, double& derivative_log_bound, double inverse_current_std, double half_size_interior, const Eigen::Vector2d& scaled_eigenvalues, const Eigen::Vector2d& dot_products)
{
    double beta = std::sqrt(2/((double)CV_PI));
    log_bound = 0;
    derivative_log_bound = 0;
    for (int k=0; k<2; ++k)
    {
        double a_k = beta*half_size_interior/(std::sqrt(scaled_eigenvalues(k)));
        double x_k = beta*dot_products(k)/(std::sqrt(scaled_eigenvalues(k)));
        
        // Log of the bound
        log_bound = log_bound + log_sinh(2*a_k*inverse_current_std) - log_cosh((x_k-a_k)*inverse_current_std) -  log_cosh((x_k+a_k)*inverse_current_std) - std::log(2);
        
        // Closed form derivative
        derivative_log_bound = derivative_log_bound + 2*a_k/my_tanh(2*a_k*inverse_current_std) - (x_k - a_k)*my_tanh((x_k-a_k)*inverse_current_std) - (x_k + a_k)*my_tanh((x_k + a_k)*inverse_current_std);
    }
}

void compute_log_non_overlap_bound_hyperbolic_one_variance(double& log_bound, double& derivative_log_bound, double& concave_log_bound, double& derivative_concave_log_bound, double inverse_current_std, double half_size_exterior, const Eigen::Vector2d& scaled_eigenvalues, const Eigen::Vector2d& dot_products)
{
    double beta = std::sqrt(2/((double)CV_PI));
    double a_1 = beta*half_size_exterior/(std::sqrt(scaled_eigenvalues(0)));
    double x_1 = beta*dot_products(0)/(std::sqrt(scaled_eigenvalues(0)));
    double a_2 = beta*half_size_exterior/(std::sqrt(scaled_eigenvalues(1)));
    double x_2 = beta*dot_products(1)/(std::sqrt(scaled_eigenvalues(1)));
    
    x_1 = std::abs(x_1);
    x_2 = std::abs(x_2);
    double tangent_extraction_point = 1; // should be high if one wants the asymptote as tangent
    
    double in_log_upper_part, deriv_in_log_upper_part;
    compute_in_log_convex_upper_part(in_log_upper_part,deriv_in_log_upper_part,inverse_current_std,x_1,x_2,a_1,a_2);
    
 //   std::cout << "In log upper part: " << in_log_upper_part << std::endl;
//  std::cout << "Deriv in log upper part: " << deriv_in_log_upper_part << std::endl;
    
    log_bound = std::log(in_log_upper_part) - log_cosh((x_1-a_1)*inverse_current_std) -  log_cosh((x_1+a_1)*inverse_current_std) - log_cosh((x_2-a_2)*inverse_current_std) -  log_cosh((x_2+a_2)*inverse_current_std)  - 2*std::log(2);
    
    derivative_log_bound = deriv_in_log_upper_part/in_log_upper_part -  (x_1 - a_1)*my_tanh((x_1 - a_1)*inverse_current_std) -  (x_1 + a_1)*my_tanh((x_1 + a_1)*inverse_current_std) -  (x_2 - a_2)*my_tanh((x_2-a_2)*inverse_current_std) -  (x_2 + a_2)*my_tanh((x_2 + a_2)*inverse_current_std);
    
    double in_log_upper_part_tangent_point, deriv_in_log_upper_part_tangent_point;
    compute_in_log_convex_upper_part(in_log_upper_part_tangent_point,deriv_in_log_upper_part_tangent_point,tangent_extraction_point,x_1,x_2,a_1,a_2);
    
  //  std::cout << "In log upper part tangent: " << in_log_upper_part_tangent_point << std::endl;
  //  std::cout << "Deriv in log upper part tangent: " << deriv_in_log_upper_part_tangent_point << std::endl;
    
    concave_log_bound = log_bound - std::log(in_log_upper_part) + (deriv_in_log_upper_part_tangent_point/in_log_upper_part_tangent_point)*(inverse_current_std - tangent_extraction_point) + std::log(in_log_upper_part_tangent_point); // we take log bound, remove its "log(upper_part)" term,  and replace it by the value of the tangent lower bounding it
    
    derivative_concave_log_bound = derivative_log_bound - deriv_in_log_upper_part/in_log_upper_part + deriv_in_log_upper_part_tangent_point/in_log_upper_part_tangent_point; // same idea with the derivatives
}

void compute_in_log_convex_upper_part(double& res, double& derivative, double inverse_estimated_std, double x_1, double x_2, double a_1, double a_2)
{
//     std::cout << "x_1: " << x_1 << std::endl;
//     std::cout << "x_2: " << x_2 << std::endl;
//     std::cout << "a_1: " << a_1 << std::endl;
//     std::cout << "a_2: " << a_2 << std::endl;
    
    res = std::cosh(2*x_1*inverse_estimated_std)*std::cosh(2*x_2*inverse_estimated_std) + std::cosh(2*x_1*inverse_estimated_std)*std::cosh(2*a_2*inverse_estimated_std) + std::cosh(2*a_1*inverse_estimated_std)*std::cosh(2*x_2*inverse_estimated_std) + std::cosh(2*(a_1 - a_2)*inverse_estimated_std);
    
    derivative = (x_1 + x_2)*std::sinh(2*(x_1 + x_2)*inverse_estimated_std) + (x_1 - x_2)*std::sinh(2*(x_1 - x_2)*inverse_estimated_std) + (x_1 + a_2)*std::sinh(2*(x_1 + a_2)*inverse_estimated_std) + (x_1 - a_2)*std::sinh(2*(x_1 - a_2)*inverse_estimated_std) + (a_1 + x_2)*std::sinh(2*(a_1 + x_2)*inverse_estimated_std) + (a_1 - x_2)*std::sinh(2*(a_1 - x_2)*inverse_estimated_std) + 2*(a_1 - a_2)*std::sinh(2*(a_1 - a_2)*inverse_estimated_std);
}

// Numerically stable for large x
double log_cosh(double x)
{
    if (x<0)
        return log_cosh(-x);
    else
        return (x + std::log(1 + std::exp(-2*x)) - std::log(2));
}

double log_sinh(double x)
{
    return (x + std::log(1 - std::exp(-2*x)) - std::log(2));
}

double my_tanh(double x)
{
    if (x<0)
        return (-my_tanh(-x));
    else
    {
        double exp_term = std::exp(-2*x);
        return ((1-exp_term)/(1 + exp_term));
    }
}



// Previous one
// void reestimate_variances_generic(std::vector<double>& estimated_variances, const std::vector<LabeledImagePair>& labeled_pairs, const std::vector<Eigen::MatrixXd>& scaled_factors_covariance_matrices_pairwise_measurements, const Eigen::MatrixXd& left_cov_propagation_global_cov_matrix, const std::vector<int>& measurement_modes, int nb_modes, int nb_optimised_variances,
//                             const std::map<std::pair<int,int>,Eigen::MatrixXd>& left_cov_propagations_relative_cov_matrix, std::map<std::pair<int,int>,double>& displacement_ij, std::map<std::pair<int,int>,Eigen::Vector2d>& mean_vector_ij,
//                             double half_size_interior, double half_size_exterior, int ind_reference_frame, double annotation_variance)
// {
//     
// 
//     
//     Eigen::MatrixXd relative_covariance_mode(12,12);
//     Eigen::Matrix2d covariance_matrix_centre_displacement;
//     Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es;
//     int nb_labeled_pairs = labeled_pairs.size();
//     
//     // Precomputed values of trace, eigenvalues, etc.
//     std::map<std::pair<int,int>,Eigen::VectorXd> traces_relative_modes, min_eigenvalues_relative_modes, max_eigenvalues_relative_modes;
//     std::map<std::pair<int,int>,Eigen::Vector2d> scaled_eigenvalues, dot_products;
//     std::map<std::pair<int,int>,Eigen::MatrixXd> covariance_matrix_relative_modes;
//     std::vector<double> instance_weights(2);
//     std::vector<int> label_count(2,0);
//     // Initialise
//     for (int ind_pair=0; ind_pair<nb_labeled_pairs; ++ind_pair)
//     {
//         int i = labeled_pairs[ind_pair].image_indices.first;
//         int j = labeled_pairs[ind_pair].image_indices.second;
//         if (labeled_pairs[ind_pair].this_pair_overlaps)
//             ++label_count[1];
//         else
//             ++label_count[0];
//         Eigen::VectorXd for_init(nb_modes);
//         Eigen::Vector2d for_init_2;
//         Eigen::MatrixXd for_init_mat(2,2*nb_modes);
//         traces_relative_modes[std::pair<int,int>(i,j)] = for_init;
//         min_eigenvalues_relative_modes[std::pair<int,int>(i,j)] = for_init;
//         max_eigenvalues_relative_modes[std::pair<int,int>(i,j)] = for_init;
//         covariance_matrix_relative_modes[std::pair<int,int>(i,j)] = for_init_mat;
//         scaled_eigenvalues[std::pair<int,int>(i,j)] = for_init_2;
//         dot_products[std::pair<int,int>(i,j)] = for_init_2;
//     }
//     
//     std::cout << "No class imbalance in variance reestimation" << std::endl;
//     for (int l=0; l<2; ++l)
//         instance_weights[l] = 1; 
//     
//     // Precompute relevant quantities (trace, eigenvalues etc) for each mode
//     //std::cout << "Precompute quantities" << std::endl;
//     Eigen::MatrixXd covariance_matrix_mode_without_ref, covariance_matrix_mode, covariance_matrix_mode_without_ref_factor;
//     for (int m=0; m<nb_modes; ++m)
//     {
//         
//         // We compute the factor of the global covariance matrix which correpsonds to the 
//         product_dense_blockdiagonal_m_th_mode( covariance_matrix_mode_without_ref_factor,m,left_cov_propagation_global_cov_matrix,scaled_factors_covariance_matrices_pairwise_measurements,measurement_modes);
//         
//         // Multiply by its transpose to form the m-th mode of the covariance matrix
//         covariance_matrix_mode_without_ref = covariance_matrix_mode_without_ref_factor*(covariance_matrix_mode_without_ref_factor.transpose());
//         
//         // Add the reference frame
//         add_reference_frame_to_global_covariance_matrix_affine(covariance_matrix_mode, covariance_matrix_mode_without_ref,ind_reference_frame);
//      
//         // Now, we precompute for each training sample the corresponding relative modes for each variance to optimise - the remaining modes are aggregated in a constant
//         for (int ind_pair=0; ind_pair<nb_labeled_pairs; ++ind_pair)
//         {
//       //      std::cout << "Process pair " << ind_pair << std::endl;
//             int i = labeled_pairs[ind_pair].image_indices.first;
//             int j = labeled_pairs[ind_pair].image_indices.second;
//             Eigen::MatrixXd left_cov_propagation = left_cov_propagations_relative_cov_matrix.at(std::pair<int,int>(i,j));
//             Eigen::MatrixXd left_cov_propagation_tr = left_cov_propagation.transpose();
//             
//         //    std::cout << "Compute relative covariance matrix" << std::endl;
//             relative_covariance_mode.block<6,6>(0,0) = covariance_matrix_mode.block<6,6>(6*i,6*i);
//             relative_covariance_mode.block<6,6>(6,0) = covariance_matrix_mode.block<6,6>(6*j,6*i);
//             relative_covariance_mode.block<6,6>(0,6) = covariance_matrix_mode.block<6,6>(6*i,6*j);
//             relative_covariance_mode.block<6,6>(6,6) = covariance_matrix_mode.block<6,6>(6*j,6*j);
// 
//          //   std::cout << "Propagate" << std::endl;
//             covariance_matrix_centre_displacement = left_cov_propagation*relative_covariance_mode*left_cov_propagation_tr;
//             
//          //   std::cout << "Compute eigenvalues" << std::endl;
//             es.computeDirect(covariance_matrix_centre_displacement);
//             Eigen::Vector2d eigenvalues = es.eigenvalues();
//             Eigen::Matrix2d cov_eigenvectors = es.eigenvectors();
//             Eigen::Matrix2d cov_eigenvectors_tr = cov_eigenvectors.transpose();            
//             
//          //   std::cout << "Update maps" << ind_pair << std::endl;
//             traces_relative_modes[std::pair<int,int>(i,j)](m) = covariance_matrix_centre_displacement.trace();
//             min_eigenvalues_relative_modes[std::pair<int,int>(i,j)](m) = eigenvalues.minCoeff();
//             max_eigenvalues_relative_modes[std::pair<int,int>(i,j)](m) = eigenvalues.maxCoeff();
//             covariance_matrix_relative_modes[std::pair<int,int>(i,j)].block<2,2>(0,2*m) = covariance_matrix_centre_displacement;
//             scaled_eigenvalues[std::pair<int,int>(i,j)] = eigenvalues;
//             dot_products[std::pair<int,int>(i,j)] = cov_eigenvectors_tr*mean_vector_ij.at(std::pair<int,int>(i,j));
//                 
// 
//             
// //             if (min_eigenvalues[ind_mode]>=0.0001)
// //             {
// //      //           std::cout << "Significant min eigenvalue: " << min_eigenvalues[ind_mode] << " for mode " << ind_mode << std::endl;
// //      //           std::cout << "Covariance matrix: " << std::endl;
// //      //           std::cout << covariance_matrix_centre_displacement << std::endl;
// //                 
// //             }
// //             else
// //             {
// //                 min_eigenvalues[ind_mode] = 0;
// //                 if (max_eigenvalues[ind_mode]<0.0001)
// //                     max_eigenvalues[ind_mode] = 0;
// //             }
//             
//             
//             
//         }
//         
//     }
//         
//         
// 
//  //   std::cout << "Start optimising" << std::endl;
//     Eigen::VectorXd current_estimate_variances(nb_modes);
//     for (int ind_mode=0; ind_mode<nb_modes; ++ind_mode)
//         current_estimate_variances[ind_mode] = estimated_variances[ind_mode];
//         // current_estimate_variances[ind_mode] = 1;
// 
//     VarianceOptimisation var_optimisation;
//     var_optimisation._labeled_pairs = labeled_pairs;
//     var_optimisation._displacement_map_ptr = &displacement_ij;
//     var_optimisation._mean_vector_ptr = &mean_vector_ij;
//     var_optimisation._traces_relative_modes_ptr = &traces_relative_modes;
//     var_optimisation._min_eigenvalues_relative_modes_ptr = &min_eigenvalues_relative_modes;
//     var_optimisation._max_eigenvalues_relative_modes_ptr = &max_eigenvalues_relative_modes;
//     var_optimisation._scaled_eigenvalues_ptr = &scaled_eigenvalues;
//     var_optimisation._dot_products_ptr = &dot_products;
//     var_optimisation._half_size_interior = half_size_interior;
//     var_optimisation._half_size_exterior = half_size_exterior;
//     var_optimisation._nb_modes = nb_modes;
//     var_optimisation._covariance_matrix_relative_modes_ptr = &covariance_matrix_relative_modes;
//     var_optimisation._nb_optimised_variances = nb_optimised_variances;
//     var_optimisation._current_estimated_variances = current_estimate_variances;
//     var_optimisation._instance_weights = instance_weights;
//     void *var_optimisation_ptr;
//     var_optimisation_ptr = &var_optimisation;
// 
//     bool use_analytical_gradients(true); // also means right now that we run the optimisation based on the hyperbolic approximation
// 
//     
//     alglib::real_1d_array x;
//     double min_standard_deviation = std::sqrt(annotation_variance);
//     if (use_analytical_gradients)
//     {
//         // here, the optimisation is done over the inverse of the standard deviation
//         current_estimate_variances[0] = 1.0/(std::sqrt(estimated_variances[0]));
//      //   current_estimate_variances[0] = -0.5*std::log(estimated_variances[0]);
//         x.setcontent(1,&(current_estimate_variances[0]));
//     }
//     else
//     {
//         x.setcontent(nb_optimised_variances,&(current_estimate_variances[0]));
//         for (int ind_var=0; ind_var<nb_optimised_variances; ++ind_var)
//             x[ind_var] = std::log(x[ind_var]);
//     }
// 
//     double epsg = 0.0001;
//     double epsf = 0.01;
//     double epsx = 0;
//     double diffstep = 1.0e-6;  
//     alglib::ae_int_t maxits = 0;
// 
//     // BLEIC
//     std::vector<double> min_x(1,0.001), max_x(1,1/min_standard_deviation);
//     alglib::real_1d_array bnd_l, bnd_u;
//     bnd_l.setcontent(1,&(min_x[0]));
//     bnd_u.setcontent(1,&(max_x[0]));
//     
//     if (use_analytical_gradients)
//     {
//      //   alglib::minlbfgscreate(1, x, state);
//     //    alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
//     //    alglib::minlbfgsoptimize(state,compute_hyperbolic_loss_one_variable,NULL,var_optimisation_ptr);
//         
//         alglib::minbleicstate state;
//         alglib::minbleicreport rep;
//         alglib::minbleiccreate(x, state);
//         alglib::minbleicsetbc(state, bnd_l, bnd_u);
//         alglib::minbleicsetcond(state, epsg, epsf, epsx, maxits);
//         alglib::minbleicoptimize(state,compute_hyperbolic_loss_one_variable,NULL,var_optimisation_ptr);
//         alglib::minbleicresults(state, x, rep);
//     }
//     else
//     {
//         std::cout << "THIS PART IS WRONG - USE ANALYTICAL GRADIENTS INSTEAD" << std::endl;
//         alglib::minlbfgsstate state;
//         alglib::minlbfgsreport rep;
//         alglib::minlbfgscreatef(nb_optimised_variances,std::min<int>(nb_optimised_variances,5),x,diffstep,state);
//         alglib::minlbfgsoptimize(state,compute_erf_loss,NULL,var_optimisation_ptr);
//         alglib::minlbfgsresults(state, x, rep);
//     }                        
//    
//         
//     if (use_analytical_gradients)
//     {
//         estimated_variances[0] = 1.0/(x[0]*x[0]);
//     }
//     else
//     {
//         for (int ind_var=0; ind_var<nb_optimised_variances; ++ind_var)            
//             estimated_variances[ind_var] = std::exp(x[ind_var]);
//     }
// //        std::cout << "Current estimated variance " << ind_var << ": " << current_estimate_variances[ind_var] << std::endl;
//     
//     std::cout << "Reestimated variances: ";
//     for (int ind_var=0; ind_var<nb_optimised_variances; ++ind_var)            
//         std::cout << estimated_variances[ind_var] << " ";
//     std::cout << std::endl;
// 
// }
// 

// Compute efficiently the Cholesky decmposition of A + coeff*A_ij*(A_ij.transpose()) where A_ij is sparse and the Cholesky decomposition of A is known
void perform_low_rank_cholesky_update(Eigen::LLT<Eigen::MatrixXd>& A_cholesky, Eigen::MatrixXd& A_inv, const Eigen::SparseMatrix<double>& A_ij, double coeff)
{

   // std::cout << "Compute QR decomposition" << std::endl;
    Eigen::SparseQR<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<int>> A_ij_QR(A_ij);
   // std::cout << "Done" << std::endl;
    int rank = A_ij_QR.rank();
    Eigen::SparseMatrix<double> Q;
    Q = A_ij_QR.matrixQ();
    Eigen::MatrixXd R = Eigen::MatrixXd(A_ij_QR.matrixR());
    Eigen::SparseMatrix<double> Q_truncated =  Q.block(0,0,A_ij.rows(),rank);
    Eigen::MatrixXd R_truncated =  R.block(0,0,rank,R.cols());
    //std::cout << "Q = " << Q_truncated << std::endl;
    //std::cout << "R = " << R_truncated << std::endl;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(R_truncated*(R_truncated.transpose()));
    Eigen::MatrixXd V = Q_truncated*es.eigenvectors();
    Eigen::VectorXd D = coeff*es.eigenvalues();
    Eigen::VectorXd A_inv_v;

    std::cout << "Rank update" << std::endl;
    for (int r=0; r<rank; ++r)
    {
      //  std::cout << "Rank update" << std::endl;
        // Update Cholesky decomposition
        A_cholesky.rankUpdate(V.col(r),D(r));
      //  std::cout << "Done" << std::endl;
        
        // Update inverse (Sherman-Morrison formula)
        A_inv_v = A_inv*(V.col(r));
        double denom = 1 + D(r)*(V.col(r).transpose())*A_inv_v;
        A_inv -= (D(r)/denom)*A_inv_v*(A_inv_v.transpose());
    } 
    std::cout << "Done" << std::endl;
}


// Compute efficiently the Cholesky decmposition of A + coeff*A_ij*(A_ij.transpose()) where A_ij is sparse and the Cholesky decomposition of A is known
void perform_low_rank_cholesky_update(Eigen::LLT<Eigen::MatrixXd>& A_cholesky, const Eigen::SparseMatrix<double>& A_ij, double coeff)
{

   // std::cout << "Compute QR decomposition" << std::endl;
    Eigen::SparseQR<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<int>> A_ij_QR(A_ij);
   // std::cout << "Done" << std::endl;
    int rank = A_ij_QR.rank();
    Eigen::SparseMatrix<double> Q;
    Q = A_ij_QR.matrixQ();
    Eigen::MatrixXd R = Eigen::MatrixXd(A_ij_QR.matrixR());
    Eigen::SparseMatrix<double> Q_truncated =  Q.block(0,0,A_ij.rows(),rank);
    Eigen::MatrixXd R_truncated =  R.block(0,0,rank,R.cols());
    //std::cout << "Q = " << Q_truncated << std::endl;
    //std::cout << "R = " << R_truncated << std::endl;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(R_truncated*(R_truncated.transpose()));
    Eigen::MatrixXd V = Q_truncated*es.eigenvectors();
    Eigen::VectorXd D = coeff*es.eigenvalues();

    std::cout << "Rank update" << std::endl;
    for (int r=0; r<rank; ++r)
    {
      //  std::cout << "Rank update" << std::endl;
        // Update Cholesky decomposition
        A_cholesky.rankUpdate(V.col(r),D(r));
      //  std::cout << "Done" << std::endl;
    } 
    std::cout << "Done" << std::endl;
}




void buildLandmarkMatrices(cv::Mat& A, cv::Mat& b, const std::vector<cv::Point2f>& landmarks_input, const std::vector<cv::Point2f>& landmarks_output)
{
    int nb_input = (int)landmarks_input.size();
    int nb_output = (int)landmarks_output.size();
    if (nb_input!=nb_output)
        std::cout << "In buildLandmarkMatrices: the provided lists of landmarks do not have the same size" << std::endl;
    A = cv::Mat(2*nb_input,6,CV_64F);
    b = cv::Mat(2*nb_input,1,CV_64F);

    // Fill A
    double *A_ptr;
    for (int i=0; i<nb_input; ++i)
    {
        double x = landmarks_input[i].x;
        double y = landmarks_input[i].y;
        A_ptr = A.ptr<double>(2*i);
        A_ptr[0] = x;
        A_ptr[1] = y;
        A_ptr[2] = 1;
        A_ptr[3] = 0;
        A_ptr[4] = 0;
        A_ptr[5] = 0;
        A_ptr = A.ptr<double>(2*i+1);
        A_ptr[0] = 0;
        A_ptr[1] = 0;
        A_ptr[2] = 0;
        A_ptr[3] = x;
        A_ptr[4] = y;
        A_ptr[5] = 1;
    }

    // Fill b
    double *b_ptr = b.ptr<double>(0);
    for (int i=0; i<nb_output; ++i)
    {
        b_ptr[2*i] = landmarks_output[i].x;
        b_ptr[2*i+1] = landmarks_output[i].y;
    }
}


// Isotropic variance * output_precomputed_covariance_matrix = real covariance matrix
void getAffineDistributionFromLandmarks(cv::Mat& output_mean_vector, cv::Mat& output_scaled_covariance_matrix, const cv::Mat& A, const cv::Mat& b)
{
    cv::Mat A_tr, pseudo_inv_A;
    cv::transpose(A,A_tr);
  //  std::cout << "Compute pseudo-inverse..." << std::endl;
    cv::invert(A_tr*A,output_scaled_covariance_matrix);
  //  std::cout << "Done" << std::endl;
    pseudo_inv_A = output_scaled_covariance_matrix*A_tr;
    output_mean_vector = pseudo_inv_A*b;
}


// This gives us (up to a factor equal to the measurement variance on the covariance matrix) the mean and covariance of the affine matrix based on the measured landmarks
void getAffineDistributionFromLandmarks(cv::Mat& output_mean_vector, cv::Mat& output_scaled_covariance_matrix, const std::vector<cv::Point2f>& landmarks_input, const std::vector<cv::Point2f>& landmarks_output)
{
    cv::Mat A, b;
    buildLandmarkMatrices(A,b,landmarks_input,landmarks_output);
    getAffineDistributionFromLandmarks(output_mean_vector,output_scaled_covariance_matrix,A,b);
}




