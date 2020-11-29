#ifndef POSITION_OVERLAP_MODEL_INCLUDED
#define POSITION_OVERLAP_MODEL_INCLUDED
#include "graph.hpp"
//#include "stdafx.h"
#include "optimization.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "structures_for_interactive_mosaicking.hpp"
#include "linear_algebra.hpp"
#include <boost/random/mersenne_twister.hpp>

// We assume affine, and based on correspondences
class PositionOverlapModel
{
    
    public:
 //   PositionOverlapModel(const std::vector<MeasuredAffineRegistration>& pairwise_measurements, const std::vector<int>& measurement_modes, const std::vector<double>& variance_parameters, int ind_reference_frame, int nb_frames, int size_square_interior, int size_square_exterior, const std::vector<cv::Point2f>& overlap_corners);
    PositionOverlapModel();
    PositionOverlapModel(const std::vector<MeasuredAffineRegistration>& pairwise_measurements, const std::vector<int>& measurement_modes, const std::vector<double>& variance_parameters, int ind_reference_frame, int nb_frames, const Eigen::Vector2d& gamma);
    PositionOverlapModel(const std::vector<MeasuredAffineRegistration>& pairwise_measurements, const std::vector<int>& measurement_modes, const std::vector<double>& variance_parameters, int ind_reference_frame, int nb_frames, const Eigen::Vector2d& gamma, double size_overlap_area);
    virtual ~PositionOverlapModel();
    virtual const Eigen::VectorXd& get_mean_estimated_absolute_positions() const {return mean_estimated_absolute_positions;};
    virtual void add_pairwise_measurement(const MeasuredAffineRegistration& new_measurement, int measurement_mode);
    virtual void set_pairwise_measurements(const std::vector<MeasuredAffineRegistration>& pairwise_measurements, const std::vector<int>& measurement_modes);
    virtual void compute_relative_displacement_distribution(Eigen::Vector2d& mean_coordinates, Eigen::Matrix2d& covariance_coordinates, int i, int j, const Eigen::Vector2d& input_point) const;
    void compute_asymmetric_relative_displacement_distribution(Eigen::Vector2d& mean_coordinates, Eigen::Matrix2d& covariance_coordinates, int i, int j, const Eigen::Vector2d& input_point) const;
    void compute_overlap_bounds_and_reward(double& lower_bound, double& upper_bound, double& reward, int i, int j, RewardType chosen_reward) const;
    void compute_overlap_probability_with_sampling(double& probability_overlap,  int i, int j, int nb_samples, boost::random::mt19937& rng) const;
    void compute_symmetrical_overlap_bounds_and_reward(double& lower_bound, double& upper_bound, double& reward, int i, int j, RewardType chosen_reward) const;
    void compute_symmetrical_overlap_probability_with_sampling(double& probability_overlap,  int i, int j, int nb_samples, boost::random::mt19937& rng) const;
    void update_variance(double variance, int mode);
    double get_variance_parameter(int mode) const {return m_variance_parameters[mode];};
    void reestimate_initial_variance(const std::vector<LabeledImagePair>& labeled_pairs, double annotation_variance);
    void estimate_size_overlap_area(const std::vector<MeasuredAffineRegistration>& pairwise_measurements);
    void estimate_size_overlap_area(const std::vector<LabeledImagePair>& labeled_pairs);
    virtual double compute_Sawhney_probability(double& sawhney_score, int i, int j, const std::vector<cv::Point2f>& image_corners) const;
    virtual void compute_mean_estimated_positions();
    virtual void compute_covariance_estimated_positions();

    

protected:
        
    // Model parameters
    std::vector<double> m_variance_parameters;
    std::vector<MeasuredAffineRegistration> m_pairwise_measurements;
    std::vector<int> m_measurement_modes;   // indicates where to look up the variance: the variance of the measurement m is m_variance_parameters[m_measurement_modes[m]]
    int m_nb_frames;
    Eigen::Vector2d m_gamma;
    double m_size_overlap_area; // this is the size a of an axis-aligned square centered on gamma defining whether two frames overlap or not
    Eigen::VectorXd mean_estimated_absolute_positions;
    
    private:
    
    int m_ind_reference_frame;

//    int m_size_square_interior;
//    int m_size_square_exterior;
//    std::vector<cv::Point2f> m_overlap_corners;

    
    // Hidden parameters that are maintained to facilitate predictions
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Eigen::LLT<Eigen::MatrixXd> A_cholesky;
    Eigen::MatrixXd A_inv;
    std::vector<Eigen::SparseMatrix<double>> linear_factor_derivatives;
    std::vector<Eigen::SparseMatrix<double>> constant_factor_derivatives;
    std::vector<Eigen::Triplet<double>> triplet_list_standard_basis;
    std::vector<Eigen::MatrixXd> scaled_factors_covariance_matrices_pairwise_measurements;
    
    // Estimated mean position
    Eigen::VectorXd estimated_absolute_positions_inv_without_ref;

    // Global covariance matrix
    Eigen::MatrixXd covariance_estimated_absolute_positions_inv;
    std::vector<Eigen::MatrixXd> scaled_covariance_matrix_modes;

};



void compute_matrices_closed_form_bundle_adjustment_on_correspondences(Eigen::MatrixXd& A, Eigen::VectorXd& b, std::vector<Eigen::SparseMatrix<double>>& linear_factor_derivatives, std::vector<Eigen::SparseMatrix<double>>& constant_factor_derivatives, std::vector<Eigen::Triplet<double>>& triplet_list_standard_basis, const std::vector<MeasuredAffineRegistration>& measured_registrations, int nb_frames, int ind_reference_frame);

void update_bundle_adjustment_on_correspondences(Eigen::MatrixXd& A, Eigen::VectorXd& b, Eigen::LLT<Eigen::MatrixXd>& A_cholesky, Eigen::MatrixXd& A_inv, std::vector<Eigen::SparseMatrix<double>>& linear_factor_derivatives, std::vector<Eigen::SparseMatrix<double>>& constant_factor_derivatives, const std::vector<Eigen::Triplet<double>>& triplet_list_standard_basis, int ind_reference_frame, const MeasuredAffineRegistration& new_measured_registration);



void fill_F_uv(Eigen::MatrixXd& F_uv, const std::vector<Eigen::SparseMatrix<double>>& linear_factors_derivatives, const std::vector<Eigen::SparseMatrix<double>>& constant_factors_derivatives, const Eigen::VectorXd& estimated_global_positions_without_ref);

void add_reference_frame_to_global_positions_affine(Eigen::VectorXd& global_positions_with_ref, const Eigen::VectorXd& global_positions_without_ref, int ind_reference_frame);

void add_reference_frame_to_global_covariance_matrix_affine(Eigen::MatrixXd& global_cov_matrix_with_ref, const Eigen::MatrixXd& global_cov_matrix_without_ref, int ind_reference_frame);

void inverse_vector_of_estimated_affine_global_positions(Eigen::VectorXd& input_vec_inv, const Eigen::VectorXd& input_vec);

void createDiagonalCovariancesCorrespondences(std::vector<Eigen::MatrixXd>& factors_covariance_matrices, const std::vector<MeasuredAffineRegistration>& measured_registrations);

void retain_blocks_i_and_j_in_covariance_matrix(Eigen::MatrixXd& covariance_matrix_ij, const Eigen::MatrixXd& full_covariance_matrix, int i, int j);

void compute_closed_form_overlap_probability_and_bounds(double& closed_form_overlap_probability_low, double& closed_form_overlap_probability_up, double& uncertainty_reward, const Eigen::Matrix2d& covariance_ij, double size_square_interior, double size_square_exterior, const Eigen::Vector2d& mean_vector_ij,  RewardType chosen_reward);

void estimateOverlapProbabilitiesWithSampling(double& probability_overlap, const Eigen::Vector2d& mean_vector_ij,
                const Eigen::Matrix2d& covariance_ij, int nb_samples, boost::random::mt19937& rng, const std::vector<cv::Point2f>& corners);

void estimateOverlapProbabilitiesWithSampling(double& probability_overlap, const Eigen::Vector2d& mean_vector_ij,
                const Eigen::Matrix2d& covariance_ij, int nb_samples, boost::random::mt19937& rng, double radius_overlap);

void reestimate_variances_generic(std::vector<double>& estimated_variances, const std::vector<LabeledImagePair>& labeled_pairs, const std::vector<Eigen::MatrixXd>& scaled_factors_covariance_matrices_pairwise_measurements, const Eigen::MatrixXd& left_cov_propagation_global_cov_matrix, const std::vector<int>& measurement_modes, int nb_modes, int nb_optimised_variances, const std::map<std::pair<int,int>,Eigen::MatrixXd>& left_cov_propagations_relative_cov_matrix, std::map<std::pair<int,int>,double>& displacement_ij, std::map<std::pair<int,int>,Eigen::Vector2d>& mean_vector_ij, double half_size_interior, double half_size_exterior, int ind_reference_frame, double annotation_variance);


void compute_hyperbolic_loss_one_variable(const alglib::real_1d_array &x, double &func, alglib::real_1d_array &grad, void *ptr);

void compute_log_overlap_bound_hyperbolic_one_variance(double& log_bound, double& derivative_log_bound, double inverse_current_std, double half_size_interior, const Eigen::Vector2d& scaled_eigenvalues, const Eigen::Vector2d& dot_products);

void compute_log_non_overlap_bound_hyperbolic_one_variance(double& log_bound, double& derivative_log_bound, double& concave_log_bound, double& derivative_concave_log_bound, double inverse_current_std, double half_size_interior, const Eigen::Vector2d& scaled_eigenvalues, const Eigen::Vector2d& dot_products);

void compute_in_log_convex_upper_part(double& res, double& derivative, double inverse_estimated_std, double x_1, double x_2, double a_1, double a_2);

double log_cosh(double x);

double log_sinh(double x);

void get_affine_matrix_from_parameters(Eigen::Matrix3d& affine_matrix, const double* parameters);

double my_tanh(double x);

void perform_low_rank_cholesky_update(Eigen::LLT<Eigen::MatrixXd>& A_cholesky, const Eigen::SparseMatrix<double>& A_ij, double coeff);

// Compute efficiently the Cholesky decmposition of A + coeff*A_ij*(A_ij.transpose()) where A_ij is sparse and the Cholesky decomposition of A is known
void perform_low_rank_cholesky_update(Eigen::LLT<Eigen::MatrixXd>& A_cholesky, Eigen::MatrixXd& A_inv, const Eigen::SparseMatrix<double>& A_ij, double coeff = 1);



void buildLandmarkMatrices(cv::Mat& A, cv::Mat& b, const std::vector<cv::Point2f>& landmarks_input, const std::vector<cv::Point2f>& landmarks_output);

// Isotropic variance * output_precomputed_covariance_matrix = real covariance matrix
void getAffineDistributionFromLandmarks(cv::Mat& output_mean_vector, cv::Mat& output_scaled_covariance_matrix, const cv::Mat& A, const cv::Mat& b);


// This gives us (up to a factor equal to the measurement variance on the covariance matrix) the mean and covariance of the affine matrix based on the measured landmarks
void getAffineDistributionFromLandmarks(cv::Mat& output_mean_vector, cv::Mat& output_scaled_covariance_matrix, const std::vector<cv::Point2f>& landmarks_input, const std::vector<cv::Point2f>& landmarks_output);














#endif
