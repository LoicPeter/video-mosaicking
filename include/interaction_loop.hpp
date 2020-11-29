#ifndef INTERACTION_LOOP_INCLUDED
#define INTERACTION_LOOP_INCLUDED

#include "structures_for_interactive_mosaicking.hpp"
#include "affine_transformation.hpp"
#include "graph.hpp"
#include "landmark_database.hpp"
#include "position_overlap_model.hpp"
#include "bow_overlap_model.hpp"
#include "linear_algebra.hpp"
#include "agent.hpp"

// Random number / probabilities / sampling
#include <boost/math/special_functions/erf.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>

#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


// Eigen for sparse matrices
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/OrderingMethods>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>



bool isScoreHigher(const UnlabeledImagePair& pair_1 , const UnlabeledImagePair& pair_2);

class CompareUnlabeledImagePair_IsHigher
{
    public:
    bool operator() (const UnlabeledImagePair& lhs, const UnlabeledImagePair& rhs) const
        {return isScoreHigher(lhs,rhs);};
};


bool isScoreLower(const UnlabeledImagePair& pair_1 , const UnlabeledImagePair& pair_2);

void createEnblendMosaic(const Eigen::VectorXd& global_positions, const InputData& input_data, const InputSequence& input_sequence, const Settings& settings, std::string folder_name);

void run_interactive_process(GroundTruthData& ground_truth_data, InputData& input_data, const InputSequence& input_sequence, InteractiveHelper& interactive_helper, Settings& settings);

void remap_affine_correspondences(std::vector<cv::Point2f>& input_points, std::vector<cv::Point2f>& output_points, const std::vector<cv::Point2f>& new_input_points);

void perform_label_count(std::vector<int>& label_count, const std::vector<LabeledImagePair>& labeled_pairs);


double getMeanDisplacementErrorAbsolute(const Eigen::VectorXd& estimated_absolute_positions, GroundTruthData& ground_truth_data);

double getMeanDisplacementErrorRelative(const Eigen::VectorXd& estimated_absolute_positions, GroundTruthData& ground_truth_data);

void symmetrise_and_save(const std::string& filename, const cv::Mat& to_save, double diagonal_value,  const std::vector<LabeledImagePair>& overlap_training_set = std::vector<LabeledImagePair>());

void symmetrise_and_save(const std::string& filename, const cv::Mat& to_save, double diagonal_value, const std::pair<int,int>& queried_pair);

void compute_overlap_probabilities_and_bounds_for_all_pairs(std::vector<UnlabeledImagePair>& list_candidate_pairs, cv::Mat& sampling_based_i_in_j, cv::Mat& closed_form_overlap_probability_low_matrix, cv::Mat& closed_form_overlap_probability_up_matrix, cv::Mat& uncertainty_reward_matrix, const cv::Mat& external_overlap_probability_matrix, const PositionOverlapModel& position_overlap_model, Settings& settings, const InputData& input_data);

void cotrain_overlap_models(ExternalOverlapModel& bow_overlap_model, PositionOverlapModel& position_overlap_model,  PositionOverlapModel& initial_position_overlap_model, std::vector<LabeledImagePair>& fixed_training_pairs, InputData& input_data, Settings& settings);


#endif // MOSAIC_INCLUDED
