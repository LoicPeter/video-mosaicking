
#ifndef STRUCTURES_FOR_INTERACTIVE_MOSAICKING_H_INCLUDED
#define STRUCTURES_FOR_INTERACTIVE_MOSAICKING_H_INCLUDED

#include <vector>
#include <string>
#include <map>
#include <Eigen/Dense>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <boost/random/mersenne_twister.hpp>
#include "graph.hpp"
#include "landmark_database.hpp"

struct MeasuredAffineRegistration
{
    std::pair<int,int> frame_indices;
    std::vector<cv::Point2f> landmarks_input;
    std::vector<cv::Point2f> landmarks_output;
    cv::Mat mean_affine_parameters;
    cv::Mat scaled_covariance_matrix;
};


struct UnlabeledImagePair
{
    std::pair<int,int> image_indices;
    double score;
};



struct LabeledImagePair
{
    std::pair<int,int> image_indices;
    bool this_pair_overlaps;
    double importance_weight;
};

struct VarianceOptimisation
{
    std::vector<LabeledImagePair> _labeled_pairs;
    std::map<std::pair<int,int>,double> *_displacement_map_ptr;
    std::map<std::pair<int,int>,Eigen::Vector2d> *_mean_vector_ptr;
    std::map<std::pair<int,int>,Eigen::VectorXd> *_traces_relative_modes_ptr;
    std::map<std::pair<int,int>,Eigen::VectorXd> *_min_eigenvalues_relative_modes_ptr;
    std::map<std::pair<int,int>,Eigen::VectorXd> *_max_eigenvalues_relative_modes_ptr;
    std::map<std::pair<int,int>,Eigen::Vector2d> *_scaled_eigenvalues_ptr;
    std::map<std::pair<int,int>,Eigen::Vector2d> *_dot_products_ptr;
    std::map<std::pair<int,int>,Eigen::MatrixXd> *_covariance_matrix_relative_modes_ptr;
    int _nb_modes;
    double _half_size_interior;
    double _half_size_exterior;
    int _current_variable_optimised;
    Eigen::VectorXd _current_estimated_variances;
    int _nb_optimised_variances;
    std::vector<double> _instance_weights;

};

struct GroundTruthData
{
    bool is_available;
  //  std::vector<double> true_absolute_positions;
  //  Graph<int> true_overlap_information;

    LandmarkDatabase gt_database; // contains all the correspondences: used for evaluation purposes
    LandmarkDatabase measurements_database; // contains all precomputed measurements that can be queried: used to automate interactions

};

enum AgentType {precomputed_agent, landmark_registration, human};

enum RewardType {our_reward, shortest_path, entropy_reward, no_reward, xia_reward};

struct InputData
{
   // Graph<cv::Mat> observed_pairwise_registrations; // should be symmetric
    Graph<int> overlap_information; // only upper half fill sufficient (ie i < j)
    std::vector<MeasuredAffineRegistration> observed_pairwise_registrations;
    Eigen::MatrixXd bow_appearance_matrix;
    int nb_bow_descriptors;
    int nb_frames;
    int nb_rows;
    int nb_cols;
    std::vector<cv::Point2f> overlap_corners;
    std::vector<int> dimensions_overlap_area;
    std::string dataset_identifier;
    cv::Ptr<cv::Feature2D> feature_extractor;
    AgentType agent_type;
};

struct InputSequence
{
    std::vector<cv::Mat> frames;
    std::vector<cv::Mat> masks;
    std::string global_video_folder;
};

struct InteractiveHelper
{
    //Graph<cv::Mat> database_annotated_registrations;
    LandmarkDatabase database_annotations;
    std::string landmark_database_identifier;
};

enum PositionalOverlapModel {no_positional_overlap_model, our_positional_overlap_model, sawhney_probability, xia_filtering, elibol_filtering_low, elibol_filtering_up};

enum ChosenExternalOverlapModel {no_external_overlap_model, our_external_overlap_model};

enum VarianceReestimationStrategy {initial_measurements_only_one_mode};

struct Settings
{
    int nb_iterations_interaction;
    int nb_possible_skips; // the number of times we allow to skip a pair because we cannot identify it visually
    int ind_reference_frame;
    double automatic_registration_variance;
    double annotation_variance;
    double beta; // for PCCA model
    double lambda; // for regularisation of PCCA model in logistic regression
    int nb_iterations_cotraining;
    int nb_samples_montecarlo_overlap_estimation;
    int seed_random_number_generator;
    int annotated_points_in_interaction;
    bool automate_interactions;
    double step_size_grid_similarity_measure;
    bool reestimate_external_overlap_model;
    bool reestimate_position_overlap_model;
    bool do_cotraining;
    bool diagonal_bow_weights;
    bool remap_correspondences;
    bool createVideos;
    bool bundle_adjustment_on_correspondences;
    bool compute_overlap_probability_for_all_pairs;
    bool random_landmarks_in_interactions;
    double factor_overlap_area; // 1 if we want Omega, a priori should be rather 2
    PositionalOverlapModel chosen_positional_overlap_model;
    ChosenExternalOverlapModel chosen_external_overlap_model;
    RewardType chosen_reward;
    boost::random::mt19937 rng;
    std::string experiment_identifier;
    
    bool reestimate_from_initial_measurements_only;
};

void perform_label_count(std::vector<int>& label_count, const std::vector<LabeledImagePair>& labeled_pairs);

#endif
