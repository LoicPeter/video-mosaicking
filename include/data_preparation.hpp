#ifndef DATA_PREPARATION_INCLUDED
#define DATA_PREPARATION_INCLUDED

#include <vector>
#include "structures_for_interactive_mosaicking.hpp"
#include "affine_transformation.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>



void createPointsforDistance(std::vector<cv::Point2f>& points_for_distance_relative, int nb_rows, int nb_cols, double grid_step_size);

cv::Mat getMask_U(const cv::Mat& frame);


void performPreprocessing(cv::Mat& preprocessedImage, const cv::Mat& image, const cv::Mat& mask_U, int sigma_blur);


void sampleDenseKeypoints(std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask, int stride, int diameter);

cv::Mat buildBagOfWords(std::vector<cv::Mat>& image_descriptors, cv::Mat& vocabulary, const std::vector<cv::Mat>& video, const std::vector<cv::Mat>& masks, int cluster_count = 300, int diameter = 15, int stride = 5);

void createSyntheticCircleDataset(InputSequence& input_sequence, InputData& input_data, GroundTruthData& ground_truth_data, Settings& settings, int random_seed);

void createSyntheticRasterScan(InputSequence& input_sequence, InputData& input_data, GroundTruthData& ground_truth_data, Settings& settings, int random_seed);

void create_square_image(cv::Mat& im, int nb_rows, int nb_cols);

void load_overlap_area_and_dimensions(InputData& input_data, double factor_overlap_area);

void loadAerialFrames(InputSequence& video_sequence, int nb_frames);

void loadFetoscopyFrames(InputSequence& video_sequence, int nb_frames);

void performInitialRegistrations_Dense(std::vector<MeasuredAffineRegistration>& measured_registrations, const InputSequence& video_sequence, const std::vector<cv::Point2f>& landmarks_input, const std::string& global_video_folder, bool load_pairwise_registrations = false, int precomputed_nb_frames = -1);

void performInitialRegistrations_Landmarks(std::vector<MeasuredAffineRegistration>& measured_registrations, InputData& input_data, const InputSequence& video_sequence, const std::vector<std::pair<int,int>>& pairs_to_register, const std::string& global_folder_name, const std::string& video_folder_name, const std::string& database_name);

void perform_landmark_based_registration(bool& is_inverse_compatible, std::vector<cv::Point2f>& landmarks_input, std::vector<cv::Point2f>& landmarks_output,  cv::Ptr<cv::Feature2D> sift, const cv::Mat& image_1, const cv::Mat& image_2, const cv::Mat& mask_1, const cv::Mat& mask_2, bool display);

void buildBagOfWordsWithDenseDetection(std::vector<cv::Mat>& image_descriptors, cv::Mat& vocabulary, const std::vector<cv::Mat>& video, const std::vector<cv::Mat>& masks, int cluster_count, int diameter, int stride);

void buildBagOfWordsWithDetection(std::vector<cv::Mat>& image_descriptors, cv::Mat& vocabulary, const std::vector<cv::Mat>& video, const std::vector<cv::Mat>& masks, int cluster_count);

void computeInitialBagOfWordsModel(Eigen::MatrixXd& bow_appearance_matrix, int& nb_bow_descriptors, const InputSequence& video_sequence, const std::string& global_video_folder, const std::string& video_folder_name, bool loadPrecomputedBOW, int nb_descriptors, int diameter, int stride, double downsize_factor = 1, bool build_with_detection = false, const std::string& bow_name = "",  bool normalise_descriptors = true);

void normaliseBagOfWordsMatrix(Eigen::MatrixXd& bow_appearance_matrix);

void loadPrecomputedFetoscopyDataset(InputData& input_data, InputSequence& input_sequence, GroundTruthData& ground_truth_data, InteractiveHelper& interactive_helper, Settings& settings);

void loadAerialDataset(InputData& input_data, InputSequence& input_sequence, GroundTruthData& ground_truth_data, InteractiveHelper& interactive_helper, Settings& settings);

void perform_PCA_reduction(Eigen::MatrixXd& output_matrix, const Eigen::MatrixXd& input_matrix, int nb_components);

void loadInitialPairwiseMeasurements(InputData& input_data, LandmarkDatabase& landmark_database, const std::vector<std::pair<int,int>>& overlapping_pairs, int nb_frames);

void loadAllPairwiseMeasurements(InputData& input_data, LandmarkDatabase& landmark_database, int nb_frames);

void prune_landmark_database(LandmarkDatabase& pruned_database, LandmarkDatabase& landmark_database, int nb_kept_points, int nb_frames);

void prune_landmark_database(LandmarkDatabase& pruned_database, LandmarkDatabase& landmark_database, int nb_kept_points, int nb_frames, bool visualizeKeypoints, const std::vector<cv::Mat>& frames);

void keep_long_range_correspondences(LandmarkDatabase& pruned_database, LandmarkDatabase& landmark_database, int nb_frames, int min_interframe_time);

void curate_landmark_database(const std::string& name_curated_database, const std::string& name_input_database, const std::vector<cv::Mat>& frames, int nb_kept_points);

void curate_landmark_database(LandmarkDatabase& curated_database, LandmarkDatabase& input_database, const std::vector<cv::Mat>& frames, int nb_kept_points);

void pick_random_points(std::vector<cv::Point2f>& new_input_points, std::vector<cv::Point2f>& new_output_points, const std::vector<cv::Point2f>& input_points, const std::vector<cv::Point2f>& output_points, int input_nb_kept_points);

void test_overlap(bool& is_i_in_j, bool& is_j_in_i, double *H_i, double *H_j, const std::vector<cv::Point2f>& corners);

void count_overlapping_points(std::vector<cv::Point2f>& overlapping_control_points_input, std::vector<cv::Point2f>& overlapping_control_points_output, double *T_ji, const std::vector<cv::Point2f>& corners, const std::vector<cv::Point2f>& control_points);

void triangulate_graph_of_constraints(Graph<cv::Mat>& pairwise_homographies, cv::Mat& pairwise_residuals);

//void getDisplacedPointsAffine(std::vector<cv::Point2f>& output_points, double *global_affine_parameters, int i, int j, const std::vector<cv::Point2f>& input_points);

//void getDisplacedPointsAffine(std::vector<cv::Point2f>& output_points, double *h_ij, const std::vector<cv::Point2f>& input_points);


void pick_least_aligned_points(std::vector<cv::Point2f>& new_input_points, std::vector<cv::Point2f>& new_output_points, const std::vector<cv::Point2f>& input_points, const std::vector<cv::Point2f>& output_points, int nb_kept_points, int random_seed = 1);

double compute_alignment_score(const std::vector<int>& ind_points, int nb_kept_points, const std::vector<cv::Point2f>& input_points, const std::vector<cv::Point2f>& output_points);

double compute_alignment_score(const std::vector<int>& ind_points, int nb_kept_points, const std::vector<cv::Point2f>& input_points);




#endif // MOSAIC_INCLUDED
