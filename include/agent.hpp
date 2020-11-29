#ifndef AGENT_INCLUDED
#define AGENT_INCLUDED

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "structures_for_interactive_mosaicking.hpp"


// Precomputed 

void automaticInteractiveAnnotationCorrespondences(int& is_there_overlap, InputData& input_data, GroundTruthData& ground_truth_data, Settings& settings, int i, int j);


// Automated registration

void automaticAnnotationViaLandmarkRegistration(int& is_there_overlap, InputData& input_data, const InputSequence& input_sequence, int i, int j, int nb_annotated_points = 3);

// Human

struct MouseParamsRegistrationAnnotation
{
    std::vector<cv::Point> landmarks;
    bool must_draw_line;
};

void manualInteractiveQuery(int& is_there_overlap, InputData& input_data, const InputSequence& input_sequence, InteractiveHelper& interactive_helper, bool& stop_labelling, int i, int j);

void show_pair_of_images(cv::Mat& joint_image, const cv::Mat& frame_a, const cv::Mat& frame_b, const std::string& joint_window_name, int size_separation);

void runInteractiveLandmarkQuery(std::vector<cv::Point2f>& landmarks_input, std::vector<cv::Point2f>& landmarks_output, const cv::Mat& frame_a, const cv::Mat& frame_b, const std::string& joint_window_name, int size_separation, int min_nb_landmarks = 3);

void onMouse(int evt, int x, int y, int flags, void* param);

#endif // MOSAIC_INCLUDED
