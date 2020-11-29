#include "agent.hpp"
#include "position_overlap_model.hpp"

#include "data_preparation.hpp" // for the registration module
#include "opencv2/highgui.hpp"


// This only annotates the pointwise correspondences
void automaticInteractiveAnnotationCorrespondences(int& is_there_overlap, InputData& input_data, GroundTruthData& ground_truth_data, Settings& settings, int i, int j)
{
    bool is_there_overlap_bool = ground_truth_data.measurements_database.isPairInDatabase(i,j);

    if (ground_truth_data.measurements_database.isPairInDatabase(i,j))
        is_there_overlap = 1;
    else
        is_there_overlap = -1;
    
    // We collect the overlap information
    input_data.overlap_information.add_edge(i,j,is_there_overlap);

    if (is_there_overlap==1)
    {

        MeasuredAffineRegistration new_measurement;
        new_measurement.frame_indices = std::pair<int,int>(i,j);
        ground_truth_data.measurements_database.loadLandmarks(i,j,new_measurement.landmarks_input,new_measurement.landmarks_output);
        
        getAffineDistributionFromLandmarks(new_measurement.mean_affine_parameters,new_measurement.scaled_covariance_matrix,new_measurement.landmarks_input,new_measurement.landmarks_output);
        
        // Add measurement to the list
        input_data.observed_pairwise_registrations.push_back(new_measurement);
    }
}

// -----------------------------------
// Registration agent
// -----------------------------------


void automaticAnnotationViaLandmarkRegistration(int& is_there_overlap, InputData& input_data, const InputSequence& input_sequence, int i, int j, int nb_annotated_points)
{
    bool display = false; // set to true to show the registration result
    
    
    bool is_there_overlap_bool;

    MeasuredAffineRegistration new_measurement;
    new_measurement.frame_indices = std::pair<int,int>(i,j);
    
    // We attempt a landmark based registration
    std::vector<cv::Point2f> matches_input, matches_output;
    perform_landmark_based_registration(is_there_overlap_bool,matches_input,matches_output,input_data.feature_extractor, input_sequence.frames[i],input_sequence.frames[j],input_sequence.masks[i],input_sequence.masks[j],false);
    
    if (is_there_overlap_bool)
        is_there_overlap = 1;
    else
        is_there_overlap = -1;
    
    input_data.overlap_information.add_edge(i,j,is_there_overlap);
    
    if (is_there_overlap==1)
    {
        pick_least_aligned_points(new_measurement.landmarks_input,new_measurement.landmarks_output,matches_input,matches_output,nb_annotated_points);
        
        getAffineDistributionFromLandmarks(new_measurement.mean_affine_parameters,new_measurement.scaled_covariance_matrix,new_measurement.landmarks_input,new_measurement.landmarks_output);
        
        input_data.observed_pairwise_registrations.push_back(new_measurement);
        if (display)
        {
            cv::Mat output_image_i, output_image_j;
            std::vector<cv::KeyPoint> input_keypoints, output_keypoints;
            for (int k=0; k<nb_annotated_points; ++k)
            {
                input_keypoints.push_back(cv::KeyPoint(new_measurement.landmarks_input[k],10));
                output_keypoints.push_back(cv::KeyPoint(new_measurement.landmarks_output[k],10));
            }

            drawKeypoints(input_sequence.frames[i],input_keypoints,output_image_i);
            drawKeypoints(input_sequence.frames[j],output_keypoints,output_image_j);
            cv::imshow("Input frame",output_image_i);
            cv::imshow("Output frame",output_image_j);
            cv::waitKey(10000);
        }
    }
    
    
       
}



// -------------------------------
// Human annotator
// -------------------------------



void manualInteractiveQuery(int& is_there_overlap, InputData& input_data, const InputSequence& input_sequence, InteractiveHelper& interactive_helper, bool& stop_labelling, int i, int j)
{
    is_there_overlap = 0;
    std::string joint_window_name = "Annotation Interface";
    cv::namedWindow(joint_window_name,cv::WINDOW_KEEPRATIO);
    cv::Mat joint_image_for_display;
    int size_separation_for_display = 1;
    show_pair_of_images(joint_image_for_display,input_sequence.frames[i],input_sequence.frames[j],joint_window_name,size_separation_for_display);

    //imwrite(global_folder_name + video_folder_name + "Displayed_Pair_" + oss_it.str() + ".png",joint_image_for_display);

    std::cout << "Do these frames overlap?" << std::endl;

    int entered_key;
    entered_key = (int)cv::waitKey(0);

    if (entered_key==27) // escape
    {
        stop_labelling = true;
    }
    else
    {
        if (entered_key==121) // Y
        {
            std::cout << "Yes" << std::endl;
            is_there_overlap = 1;
        }
        else
        {
            if (entered_key==100) // D
            {
                std::cout << "Not sure / Please ignore" << std::endl;
                is_there_overlap = 0;
            }
            else
            {
                if (entered_key==110) // N
                {
                    std::cout << "No" << std::endl;
                    is_there_overlap = -1;
                }
            }
        }
    }

    if (stop_labelling==false)
    {
        input_data.overlap_information.add_edge(i,j,is_there_overlap);

        if (entered_key==121)
        {
            bool query_manual_annotation(true);

            MeasuredAffineRegistration new_measurement;

            // The measurement registers the frame i and the frame j
            new_measurement.frame_indices = std::pair<int,int>(i,j);


            if (interactive_helper.database_annotations.isPairInDatabase(i,j))
            {
                std::cout << "Existing annotations found. Do you want to use them?" << std::endl;
                entered_key = (int)cv::waitKey(0);
                if (entered_key==121) // Y
                {
                    std::cout << "Yes" << std::endl;
                    query_manual_annotation = false;
                    interactive_helper.database_annotations.loadLandmarks(i,j,new_measurement.landmarks_input,new_measurement.landmarks_output);
                }
                else
                {
                    std::cout << "No" << std::endl;
                    interactive_helper.database_annotations.eraseFromDatabase(i,j);
                }
            }

            if (query_manual_annotation)
            {

                // Old one
                //std::cout << "Please provide 4 matches" << std::endl;
                //manually_annotate_registration(homography_user,interactive_helper.frames[i],interactive_helper.frames[j],joint_window_name,size_separation_for_display);
                //interactive_helper.database_annotated_registrations.add_edge(i,j,homography_user);

                // Collect landmarks
                runInteractiveLandmarkQuery(new_measurement.landmarks_input,new_measurement.landmarks_output,input_sequence.frames[i],input_sequence.frames[j],joint_window_name,size_separation_for_display);

                // Add to the database
                interactive_helper.database_annotations.addToDatabase(i,j,new_measurement.landmarks_input,new_measurement.landmarks_output);
            }


            // Get matrices of transformation (mean and scaled covariance)
            getAffineDistributionFromLandmarks(new_measurement.mean_affine_parameters,new_measurement.scaled_covariance_matrix,new_measurement.landmarks_input,new_measurement.landmarks_output);

            // Add measurement to the list
            input_data.observed_pairwise_registrations.push_back(new_measurement);
        }

    }

}

void show_pair_of_images(cv::Mat& joint_image, const cv::Mat& frame_a, const cv::Mat& frame_b, const std::string& joint_window_name, int size_separation)
{
    int nb_rows_a = frame_a.rows;
    int nb_rows_b = frame_b.rows;
    int nb_cols_a = frame_a.cols;
    int nb_cols_b = frame_b.cols;
    if ((nb_rows_a != nb_rows_b) || (nb_cols_a != nb_cols_b))
        std::cout << "In show_pair_of_images: the two images are not from the same size" << std::endl;
    int nb_rows = nb_rows_a;
    int nb_cols = nb_cols_a + size_separation + nb_cols_b;
    joint_image = cv::Mat(nb_rows,nb_cols,CV_8UC3,cv::Scalar::all(0));
    cv::Rect region_a(0,0,nb_cols_a,nb_rows_a);
    cv::Rect region_b(nb_cols_a + size_separation,0,nb_cols_b,nb_rows_b);
    frame_a.copyTo(joint_image(region_a));
    frame_b.copyTo(joint_image(region_b));

    cv::imshow(joint_window_name,joint_image);
    cv::waitKey(1);
}



void runInteractiveLandmarkQuery(std::vector<cv::Point2f>& landmarks_input, std::vector<cv::Point2f>& landmarks_output, const cv::Mat& frame_a, const cv::Mat& frame_b, const std::string& joint_window_name, int size_separation, int min_nb_landmarks)
{
    // Parameters
    int line_thickness = 2;
    int line_type = cv::LINE_8;
    cv::Point start_point, end_point;
    int nb_cols_a = frame_a.cols;

    // Preparation joint image
    cv::Mat joint_image;
    show_pair_of_images(joint_image,frame_a,frame_b,joint_window_name,size_separation);

    // Collect the landmarks from user interaction
    bool exit_requested(false);
    bool must_draw_line(false);
    int entered_key;

    std::cout << "Please provide at least " << min_nb_landmarks << " non-aligned landmark correspondences, and press Esc when done" << std::endl;

    int nb_lines_drawn(0);
    MouseParamsRegistrationAnnotation mouse_params;
    mouse_params.must_draw_line = false;
    cv::setMouseCallback(joint_window_name, onMouse,(void*)(&mouse_params));
    //while ((nb_lines_drawn<min_nb_landmarks) && (exit_requested==false))
    while (exit_requested==false)
    {
        // Show image
        cv::imshow(joint_window_name,joint_image);

        // Draw line
        if (mouse_params.must_draw_line)
        {
            int nb_landmarks = (int)mouse_params.landmarks.size();
           // std::cout << "Landmarks:" << nb_landmarks << " " << std::endl;
            line(joint_image,mouse_params.landmarks[nb_landmarks-2],mouse_params.landmarks[nb_landmarks-1],cv::Scalar(255,0,0),line_thickness,line_type);
            mouse_params.must_draw_line = false;
            ++nb_lines_drawn;
        }

        entered_key = (int)cv::waitKey(1);

        if ((entered_key==27) && (nb_lines_drawn >= min_nb_landmarks)) // escape
            exit_requested = true;
    }

    // Draw last line
    int nb_landmarks = (int)mouse_params.landmarks.size();
    line(joint_image,mouse_params.landmarks[nb_landmarks-2],mouse_params.landmarks[nb_landmarks-1],cv::Scalar(255,0,0),line_thickness,line_type);
    cv::waitKey(1);

    cv::setMouseCallback(joint_window_name,NULL);
    
    // Correct landmarks from the second image
    for (int k=0; k<nb_lines_drawn; ++k)
        mouse_params.landmarks[2*k+1].x -= nb_cols_a + size_separation;

    for (int k=0; k<nb_lines_drawn; ++k)
    {
        landmarks_input.push_back(mouse_params.landmarks[2*k]);
        landmarks_output.push_back(mouse_params.landmarks[2*k+1]);
    }

    std::cout << nb_lines_drawn << " correspondences collected" << std::endl;
}



void onMouse(int evt, int x, int y, int flags, void* mouse_params)
{
    if (evt == CV_EVENT_LBUTTONDOWN)
    {
        MouseParamsRegistrationAnnotation* ptPtr = (MouseParamsRegistrationAnnotation*)mouse_params;
        (ptPtr->landmarks).push_back(cv::Point(x,y));
        //std::cout << x << " " << y << std::endl;
        if (((ptPtr->landmarks).size() % 2)==0)
            ptPtr->must_draw_line = true;
    }

}

