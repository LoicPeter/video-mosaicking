#include "interaction_loop.hpp"
#include "agent.hpp"
#include <queue>


bool isScoreHigher(const UnlabeledImagePair& pair_1 , const UnlabeledImagePair& pair_2)
{
    return (pair_1.score > pair_2.score);
}

bool isScoreLower(const UnlabeledImagePair& pair_1 , const UnlabeledImagePair& pair_2)
{
    return (pair_1.score < pair_2.score);
}

void createEnblendMosaic(const Eigen::VectorXd& global_positions, const InputData& input_data, const InputSequence& input_sequence, const Settings& settings, std::string folder_name)
{

    int nb_frames = input_data.nb_frames;
    int initial_nb_rows = input_data.nb_rows;
    int initial_nb_cols = input_data.nb_cols;
    int factor_size_mosaic_rows, factor_size_mosaic_cols, resize_factor, subsampling, x0, y0, nb_rows, nb_cols;
    bool enblend_fusion;
    std::vector<int> indices_of_interest;
    
    if (input_data.dataset_identifier=="Fetoscopy")
    {
        factor_size_mosaic_rows = 5;
        factor_size_mosaic_cols = 5;
        resize_factor = 1;
        subsampling = 2;
        nb_rows = factor_size_mosaic_rows*initial_nb_rows/resize_factor;
        nb_cols = factor_size_mosaic_cols*initial_nb_cols/resize_factor;
        x0 = nb_cols/2;
        y0 = nb_rows/2;
        enblend_fusion = true;
    }
    
    
    if (input_data.dataset_identifier=="Aerial")
    {
        factor_size_mosaic_rows = 27;
        factor_size_mosaic_cols = 20;
        resize_factor = 7;
        subsampling = 2;
        nb_rows = factor_size_mosaic_rows*initial_nb_rows/resize_factor;
        nb_cols = factor_size_mosaic_cols*initial_nb_cols/resize_factor;
        x0 = 50;
        y0 = nb_rows - 700 - initial_nb_rows/resize_factor;
        enblend_fusion = true;
    }

    
    if (input_data.dataset_identifier=="CircleWrongExternal")
    {
        factor_size_mosaic_rows = 10;
        factor_size_mosaic_cols = 9;
        resize_factor = 1;
        subsampling = 25;
        nb_rows = factor_size_mosaic_rows*initial_nb_rows/resize_factor;
        nb_cols = factor_size_mosaic_cols*initial_nb_cols/resize_factor;
        x0 = 50;
        y0 = nb_rows/4;
        enblend_fusion = false;
    }

    if (input_data.dataset_identifier=="Raster")
    {
        factor_size_mosaic_rows = 12;
        factor_size_mosaic_cols = 60;
        resize_factor = 2;
        subsampling = 2;
        nb_rows = factor_size_mosaic_rows*initial_nb_rows/resize_factor;
        nb_cols = factor_size_mosaic_cols*initial_nb_cols/resize_factor;
        x0 = 250;
        y0 = 14*nb_rows/60;
        enblend_fusion = false;
        for (int i=0; i<=148; i+=2)
            indices_of_interest.push_back(i);
     //   indices_of_interest.push_back(149);
        for (int i=151; i<300; i+=2)
            indices_of_interest.push_back(i);
    }

    if (indices_of_interest.size()==0)
    {
        for (int i=0; i<nb_frames; i+=subsampling)
            indices_of_interest.push_back(i);
    }
    
    cv::Mat mosaic(nb_rows,nb_cols,CV_8UC3,cv::Scalar::all(0)), white_mosaic(nb_rows,nb_cols,CV_8UC3,cv::Scalar::all(255));
    cv::Mat mosaic_temp(nb_rows,nb_cols,CV_8UC3,cv::Scalar::all(0)), deformed_mosaic_temp(nb_rows,nb_cols,CV_8UC3,cv::Scalar::all(0));
    cv::Mat mask_temp(nb_rows,nb_cols,CV_8UC1,cv::Scalar::all(0)), deformed_mask_temp(nb_rows,nb_cols,CV_8UC1,cv::Scalar::all(0));
    cv::Mat frame_with_alpha_channel(nb_rows,nb_cols,CV_8UC4,cv::Scalar::all(0));

    cv::Mat ref_homography;
    affine_vector_to_matrix(ref_homography,global_positions.segment<6>(6*settings.ind_reference_frame));
    cv::Mat inv_ref_homography = ref_homography.inv();

    cv::Mat rescaling_matrix = (1/((double)resize_factor))*cv::Mat::eye(3,3,CV_64F);
    rescaling_matrix.at<double>(0,2) = x0;
    rescaling_matrix.at<double>(1,2) = y0;
    rescaling_matrix.at<double>(2,2) = 1;
    cv::Mat rescaling_matrix_inv = rescaling_matrix.inv();
    
    
    std::ofstream infile(input_data.dataset_identifier + "/" + folder_name + "/list_files.txt");
    for (int i : indices_of_interest)
    {
        std::ostringstream oss_i;
        oss_i << i;

        cv::Mat current_resized_frame, current_resized_mask;
        cv::resize(input_sequence.frames[i],current_resized_frame,cv::Size(),1/((double)resize_factor),1/((double)resize_factor),cv::INTER_AREA);
        cv::resize(input_sequence.masks[i],current_resized_mask,cv::Size(),1/((double)resize_factor),1/((double)resize_factor),cv::INTER_NEAREST);
        
        // Compute central region (frame dependent)
        cv::Rect central_region(x0,y0,current_resized_frame.cols,current_resized_frame.rows);
        current_resized_frame.copyTo(mosaic_temp(central_region),current_resized_mask);
        current_resized_mask.copyTo(mask_temp(central_region));

        // Compute the matrix (if we fixed a reference frame, the new global homography is computed for each frame without changing the original mosaic)
        cv::Mat adjusted_matrix, current_homography;
        affine_vector_to_matrix(current_homography,global_positions.segment<6>(6*i));
        adjusted_matrix = rescaling_matrix*current_homography*rescaling_matrix_inv;

        // Warp frame i
        cv::warpPerspective(mosaic_temp, deformed_mosaic_temp, adjusted_matrix, mosaic_temp.size(), cv::WARP_INVERSE_MAP);
        cv::warpPerspective(mask_temp, deformed_mask_temp, adjusted_matrix, mask_temp.size(), cv::WARP_INVERSE_MAP);

        // Split frame
        std::vector<cv::Mat> split_color_channels(3);
        cv::split(deformed_mosaic_temp,split_color_channels);
        split_color_channels.push_back(deformed_mask_temp);
        cv::merge(split_color_channels,frame_with_alpha_channel);

        // Save frame i
        std::string image_name = "Frame" + oss_i.str() + ".png";
        cv::imwrite(input_data.dataset_identifier + "/" + folder_name + "/" + image_name,frame_with_alpha_channel);
        //if ((i % 2)==0)
            infile << image_name << std::endl;
            
        if (!enblend_fusion)
        {
           cv::add(mosaic,deformed_mosaic_temp,mosaic);
        }

    }
    
    infile.close();
    
    if (enblend_fusion)
    {
        std::string cmd_enblend = "enfuse --hard-mask --output=" + input_data.dataset_identifier + "/" + folder_name + "/mosaic.png @" + input_data.dataset_identifier + "/" + folder_name + "/list_files.txt";
        int success = system(cmd_enblend.c_str());
    }
    else
    {
        cv::imwrite(input_data.dataset_identifier + "/" + folder_name + "/mosaic.png",white_mosaic - mosaic);
    }

}


void remap_affine_correspondences(std::vector<cv::Point2f>& input_points, std::vector<cv::Point2f>& output_points, const std::vector<cv::Point2f>& new_input_points)
{
    
    // Estimate the affine transformation
    cv::Mat mean_transformation, dummy_output_scaled_covariance_matrix;
    getAffineDistributionFromLandmarks(mean_transformation, dummy_output_scaled_covariance_matrix,input_points,output_points);
    
    // Map the points
    input_points = new_input_points;
    getDisplacedPointsAffine(output_points,(double*)(mean_transformation.data),input_points);
}


void perform_label_count(std::vector<int>& label_count, const std::vector<LabeledImagePair>& labeled_pairs)
{
    label_count = std::vector<int>(2,0);
    int nb_labeled_pairs = (int)labeled_pairs.size();
    for (int ind_pair=0; ind_pair<nb_labeled_pairs; ++ind_pair)
    {
        if (labeled_pairs[ind_pair].this_pair_overlaps)
            ++label_count[1];
        else
            ++label_count[0];
    }
}

double getMeanDisplacementErrorRelative(const Eigen::VectorXd& estimated_absolute_positions, GroundTruthData& ground_truth_data)
{
    int nb_frames = ground_truth_data.gt_database.getNbFrames();
    double mean_displacement_error(0);
    std::vector<double> displacement_errors;
    int total_nb_points_for_evaluation(0);
    std::vector<cv::Point2f> landmarks_i, displaced_landmarks_j, landmarks_j;
    Eigen::Matrix3d T_ij, T_i, T_j;
    for (int i=0; i<nb_frames; ++i)
    {
        affine_vector_to_matrix(T_i,estimated_absolute_positions.segment<6>(6*i));
        for (int j=0; j<nb_frames; ++j) // seems better to incldue all pairs in both directions to avoid biases
        {
            if ((j!=i) && (ground_truth_data.gt_database.isPairInDatabase(i,j)))
            {
                //std::cout << "Pair (" << i << "," << j << ")" << std::endl;
                ground_truth_data.gt_database.loadLandmarks(i,j,landmarks_i,landmarks_j);
                affine_vector_to_matrix(T_j,estimated_absolute_positions.segment<6>(6*j));
                T_ij = T_i*(T_j.inverse());
                T_ij.transposeInPlace(); // we do this because we are going to call "data()", and Eigen stores column-major by default (and our vectorization  follows a row-major ordering)
                getDisplacedPointsAffine(displaced_landmarks_j,T_ij.data(),landmarks_j);
                int nb_landmarks = (int)landmarks_j.size();
                for (int p=0; p<nb_landmarks; ++p)
                {
                    double error_landmark = sqrt(std::pow<double>(displaced_landmarks_j[p].x - landmarks_i[p].x,2) + std::pow<double>(displaced_landmarks_j[p].y - landmarks_i[p].y,2));
                    mean_displacement_error += error_landmark;
                    displacement_errors.push_back(error_landmark);
                    
//                     if (p==0)
//                        std::cout << "(" << i << "," << j << "): Error on landmark #" << p << " = " << error_landmark << std::endl;
                }
                
                total_nb_points_for_evaluation += nb_landmarks;
            }
        }
    }

    

    
    
    mean_displacement_error = std::sqrt(mean_displacement_error/((double)total_nb_points_for_evaluation));

    return mean_displacement_error;
}

double getMeanDisplacementErrorAbsolute(const Eigen::VectorXd& estimated_absolute_positions, GroundTruthData& ground_truth_data)
{
    int nb_frames = ground_truth_data.gt_database.getNbFrames();
    double mean_displacement_error(0);
    std::vector<double> displacement_errors;
    int total_nb_points_for_evaluation(0);
    std::vector<cv::Point2f> landmarks_i, landmarks_j, displaced_landmarks_i, displaced_landmarks_j;
    Eigen::Matrix3d T_ij, T_i, T_j, T_ri, T_rj;
    for (int i=0; i<nb_frames; ++i)
    {
        affine_vector_to_matrix(T_i,estimated_absolute_positions.segment<6>(6*i));
        T_ri = T_i.inverse();
        T_ri.transposeInPlace(); // we do this because we are going to call "data()", and Eigen stores column-major by default (and our vectorization  follows a row-major ordering)

        for (int j=0; j<nb_frames; ++j) // seems better to incldue all pairs in both directions to avoid biases
        {
            if ((j!=i) && (ground_truth_data.gt_database.isPairInDatabase(i,j)))
            {
                //std::cout << "Pair (" << i << "," << j << ")" << std::endl;
                ground_truth_data.gt_database.loadLandmarks(i,j,landmarks_i,landmarks_j);
                affine_vector_to_matrix(T_j,estimated_absolute_positions.segment<6>(6*j));
                T_rj = T_j.inverse();
                T_rj.transposeInPlace(); // we do this because we are going to call "data()", and Eigen stores column-major by default (and our vectorization  follows a row-major ordering)
                
                getDisplacedPointsAffine(displaced_landmarks_i,T_ri.data(),landmarks_i);
                getDisplacedPointsAffine(displaced_landmarks_j,T_rj.data(),landmarks_j);
                int nb_landmarks = (int)landmarks_j.size();
                for (int p=0; p<nb_landmarks; ++p)
                {
                    double error_landmark = std::pow<double>(displaced_landmarks_j[p].x - displaced_landmarks_i[p].x,2) + std::pow<double>(displaced_landmarks_j[p].y - displaced_landmarks_i[p].y,2);
                    mean_displacement_error += error_landmark;
                    displacement_errors.push_back(error_landmark);
                    
//                     if (p==0)
//                         std::cout << "(" << i << "," << j << "): Error on landmark #" << p << " = " << error_landmark << std::endl;
                }
                
                total_nb_points_for_evaluation += nb_landmarks;
            }
        }
    }

    

    
    
    mean_displacement_error = std::sqrt(mean_displacement_error/((double)total_nb_points_for_evaluation));

   // std::sort(displacement_errors.begin(),displacement_errors.end());
  //  double median_displacement_error = displacement_errors[(int)displacement_errors.size()/2];
    
    return mean_displacement_error;
}

void symmetrise_and_save(const std::string& filename, const cv::Mat& to_save, double diagonal_value, const std::vector<LabeledImagePair>& overlap_training_set)
{
    int nb_frames = to_save.rows;
    cv::Mat to_save_symmetric(nb_frames,nb_frames,CV_64F,cv::Scalar::all(diagonal_value));

    for (int i=0; i<nb_frames; ++i)
    {
        for (int j=(i+1); j<nb_frames; ++j)
        {
            to_save_symmetric.at<double>(i,j) = to_save.at<double>(i,j);
            to_save_symmetric.at<double>(j,i) = to_save.at<double>(i,j);
        }
    }
    
    cv::Mat to_save_uchar;
    to_save_symmetric.convertTo(to_save_uchar,CV_8UC1,255.0);   
    cv::Mat to_save_jet;
    cv::applyColorMap(to_save_uchar,to_save_jet,cv::COLORMAP_OCEAN);
    
    int nb_labeled_pairs = (int)overlap_training_set.size();
    for (int k=0; k<nb_labeled_pairs; ++k)
    {
        int i = overlap_training_set[k].image_indices.first;
        int j = overlap_training_set[k].image_indices.second;
        if (abs(i-j)>1)
        {
            cv::Scalar color;
            if (overlap_training_set[k].this_pair_overlaps)
                color = cv::Scalar(0,200,0);
            else
                color = cv::Scalar(0,0,0);
            cv::drawMarker(to_save_jet,cv::Point2f(i,j),color, cv::MARKER_CROSS,50,3);
            cv::drawMarker(to_save_jet,cv::Point2f(j,i),color, cv::MARKER_CROSS,50,3);
        }
    }
        
       
    cv::imwrite(filename,to_save_jet);
}




void symmetrise_and_save(const std::string& filename, const cv::Mat& to_save, double diagonal_value, const std::pair<int,int>& queried_pair)
{
    int nb_frames = to_save.rows;
    cv::Mat to_save_symmetric(nb_frames,nb_frames,CV_64F,cv::Scalar::all(diagonal_value));

    for (int i=0; i<nb_frames; ++i)
    {
        for (int j=(i+1); j<nb_frames; ++j)
        {
            to_save_symmetric.at<double>(i,j) = to_save.at<double>(i,j);
            to_save_symmetric.at<double>(j,i) = to_save.at<double>(i,j);
        }
    }
    
    cv::Mat to_save_uchar;
    to_save_symmetric.convertTo(to_save_uchar,CV_8UC1,255.0);   
    cv::Mat to_save_jet;
    cv::applyColorMap(to_save_uchar,to_save_jet,cv::COLORMAP_OCEAN);
    
    int i = queried_pair.first;
    int j = queried_pair.second;
    cv::Scalar color= cv::Scalar(0,200,200);
    //cv::drawMarker(to_save_jet,cv::Point2f(i,j),color, cv::MARKER_CROSS,50,3);
    //cv::drawMarker(to_save_jet,cv::Point2f(j,i),color, cv::MARKER_CROSS,50,3);
        
       
    cv::imwrite(filename,to_save_jet);
}


void compute_overlap_probabilities_and_bounds_for_all_pairs(std::vector<UnlabeledImagePair>& list_candidate_pairs, cv::Mat& sampling_based_i_in_j, cv::Mat& closed_form_overlap_probability_low_matrix, cv::Mat& closed_form_overlap_probability_up_matrix, cv::Mat& uncertainty_reward_matrix, const cv::Mat& external_overlap_probability_matrix, const PositionOverlapModel& position_overlap_model, Settings& settings, const InputData& input_data)
{
 
    int nb_frames = input_data.nb_frames;
    
    cv::Mat shortest_path_matrix;

    // Prepare the list of candidate pairs ranked in decreasing order of external overlap probability
    int nb_pairs = (nb_frames-1)*(nb_frames-2)/2;
    std::vector<UnlabeledImagePair> list_candidate_pairs_external_overlap(nb_pairs);
    int ind=0;
    for (int i=0; i<nb_frames; ++i)
    {
        for (int j=(i+2); j<nb_frames; ++j)
        {
            
            if (i<j)
            {
                list_candidate_pairs_external_overlap[ind].image_indices = std::pair<int,int>(i,j);
                list_candidate_pairs_external_overlap[ind].score = external_overlap_probability_matrix.at<double>(i,j);
                ++ind;
            }
        }
    }
    std::sort(list_candidate_pairs_external_overlap.begin(),list_candidate_pairs_external_overlap.end(),isScoreHigher);
    
    
    // input_data.observed_pairwise_registrations[(int)input_data.observed_pairwise_registrations.size()-1]
    Graph<double> connectivity_graph(nb_frames);
    if (settings.chosen_reward == shortest_path)
    {
        int nb_observed_registrations = input_data.observed_pairwise_registrations.size();
        for (int k=0; k<nb_observed_registrations; ++k)
        {
            int i = input_data.observed_pairwise_registrations[k].frame_indices.first;
            int j = input_data.observed_pairwise_registrations[k].frame_indices.second;
            double sawhney_score;
            double sawhney_proba = position_overlap_model.compute_Sawhney_probability(sawhney_score,i,j,input_data.overlap_corners);
            connectivity_graph.add_edge(i,j,sawhney_score);
        }
        
        getDistanceBetweenAllVertices(shortest_path_matrix,connectivity_graph);
    }
    
    // Build a min-heap in which the top "nb_possible_skips" candidates in terms of expected reward are stored
    int nb_possible_skips = settings.nb_possible_skips;
    std::priority_queue<UnlabeledImagePair,std::vector<UnlabeledImagePair>,CompareUnlabeledImagePair_IsHigher> top_pairs;
    
    uncertainty_reward_matrix = cv::Mat(nb_frames,nb_frames,CV_64F,cv::Scalar::all(0));
    
    for (int ind=0; ind<nb_pairs; ++ind)
    {
        int i = list_candidate_pairs_external_overlap[ind].image_indices.first;
        int j = list_candidate_pairs_external_overlap[ind].image_indices.second;
             
        sampling_based_i_in_j.at<double>(i,j) = 0; // initialization (we remove previous information)
        uncertainty_reward_matrix.at<double>(i,j) = 0;
        double external_overlap_probability = list_candidate_pairs_external_overlap[ind].score;
        
        double closed_form_overlap_probability_low, closed_form_overlap_probability_up, uncertainty_reward, position_based_overlap_probability(-1);
        
//        double reverse_reward;
//        position_overlap_model.compute_overlap_bounds_and_reward(closed_form_overlap_probability_low, closed_form_overlap_probability_up, reverse_reward,j,i,settings.chosen_reward);
//        uncertainty_reward_matrix.at<double>(j,i) = reverse_reward;
       position_overlap_model.compute_symmetrical_overlap_bounds_and_reward(closed_form_overlap_probability_low, closed_form_overlap_probability_up, uncertainty_reward,i,j,settings.chosen_reward);
        
        if (settings.chosen_reward == shortest_path)
        {
            double eps = 0.00001;
            double arc_length = connectivity_graph.get_edge(i,j);
            uncertainty_reward = std::max<double>(shortest_path_matrix.at<double>(i,j)/(arc_length + eps) - 1,0);
        }
        
        if (settings.chosen_reward == no_reward)
            uncertainty_reward = 1;
        
        if (settings.chosen_reward == xia_reward)
        {
            double index_pair = i*nb_frames + j;
            uncertainty_reward = nb_frames*nb_frames - index_pair;
        }

        uncertainty_reward_matrix.at<double>(i,j) = uncertainty_reward;
       
       
        if ((ind<nb_possible_skips) || (settings.compute_overlap_probability_for_all_pairs)) // if we do not have enough candidates yet
        {
            // We run a more complete estimation of the overlap probability
            if (settings.chosen_positional_overlap_model == no_positional_overlap_model)
            {
                closed_form_overlap_probability_low = 1;
                closed_form_overlap_probability_up = 1;
                position_based_overlap_probability = 1;
            }
            
            if ((settings.chosen_positional_overlap_model == sawhney_probability) || (settings.chosen_positional_overlap_model == xia_filtering))
            {
                double sawhney_score;
                double sawhney_proba = position_overlap_model.compute_Sawhney_probability(sawhney_score,i,j,input_data.overlap_corners);
                closed_form_overlap_probability_low = sawhney_proba;
                closed_form_overlap_probability_up = sawhney_proba;
                position_based_overlap_probability = sawhney_proba;
                if (settings.chosen_positional_overlap_model == xia_filtering)
                {
                    if (sawhney_proba>0)
                    {
                        position_based_overlap_probability = 1;
                        closed_form_overlap_probability_low = 1;
                        closed_form_overlap_probability_up = 1;
                    }
                }
            }
            
            if ((settings.chosen_positional_overlap_model == our_positional_overlap_model) || ((settings.chosen_positional_overlap_model == elibol_filtering_low) && (closed_form_overlap_probability_low>=0.01)) || ((settings.chosen_positional_overlap_model == elibol_filtering_up) && (closed_form_overlap_probability_up>=0.99)))
                position_overlap_model.compute_symmetrical_overlap_probability_with_sampling(position_based_overlap_probability,i,j,settings.nb_samples_montecarlo_overlap_estimation,settings.rng);
           
            // Expected reward
            double expected_reward = position_based_overlap_probability*external_overlap_probability*uncertainty_reward;
            
            UnlabeledImagePair current_pair_expected_reward;
            current_pair_expected_reward.image_indices = std::pair<int,int>(i,j);
            current_pair_expected_reward.score = expected_reward;
            top_pairs.push(current_pair_expected_reward);
            
            // Update the map in any case
            sampling_based_i_in_j.at<double>(i,j) = position_based_overlap_probability;
            
        }
        else
        {
            double upper_bound_expected_reward =  closed_form_overlap_probability_up*external_overlap_probability*uncertainty_reward;
            
            UnlabeledImagePair worst_top_pair = top_pairs.top();
            if  (upper_bound_expected_reward>worst_top_pair.score) // check if the upper bound is higher than the worst expected reward in the top pairs
            {
                // We run a more complete estimation of the overlap probability
                if (settings.chosen_positional_overlap_model == no_positional_overlap_model)
                {
                    closed_form_overlap_probability_low = 1;
                    closed_form_overlap_probability_up = 1;
                    position_based_overlap_probability = 1;
                }
                
                if ((settings.chosen_positional_overlap_model == sawhney_probability) || (settings.chosen_positional_overlap_model == xia_filtering))
                {
                    double sawhney_score;
                    double sawhney_proba = position_overlap_model.compute_Sawhney_probability(sawhney_score,i,j,input_data.overlap_corners);
                    closed_form_overlap_probability_low = sawhney_proba;
                    closed_form_overlap_probability_up = sawhney_proba;
                    position_based_overlap_probability = sawhney_proba;
                    if (settings.chosen_positional_overlap_model == xia_filtering)
                    {
                        if (sawhney_proba>0)
                        {
                            position_based_overlap_probability = 1;
                            closed_form_overlap_probability_low = 1;
                            closed_form_overlap_probability_up = 1;
                        }
                    }
                }
                
                if ((settings.chosen_positional_overlap_model == our_positional_overlap_model) || ((settings.chosen_positional_overlap_model == elibol_filtering_low) && (closed_form_overlap_probability_low>=0.01)) || ((settings.chosen_positional_overlap_model == elibol_filtering_up) && (closed_form_overlap_probability_up>=0.99)))
                    position_overlap_model.compute_symmetrical_overlap_probability_with_sampling(position_based_overlap_probability,i,j,settings.nb_samples_montecarlo_overlap_estimation,settings.rng);
                
                double expected_reward = position_based_overlap_probability*external_overlap_probability*uncertainty_reward;
                
                if (expected_reward>worst_top_pair.score)
                {
                    UnlabeledImagePair current_pair_expected_reward;
                    current_pair_expected_reward.image_indices = std::pair<int,int>(i,j);
                    current_pair_expected_reward.score = expected_reward;
                    top_pairs.pop(); // we remove the worst top pair
                    top_pairs.push(current_pair_expected_reward); // we add the better one we just found
                }
            
                // Update the map in any case
                sampling_based_i_in_j.at<double>(i,j) = position_based_overlap_probability;
            }
            
        }

        // Update the visualization maps
        closed_form_overlap_probability_low_matrix.at<double>(i,j) = closed_form_overlap_probability_low;
        closed_form_overlap_probability_up_matrix.at<double>(i,j) = closed_form_overlap_probability_up;
        uncertainty_reward_matrix.at<double>(i,j) = uncertainty_reward;
        
        if ((position_based_overlap_probability>0.01) && (position_based_overlap_probability<0.99)) // if we've computed it and it is significant
        {
            if ((closed_form_overlap_probability_up<position_based_overlap_probability) || (closed_form_overlap_probability_low>position_based_overlap_probability))
                std::cout << "WARNING! Lower bound: " << closed_form_overlap_probability_low << " - Numerical Probability: " << position_based_overlap_probability << " - Upper bound: " << closed_form_overlap_probability_up << std::endl;
        }
    }

    
    // Finally, we create the list of top candidates sorted in order of decreasing expected reward
    int nb_top_pairs = (int)top_pairs.size(); // should be nb_possible_skips
    list_candidate_pairs.resize(nb_top_pairs);
    for (int ind=(nb_top_pairs-1); ind>=0; --ind)
    {
        list_candidate_pairs[ind] = top_pairs.top();
        top_pairs.pop();
    }
}



void cotrain_overlap_models(ExternalOverlapModel& bow_overlap_model, PositionOverlapModel& position_overlap_model,  PositionOverlapModel& initial_position_overlap_model, std::vector<LabeledImagePair>& fixed_training_pairs, InputData& input_data, Settings& settings)
{    
    // Count number of positives and negatives
    int nb_fixed_pairs = (int)fixed_training_pairs.size();
    std::vector<int> label_count;
    perform_label_count(label_count,fixed_training_pairs);

    if ((settings.reestimate_external_overlap_model) && (label_count[0]!=0))
    {
        // Preallocate the training sets
        std::vector<LabeledImagePair> training_set_external(nb_fixed_pairs);
        for (int k=0; k<nb_fixed_pairs; ++k)
        {
            training_set_external[k] = fixed_training_pairs[k];
            training_set_external[k].importance_weight = 1;
        }
        
        std::cout << "Train external model" << std::endl;
        bow_overlap_model.train(training_set_external,input_data.bow_appearance_matrix);
    }
}



void run_interactive_process(GroundTruthData& ground_truth_data, InputData& input_data, const InputSequence& input_sequence, InteractiveHelper& interactive_helper, Settings& settings)
{
      // Window for display
    std::string expected_reward_window_name = "Expected Reward";
    std::string uncertainty_window_name = "Uncertainty";
    std::string montecarlo_uncertainty_window_name = "MC Uncertainty";
    std::string empirical_overlap_position_window_name = "Empirical Overlap Probabilities Position";
    std::string upper_bound_overlap_position_window_name = "Upper Bound";
    std::string lower_bound_overlap_position_window_name = "Lower Bound";
    std::string overlap_visual_window_name = "Overlap Probabilities Visual";
////    std::string sum_variances_window_name = "Sum X-Variances";
    cv::namedWindow(expected_reward_window_name,cv::WINDOW_KEEPRATIO);
    cv::namedWindow(uncertainty_window_name,cv::WINDOW_KEEPRATIO);
    cv::namedWindow(empirical_overlap_position_window_name,cv::WINDOW_KEEPRATIO);
    cv::namedWindow(upper_bound_overlap_position_window_name,cv::WINDOW_KEEPRATIO);
    cv::namedWindow(lower_bound_overlap_position_window_name,cv::WINDOW_KEEPRATIO);
    cv::namedWindow(overlap_visual_window_name,cv::WINDOW_KEEPRATIO);
    cv::namedWindow(montecarlo_uncertainty_window_name,cv::WINDOW_KEEPRATIO);
//   // namedWindow(sum_variances_window_name,WINDOW_KEEPRATIO);

    // Clock
    clock_t elapsed_time; 
    
    int nb_frames = input_data.nb_frames;
    int nb_added_annotations(0);
    int nb_measurements = input_data.observed_pairwise_registrations.size();
    

    if (settings.remap_correspondences)
    {
        for (int m=0; m<nb_measurements; ++m)
            remap_affine_correspondences(input_data.observed_pairwise_registrations[m].landmarks_input,input_data.observed_pairwise_registrations[m].landmarks_output,input_data.overlap_corners);
    }
    
    // Here, we define what measurements have the same variances
    int nb_modes;
    if (input_data.agent_type==landmark_registration)
        nb_modes = 1; // number of terms that have a different variance weight (here, one for automated registrations, one for user-based ones)
    else
        nb_modes = 2;
    std::vector<double> estimated_variances(nb_modes,settings.annotation_variance); // default variance: annotation one
    estimated_variances[0] = settings.automatic_registration_variance;
    estimated_variances[1] = settings.annotation_variance;
    std::vector<int> measurement_modes(nb_measurements,0);

    std::cout << "Compute position overlap model" << std::endl;
    
    Eigen::Vector2d gamma;
    gamma(0) = ((double)input_data.nb_cols)/2;
    gamma(1) = ((double)input_data.nb_rows)/2;
    
    
    //double size_overlap_area = std::min<int>(input_data.dimensions_overlap_area[0],input_data.dimensions_overlap_area[1]);
    double size_overlap_area = std::sqrt((double)std::pow<int>(input_data.dimensions_overlap_area[0],2) + (double)std::pow<int>(input_data.dimensions_overlap_area[1],2));
    PositionOverlapModel position_overlap_model(input_data.observed_pairwise_registrations,measurement_modes,estimated_variances,settings.ind_reference_frame,nb_frames,gamma,size_overlap_area);
    PositionOverlapModel initial_position_overlap_model = position_overlap_model;
    std::cout << "Position overlap model created" << std::endl;
    
    
    // Training set of overlapping pairs
    std::vector<LabeledImagePair> overlap_training_set;
    for (int ind_measurement=0; ind_measurement<nb_measurements; ++ind_measurement)
    {
        LabeledImagePair pair_to_add;
        pair_to_add.this_pair_overlaps = true;
        pair_to_add.image_indices = input_data.observed_pairwise_registrations[ind_measurement].frame_indices;
        pair_to_add.importance_weight = 1;
        overlap_training_set.push_back(pair_to_add);
    }

    if (settings.createVideos)
    {
//         estimated_absolute_positions_without_ref = A_cholesky.solve(b);
//         add_reference_frame_to_global_positions_affine(estimated_absolute_positions,estimated_absolute_positions_without_ref,settings.ind_reference_frame);
//         
//         Eigen::VectorXd estimated_absolute_positions_inv; // we must invert it if  BA is now done on inverses
//         if (settings.bundle_adjustment_on_correspondences)
//             inverse_vector_of_estimated_affine_global_positions(estimated_absolute_positions_inv,estimated_absolute_positions);
//         else
//             estimated_absolute_positions_inv = estimated_absolute_positions;
 //       createMosaicVideo("TestVideo-BeforeInteraction.avi",(double*)estimated_absolute_positions_inv.data(),input_data,input_sequence);
        
        createEnblendMosaic(position_overlap_model.get_mean_estimated_absolute_positions(),input_data,input_sequence,settings,"InitialMosaic");
    }
    
    // Initialise matrices
    cv::Mat numerical_overlap_probabilities(nb_frames,nb_frames,CV_64F,cv::Scalar::all(1)), predicted_overlap_probabilities(nb_frames,nb_frames,CV_64F,cv::Scalar::all(1));
    cv::Mat overlap_probabilities_visual(nb_frames,nb_frames,CV_64F,cv::Scalar::all(1));
    cv::Mat joint_overlap_probabilities(nb_frames,nb_frames,CV_64F,cv::Scalar::all(1));
    cv::Mat uncertainty_matrix(nb_frames,nb_frames,CV_64F,cv::Scalar::all(0)), product_matrix(nb_frames,nb_frames,CV_64F,cv::Scalar::all(0));
    cv::Mat montecarlo_uncertainty_matrix(nb_frames,nb_frames,CV_64F,cv::Scalar::all(0));
    cv::Mat expected_uncertainty_matrix(nb_frames,nb_frames,CV_64F,cv::Scalar::all(0));
    cv::Mat closed_form_overlap_probability_low_matrix(nb_frames,nb_frames,CV_64F,cv::Scalar::all(1)); // "exact" ones
    cv::Mat closed_form_overlap_probability_up_matrix(nb_frames,nb_frames,CV_64F,cv::Scalar::all(1));
    cv::Mat lower_bound_overlap_probability_matrix(nb_frames,nb_frames,CV_64F,cv::Scalar::all(0)); // approximate simpler ones
    cv::Mat upper_bound_overlap_probability_matrix(nb_frames,nb_frames,CV_64F,cv::Scalar::all(0));

    // Visual overlap model
    std::cout << "Compute visual overlap model" << std::endl;
//    ExternalOverlapModel *bow_overlap_model = new FullPCCAOverlapModel(std::min<int>(20,input_data.nb_bow_descriptors),input_data.nb_bow_descriptors,settings.beta);
    ExternalOverlapModel *bow_overlap_model;
    if (settings.chosen_external_overlap_model == no_external_overlap_model)
        bow_overlap_model = new TrivialExternalModel;
    if (settings.chosen_external_overlap_model == our_external_overlap_model)
        bow_overlap_model = new NonIncreasingDiagonalPCCAOverlapModel(input_data.nb_bow_descriptors,settings.beta);

    bow_overlap_model->_getOverlapProbabilities(overlap_probabilities_visual,input_data.bow_appearance_matrix);


    // Labeling interface
    double max_uncertainty_first_iteration;
    bool stop_labelling(false);
    bool is_there_a_non_overlapping_pair(false);
    int is_there_overlap;
    int nb_pairs = (nb_frames*(nb_frames-1))/2;
    std::vector<UnlabeledImagePair> candidate_pairs(nb_pairs);
    std::map<std::pair<int,int>,Eigen::MatrixXd> left_cov_propagations_relative_cov_matrix;
    std::map<std::pair<int,int>,double> displacement_ij;
    std::map<std::pair<int,int>,Eigen::Vector2d> mean_vector_ij;
    
    // Store the results
    std::ofstream infile(input_data.dataset_identifier + "/" + settings.experiment_identifier + ".txt");

    for (int it=0; it<settings.nb_iterations_interaction; ++it)
    {

        if (ground_truth_data.is_available)
        {
            double error;
            error = getMeanDisplacementErrorAbsolute(position_overlap_model.get_mean_estimated_absolute_positions(),ground_truth_data);
            std::cout << "Mean displacement error (absolute): " << error << std::endl;
            infile << error << " ";
            error = getMeanDisplacementErrorRelative(position_overlap_model.get_mean_estimated_absolute_positions(),ground_truth_data);
            std::cout << "Mean displacement error (relative): " << error << std::endl;
            infile << error << " ";
        }
 
        
        if ((settings.reestimate_external_overlap_model) || (settings.reestimate_position_overlap_model))
        {
            std::cout << "Cotrain models" << std::endl;
            cotrain_overlap_models(*bow_overlap_model,position_overlap_model,initial_position_overlap_model,overlap_training_set,input_data,settings);
        }
        
        std::cout << "Compute external probabilities" << std::endl;
        bow_overlap_model->_getOverlapProbabilities(overlap_probabilities_visual,input_data.bow_appearance_matrix);
        
        
        std::cout << "Create list of top pairs" << std::endl;
        elapsed_time = clock();
        compute_overlap_probabilities_and_bounds_for_all_pairs(candidate_pairs,predicted_overlap_probabilities,closed_form_overlap_probability_low_matrix, closed_form_overlap_probability_up_matrix,uncertainty_matrix,overlap_probabilities_visual,position_overlap_model,settings,input_data);
        elapsed_time = clock() - elapsed_time;
        
        
        if (it==0)
        {
            double min_dummy;
            cv::minMaxLoc(uncertainty_matrix,&min_dummy,&max_uncertainty_first_iteration);
        }

        
       // For display
       cv::imshow(empirical_overlap_position_window_name,predicted_overlap_probabilities);
       cv::imshow(upper_bound_overlap_position_window_name,closed_form_overlap_probability_up_matrix);
       cv::imshow(lower_bound_overlap_position_window_name,closed_form_overlap_probability_low_matrix);
       cv::Mat uncertainty_matrix_normalised = (1/max_uncertainty_first_iteration)*uncertainty_matrix;
       //cv::normalize(uncertainty_matrix,uncertainty_matrix_normalised,1,0,cv::NORM_MINMAX);
       cv::imshow(uncertainty_window_name,uncertainty_matrix_normalised);
       cv::imshow(overlap_visual_window_name,overlap_probabilities_visual);
       cv::Mat joint_overlap_probabilities = overlap_probabilities_visual.mul(predicted_overlap_probabilities);
       cv::Mat expected_reward_matrix = joint_overlap_probabilities.mul(uncertainty_matrix);
       cv::Mat expected_reward_matrix_normalised = joint_overlap_probabilities.mul(uncertainty_matrix_normalised);
       cv::normalize(expected_reward_matrix,expected_reward_matrix_normalised,1,0,cv::NORM_MINMAX);
       cv::imshow(expected_reward_window_name,expected_reward_matrix);
       
       
       
       // Save
       std::ostringstream oss_it;
       oss_it << it;
       std::string path_figures = input_data.dataset_identifier + "/Figures/";
       symmetrise_and_save(path_figures + "PositionOverlap-" + oss_it.str() + ".png",predicted_overlap_probabilities,1,overlap_training_set);
       symmetrise_and_save(path_figures + "ExternalOverlap-" + oss_it.str() + ".png",overlap_probabilities_visual,1,overlap_training_set);
       symmetrise_and_save(path_figures + "Reward-" + oss_it.str() + ".png",uncertainty_matrix_normalised,0,overlap_training_set);

       symmetrise_and_save(path_figures + "PositionOverlapNoCross-" + oss_it.str() + ".png",predicted_overlap_probabilities,1);
       symmetrise_and_save(path_figures + "LowerBound-" + oss_it.str() + ".png",closed_form_overlap_probability_low_matrix,1);
       symmetrise_and_save(path_figures + "UpperBound-" + oss_it.str() + ".png",closed_form_overlap_probability_up_matrix,1);
       
       cv::waitKey(1000);

        std::cout << "Query" << std::endl;
        int ind_max_reward(0);
        int i_query, j_query;
        for (int ind_max_reward=0; ind_max_reward<nb_pairs; ++ind_max_reward)
        {
            i_query = candidate_pairs[ind_max_reward].image_indices.first;
            j_query = candidate_pairs[ind_max_reward].image_indices.second;

            std::cout << "Asked pair: (" << i_query << "," << j_query << ")" << std::endl;
            std::cout << "Reward: " << candidate_pairs[ind_max_reward].score << std::endl;
            std::cout << "Probability overlap based on position: " << predicted_overlap_probabilities.at<double>(i_query,j_query) << std::endl;
            std::cout << "Probability overlap based on visual similarity: " << overlap_probabilities_visual.at<double>(i_query,j_query) << std::endl;

            
            cv::imwrite(path_figures + "QueriedFrame-i-" + oss_it.str() + ".png",input_sequence.frames[i_query]);
            cv::imwrite(path_figures + "QueriedFrame-j-" + oss_it.str() + ".png",input_sequence.frames[j_query]);
            
            if (input_data.overlap_information.is_there_edge(i_query,j_query)==false) // if we do not know yet if the two pairs overlap (in particular, we do not ask a pair of consecutive images)
            {
                if (input_data.agent_type==precomputed_agent)
                    automaticInteractiveAnnotationCorrespondences(is_there_overlap,input_data,ground_truth_data,settings,i_query,j_query);
                if (input_data.agent_type==landmark_registration)
                    automaticAnnotationViaLandmarkRegistration(is_there_overlap,input_data,input_sequence,i_query,j_query,settings.annotated_points_in_interaction);
                if (input_data.agent_type==human)
                    manualInteractiveQuery(is_there_overlap,input_data,input_sequence,interactive_helper,stop_labelling,i_query,j_query);
                
                if (is_there_overlap==1)
                    std::cout << "The pair does overlap" << std::endl;
                if (is_there_overlap==(-1))
                    std::cout << "The pair does not overlap" << std::endl;

                if ((is_there_overlap!=0) || (stop_labelling)) // if the pair was not declared as doubtful - otherwise suggest the next pair
                    break;
            }
//             else
//             {
//                  // Add pair to training set
//                 LabeledImagePair new_pair;
//                 new_pair.image_indices = std::pair<int,int>(i_query,j_query);
//                 if (input_data.overlap_information.get_edge(i_query,j_query)==1)
//                     new_pair.this_pair_overlaps = true;
//                 else
//                     new_pair.this_pair_overlaps = false;
//                 new_pair.importance_weight = 1;
//                 overlap_training_set.push_back(new_pair);
//             }
        }

        if ((stop_labelling) || (ind_max_reward==nb_pairs))
            break;


        // Add pair to training set
        LabeledImagePair new_pair;
        new_pair.image_indices = std::pair<int,int>(i_query,j_query);
        if (is_there_overlap==1)
            new_pair.this_pair_overlaps = true;
        else
        {
            new_pair.this_pair_overlaps = false;
            is_there_a_non_overlapping_pair = true;
        }
        new_pair.importance_weight = 1;
        overlap_training_set.push_back(new_pair);
        
        infile << i_query << " " << j_query << " " << is_there_overlap << " " << candidate_pairs[ind_max_reward].score << std::endl;
       
        symmetrise_and_save(path_figures + "ExpectedReward-" + oss_it.str() + ".png",expected_reward_matrix_normalised,0,std::pair<int,int>(i_query,j_query));
        
        if (is_there_overlap==1)
        {
                        
            if (settings.remap_correspondences)
                remap_affine_correspondences(input_data.observed_pairwise_registrations[(int)input_data.observed_pairwise_registrations.size()-1].landmarks_input,input_data.observed_pairwise_registrations[(int)input_data.observed_pairwise_registrations.size()-1].landmarks_output,input_data.overlap_corners);
            
            int measurement_mode = nb_modes-1;
            position_overlap_model.add_pairwise_measurement(input_data.observed_pairwise_registrations[(int)input_data.observed_pairwise_registrations.size()-1],measurement_mode);
     
        }


    }

    std::cout << "Interaction loop over" << std::endl;

    if (settings.createVideos)
    {

        std::cout << "Saving video..." << std::endl;
      //  createMosaicVideo("TestVideo-AfterInteraction.avi",(double*)estimated_absolute_positions_inv.data(),input_data,input_sequence);
        createEnblendMosaic(position_overlap_model.get_mean_estimated_absolute_positions(),input_data,input_sequence,settings,"FinalMosaic");
    }
    
    if (ground_truth_data.is_available)
    {
        double error_abs = getMeanDisplacementErrorAbsolute(position_overlap_model.get_mean_estimated_absolute_positions(),ground_truth_data);
        std::cout << "Mean displacement error (absolute): " << error_abs << std::endl;
        double error_rel = getMeanDisplacementErrorRelative(position_overlap_model.get_mean_estimated_absolute_positions(),ground_truth_data);
        std::cout << "Mean displacement error (relative): " << error_rel << std::endl;
        infile << error_abs << " " << error_rel << " -1 -1 0 -1" << std::endl;
    }

    // Save the database of landmarks, if any
    if (settings.automate_interactions==false)
        interactive_helper.database_annotations.save(interactive_helper.landmark_database_identifier);

    infile.close();
    
    delete bow_overlap_model; 
}


