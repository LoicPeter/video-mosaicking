#include "data_preparation.hpp"
#include "linear_algebra.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv_modules.hpp"
#include <opencv2/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/stitching.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "bow_overlap_model.hpp" // for cosine similarity
#include "position_overlap_model.hpp" // for affine distribution from landmarks

// Random number / probabilities / sampling
#include <boost/math/special_functions/erf.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>



void createPointsforDistance(std::vector<cv::Point2f>& points_for_distance_relative, int nb_rows, int nb_cols, double grid_step_size)
{
    points_for_distance_relative.clear();
    for (double t=0; t<=1; t+=grid_step_size)
    {
        for (double u=0; u<=1; u+=grid_step_size)
        {
            points_for_distance_relative.push_back(cv::Point2f(t*nb_cols,u*nb_rows));
        }
    }
}

cv::Mat getMask_U(const cv::Mat& frame)
{
    CV_Assert(frame.depth() == CV_8U);

    int nbRows = frame.rows;
    int nbCols = frame.cols;
    int nbChannels = frame.channels();
    cv::Mat mask(nbRows,nbCols,CV_8UC1);

    // Transform mask
    int border = 10;
    cv::Mat extended_mask(nbRows+2*border,nbCols+2*border,mask.type(),cv::Scalar::all(0));
    cv::Rect roi(border,border,nbCols,nbRows);

    cv::Mat img_grayscale;
    cv::cvtColor(frame, img_grayscale, CV_RGB2GRAY);
    cv::threshold(img_grayscale,mask,10,255,cv::THRESH_BINARY);
    mask.copyTo(extended_mask(roi));
    cv::erode(extended_mask,extended_mask,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(21,21)));
    return extended_mask(roi);
}




void performPreprocessing(cv::Mat& preprocessedImage, const cv::Mat& image, const cv::Mat& mask_U, int sigma_blur) // preprocess before registering: smooth and normalise between 0 and 1
{
    int nbChannels = preprocessedImage.channels();
    if (nbChannels==1) // we want a grayscale matching
    {
        cv::Mat img_grayscale;
        cv::cvtColor(image, img_grayscale, CV_RGB2GRAY);
        img_grayscale.convertTo(preprocessedImage,CV_MAKETYPE(CV_64F,nbChannels));
    }
    else
        image.convertTo(preprocessedImage,CV_MAKETYPE(CV_64F,nbChannels));
    preprocessedImage = preprocessedImage/255;
    //std::cout << sigma_blur << std::endl;
    cv::GaussianBlur(preprocessedImage,preprocessedImage, cv::Size(sigma_blur,sigma_blur), 0, 0, cv::BORDER_DEFAULT );
    //cv::normalize(preprocessedImage,preprocessedImage,1000,0,NORM_L2,-1,mask_U);
}



void sampleDenseKeypoints(std::vector<cv::KeyPoint>& keypoints, const cv::Mat& mask, int stride, int diameter)
{
    CV_Assert(mask.depth() == CV_8U);

    keypoints.clear();
    int nb_rows = mask.rows;
    int nb_cols = mask.cols;
    for (int y=0; y<nb_rows; y+=stride)
    {
        for (int x=0; x<nb_cols; x+=stride)
        {
            if (mask.at<unsigned char>(y,x)!=0)
                keypoints.push_back(cv::KeyPoint(cv::Point2f(x,y),diameter));
        }

    }

}


cv::Mat buildBagOfWords(std::vector<cv::Mat>& image_descriptors, cv::Mat& vocabulary, const std::vector<cv::Mat>& video, const std::vector<cv::Mat>& masks, int cluster_count, int diameter, int stride)
{

    int nb_frames = (int)video.size();

    bool try_cuda = false;
    std::string features_type = "orb";
    std::string matcher_type = "homography";
    float match_conf = 0.05;
    int range_width = -1;
    //Ptr<SURF> sift = cv::xfeatures2d::SURF::create(100,3,1,true,false);
    //Ptr<SIFT> sift_select = cv::xfeatures2d::SIFT::create(70,2,0.04,20,2);

    cv::Ptr<cv::xfeatures2d::VGG> sift = cv::xfeatures2d::VGG::create();
 //   cv::Ptr<cv::xfeatures2d::DAISY> sift = cv::xfeatures2d::DAISY::create();
 //   cv::Ptr<cv::xfeatures2d::SURF> sift = cv::xfeatures2d::SURF::create(300);

    // Bag of words object
    cv::BFMatcher matcher;
    cv::Ptr<cv::BFMatcher> ptr_matcher = matcher.create();
    cv::BOWImgDescriptorExtractor bow_extractor(sift,ptr_matcher);

  //  Ptr<SURF> sift = cv::xfeatures2d::SURF::create(200);

    // IPCAI parameters
//    int cluster_count = 300;
//    int diameter = 25;
//    int stride = 15;

  //  int cluster_count = 100;
  //  int diameter = 15;
  //  int stride = 5;

    // Color extractor
  //  cv::OpponentColorDescriptorExtractor color;
    
    // Bag of words vocabulary
   // cv::TermCriteria term_criteria(1,10,0);
   // cv::BOWKMeansTrainer bow_trainer(cluster_count,term_criteria,1,cv::KMEANS_RANDOM_CENTERS); // Bag of words vocabulary
    cv::BOWKMeansTrainer bow_trainer(cluster_count);
    for (int ind_frame=0; ind_frame<nb_frames; ++ind_frame)
    {
        std::vector<cv::KeyPoint> keypoints;
        sampleDenseKeypoints(keypoints,masks[ind_frame],stride,diameter);
        cv::Mat descriptors;
        sift->compute(video[ind_frame],keypoints,descriptors);
        bow_trainer.add(descriptors);
    }

    vocabulary = bow_trainer.cluster();
    std::cout << vocabulary.size() << std::endl;


    bow_extractor.setVocabulary(vocabulary);


    //namedWindow("Keypoints", WINDOW_KEEPRATIO);
    for (int ind_frame=0; ind_frame<nb_frames; ++ind_frame)
    {
        std::vector<cv::KeyPoint> keypoints;
        sampleDenseKeypoints(keypoints,masks[ind_frame],stride,diameter);
        cv::Mat descriptors;
        sift->compute(video[ind_frame],keypoints,descriptors);
        cv::Mat output_frame;
        bow_extractor.compute(video[ind_frame],keypoints,image_descriptors[ind_frame]);
        //std::cout << "Frame " << ind_frame << ":" << std::endl;
        //std::cout << "Descriptor: " << image_descriptors[ind_frame] << std::endl;

        cv::Mat im_keypoints;
        drawKeypoints(video[ind_frame],keypoints,im_keypoints, cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        //imshow("Keypoints",im_keypoints);
        //cv::waitKey(1);

//        int channels[] = {0, 1, 2};
//        int histSize[] = {128};
//        float hranges[] = { 0, 255 };
//        const float* ranges[1];
//        ranges[0] = hranges;
//        calcHist(&video[ind_frame],1,channels,masks[ind_frame],image_descriptors[ind_frame],1,histSize,ranges);

    }

    // Build similarity matrix
    cv::Mat similarity_matrix(nb_frames,nb_frames,CV_64F);
    //CV_Assert(image_descriptors[0].depth() == CV_64F);
    for (int i=0; i<nb_frames; ++i)
    {
        for (int j=i; j<nb_frames; ++j)
        {
            double norm_1 = norm(image_descriptors[i],cv::NORM_L2);
            double norm_2 = norm(image_descriptors[j],cv::NORM_L2);
            double dot_product = image_descriptors[i].dot(image_descriptors[j]);
            similarity_matrix.at<double>(i,j) = dot_product/(norm_1*norm_2);
            similarity_matrix.at<double>(j,i) = similarity_matrix.at<double>(i,j);
        }
    }
    cv::namedWindow("Similarity", cv::WINDOW_KEEPRATIO);
    //namedWindow("Current frame", WINDOW_KEEPRATIO);
    //namedWindow("Closest match", WINDOW_KEEPRATIO);
    cv::imshow("Similarity",similarity_matrix);
    std::ostringstream oss_cluster_count;
    oss_cluster_count << cluster_count;
//    cv::imwrite("saved_graph_300_for_tests/Similarity" + oss_cluster_count.str() + ".png",similarity_matrix);
//    writeMatToFile(similarity_matrix,"saved_graph_300_for_tests/Similarity" + oss_cluster_count.str() + ".txt");
//    cv::waitKey(1);

    return similarity_matrix;
}


void createSyntheticCircleDataset(InputSequence& input_sequence, InputData& input_data, GroundTruthData& ground_truth_data, Settings& settings, int random_seed)
{

    input_data.nb_frames = 1000;
    input_data.nb_rows = 100;
    input_data.nb_cols = 100;
    input_data.nb_bow_descriptors = 2;
    input_data.bow_appearance_matrix = Eigen::MatrixXd(input_data.nb_bow_descriptors,input_data.nb_frames);
    input_data.dataset_identifier = "CircleWrongExternal";
    input_data.agent_type = precomputed_agent;
    
    
    bool error_in_translation_only = false;
    int nb_points_defining_an_overlap = 3;
    int nb_neighbours = 1; // number of neighbours in the initial registrations
    double radius_circle = 250;
    double period = 1000;
    double step_theta = 2*3.14/period;
    double current_theta = 0;
    double bag_of_words_factor = 2; // 1 = perfect position information, 2 = cannot discriminate opposite parts of the circle
    double true_auto_standard_deviation(1);
    boost::normal_distribution<double> auto_dist(0.0,true_auto_standard_deviation); // error made by the registration algorithm
    boost::normal_distribution<double> agent_dist(0.0,std::sqrt(settings.annotation_variance)); // error made by the agent

    boost::random::mt19937 rng(random_seed);
    
    // Create fake input sequence
    input_sequence.frames.resize(input_data.nb_frames);
    input_sequence.masks.resize(input_data.nb_frames);
    cv::Mat mask(input_data.nb_rows,input_data.nb_cols,CV_8UC1,cv::Scalar::all(255));
    for (int k=0; k<input_data.nb_frames; ++k)
    {
        create_square_image(input_sequence.frames[k],input_data.nb_rows,input_data.nb_cols);
        mask.copyTo(input_sequence.masks[k]);
    }    
    
    std::vector<cv::Point2f> dense_grid_input, corners, sparse_grid_input; // fictive input landmarks
    createPointsforDistance(dense_grid_input,input_data.nb_rows,input_data.nb_cols,0.5);
    createPointsforDistance(corners,input_data.nb_rows,input_data.nb_cols,1);
    createPointsforDistance(sparse_grid_input,input_data.nb_rows,input_data.nb_cols,0.5);
    
    // These are the corners used to predict whether two frames overlap - including a buffer zone around the domain Omega
    double ext_rows = 0.5*(settings.factor_overlap_area - 1)*input_data.nb_rows;
    double ext_cols = 0.5*(settings.factor_overlap_area - 1)*input_data.nb_cols;
    input_data.overlap_corners = std::vector<cv::Point2f>(4);
    input_data.overlap_corners[0] = cv::Point2f(-ext_cols,-ext_rows);
    input_data.overlap_corners[1] = cv::Point2f(-ext_cols,input_data.nb_rows + ext_rows);
    input_data.overlap_corners[2] = cv::Point2f(input_data.nb_cols + ext_cols,-ext_rows);
    input_data.overlap_corners[3] = cv::Point2f(input_data.nb_cols + ext_cols,input_data.nb_rows + ext_rows);
    input_data.dimensions_overlap_area.resize(2);
    input_data.dimensions_overlap_area[0] = settings.factor_overlap_area*input_data.nb_rows;
    input_data.dimensions_overlap_area[1] = settings.factor_overlap_area*input_data.nb_cols;

    // Ground truth
    ground_truth_data.is_available = true;
    settings.automate_interactions = true;
    std::vector<double> true_absolute_positions(6*input_data.nb_frames,0);

    // Create true trajectory
    for (int ind_frame=0; ind_frame<input_data.nb_frames; ++ind_frame)
    {
        current_theta = current_theta + step_theta;
        //true_x[ind_frame] = radius_circle*cos(current_theta);
        //true_y[ind_frame] = radius_circle*sin(current_theta);
        true_absolute_positions[6*ind_frame] = 1;
        true_absolute_positions[6*ind_frame+2] = radius_circle*cos(current_theta);
        true_absolute_positions[6*ind_frame+4] = 1;
        true_absolute_positions[6*ind_frame+5] = radius_circle*sin(current_theta);

        // BOW descriptor
        input_data.bow_appearance_matrix(0,ind_frame) = cos(bag_of_words_factor*current_theta);
        input_data.bow_appearance_matrix(1,ind_frame) = sin(bag_of_words_factor*current_theta);
    }


    // Compute ground truth correspondences, and their pruned, noisy versions used as simulated measurements
    ground_truth_data.gt_database = LandmarkDatabase(input_data.nb_frames);
    ground_truth_data.measurements_database = LandmarkDatabase(input_data.nb_frames);
    std::vector<cv::Point2f> overlapping_control_points_input, overlapping_control_points_output, pruned_input_points, pruned_output_points;
    double T_j_inv[6], T_ij[6];
    for (int j=0; j<input_data.nb_frames; ++j)
    {
        for (int i=(j+1); i<input_data.nb_frames; ++i)
        {
            
            inverse_affine_matrix(T_j_inv,&(true_absolute_positions[6*j]));
            multiply_affine_and_affine(T_ij,&(true_absolute_positions[6*i]),T_j_inv);
            count_overlapping_points(overlapping_control_points_input,overlapping_control_points_output,T_ij,corners,dense_grid_input);
            
            int nb_overlapping_points = overlapping_control_points_input.size();
            if (nb_overlapping_points>=nb_points_defining_an_overlap)
            {
                
                // Exact, ground truth points
             //   ground_truth_data.gt_database.addToDatabase(i,j,overlapping_control_points_output,overlapping_control_points_input);
                ground_truth_data.gt_database.addToDatabase(j,i,overlapping_control_points_input,overlapping_control_points_output);

    
                // Pick random set of points that are not aligned to be used as measurements
             //   pick_least_aligned_points(pruned_input_points,pruned_output_points,overlapping_control_points_input,overlapping_control_points_output,nb_points_defining_an_overlap,random_seed);
                pruned_input_points =  overlapping_control_points_input;
                pruned_output_points = overlapping_control_points_output;
                
                
                // Add random error to each landmark
                double delta_x, delta_y;
                if (abs(i-j)<=nb_neighbours)
                {
                    delta_x = auto_dist(rng);
                    delta_y = auto_dist(rng);
                }
                else
                {
                    delta_x = agent_dist(rng);
                    delta_y = agent_dist(rng);
                }
                for (int p=0; p<nb_points_defining_an_overlap; ++p)
                {
                    if (error_in_translation_only==false) // different error for each landmark
                    {
                        if (abs(i-j)<=nb_neighbours)
                        {
                            delta_x = auto_dist(rng);
                            delta_y = auto_dist(rng);
                        }
                        else
                        {
                            delta_x = agent_dist(rng);
                            delta_y = agent_dist(rng);
                        }
                    }
                    pruned_output_points[p].x += delta_x;
                    pruned_output_points[p].y += delta_y;
                    
                }
                

                
                // Store in database
                ground_truth_data.measurements_database.addToDatabase(j,i,pruned_input_points,pruned_output_points);
            }

        }
    }
   
    // Load initial input measurements
    std::vector<std::pair<int,int>> consecutive_pairs;
    for (int i=0; i<input_data.nb_frames; ++i)
    {
        for (int j=(i+1); j<input_data.nb_frames; ++j)
        {
            if (abs(i-j)<=nb_neighbours)
                consecutive_pairs.push_back(std::pair<int,int>(i,j));
        }
    }
    loadInitialPairwiseMeasurements(input_data,ground_truth_data.measurements_database,consecutive_pairs,input_data.nb_frames);
}



void createSyntheticRasterScan(InputSequence& input_sequence, InputData& input_data, GroundTruthData& ground_truth_data, Settings& settings, int random_seed)
{

    input_data.nb_frames = 300;
    input_data.nb_rows = 100;
    input_data.nb_cols = 100; // must be a multiple of step_size


    input_data.dataset_identifier = "Raster";
    input_data.agent_type = precomputed_agent;
    
    
    bool error_in_translation_only = false;
    int nb_points_defining_an_overlap = 3;
    int nb_neighbours = 1; // number of neighbours in the initial registrations
    int nb_layers = 2;
    int step_size = 3;
    int nb_frames_per_layers = input_data.nb_frames/nb_layers;
    int nb_x_quantums = step_size + nb_frames_per_layers - 1;
    int nb_y_quantums = step_size + nb_layers - 1;
    double step_x = input_data.nb_cols/step_size;
    double step_y = input_data.nb_rows/step_size;
    double current_x = 0;
    double current_y = 0;
    double bag_of_words_factor = 1; // 1 = perfect position information, 2 = cannot discriminate opposite parts of the circle
    double true_auto_standard_deviation(1);
    
    input_data.nb_bow_descriptors = nb_x_quantums*nb_y_quantums;
    input_data.bow_appearance_matrix = Eigen::MatrixXd::Zero(input_data.nb_bow_descriptors,input_data.nb_frames);
    
    
    boost::normal_distribution<double> auto_dist(0.0,true_auto_standard_deviation); // error made by the registration algorithm
    boost::normal_distribution<double> agent_dist(0.0,std::sqrt(settings.annotation_variance)); // error made by the agent

    boost::random::mt19937 rng(random_seed);
    
    // Create fake input sequence
    input_sequence.frames.resize(input_data.nb_frames);
    input_sequence.masks.resize(input_data.nb_frames);
    cv::Mat mask(input_data.nb_rows,input_data.nb_cols,CV_8UC1,cv::Scalar::all(255));
    for (int k=0; k<input_data.nb_frames; ++k)
    {
        create_square_image(input_sequence.frames[k],input_data.nb_rows,input_data.nb_cols);
        mask.copyTo(input_sequence.masks[k]);
    }    
    
    std::vector<cv::Point2f> dense_grid_input, corners, sparse_grid_input; // fictive input landmarks
    createPointsforDistance(dense_grid_input,input_data.nb_rows,input_data.nb_cols,0.5);
    createPointsforDistance(corners,input_data.nb_rows,input_data.nb_cols,1);
    createPointsforDistance(sparse_grid_input,input_data.nb_rows,input_data.nb_cols,0.5);
    
    // These are the corners used to predict whether two frames overlap - including a buffer zone around the domain Omega
    double ext_rows = 0.5*(settings.factor_overlap_area - 1)*input_data.nb_rows;
    double ext_cols = 0.5*(settings.factor_overlap_area - 1)*input_data.nb_cols;
    input_data.overlap_corners = std::vector<cv::Point2f>(4);
    input_data.overlap_corners[0] = cv::Point2f(-ext_cols,-ext_rows);
    input_data.overlap_corners[1] = cv::Point2f(-ext_cols,input_data.nb_rows + ext_rows);
    input_data.overlap_corners[2] = cv::Point2f(input_data.nb_cols + ext_cols,-ext_rows);
    input_data.overlap_corners[3] = cv::Point2f(input_data.nb_cols + ext_cols,input_data.nb_rows + ext_rows);
    input_data.dimensions_overlap_area.resize(2);
    input_data.dimensions_overlap_area[0] = settings.factor_overlap_area*input_data.nb_rows;
    input_data.dimensions_overlap_area[1] = settings.factor_overlap_area*input_data.nb_cols;

    // Ground truth
    ground_truth_data.is_available = true;
    settings.automate_interactions = true;
    std::vector<double> true_absolute_positions(6*input_data.nb_frames,0);

    // Create true trajectory
    for (int ind_frame=0; ind_frame<input_data.nb_frames; ++ind_frame)
    {
        int ind_layer = ind_frame/nb_frames_per_layers;
        
        if ((ind_frame>0) && ((ind_frame % nb_frames_per_layers)==0))
           current_y += step_y;
        else
        {
            if ((ind_layer % 2)==0)
                current_x += step_x;
            else
                current_x -= step_x;
        }
        
        //true_x[ind_frame] = radius_circle*cos(current_theta);
        //true_y[ind_frame] = radius_circle*sin(current_theta);
        true_absolute_positions[6*ind_frame] = 1;
        true_absolute_positions[6*ind_frame+2] = -current_x;
        true_absolute_positions[6*ind_frame+4] = 1;
        true_absolute_positions[6*ind_frame+5] = current_y;


        int ind_blx = current_x/step_x + nb_x_quantums*ind_layer;
        int nb_quantums_in_frame = step_size*step_size;
        double bow_constant = std::sqrt(1.0/(double)nb_quantums_in_frame);
        for (int dx = 0; dx<step_size; ++dx)
        {
            for (int dy = 0; dy<step_size; ++dy)
            {
                input_data.bow_appearance_matrix(ind_blx + dy*nb_x_quantums + dx,ind_frame) = bow_constant;
            }
            
        }
        
    }


    // Compute ground truth correspondences, and their pruned, noisy versions used as simulated measurements
    ground_truth_data.gt_database = LandmarkDatabase(input_data.nb_frames);
    ground_truth_data.measurements_database = LandmarkDatabase(input_data.nb_frames);
    std::vector<cv::Point2f> overlapping_control_points_input, overlapping_control_points_output, pruned_input_points, pruned_output_points;
    double T_j_inv[6], T_ij[6];
    for (int j=0; j<input_data.nb_frames; ++j)
    {
        for (int i=(j+1); i<input_data.nb_frames; ++i)
        {
            inverse_affine_matrix(T_j_inv,&(true_absolute_positions[6*j]));
            multiply_affine_and_affine(T_ij,&(true_absolute_positions[6*i]),T_j_inv);
            count_overlapping_points(overlapping_control_points_input,overlapping_control_points_output,T_ij,corners,dense_grid_input);
            
            int nb_overlapping_points = overlapping_control_points_input.size();
            if (nb_overlapping_points>=nb_points_defining_an_overlap)
            {
                
                // Exact, ground truth points
                ground_truth_data.gt_database.addToDatabase(j,i,overlapping_control_points_input,overlapping_control_points_output);
    
                // Pick random set of points that are not aligned to be used as measurements
                //pick_least_aligned_points(pruned_input_points,pruned_output_points,overlapping_control_points_input,overlapping_control_points_output,nb_points_defining_an_overlap,random_seed);
                pruned_input_points =  overlapping_control_points_input;
                pruned_output_points = overlapping_control_points_output;
                
                
                // Add random error to each landmark
                double delta_x, delta_y;
                if (abs(i-j)<=nb_neighbours)
                {
                    delta_x = auto_dist(rng);
                    delta_y = auto_dist(rng);
                }
                else
                {
                    delta_x = agent_dist(rng);
                    delta_y = agent_dist(rng);
                }
                
                int nb_points = pruned_input_points.size();
                for (int p=0; p<nb_points; ++p)
                {
                    if (error_in_translation_only==false) // different error for each landmark
                    {
                        if (abs(i-j)<=nb_neighbours)
                        {
                            delta_x = auto_dist(rng);
                            delta_y = auto_dist(rng);
                        }
                        else
                        {
                            delta_x = agent_dist(rng);
                            delta_y = agent_dist(rng);
                        }
                    }
                    pruned_output_points[p].x += delta_x;
                    pruned_output_points[p].y += delta_y;
                    
                }
                

                
                // Store in database
                ground_truth_data.measurements_database.addToDatabase(j,i,pruned_input_points,pruned_output_points);
               // ground_truth_data.measurements_database.addToDatabase(i,j,pruned_output_points,pruned_input_points);
            }

        }
    }
   
    // Load initial input measurements
    std::vector<std::pair<int,int>> consecutive_pairs;
    for (int i=0; i<input_data.nb_frames; ++i)
    {
        for (int j=0; j<input_data.nb_frames; ++j)
        {
            if ((abs(i-j)<=nb_neighbours) && (i<j))
                consecutive_pairs.push_back(std::pair<int,int>(i,j));
        }
    }
    loadInitialPairwiseMeasurements(input_data,ground_truth_data.measurements_database,consecutive_pairs,input_data.nb_frames);
}

void create_square_image(cv::Mat& im, int nb_rows, int nb_cols)
{
    int thickness = 5;
    im = cv::Mat(nb_rows,nb_cols,CV_8UC3,cv::Scalar::all(255));
    cv::Rect interior(thickness,thickness,nb_cols-2*thickness,nb_rows-2*thickness);
    im(interior) = cv::Scalar::all(0);
}


void load_overlap_area_and_dimensions(InputData& input_data, double factor_overlap_area)
{
    // These are the corners used to predict whether two frames overlap - including a buffer zone around the domain Omega
    double ext_rows = 0.5*(factor_overlap_area - 1)*input_data.nb_rows;
    double ext_cols = 0.5*(factor_overlap_area - 1)*input_data.nb_cols;
    input_data.overlap_corners = std::vector<cv::Point2f>(4);
    input_data.overlap_corners[0] = cv::Point2f(-ext_cols,-ext_rows);
    input_data.overlap_corners[1] = cv::Point2f(-ext_cols,input_data.nb_rows + ext_rows);
    input_data.overlap_corners[2] = cv::Point2f(input_data.nb_cols + ext_cols,-ext_rows);
    input_data.overlap_corners[3] = cv::Point2f(input_data.nb_cols + ext_cols,input_data.nb_rows + ext_rows);
    input_data.dimensions_overlap_area.resize(2);
    input_data.dimensions_overlap_area[0] = factor_overlap_area*input_data.nb_rows;
    input_data.dimensions_overlap_area[1] = factor_overlap_area*input_data.nb_cols;
}

void loadAerialFrames(InputSequence& video_sequence, int nb_frames)
{
   // cv::namedWindow("Frame",cv::WINDOW_KEEPRATIO);
   // cv::namedWindow("Mask",cv::WINDOW_KEEPRATIO);

    // Creat list filenames
    std::string data_folder = "Aerial/Sequence/";
    video_sequence.frames.clear();
    video_sequence.masks.clear();
    for (int f=1; f<=24; ++f)
    {
        std::ostringstream oss_first;
        if (f<10)
            oss_first << "0";
        oss_first << f;
        for (int s=1; s<=31; ++s)
        {
            std::ostringstream oss_second;
            if (s<10)
                oss_second << "0";
            oss_second << s;

            std::string filename = data_folder + oss_first.str() + "_" + oss_second.str() + ".jpg";
            cv::Mat frame = cv::imread(filename);
            cv::Mat mask(frame.rows,frame.cols,CV_8U,cv::Scalar::all(255));
            video_sequence.frames.push_back(frame);
            video_sequence.masks.push_back(mask);

          //  cv::imshow("Frame",frame);
          //  cv::imshow("Mask",mask);
          //  cv::waitKey(1);

            if (video_sequence.frames.size()>=nb_frames)
                break;
        }

        if (video_sequence.frames.size()>=nb_frames)
            break;

    }

}




void loadFetoscopyFrames(InputSequence& video_sequence, int nb_frames)
{
  //  cv::namedWindow("Frame",cv::WINDOW_KEEPRATIO);
  //  cv::namedWindow("Mask",cv::WINDOW_KEEPRATIO);

    // Creat list filenames
    std::string data_folder = "Fetoscopy/Sequence/";
    cv::Mat mask, frame;
    video_sequence.frames.clear();
    video_sequence.masks.clear();
    for (int f=1; f<=nb_frames; ++f)
    {
        std::ostringstream oss_first;
        if (f<10)
            oss_first << "00";
        else
        {
            if (f<100)
                oss_first << "0";
        }
        oss_first << f;

        std::string filename = data_folder + "fetoscopy-" + oss_first.str() + ".png";
        frame = cv::imread(filename);

        if (f==1)
             mask = getMask_U(frame);

        video_sequence.frames.push_back(frame);
        video_sequence.masks.push_back(mask);
    }

}



// Prepare registrations based on Lucas-Kanade
void performInitialRegistrations_Dense(std::vector<MeasuredAffineRegistration>& measured_registrations, const InputSequence& video_sequence,
            const std::vector<cv::Point2f>& landmarks_input, const std::string& global_folder_name, bool load_pairwise_registrations, int precomputed_nb_frames)
{
    // Parameters (possibly better to put eventually in a RegistrationSettings object)
    int nbChannels = 1;
    int sigma_blur = 5;
    int nb_scales_pyramid = 6;
    double dense_grid_step_size(0.1);
    int half_mosaic_size = 2; // not sure if it matters here
    double threshold_confidence(0.8); // should not be there either...
   // RegistrationMethod registration_method(gradient_orientation_ours);

    // Registrations are stored in a graph
    int nb_frames = (int)video_sequence.frames.size();
    if (precomputed_nb_frames==(-1))
        precomputed_nb_frames = nb_frames;
    Graph<cv::Mat> graph_pairwise_registrations(precomputed_nb_frames);
    cv::Mat pairwise_residuals(precomputed_nb_frames,precomputed_nb_frames,CV_64F,cv::Scalar::all(-1));
    if (load_pairwise_registrations)
    {
        std::cout << "Load registrations" << std::endl;
        // Load file
        graph_pairwise_registrations.load(global_folder_name + "Registrations","Pairwise.txt",precomputed_nb_frames);
        loadMatFile(pairwise_residuals,global_folder_name + "Registrations/Residuals.txt",precomputed_nb_frames,precomputed_nb_frames);
        std::cout << "Done" << std::endl;
    }
    else
    {
        
        std::cout << "Need to define a (dense) registration module" << std::endl;
        
        
//         // We first prepare a multiscale helper for the mosaic
//         std::cout << "Preprocess all frames..." << std::endl;
//         std::vector<cv::Mat> preprocessed_frames(nb_frames);
//         for (int i=0; i<nb_frames; ++i)
//         {
//             preprocessed_frames[i] = cv::Mat(video_sequence.frames[i].size(),CV_MAKETYPE(CV_64F,nbChannels));
//             performPreprocessing(preprocessed_frames[i],video_sequence.frames[i],video_sequence.masks[i],sigma_blur);
//         }
//         std::cout << "Compute multiscale helper..." << std::endl;
//         Multiscale_Mosaic_Helper multiscale_helper(preprocessed_frames,video_sequence.masks[0],nb_scales_pyramid);
//         preprocessed_frames.clear();
//         std::cout << "Done!" << std::endl;
// 
//         // Create a mosaic object (this part is based on an old structure, and as the rest of this function could be simplified)
// 
//         int nb_rows = video_sequence.frames[0].rows;
//         int nb_cols = video_sequence.frames[0].cols;
//         int x_0(half_mosaic_size*nb_cols), y_0(half_mosaic_size*nb_rows);
//         Mosaic mosaic((2*half_mosaic_size+1)*nb_rows,(2*half_mosaic_size+1)*nb_cols,x_0,y_0,dense_grid_step_size);
//         for (int i=0; i<nb_frames; ++i)
//             mosaic.addFrame(video_sequence.frames[i],video_sequence.masks[i]);
// 
//         // Register pairwises
//         cv::namedWindow("Fixed image", cv::WINDOW_KEEPRATIO); //resizable window;
//         cv::namedWindow("Mobile image", cv::WINDOW_KEEPRATIO); //resizable window;
//         addWindowedConstraints(mosaic,multiscale_helper,graph_pairwise_registrations,pairwise_residuals,2,registration_method,threshold_confidence);
// 
//         // We save the pairwise content
//         std::cout << "Saving graph" << std::endl;
//         graph_pairwise_registrations.save(global_folder_name + "Registrations","Pairwise.txt");
//         writeMatToFile(pairwise_residuals,global_folder_name + "Registrations/Residuals.txt");
    }

    // Reduces the graph so that there is only (at most) one registration per pair (i,j), with i<j
    triangulate_graph_of_constraints(graph_pairwise_registrations,pairwise_residuals);

    // Fill measured registrations
    std::cout << "Compute landmarks" << std::endl;
    for (int i=0; i<(nb_frames-1); ++i)
    {
        MeasuredAffineRegistration new_measurement;
        new_measurement.frame_indices = std::pair<int,int>(i,i+1);
        new_measurement.landmarks_input = landmarks_input;
        getDisplacedPointsAffine(new_measurement.landmarks_output,(double*)(graph_pairwise_registrations.get_edge_ptr(i,i+1)->data),new_measurement.landmarks_input);
        measured_registrations.push_back(new_measurement);
    }
    std::cout << "Done" << std::endl;

}



// Prepare registrations based on Lucas-Kanade
// void performInitialRegistrations_Landmarks(std::vector<MeasuredAffineRegistration>& measured_registrations, const InputSequence& video_sequence,
//             const std::vector<cv::Point2f>& landmarks_input, const std::string& global_folder_name, const std::string& video_folder_name)
// {
// 
//     std::string features_type = "orb";
//     std::string matcher_type = "homography";
//     float match_conf = 0.05;
//     int range_width = -1;
//     int nb_frames = (int)video_sequence.frames.size();
// 
//      for (int i=0; i<(nb_frames-1); ++i)
//      {
// 
// 
// //    // detecting keypoints
// //    Ptr<FeatureDetector> detector = FastFeatureDetector::create(20);
//     std::vector<cv::KeyPoint> keypoints1, keypoints2;
// //    detector->detect(fixed_image, keypoints1, mask_U);
// //    detector->detect(mobile_image, keypoints2, mask_U);
// //
// //    // computing descriptors
//     //Ptr<SIFT> extractor = SIFT::create();
//     cv::Mat descriptors1, descriptors2;
// //    extractor->compute(fixed_image, keypoints1, descriptors1);
// //    extractor->compute(mobile_image, keypoints2, descriptors2);
// 
//     cv::Ptr<cv::xfeatures2d::SURF> sift = cv::xfeatures2d::SURF::create(100);
//     //cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(0,3,0.03);
//     sift->detectAndCompute(video_sequence.frames[i],video_sequence.masks[i],keypoints1,descriptors1,false);
//     sift->detectAndCompute(video_sequence.frames[i+1],video_sequence.masks[i+1],keypoints2,descriptors2,false);
// 
// 
//     // matching descriptors
//     cv::BFMatcher matcher;
//     std::vector<cv::DMatch> matches;
//     matcher.match(descriptors1, descriptors2, matches);
// 
//     namedWindow("Matches", cv::WINDOW_KEEPRATIO);
// 	cv::Mat outputImage;
// 	drawMatches(video_sequence.frames[i],keypoints1,video_sequence.frames[i+1],keypoints2,matches,outputImage);
// 	imshow("Matches",outputImage);
// 	cv::waitKey(1);
// 
//     MeasuredAffineRegistration new_measurement;
//     new_measurement.frame_indices = std::pair<int,int>(i,i+1);
// 
//     std::vector<cv::Point2f> obj, scene;
//     for( int k = 0; k < matches.size(); k++ )
//     {
//         // For simplicity, but we should first do this with obj and scene, then run Ransac to keep only the inliers landmarks before adding them to the measurement
//         obj.push_back( keypoints1[ matches[k].queryIdx ].pt );
//         scene.push_back( keypoints2[ matches[k].trainIdx ].pt );
// 
//     }
// 
//     std::cout << "We filter out the outliers" << std::endl;
//     cv::Mat inlier_mask;
//     cv::Mat H = findHomography(obj,scene,CV_RANSAC,3,inlier_mask);
// 
//     for( int k = 0; k < matches.size(); k++ )
//     {
//         if (inlier_mask.at<uchar>(k)==1) // if inlier
//         {
//              new_measurement.landmarks_input.push_back( keypoints1[ matches[k].queryIdx ].pt );
//              new_measurement.landmarks_output.push_back( keypoints2[ matches[k].trainIdx ].pt );
//         }
//     }
// 
//     measured_registrations.push_back(new_measurement);
// 
//     }
// 
// }




void performInitialRegistrations_Landmarks(std::vector<MeasuredAffineRegistration>& measured_registrations, InputData& input_data, const InputSequence& video_sequence, const std::vector<std::pair<int,int>>& pairs_to_register, const std::string& global_folder_name, const std::string& video_folder_name, const std::string& database_name)
{


    bool display(true);
    int nb_frames = (int)video_sequence.frames.size();
    int nb_pairs = (int)pairs_to_register.size();

    LandmarkDatabase full_landmark_database(nb_frames);
    
    
    omp_lock_t writelock;
    omp_init_lock(&writelock);
    int nb_processed_pairs(0);
    #pragma omp parallel for num_threads(1)
    for (int k=0; k<nb_pairs; ++k)
    {

        int i = pairs_to_register[k].first;
        int j = pairs_to_register[k].second;
         
        std::cout << "Process pair (" << i << "," << j << ")" << std::endl;

         
        MeasuredAffineRegistration new_measurement;
        new_measurement.frame_indices = std::pair<int,int>(i,j);
         
       
 //       sift = cv::ORB::create(500,1.2,3,31,0,2,cv::ORB::HARRIS_SCORE,21,20);
 //       sift = cv::xfeatures2d::SURF::create(400,5,2,true);
//        sift = cv::xfeatures2d::SIFT::create(500,3,0.1,10,3);
//        sift = cv::xfeatures2d::SIFT::create(200,3,0.05,10,4);
         
        bool is_registration_successful(false);
        perform_landmark_based_registration(is_registration_successful,new_measurement.landmarks_input,new_measurement.landmarks_output,input_data.feature_extractor,video_sequence.frames[i],video_sequence.frames[j],video_sequence.masks[i],video_sequence.masks[j],display);
  
        if (is_registration_successful)
        {
            omp_set_lock(&writelock);
            measured_registrations.push_back(new_measurement);
            full_landmark_database.addToDatabase(i,j,new_measurement.landmarks_input,new_measurement.landmarks_output);
            omp_unset_lock(&writelock);
        }     
        
        omp_set_lock(&writelock);
        ++nb_processed_pairs;
        std::cout << "Processed pairs: " << nb_processed_pairs << "/" << nb_pairs << std::endl;
        omp_unset_lock(&writelock);
    }

    omp_destroy_lock(&writelock);
    full_landmark_database.save(database_name);

}


void perform_landmark_based_registration(bool& is_inverse_compatible, std::vector<cv::Point2f>& landmarks_input, std::vector<cv::Point2f>& landmarks_output,  cv::Ptr<cv::Feature2D> sift, const cv::Mat& image_1, const cv::Mat& image_2, const cv::Mat& mask_1, const cv::Mat& mask_2, bool display)
{
    // Parameters
    double maxIters = 5000;
    double confidence = 0.9;
    double ransac_threshold = 3;
    double threshold_inverse_compatibility = 1;
    int min_nb_matches = 30;
    cv::Mat I = cv::Mat::eye(3,3,CV_64F);
    
    landmarks_input.clear();
    landmarks_output.clear();
    
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2, inlier_mask, inlier_mask_2;
    
    // Detect keypoints and their descriptors
    sift->detectAndCompute(image_1,mask_1,keypoints1,descriptors1,false);
    sift->detectAndCompute(image_2,mask_2,keypoints2,descriptors2,false);

     // Matching descriptors
    cv::BFMatcher matcher(cv::NORM_L2,true); // we crosscheck
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    if (display)
    {
        cv::Mat outputImage;
        drawMatches(image_1,keypoints1,image_2,keypoints2,matches,outputImage);
        imshow("Matches",outputImage);
        cv::waitKey(2000);
    }
        
    std::vector<cv::Point2f> obj, scene;
    for( int m = 0; m < matches.size(); m++ )
    {
        obj.push_back( keypoints1[ matches[m].queryIdx ].pt );
        scene.push_back( keypoints2[ matches[m].trainIdx ].pt );
    }

    is_inverse_compatible = false;
    try
    {
        cv::Mat H = findHomography(obj,scene,CV_RANSAC,ransac_threshold,inlier_mask,maxIters,confidence);
        cv::Mat H2 = findHomography(scene,obj,CV_RANSAC,ransac_threshold,inlier_mask_2,maxIters,confidence);
        cv::Mat E = H*H2 - I;
  //      std::cout << H << std::endl;
   //     std::cout << "Determinant: " << H.at<double>(0,0)*H.at<double>(1,1) - H.at<double>(0,1)*H.at<double>(1,0) << std::endl;
        double score = abs(E.at<double>(0,0)) + abs(E.at<double>(1,1)) + abs(E.at<double>(2,2));
        if (score<threshold_inverse_compatibility)
            is_inverse_compatible = true;
        else
        {
            std::cout << "Transformation not compatible with its inverse" << std::endl;
        }
    }
    catch (std::string s)
    {
        std::cout << "findHomography was unable to estimate homographies: Thrown exception: " << s << std::endl;
    }
        

    if (is_inverse_compatible)
    {
        std::vector<cv::KeyPoint> inliers_keypoints1, inliers_keypoints2;
        std::vector<cv::DMatch> inlier_matches;
        for( int m = 0; m < matches.size(); m++ )
        {
            if (inlier_mask.at<uchar>(m)==1) // if inlier
            {
                inlier_matches.push_back(matches[m]);
                landmarks_input.push_back( keypoints1[ matches[m].queryIdx ].pt );
                landmarks_output.push_back( keypoints2[ matches[m].trainIdx ].pt );
            }
        }
            
        if  (inlier_matches.size()<min_nb_matches)    
        {
            std::cout << "Not enough inlier matches found" << std::endl;
            is_inverse_compatible = false;
        }
        else
        {
        if (display)
        {
            cv::Mat outputImage;
            drawMatches(image_1,keypoints1,image_2,keypoints2,inlier_matches,outputImage);
            imshow("Matches",outputImage);
            cv::waitKey(1);
        }
        }
    }

}


void buildBagOfWordsWithDenseDetection(std::vector<cv::Mat>& image_descriptors, cv::Mat& vocabulary, const std::vector<cv::Mat>& video, const std::vector<cv::Mat>& masks, int cluster_count, int diameter, int stride)
{
    int nb_frames = (int)video.size();
    cv::Ptr<cv::Feature2D> sift, vgg;
   // sift = cv::xfeatures2d::SURF::create(500);
   sift = cv::xfeatures2d::SURF::create(300,4,3,false,false);
//    sift = cv::xfeatures2d::SIFT::create(1,3,0,100); 
//    sift = cv::ORB::create(1); 
    vgg = cv::xfeatures2d::VGG::create();
    int radius = diameter/2;

    std::vector<std::vector<cv::KeyPoint>> image_keypoints(nb_frames);
    // Bag of words objects
    cv::TermCriteria term_criteria(1,10,0);
    cv::BFMatcher matcher;
    cv::Ptr<cv::BFMatcher> ptr_matcher = matcher.create();
    cv::BOWImgDescriptorExtractor bow_extractor(vgg,ptr_matcher);
    cv::BOWKMeansTrainer bow_trainer(cluster_count); // Bag of words vocabulary

//     std::vector<cv::KeyPoint> local_keypoints;
//     for (int ind_frame=0; ind_frame<nb_frames; ++ind_frame)
//     {
//         std::cout << "Detect and compute keypoints in frame " << ind_frame + 1 << "/" << nb_frames << std::endl;
//         cv::Mat current_frame = video[ind_frame];
//         cv::Mat current_mask = masks[ind_frame];
//         
//         int dimy = current_frame.rows;
//         int dimx = current_frame.cols;
//         for (int x=radius; x<(dimx - radius - 1); x+=stride)
//         {
//             for (int y=radius; y<(dimy - radius - 1); y+=stride)
//             { 
//                 cv::Mat descriptors, cropped_frame, cropped_mask;
//                 cv::Rect roi(x-radius,y-radius,diameter,diameter);
//                // std::cout << "Crop frames" << std::endl;
//                 cropped_frame = current_frame(roi);
//                 cropped_mask = current_mask(roi);
//              //   std::cout << "Done" << std::endl;
//                 sift->detectAndCompute(cropped_frame,cropped_mask,local_keypoints,descriptors,false);
//                 if (local_keypoints.size()>0)
//                 {
//                     int strongest_keypoint(0);
//                     float strongest_response(local_keypoints[0].response);
//                     for (int k=1; k<local_keypoints.size(); ++k)
//                     {
//                         if (local_keypoints[k].response>strongest_response)
//                         {
//                             strongest_response = local_keypoints[k].response;
//                             strongest_keypoint = k;
//                         }
//                     }
//                     
//                     cv::Rect roi_bestkeypoint(0,strongest_keypoint,descriptors.cols,1);
//                     bow_trainer.add(descriptors(roi_bestkeypoint));
//                     local_keypoints[strongest_keypoint].pt.x += x-radius;
//                     local_keypoints[strongest_keypoint].pt.y += y-radius;
//                     image_keypoints[ind_frame].push_back(local_keypoints[strongest_keypoint]);
//                 }
//             }
//         }
        
    
    
    for (int ind_frame=0; ind_frame<nb_frames; ++ind_frame)
    {
        std::cout << "Detect and compute keypoints in frame " << ind_frame + 1 << "/" << nb_frames << std::endl;
        cv::Mat current_frame = video[ind_frame];
        cv::Mat current_mask = masks[ind_frame];
        std::vector<cv::KeyPoint> all_image_keypoints;
        int dimy = current_frame.rows;
        int dimx = current_frame.cols;
        cv::Mat descriptors;
     //   sift->detectAndCompute(video[ind_frame],masks[ind_frame],all_image_keypoints,descriptors,false);
        
        sift->detect(video[ind_frame],all_image_keypoints,masks[ind_frame]);
        vgg->compute(video[ind_frame],all_image_keypoints,descriptors);
        
        for (int x=0; x<dimx; x+=stride)
        {
            for (int y=0; y<dimy; y+=stride)
            { 
                cv::Rect roi(x-radius,y-radius,diameter,diameter);
                
                int strongest_keypoint(-1);
                float strongest_response(0);
                float size_strongest_keypoint(0);
                for (int k=0; k<all_image_keypoints.size(); ++k)
                {
                    if (roi.contains(all_image_keypoints[k].pt))
                    {
                        
                        if ((strongest_keypoint==(-1)) || ((all_image_keypoints[k].response>strongest_response) || ((all_image_keypoints[k].response==strongest_response) && (all_image_keypoints[k].size>size_strongest_keypoint))))
                        {
                            strongest_response = all_image_keypoints[k].response;
                            size_strongest_keypoint = all_image_keypoints[k].size;
                            strongest_keypoint = k;
                        }
                    }
                }
                
                if (strongest_keypoint!=(-1))
                {
                    cv::Rect roi_bestkeypoint(0,strongest_keypoint,descriptors.cols,1);
                    bow_trainer.add(descriptors(roi_bestkeypoint));
               //     all_image_keypoints[strongest_keypoint].pt.x += x-radius;
               //     all_image_keypoints[strongest_keypoint].pt.y += y-radius;
                    image_keypoints[ind_frame].push_back(all_image_keypoints[strongest_keypoint]);
                }
            }
        }
    
    
//         cv::Mat im_keypoints;
//         drawKeypoints(current_frame,image_keypoints[ind_frame],im_keypoints,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//         imshow("Keypoints",im_keypoints);
//         std::ostringstream oss_ind_frame;
//         oss_ind_frame << ind_frame;
//         cv::imwrite("DenseDetectedKeypoints" + oss_ind_frame.str() + ".png",im_keypoints);
//         cv::waitKey(2000);
    }

    vocabulary = bow_trainer.cluster();
    std::cout << vocabulary.size() << std::endl;
    bow_extractor.setVocabulary(vocabulary);


    //namedWindow("Keypoints", WINDOW_KEEPRATIO);
    for (int ind_frame=0; ind_frame<nb_frames; ++ind_frame)
    {
        bow_extractor.compute(video[ind_frame],image_keypoints[ind_frame],image_descriptors[ind_frame]);
       // image_descriptors[ind_frame] = ((double)image_keypoints[ind_frame].size())*image_descriptors[ind_frame];
       // std::cout << image_descriptors[ind_frame] << std::endl;
    }
    
    
    // Build similarity matrix
    cv::Mat similarity_matrix(nb_frames,nb_frames,CV_64F);
    //CV_Assert(image_descriptors[0].depth() == CV_64F);
    for (int i=0; i<nb_frames; ++i)
    {
        for (int j=i; j<nb_frames; ++j)
        {
            double norm_1 = norm(image_descriptors[i],cv::NORM_L2);
            double norm_2 = norm(image_descriptors[j],cv::NORM_L2);
            double dot_product = image_descriptors[i].dot(image_descriptors[j]);
            similarity_matrix.at<double>(i,j) = dot_product/(norm_1*norm_2);
            similarity_matrix.at<double>(j,i) = similarity_matrix.at<double>(i,j);
        }
    }
    cv::namedWindow("Similarity", cv::WINDOW_KEEPRATIO);
    cv::imshow("Similarity",similarity_matrix);

}







void buildBagOfWordsWithDetection(std::vector<cv::Mat>& image_descriptors, cv::Mat& vocabulary, const std::vector<cv::Mat>& video, const std::vector<cv::Mat>& masks, int cluster_count)
{

    int nb_frames = (int)video.size();
    cv::Ptr<cv::Feature2D> sift;
   // sift = cv::xfeatures2d::SURF::create(500);
    sift = cv::xfeatures2d::SURF::create(20000,4,3,true,false);
 //   sift = cv::xfeatures2d::SIFT::create(100,3,0.04,10,1.6); 		
 //    sift = cv::ORB::create(50);
//     sift = cv::xfeatures2d::DAISY::create();

    // Bag of words objects
    cv::TermCriteria term_criteria(1,10,0);
    cv::BFMatcher matcher;
    cv::Ptr<cv::BFMatcher> ptr_matcher = matcher.create();
    cv::BOWImgDescriptorExtractor bow_extractor(sift,ptr_matcher);
    cv::BOWKMeansTrainer bow_trainer(cluster_count); // Bag of words vocabulary

    std::vector<std::vector<cv::KeyPoint>> image_keypoints(nb_frames);
    std::vector<cv::Mat> image_keypoints_descriptors(nb_frames);
    for (int ind_frame=0; ind_frame<nb_frames; ++ind_frame)
    {
        std::cout << "Detect and compute keypoints in frame " << ind_frame + 1 << "/" << nb_frames << std::endl;
        sift->detectAndCompute(video[ind_frame],masks[ind_frame],image_keypoints[ind_frame],image_keypoints_descriptors[ind_frame],false);
        bow_trainer.add(image_keypoints_descriptors[ind_frame]);
//        std::cout << image_keypoints_descriptors[ind_frame] << std::endl;
//         cv::Mat im_keypoints;
//         drawKeypoints(video[ind_frame],image_keypoints[ind_frame],im_keypoints, cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//         imshow("Keypoints",im_keypoints);
//         cv::waitKey(1000);
    }

    std::cout << "Build vocabulary" << std::endl;
    
    
    
    
    vocabulary = bow_trainer.cluster();
    std::cout << vocabulary.size() << std::endl;
    bow_extractor.setVocabulary(vocabulary);


    //namedWindow("Keypoints", WINDOW_KEEPRATIO);
    for (int ind_frame=0; ind_frame<nb_frames; ++ind_frame)
    {
        bow_extractor.compute(video[ind_frame],image_keypoints[ind_frame],image_descriptors[ind_frame]);
        image_descriptors[ind_frame] = ((double)image_keypoints[ind_frame].size())*image_descriptors[ind_frame];
       // std::cout << image_descriptors[ind_frame] << std::endl;
    }
    
    
    // Build similarity matrix
    cv::Mat similarity_matrix(nb_frames,nb_frames,CV_64F);
    //CV_Assert(image_descriptors[0].depth() == CV_64F);
    for (int i=0; i<nb_frames; ++i)
    {
        for (int j=i; j<nb_frames; ++j)
        {
            double norm_1 = norm(image_descriptors[i],cv::NORM_L2);
            double norm_2 = norm(image_descriptors[j],cv::NORM_L2);
            double dot_product = image_descriptors[i].dot(image_descriptors[j]);
            similarity_matrix.at<double>(i,j) = dot_product/(norm_1*norm_2);
            similarity_matrix.at<double>(j,i) = similarity_matrix.at<double>(i,j);
        }
    }
    cv::namedWindow("Similarity", cv::WINDOW_KEEPRATIO);
    cv::imshow("Similarity",similarity_matrix);
}


void computeInitialBagOfWordsModel(Eigen::MatrixXd& bow_appearance_matrix, int& nb_bow_descriptors, const InputSequence& video_sequence, const std::string& global_folder_name, const std::string& video_folder_name,  bool load_precomputed_appearance_bag_of_words, int nb_descriptors, int diameter, int stride, double downsize_factor, bool build_with_detection, const std::string& bow_name, bool normalise_descriptors)
{


    // Appearance bag of words - many simplifications could be done here too
    int nb_frames = (int)video_sequence.frames.size();
    cv::Mat similarity_matrix, appearance_vocabulary;
    std::vector<cv::Mat> appearance_descriptors(nb_frames);
    if (load_precomputed_appearance_bag_of_words)
    {
        std::cout << "Load appearance bag of words" << std::endl;
        //loadMatFile(similarity_matrix,global_folder_name + video_folder_name + "AppearanceDescriptors/Similarity.txt",nb_frames,nb_frames);
        for (int ind=0; ind<nb_frames; ++ind)
        {
            std::ostringstream oss_ind;
            oss_ind << ind;
            std::string filename = global_folder_name + video_folder_name + "BOW/Descriptor" + bow_name + "-Frame" + oss_ind.str() + ".txt";
            loadFloatMatFile(appearance_descriptors[ind],filename,1,nb_descriptors);
        }
        //  loadVector<int>(appearance_vocabulary_dimensions,global_folder_name + video_folder_name + "AppearanceDescriptors/Vocabulary_Dimensions.txt");
        //  loadFloatMatFile(appearance_vocabulary,global_folder_name + video_folder_name + "AppearanceDescriptors/Vocabulary.txt",appearance_vocabulary_dimensions[0],appearance_vocabulary_dimensions[1]);
        std::cout << "Bag of words loaded" << std::endl;
    }
    else
    {
        std::vector<cv::Mat> resized_frames = video_sequence.frames;
        std::vector<cv::Mat> resized_masks = video_sequence.masks;
        if (downsize_factor!=1)
        {
            std::cout << "Downsize input frames and masks for the computation of bag of words" << std::endl;
            for (int k=0; k<nb_frames; ++k)
            {
                cv::resize(video_sequence.frames[k],resized_frames[k],cv::Size(),1/((double)downsize_factor),1/((double)downsize_factor),cv::INTER_AREA);
                cv::resize(video_sequence.masks[k],resized_masks[k],cv::Size(),1/((double)downsize_factor),1/((double)downsize_factor),cv::INTER_NEAREST);
            }
        }
        
        
        std::cout << "Build bag of words" << std::endl;
        if (build_with_detection)
            buildBagOfWordsWithDenseDetection(appearance_descriptors,appearance_vocabulary,resized_frames,resized_masks,nb_descriptors,diameter,stride);
            //buildBagOfWordsWithDetection(appearance_descriptors,appearance_vocabulary,resized_frames,resized_masks,nb_descriptors);
        else
            cv::Mat similarity_matrix = buildBagOfWords(appearance_descriptors,appearance_vocabulary,resized_frames,resized_masks,nb_descriptors,diameter,stride);
        //
        //  appearance_vocabulary_dimensions[0] = appearance_vocabulary.rows;
        //  appearance_vocabulary_dimensions[1] = appearance_vocabulary.cols;
        for (int ind=0; ind<nb_frames; ++ind)
        {
            std::ostringstream oss_ind;
            oss_ind << ind;
            std::string filename = global_folder_name + video_folder_name + "BOW/Descriptor" + bow_name + "-Frame" + oss_ind.str() + ".txt";
            writeFloatMatToFile(appearance_descriptors[ind],filename);
        }
        // writeMatToFile(similarity_matrix,global_folder_name + video_folder_name + "AppearanceDescriptors/Similarity.txt");
       //  saveVector<int>(appearance_vocabulary_dimensions,global_folder_name + video_folder_name + "AppearanceDescriptors/Vocabulary_Dimensions.txt");
       //  writeFloatMatToFile(appearance_vocabulary,global_folder_name + video_folder_name + "AppearanceDescriptors/Vocabulary.txt");
        //imwrite(global_folder_name + video_folder_name + "BOW/Similarity.png",255*similarity_matrix);
    }

    nb_bow_descriptors = nb_descriptors;
    bow_appearance_matrix = Eigen::MatrixXd(nb_bow_descriptors,nb_frames);

    // Create the appearance matrix
    for (int ind_frame=0; ind_frame<nb_frames; ++ind_frame)
    {
        for (int d=0; d<nb_descriptors; ++d)
            bow_appearance_matrix(d,ind_frame) = appearance_descriptors[ind_frame].at<float>(0,d);
    }
    
    if (normalise_descriptors)
        normaliseBagOfWordsMatrix(bow_appearance_matrix);
    
    // Here, we display the similarity matrix
    similarity_matrix = cv::Mat(nb_frames,nb_frames,CV_64F,cv::Scalar::all(0));
    for (int i=0; i<nb_frames; ++i)
    {
        for (int j=0; j<nb_frames; ++j)
        {
        //    similarity_matrix.at<double>(i,j) = countNbCommonPoints(bow_appearance_matrix.block(0,i,nb_descriptors,1),bow_appearance_matrix.block(0,j,nb_descriptors,1));
            similarity_matrix.at<double>(i,j) = computeCosineSimilarity(bow_appearance_matrix.block(0,i,nb_descriptors,1),bow_appearance_matrix.block(0,j,nb_descriptors,1));
        }
    }
    
    cv::Mat visualisation_descriptors(nb_frames,nb_bow_descriptors,CV_8U);
    for (int i=0; i<nb_frames; ++i)
    {
        for (int j=0; j<nb_descriptors; ++j)
        {
            visualisation_descriptors.at<unsigned char>(i,j) = 255*bow_appearance_matrix(j,i);
        }
    }
    
//     cv::Mat visualisation_descriptors_jet;
//     cv::applyColorMap(visualisation_descriptors,visualisation_descriptors_jet,cv::COLORMAP_JET);
//     
//     cv::namedWindow("Visualisation descriptors",cv::WINDOW_NORMAL);
//     cv::imshow("Visualisation descriptors",visualisation_descriptors_jet);
//     cv::waitKey(-1);

}

void normaliseBagOfWordsMatrix(Eigen::MatrixXd& bow_appearance_matrix)
{
    int nb_frames = bow_appearance_matrix.cols();
    int nb_descriptors = bow_appearance_matrix.rows();
    
    for (int ind_frame=0; ind_frame<nb_frames; ++ind_frame)
    {
        double sum_descriptor(0);
        for (int d=0; d<nb_descriptors; ++d)
            sum_descriptor += bow_appearance_matrix(d,ind_frame)*bow_appearance_matrix(d,ind_frame);
        
        double norm_descriptor = sqrt(sum_descriptor);
        for (int d=0; d<nb_descriptors; ++d)
            bow_appearance_matrix(d,ind_frame) = bow_appearance_matrix(d,ind_frame)/norm_descriptor;
    }
}

void loadPrecomputedFetoscopyDataset(InputData& input_data, InputSequence& input_sequence, GroundTruthData& ground_truth_data, InteractiveHelper& interactive_helper, Settings& settings)
{
    std::string global_folder_name = "Fetoscopy/";
    std::string video_folder_name = "";
   // int starting_frame = 10;
    
    // BOW
    int cluster_count = 600;
    int diameter = 15;
    int stride = 5;
    

    input_data.nb_frames = 600;
    int precomputed_nb_frames = 700;

    input_data.dataset_identifier = "Fetoscopy";

    // Load the frames and their mask
 //   loadFetoscopyFrames(input_sequence,video_filename,starting_frame,input_data.nb_frames);
    loadFetoscopyFrames(input_sequence,input_data.nb_frames);

    input_data.nb_rows = input_sequence.frames[0].rows;
    input_data.nb_cols = input_sequence.frames[0].cols;
    input_data.agent_type = human;
    
    
    // Define corners encoding whether overlap or not (not the real corners of the image because of the mask)
    int delta_overlap = 100;
    cv::Point2f centre_point = cv::Point2f(input_sequence.frames[0].cols/2,input_sequence.frames[0].rows/2);
    input_data.overlap_corners.clear();
    for (int dx=-1; dx<=1; dx+=2)
    {
         for (int dy=-1; dy<=1; dy+=2)
            input_data.overlap_corners.push_back(centre_point + cv::Point2f(((double)dx*delta_overlap),((double)dy*delta_overlap)));
    }
    input_data.dimensions_overlap_area.resize(2);
    input_data.dimensions_overlap_area[0] = 2*delta_overlap + 1;
    input_data.dimensions_overlap_area[1] = 2*delta_overlap + 1;

    std::cout << "For the fetoscopy dataset, the scaling factor for the overlap area has no effect" << std::endl;
    
    // Bag of words
    bool loadPrecomputedBOW(true);
    computeInitialBagOfWordsModel(input_data.bow_appearance_matrix,input_data.nb_bow_descriptors,input_sequence,global_folder_name,video_folder_name,loadPrecomputedBOW,cluster_count,diameter,stride);

    // PCA
//     input_data.nb_bow_descriptors = 30;
//     Eigen::MatrixXd high_dimensional_bow_appearance_matrix = input_data.bow_appearance_matrix;
//     perform_PCA_reduction(input_data.bow_appearance_matrix,high_dimensional_bow_appearance_matrix,sinput_data.nb_bow_descriptors);
    
    
    // Fictive input landmarks
    std::vector<cv::Point2f> landmarks_input, corners; // fictive input landmarks
    createPointsforDistance(landmarks_input,input_data.nb_rows,input_data.nb_cols,1);
    createPointsforDistance(corners,input_data.nb_rows,input_data.nb_cols,1);

    // Initial registrations
    bool load_precomputed_registrations(true);
    input_data.observed_pairwise_registrations.clear();
    input_data.overlap_information = Graph<int>(input_data.nb_frames);
    performInitialRegistrations_Dense(input_data.observed_pairwise_registrations,input_sequence,landmarks_input,global_folder_name,load_precomputed_registrations,precomputed_nb_frames);
    for (int ind_frame=0; ind_frame<(input_data.nb_frames-1); ++ind_frame)
    {
        // Get matrices of transformation (mean and scaled covariance)
        getAffineDistributionFromLandmarks(input_data.observed_pairwise_registrations[ind_frame].mean_affine_parameters,input_data.observed_pairwise_registrations[ind_frame].scaled_covariance_matrix,input_data.observed_pairwise_registrations[ind_frame].landmarks_input,input_data.observed_pairwise_registrations[ind_frame].landmarks_output);
        input_data.overlap_information.add_edge(ind_frame,ind_frame+1,1);
    }

    std::cout << "Load ground truth" << std::endl;
    // Ground truth
    ground_truth_data.is_available = true;
    ground_truth_data.gt_database = LandmarkDatabase(input_data.nb_frames);
    ground_truth_data.gt_database.load("Fetoscopy/GroundTruth/ManualAnnotationsFetoscopy.txt");
    
    // Compute stats for paper
    int nb_annotated_pairs(0), nb_gt_landmarks(0), nb_pairs_above_similarity_threshold(0);
    for (int i=0; i<input_data.nb_frames; ++i)
    {
        for (int j=(i+1); j<input_data.nb_frames; ++j)
        {
            if (ground_truth_data.gt_database.isPairInDatabase(i,j))
            {
                std::cout << i << " " << j << std::endl;
                ++nb_annotated_pairs;
                std::vector<cv::Point2f> landmarks_i, landmarks_j;
                ground_truth_data.gt_database.loadLandmarks(i,j,landmarks_i, landmarks_j);
                nb_gt_landmarks += (int)landmarks_i.size();
                
            }
            
            double similarity_score = computeCosineSimilarity(input_data.bow_appearance_matrix.block(0,i,input_data.nb_bow_descriptors,1),input_data.bow_appearance_matrix.block(0,j,input_data.nb_bow_descriptors,1));
            if (similarity_score>=0.8)
                ++nb_pairs_above_similarity_threshold;
            
        }
    }
    std::cout << "Nb pairs: " << nb_annotated_pairs << std::endl;
    std::cout << "Nb correspondences: " << nb_gt_landmarks << std::endl;
    std::cout << "Nb pairs above 0.8: " << nb_pairs_above_similarity_threshold << std::endl; 
        
    // Manual annotation of landmarks to construct a ground truth
//     std::string manual_gt_database_filename = "ManualAnnotationsFetoscopy.txt";
//     LandmarkDatabase manual_gt_annotations(input_data.nb_frames);
//     manual_gt_annotations.load(manual_gt_database_filename);
//     int frame_subsampling_rate_i(50); // we subsample the sequence to make the annotation faster
//     int frame_subsampling_rate_j(10); // we subsample the sequence to make the annotation faster
//     bool stop_labelling(false);
//     for (int i=250; i<input_data.nb_frames; i+=frame_subsampling_rate_i)
//     {
//         for (int j=(i+frame_subsampling_rate_j); j<input_data.nb_frames; j+=frame_subsampling_rate_j)
//         {
//             if ((!stop_labelling) && (!(manual_gt_annotations.isPairInDatabase(i,j))))
//                 manualLandmarkAnnotation(stop_labelling,manual_gt_annotations,input_sequence,i,j);
//             
//         }
//     }
//     manual_gt_annotations.save(manual_gt_database_filename);
    
    
    settings.automate_interactions = false;

    

    
    // Interactive helper
    interactive_helper.landmark_database_identifier = input_data.dataset_identifier + "-LandmarkDatabase.txt";
    interactive_helper.database_annotations = LandmarkDatabase(input_data.nb_frames);
    interactive_helper.database_annotations.load(interactive_helper.landmark_database_identifier);

   std::cout << "Fetoscopy dataset loaded" << std::endl;
   
   
}

void loadAerialDataset(InputData& input_data, InputSequence& input_sequence, GroundTruthData& ground_truth_data, InteractiveHelper& interactive_helper, Settings& settings)
{

    std::string global_folder_name = "Aerial/";
    std::string video_folder_name = "";
    std::string raw_database_name = global_folder_name + "Registrations/TrueCorrespondences-AllFrames-Raw.txt";
    std::string curated_database_name = global_folder_name + "Registrations/TrueCorrespondences-AllFrames-Curated.txt";
    std::string measurements_database_name = global_folder_name + "Registrations/Measurements-AllFrames-Curated.txt";
    
    
//    std::string raw_database_name = global_folder_name + "Registrations/ForTests-Raw.txt";
//    std::string curated_database_name = global_folder_name + "Registrations/ForTests-Curated.txt";
//    std::string measurements_database_name = global_folder_name + "Registrations/MeasurementsForTests-Curated.txt";
    
  //  int nb_neighbours = 80;
    int nb_neighbours = 3;
    int nb_neighbours_consecutive = 1;
    int nb_points_pruning = settings.annotated_points_in_interaction;
    input_data.nb_frames = 744;
  //  input_data.nb_frames = 200;
    
    // BOW
    int downsize_factor_BOW = 1;
    int nb_descriptors = 1000;
    int diameter = 100;
    int stride = 100;
    bool build_with_detection = true;
    std::string bow_name = "SIFT-DenseDetection";
    
    
    bool loadPrecomputedBOW(true);
    bool loadPrecomputedRegistrations(true);
    bool curateDatabase(false);
    bool create_measurements_database(false);
    loadAerialFrames(input_sequence,input_data.nb_frames);
    input_data.nb_frames = (int)input_sequence.frames.size();
    input_data.dataset_identifier = "Aerial";
    input_data.nb_rows = input_sequence.frames[0].rows;
    input_data.nb_cols = input_sequence.frames[0].cols;
    input_data.feature_extractor = cv::xfeatures2d::SURF::create(300);
  //  input_data.feature_extractor = cv::xfeatures2d::SURF::create(20000,4,3,true,false);
//    input_data.feature_extractor = cv::xfeatures2d::SIFT::create(10,3,0.01,1,4); 		
 //   input_data.feature_extractor = cv::ORB::create(50); 
    input_data.agent_type = landmark_registration;
    
    // Image corners
    std::vector<cv::Point2f> corners;
    createPointsforDistance(corners,input_data.nb_rows,input_data.nb_cols,1);
  //  input_data.overlap_corners = corners;

    load_overlap_area_and_dimensions(input_data,settings.factor_overlap_area);
    
    
    // This part is to create the gt database
//   std::vector<std::pair<int,int>> consecutive_pairs;
//     for (int i=0; i<input_data.nb_frames; ++i)
//     {
//         for (int j=(i+1); j<input_data.nb_frames; ++j)
//         {
//             if (abs(i-j)<=nb_neighbours)
//                 consecutive_pairs.push_back(std::pair<int,int>(i,j));
//         }
//     }
//     loadInitialPairwiseMeasurements(input_data,pruned_database,consecutive_pairs,input_data.nb_frames);
    
    
    if (loadPrecomputedRegistrations==false)
    {
        std::vector<std::pair<int,int>> pairs_to_register;     
        for (int i=0; i<input_data.nb_frames; ++i)
        {
            for (int j=i+1; j<input_data.nb_frames; ++j)
            {
                if (abs(i-j)<=nb_neighbours)
                    pairs_to_register.push_back(std::pair<int,int>(i,j));
            }
        }
        std::vector<MeasuredAffineRegistration> dummy_raw_database;
        performInitialRegistrations_Landmarks(dummy_raw_database,input_data,input_sequence,pairs_to_register,global_folder_name,video_folder_name,raw_database_name);
    }
    

    // Load (possibly curated) database of available correspondences as ground truth
    LandmarkDatabase gt_all_correspondences(input_data.nb_frames);
    if (curateDatabase)
    {
        int nb_kept_points = 50;
        curate_landmark_database(curated_database_name,raw_database_name,input_sequence.frames,nb_kept_points);
        gt_all_correspondences.load(curated_database_name);
    }
    else
        gt_all_correspondences.load(raw_database_name);
    ground_truth_data.gt_database = LandmarkDatabase(input_data.nb_frames);
    keep_long_range_correspondences(ground_truth_data.gt_database,gt_all_correspondences,input_data.nb_frames,0);
    
    
    // Bag of words
    int total_nb_descriptors;
    computeInitialBagOfWordsModel(input_data.bow_appearance_matrix,input_data.nb_bow_descriptors,input_sequence,global_folder_name,video_folder_name,loadPrecomputedBOW,nb_descriptors,diameter,stride,downsize_factor_BOW,build_with_detection,bow_name,true);
    
    // Apply PCA on it
//     std::cout << "Run PCA" << std::endl;
//     input_data.nb_bow_descriptors = 20;
//     Eigen::MatrixXd full_bow_appearance_matrix = input_data.bow_appearance_matrix;
//     full_bow_appearance_matrix.transposeInPlace();
//     perform_PCA_reduction(input_data.bow_appearance_matrix,full_bow_appearance_matrix,input_data.nb_bow_descriptors);
//     input_data.bow_appearance_matrix.transposeInPlace();
//     normaliseBagOfWordsMatrix(input_data.bow_appearance_matrix);
    
   // Create the measurements from the full database 
    std::cout << "Prune database" << std::endl;
    ground_truth_data.measurements_database = LandmarkDatabase(input_data.nb_frames);
    if (create_measurements_database)
    {
        prune_landmark_database(ground_truth_data.measurements_database,ground_truth_data.gt_database,nb_points_pruning,input_data.nb_frames);
        ground_truth_data.measurements_database.save(measurements_database_name);
    }
    else
        ground_truth_data.measurements_database.load(measurements_database_name);
    std::cout << "Done" << std::endl;
    
    
    // Load initial input measurements
    int count_overlapping_pairs(0);
    std::vector<std::pair<int,int>> consecutive_pairs;
    for (int i=0; i<input_data.nb_frames; ++i)
    {
        for (int j=(i+1); j<input_data.nb_frames; ++j)
        {
            if (abs(i-j)<=nb_neighbours_consecutive)
                consecutive_pairs.push_back(std::pair<int,int>(i,j));
            
            if (ground_truth_data.measurements_database.isPairInDatabase(i,j))
            {
                std::cout << i << j << std::endl;
                ++count_overlapping_pairs;
            }
            
        }
    }
    loadInitialPairwiseMeasurements(input_data,ground_truth_data.measurements_database,consecutive_pairs,input_data.nb_frames);
    std::cout << "Total: " << count_overlapping_pairs << std::endl;
    // Or, instead, load all the availabe measurements to have the gold standard bundle adjustment
//    loadAllPairwiseMeasurements(input_data,pruned_database,input_data.nb_frames);
    
  
    // Ground truth
    ground_truth_data.is_available = true;
    settings.automate_interactions = true;
    
    
    // Interactive helper
    interactive_helper.landmark_database_identifier = input_data.dataset_identifier + "-LandmarkDatabase.txt";
    interactive_helper.database_annotations = LandmarkDatabase(input_data.nb_frames);
    //interactive_helper.database_annotations.load(interactive_helper.landmark_database_identifier);
   std::cout << "Aerial dataset loaded" << std::endl;
}

void perform_PCA_reduction(Eigen::MatrixXd& output_matrix, const Eigen::MatrixXd& input_matrix, int nb_components)
{
    int n = input_matrix.rows();
    int p = input_matrix.cols();

    
    // Each column has mean 0
    Eigen::MatrixXd X(n,p);
    Eigen::VectorXd mean_values(p);
    mean_values = input_matrix.colwise().mean().transpose();
    for (int j=0; j<p; ++j)
        X.block(0,j,n,1) = input_matrix.block(0,j,n,1) - Eigen::MatrixXd::Constant(n,1,mean_values(j));
    
   // std::cout << mean_values << std::endl;
    
    // Compute SVD
    Eigen::BDCSVD<Eigen::MatrixXd> X_svd(n,p,Eigen::ComputeFullU | Eigen::ComputeFullV);
    X_svd.compute(X);
    
    // Truncate
    Eigen::MatrixXd U = X_svd.matrixU();
    Eigen::VectorXd singular_values = X_svd.singularValues(); 
    output_matrix = U.block(0,0,n,nb_components)*(singular_values.segment(0,nb_components).asDiagonal());
    
    // Transformation matrix
    Eigen::MatrixXd V = X_svd.matrixV();
    Eigen::VectorXd transformed_mean = mean_values*V.block(0,0,p,nb_components);
    
    for (int i=0; i<n; ++i)
        output_matrix.block(i,0,1,nb_components) += transformed_mean;
}

// void loadGroundTruthData(GroundTruthData& ground_truth_data, LandmarkDatabase& landmark_database, int nb_frames)
// {
//     //ground_truth_data.true_overlap_information = Graph<int>(nb_frames);
//     //ground_truth_data.true_correspondences.clear();
//     for (int i=0; i<nb_frames; ++i)
//     {
//         for (int j=(i+1); j<nb_frames; ++j)
//         {
//             if (landmark_database.isPairInDatabase(i,j))
//             {
//                 ground_truth_data.true_overlap_information.add_edge(i,j,1);
//                 MeasuredAffineRegistration new_corr;
//                 new_corr.frame_indices = std::pair<int,int>(i,j);
//                 landmark_database.loadLandmarks(i,j,new_corr.landmarks_input,new_corr.landmarks_output);
//                 ground_truth_data.true_correspondences.push_back(new_corr);
//             }
//         }
//         
//     }
// }

void loadInitialPairwiseMeasurements(InputData& input_data, LandmarkDatabase& landmark_database, const std::vector<std::pair<int,int>>& overlapping_pairs, int nb_frames)
{
    input_data.overlap_information =  Graph<int>(nb_frames);
    input_data.observed_pairwise_registrations.clear();
    int nb_overlapping_pairs = (int)overlapping_pairs.size(); 
    for (int k=0; k<nb_overlapping_pairs; ++k)
    {
        int i = overlapping_pairs[k].first;
        int j = overlapping_pairs[k].second;
        if (landmark_database.isPairInDatabase(i,j))
        {
            input_data.overlap_information.add_edge(i,j,1);
            MeasuredAffineRegistration new_corr;
            new_corr.frame_indices = std::pair<int,int>(i,j);
            landmark_database.loadLandmarks(i,j,new_corr.landmarks_input,new_corr.landmarks_output);
            
            getAffineDistributionFromLandmarks(new_corr.mean_affine_parameters,new_corr.scaled_covariance_matrix,new_corr.landmarks_input,new_corr.landmarks_output);

            input_data.observed_pairwise_registrations.push_back(new_corr);
        }
        else
            std::cout << "Warning! In loadInitialPairwiseMeasurements, the input pair (" << i << "," << j << ")  is supposed to overlap, but was not found in the database" << std::endl;
    }
}

void loadAllPairwiseMeasurements(InputData& input_data, LandmarkDatabase& landmark_database, int nb_frames)
{
    input_data.overlap_information =  Graph<int>(nb_frames);
    input_data.observed_pairwise_registrations.clear();
    for (int i=0; i<nb_frames; ++i)
    {
        for (int j=(i+1); j<nb_frames; ++j)
        {
            if (landmark_database.isPairInDatabase(i,j))
            {
                input_data.overlap_information.add_edge(i,j,1);
                MeasuredAffineRegistration new_corr;
                new_corr.frame_indices = std::pair<int,int>(i,j);
                landmark_database.loadLandmarks(i,j,new_corr.landmarks_input,new_corr.landmarks_output);
                getAffineDistributionFromLandmarks(new_corr.mean_affine_parameters,new_corr.scaled_covariance_matrix,new_corr.landmarks_input,new_corr.landmarks_output);
                input_data.observed_pairwise_registrations.push_back(new_corr);
            }
        }
    }
}

void prune_landmark_database(LandmarkDatabase& pruned_database, LandmarkDatabase& landmark_database, int nb_kept_points, int nb_frames)
{
    bool visualizeKeypoints(false);
    std::vector<cv::Mat> frames(0);
    prune_landmark_database(pruned_database,landmark_database,nb_kept_points,nb_frames,visualizeKeypoints,frames);
}

void prune_landmark_database(LandmarkDatabase& pruned_database, LandmarkDatabase& landmark_database, int nb_kept_points, int nb_frames, bool visualizeKeypoints, const std::vector<cv::Mat>& frames)
{
    if (visualizeKeypoints)
    {
        cv::namedWindow("Input frame");
        cv::namedWindow("Output frame");
    }
    
    std::vector<cv::Point2f> input_points, output_points, pruned_input_points, pruned_output_points;
    for (int i=0; i<nb_frames; ++i)
    {
        for (int j=(i+1); j<nb_frames; ++j)
        {
            if (landmark_database.isPairInDatabase(i,j))
            {
                // Load correspondences from database
                landmark_database.loadLandmarks(i,j,input_points,output_points);
                
                // Pick random set of points that are not aligned
        //        pick_random_points(pruned_input_points,pruned_output_points,input_points,output_points,nb_kept_points);
                pick_least_aligned_points(pruned_input_points,pruned_output_points,input_points,output_points,nb_kept_points);
                
                // Add to pruned database
                if (pruned_database.isPairInDatabase(i,j))
                    pruned_database.eraseFromDatabase(i,j);
                pruned_database.addToDatabase(i,j,pruned_input_points,pruned_output_points);
                
                // Compute vector of keypoints
                if (visualizeKeypoints)
                {
                    std::vector<cv::KeyPoint> retained_input_keypoints, retained_output_keypoints;
                    for (int k=0; k<nb_kept_points; ++k)
                    {
                        retained_input_keypoints.push_back(cv::KeyPoint(pruned_input_points[k],1));
                        retained_output_keypoints.push_back(cv::KeyPoint(pruned_output_points[k],1));
                    }
                    cv::Mat output_image_i, output_image_j;
                    drawKeypoints(frames[i],retained_input_keypoints,output_image_i);
                    drawKeypoints(frames[j],retained_output_keypoints,output_image_j);
                    imshow("Input frame",output_image_i);
                    imshow("Output frame",output_image_j);
                    cv::waitKey(10000);
                }
            }
        }
    }
}

void keep_long_range_correspondences(LandmarkDatabase& pruned_database, LandmarkDatabase& landmark_database, int nb_frames, int min_interframe_time)
{
    std::vector<cv::Point2f> input_points, output_points;
    for (int i=0; i<nb_frames; ++i)
    {
        for (int j=(i+min_interframe_time); j<nb_frames; ++j)
        {
            if (landmark_database.isPairInDatabase(i,j))
            {
                // Load correspondences from database
                landmark_database.loadLandmarks(i,j,input_points,output_points);
                
                // Add to pruned database
                if (pruned_database.isPairInDatabase(i,j))
                    pruned_database.eraseFromDatabase(i,j);
                pruned_database.addToDatabase(i,j,input_points,output_points);
            }
        }
    }
}


void curate_landmark_database(const std::string& name_curated_database, const std::string& name_input_database, const std::vector<cv::Mat>& frames, int nb_kept_points)
{
    int nb_frames = (int)frames.size();
    LandmarkDatabase curated_database(nb_frames), input_database(nb_frames);
    curated_database.load(name_curated_database);
    input_database.load(name_input_database);
    curate_landmark_database(curated_database,input_database,frames,nb_kept_points);
    curated_database.save(name_curated_database);
  //  input_database.save(name_input_database);
}


// In this function, we go through all the landmark correpsondences in a database and ask the user to confirm/discard the matches via visual inspection
void curate_landmark_database(LandmarkDatabase& curated_database, LandmarkDatabase& input_database, const std::vector<cv::Mat>& frames, int nb_kept_points)
{
    cv::namedWindow("Matches");
    bool quit_interface(false);
    std::vector<cv::Point2f> input_points, output_points, pruned_input_points, pruned_output_points;
    int nb_frames = (int)frames.size();
    for (int i=0; i<nb_frames; ++i)
    {
        for (int j=(i+1); j<nb_frames; ++j)
        {
            if ((input_database.isPairInDatabase(i,j)) && (curated_database.isPairInDatabase(i,j)==false))
            {
                // Load correspondences from database
                input_database.loadLandmarks(i,j,input_points,output_points);
                
                // Pick random subset of points
                pick_random_points(pruned_input_points,pruned_output_points,input_points,output_points,nb_kept_points);


                std::vector<cv::KeyPoint> retained_input_keypoints, retained_output_keypoints;
                std::vector<cv::DMatch> matches;
                for (int k=0; k<pruned_input_points.size(); ++k)
                {
                    retained_input_keypoints.push_back(cv::KeyPoint(pruned_input_points[k],1));
                    retained_output_keypoints.push_back(cv::KeyPoint(pruned_output_points[k],1));
                    matches.push_back(cv::DMatch(k,k,0));
                   
                }
                cv::Mat outputImage;
                drawMatches(frames[i],retained_input_keypoints,frames[j],retained_output_keypoints,matches,outputImage);
                cv::imshow("Matches",outputImage);
                std::cout << "Pair (" << i << "," << j << "): Press Y if the proposed match is correct. Press Esc to leave the interface" << std::endl;
                int entered_key;
                entered_key = (int)cv::waitKey(0);

                if (entered_key==27) // escape
                    quit_interface = true;
                else
                {
                    if (entered_key==121) // Y
                    {
                        std::cout << "Correct match" << std::endl;
                        curated_database.addToDatabase(i,j,pruned_input_points,pruned_output_points);
                    }
                //    else
                //        input_database.eraseFromDatabase(i,j);
                }
                
            }
            
            if (quit_interface)
                break;
        }
        
        if (quit_interface)
                break;
    }
}


void pick_random_points(std::vector<cv::Point2f>& new_input_points, std::vector<cv::Point2f>& new_output_points, const std::vector<cv::Point2f>& input_points, const std::vector<cv::Point2f>& output_points, int input_nb_kept_points)
{
    int nb_points = input_points.size();
    int nb_kept_points = std::min<int>(input_nb_kept_points,nb_points);

    // Create vector of indices for random sampling
    std::vector<int> ind_points(nb_points);
    for (int k=0; k<nb_points; ++k)
        ind_points[k] = k;
    
    // We shuffle
    std::random_shuffle(ind_points.begin(),ind_points.end());

    new_input_points.clear();
    new_output_points.clear();
    for (int k=0; k<nb_kept_points; ++k)
    {
            new_input_points.push_back(input_points[ind_points[k]]);
            new_output_points.push_back(output_points[ind_points[k]]);
    }
}




void test_overlap(bool& is_i_in_j, bool& is_j_in_i, double *H_i, double *H_j, const std::vector<cv::Point2f>& corners)
{
    std::vector<cv::Point2f> centre(1);
    centre[0] = (0.25)*(corners[0] + corners[1] + corners[2] + corners[3]);
    std::vector<cv::Point2f> displaced_corners_i(4), displaced_corners_j(4), displaced_centre_i(1), displaced_centre_j(1);
    getDisplacedPointsAffine(displaced_corners_i,H_i,corners);
    getDisplacedPointsAffine(displaced_centre_i,H_i,centre);
    getDisplacedPointsAffine(displaced_corners_j,H_j,corners);
    getDisplacedPointsAffine(displaced_centre_j,H_j,centre);

    is_i_in_j = is_point_inside_parallelogram(displaced_centre_i[0],displaced_corners_j);
    is_j_in_i = is_point_inside_parallelogram(displaced_centre_j[0],displaced_corners_i);
}


// j = fixed image, i = moving image, j < i
void count_overlapping_points(std::vector<cv::Point2f>& overlapping_control_points_input, std::vector<cv::Point2f>& overlapping_control_points_output, double *T_ij, const std::vector<cv::Point2f>& corners, const std::vector<cv::Point2f>& control_points)
{
    int nb_control_points = (int)control_points.size();
    std::vector<cv::Point2f> displaced_control_points, displaced_corners;
    overlapping_control_points_input.clear();
    overlapping_control_points_output.clear();
    double T_ji[6];
    inverse_affine_matrix(T_ji,T_ij);
    getDisplacedPointsAffine(displaced_control_points,T_ij,control_points);
    getDisplacedPointsAffine(displaced_corners,T_ji,corners);
    for (int k=0; k<nb_control_points; ++k)
    {
        if (is_point_inside_parallelogram(control_points[k],displaced_corners))
        {
             overlapping_control_points_input.push_back(control_points[k]);
             overlapping_control_points_output.push_back(displaced_control_points[k]);
        }
    }
}



void triangulate_graph_of_constraints(Graph<cv::Mat>& pairwise_homographies, cv::Mat& pairwise_residuals)
{
    int nb_frames = (int)pairwise_homographies.m_graph_relationships.size();
    for (int i = 0; i <nb_frames; ++i)
    {
        for(int j=(i+1); j<nb_frames; ++j)
        {
            if ((pairwise_homographies.is_there_edge(i,j)) && (pairwise_homographies.is_there_edge(j,i)))
            {

                if (pairwise_residuals.at<double>(i,j) > pairwise_residuals.at<double>(j,i)) // if the "inverse" is better, we update
                {
                    cv::Mat h_ji = pairwise_homographies.m_graph_relationships[j][i];
                    pairwise_homographies.m_graph_relationships[i][j] = h_ji.inv();
                    pairwise_residuals.at<double>(i,j) = pairwise_residuals.at<double>(j,i);
                }
                pairwise_homographies.remove_edge(j,i);
            }
            else
            {
                if (pairwise_homographies.is_there_edge(j,i)) // i.e, if only the "wrong side" of the graph has a constraint
                {
                    cv::Mat h_ji = pairwise_homographies.m_graph_relationships[j][i];
                    pairwise_homographies.add_edge(i,j,h_ji.inv());
                    pairwise_residuals.at<double>(i,j) = pairwise_residuals.at<double>(j,i);
                    pairwise_homographies.remove_edge(j,i);
                }
            }
        }
    }
}



// void getDisplacedPointsAffine(std::vector<cv::Point2f>& output_points, double *global_affine_parameters, int i, int j, const std::vector<cv::Point2f>& input_points)
// {
//     double *H_i_inv_ptr = new double[6];
//     double *h_ij = new double[6];   // h_ij = H_j * H_i^(-1)
//     inverse_affine_matrix<double>(H_i_inv_ptr,global_affine_parameters + 6*i);
//     multiply_affine_and_affine<double>(h_ij,global_affine_parameters + 6*j,H_i_inv_ptr);
//     getDisplacedPointsAffine(output_points,h_ij,input_points);
// 
//     delete[] H_i_inv_ptr;
//     delete[] h_ij;
// }


void pick_least_aligned_points(std::vector<cv::Point2f>& new_input_points, std::vector<cv::Point2f>& new_output_points, const std::vector<cv::Point2f>& input_points, const std::vector<cv::Point2f>& output_points, int input_nb_kept_points, int random_seed)
{
    std::srand(random_seed);
    int nb_points = input_points.size();
    int nb_kept_points = std::min<int>(input_nb_kept_points,nb_points);
    int nb_iterations = 1000;
    double min_score;
    // Create vector of indices for random sampling
    std::vector<int> ind_points(nb_points), best_indices(nb_kept_points);
    for (int k=0; k<nb_points; ++k)
        ind_points[k] = k;
    
    // We take the combination of points leading to the lowest conditon number
    for (int it=0; it<nb_iterations; ++it)
    {
        std::random_shuffle(ind_points.begin(),ind_points.end());
        double score = compute_alignment_score(ind_points,nb_kept_points,input_points,output_points);
        if ((it==0) || (score<min_score))
        {
            min_score = score;
            for (int k=0; k<nb_kept_points; ++k)
                best_indices[k] = ind_points[k];
        }
    }
    

    new_input_points.clear();
    new_output_points.clear();
   // if (min_score==0)
   // {
       // std::cout << "Could not find points that were not aligned. List of available points: " << std::endl;
      //  for (int k=0; k<nb_points; ++k)
      //      std::cout << input_points[k] << std::endl;
   // }
    
    for (int k=0; k<nb_kept_points; ++k)
    {
            new_input_points.push_back(input_points[best_indices[k]]);
            new_output_points.push_back(output_points[best_indices[k]]);
    }
}



double compute_alignment_score(const std::vector<int>& ind_points, int nb_kept_points, const std::vector<cv::Point2f>& input_points, const std::vector<cv::Point2f>& output_points)
{
    return std::max<double>(compute_alignment_score(ind_points,nb_kept_points,input_points),compute_alignment_score(ind_points,nb_kept_points,output_points));
}

double compute_alignment_score(const std::vector<int>& ind_points, int nb_kept_points, const std::vector<cv::Point2f>& input_points)
{
    double eps = 0.00000000000001;
    Eigen::MatrixXd X;
    X = Eigen::MatrixXd::Constant(3,nb_kept_points,1);
    for (int k=0; k<nb_kept_points; ++k)
    {
        X(0,k) = input_points[ind_points[k]].x;
        X(1,k) = input_points[ind_points[k]].y;
    }
    
    Eigen::Matrix3d A = X*(X.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;
    es.computeDirect(A);
    Eigen::Vector3d lambda = es.eigenvalues();
    double min_eigenvalue = lambda.minCoeff();
    double max_eigenvalue = lambda.maxCoeff();
  //  std::cout << min_eigenvalue << " " << max_eigenvalue << std::endl;
   // return max_eigenvalue/(std::max<double>(min_eigenvalue,eps));
    return -(A.determinant());
}
