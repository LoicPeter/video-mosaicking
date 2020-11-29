#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <sstream>
#include <ctime>
#include "structures_for_interactive_mosaicking.hpp"
#include "graph.hpp"
#include "utilities.hpp"
#include "position_overlap_model.hpp"
#include "bow_overlap_model.hpp"
#include <omp.h>
#include "interaction_loop.hpp"
#include "data_preparation.hpp"

// For solving system
#include "stdafx.h"
#include "optimization.h"


#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

// Random number / probabilities / sampling
#include <boost/math/special_functions/erf.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/normal_distribution.hpp>


void run_experiments();

int main()
{
    // Mosaicking
    run_experiments();
}



void run_experiments()
{

    Settings settings;

    // Set up settings
    settings.nb_iterations_interaction = 50;
    settings.nb_possible_skips = settings.nb_iterations_interaction;
    settings.automatic_registration_variance = 1;
    settings.annotation_variance = 1;
    settings.ind_reference_frame = 0;
    settings.beta = 3; // for PCCA model
    settings.lambda = 0.01; // for regularisation of PCCA model in logistic regression
    settings.nb_iterations_cotraining = 10;
    settings.nb_samples_montecarlo_overlap_estimation = 2000;
    settings.seed_random_number_generator = 1;
    settings.factor_overlap_area = 1;
    settings.annotated_points_in_interaction = 3;
    settings.automate_interactions = true;
    settings.step_size_grid_similarity_measure = 1;
    settings.remap_correspondences = true;
    settings.do_cotraining = false;
    settings.diagonal_bow_weights = true;
    settings.createVideos = false;
    settings.bundle_adjustment_on_correspondences = true;
    settings.compute_overlap_probability_for_all_pairs = true;
    settings.random_landmarks_in_interactions = false;
    settings.chosen_positional_overlap_model = our_positional_overlap_model;
    settings.chosen_external_overlap_model = our_external_overlap_model;
    settings.chosen_reward = our_reward;
    settings.reestimate_from_initial_measurements_only = true;
    settings.reestimate_position_overlap_model = false;
    settings.reestimate_external_overlap_model = true;
    settings.experiment_identifier = "Tests";

    
    std::vector<Settings> experiments_settings(19,settings);

    
    // The new, final series of experiments
    
    // External baselines
    experiments_settings[0].chosen_positional_overlap_model = our_positional_overlap_model;
    experiments_settings[0].chosen_external_overlap_model = our_external_overlap_model;
    experiments_settings[0].chosen_reward = our_reward;
    experiments_settings[0].experiment_identifier = "Ours-External-Ours";
    
    experiments_settings[1].chosen_positional_overlap_model = sawhney_probability;
    experiments_settings[1].chosen_external_overlap_model = our_external_overlap_model;
    experiments_settings[1].chosen_reward = shortest_path;
    experiments_settings[1].experiment_identifier = "Sawhney-External-Sawhney";
    
    experiments_settings[2].chosen_positional_overlap_model = elibol_filtering_up;
    experiments_settings[2].chosen_external_overlap_model = our_external_overlap_model;
    experiments_settings[2].chosen_reward = entropy_reward;
    experiments_settings[2].experiment_identifier = "ElibolUp-External-Elibol";
    
    experiments_settings[3].chosen_positional_overlap_model = no_positional_overlap_model;
    experiments_settings[3].chosen_external_overlap_model = our_external_overlap_model;
    experiments_settings[3].chosen_reward = our_reward;
    experiments_settings[3].experiment_identifier = "None-External-Ours";
    
    experiments_settings[4].chosen_positional_overlap_model = sawhney_probability;
    experiments_settings[4].chosen_external_overlap_model = no_external_overlap_model;
    experiments_settings[4].chosen_reward = shortest_path;
    experiments_settings[4].experiment_identifier = "Sawhney-None-Sawhney";
    
    experiments_settings[5].chosen_positional_overlap_model = elibol_filtering_up;
    experiments_settings[5].chosen_external_overlap_model = no_external_overlap_model;
    experiments_settings[5].chosen_reward = entropy_reward;
    experiments_settings[5].experiment_identifier = "ElibolUp-None-Elibol";
    
    experiments_settings[6].chosen_positional_overlap_model = sawhney_probability;
    experiments_settings[6].chosen_external_overlap_model = our_external_overlap_model;
    experiments_settings[6].chosen_reward = our_reward;
    experiments_settings[6].experiment_identifier = "Sawhney-External-Ours";
    
    experiments_settings[7].chosen_positional_overlap_model = elibol_filtering_up;
    experiments_settings[7].chosen_external_overlap_model = our_external_overlap_model;
    experiments_settings[7].chosen_reward = our_reward;
    experiments_settings[7].experiment_identifier = "ElibolUp-External-Ours";
    
    experiments_settings[8].chosen_positional_overlap_model = xia_filtering;
    experiments_settings[8].chosen_external_overlap_model = our_external_overlap_model;
    experiments_settings[8].chosen_reward = our_reward;
    experiments_settings[8].experiment_identifier = "Xia-External-Ours";
    
    experiments_settings[9].chosen_positional_overlap_model = our_positional_overlap_model;
    experiments_settings[9].chosen_external_overlap_model = our_external_overlap_model;
    experiments_settings[9].chosen_reward = shortest_path;
    experiments_settings[9].experiment_identifier = "Ours-External-Sawhney";
    
    experiments_settings[10].chosen_positional_overlap_model = our_positional_overlap_model;
    experiments_settings[10].chosen_external_overlap_model = our_external_overlap_model;
    experiments_settings[10].chosen_reward = entropy_reward;
    experiments_settings[10].experiment_identifier = "Ours-External-Elibol";

    experiments_settings[11].chosen_positional_overlap_model = elibol_filtering_up;
    experiments_settings[11].chosen_external_overlap_model = our_external_overlap_model;
    experiments_settings[11].chosen_reward = entropy_reward;
    experiments_settings[11].experiment_identifier = "ElibolUp-External-Elibol";
    
    experiments_settings[12].chosen_positional_overlap_model = elibol_filtering_up;
    experiments_settings[12].chosen_external_overlap_model = our_external_overlap_model;
    experiments_settings[12].chosen_reward = our_reward;
    experiments_settings[12].experiment_identifier = "ElibolUp-External-Ours";
    
    experiments_settings[13].chosen_positional_overlap_model = no_positional_overlap_model;
    experiments_settings[13].chosen_external_overlap_model = our_external_overlap_model;
    experiments_settings[13].chosen_reward = no_reward;
    experiments_settings[13].experiment_identifier = "None-External-None";
    
    experiments_settings[14].chosen_positional_overlap_model = xia_filtering;
    experiments_settings[14].chosen_external_overlap_model = no_external_overlap_model;
    experiments_settings[14].chosen_reward = xia_reward;
    experiments_settings[14].experiment_identifier = "Xia-None-Xia";
    
    experiments_settings[15].chosen_positional_overlap_model = our_positional_overlap_model;
    experiments_settings[15].chosen_external_overlap_model = no_external_overlap_model;
    experiments_settings[15].chosen_reward = no_reward;
    experiments_settings[15].experiment_identifier = "Ours-None-None";
    
    experiments_settings[16].chosen_positional_overlap_model = our_positional_overlap_model;
    experiments_settings[16].chosen_external_overlap_model = our_external_overlap_model;
    experiments_settings[16].chosen_reward = no_reward;
    experiments_settings[16].experiment_identifier = "Ours-External-None";
    
    experiments_settings[17].chosen_positional_overlap_model = our_positional_overlap_model;
    experiments_settings[17].chosen_external_overlap_model = no_external_overlap_model;
    experiments_settings[17].chosen_reward = our_reward;
    experiments_settings[17].experiment_identifier = "Ours-None-Ours";
    
    experiments_settings[18].chosen_positional_overlap_model = xia_filtering;
    experiments_settings[18].chosen_external_overlap_model = our_external_overlap_model;
    experiments_settings[18].chosen_reward = no_reward;
    experiments_settings[18].experiment_identifier = "Xia-External-None";

  //  int ind_exps[4] = {0,1,2,18};
  //  int ind_exps[5] = {0,3,17};
    int ind_exps[5] = {0,3,4,5,17}; // the 5 experiments in the paper
  //  int ind_exps[3] = {3,5,17};
    
    // Fetoscopy
    for (int ind_ind_exp = 0; ind_ind_exp<0; ++ind_ind_exp)
    {
        int ind_exp = ind_exps[ind_ind_exp];
        
        InputData input_data;
        GroundTruthData ground_truth_data;
        InputSequence input_sequence;
        InteractiveHelper interactive_helper;
        
        experiments_settings[ind_exp].nb_iterations_interaction = 10;
        experiments_settings[ind_exp].nb_possible_skips = 100;
    //    loadAerialDataset(input_data,input_sequence,ground_truth_data,interactive_helper,experiments_settings[ind_exp]);
        loadPrecomputedFetoscopyDataset(input_data,input_sequence,ground_truth_data,interactive_helper,experiments_settings[ind_exp]);
        run_interactive_process(ground_truth_data,input_data,input_sequence,interactive_helper,experiments_settings[ind_exp]);
        
    }
    
        
    // Aerial
    for (int ind_ind_exp = 0; ind_ind_exp<0; ++ind_ind_exp)
    {
        int ind_exp = ind_exps[ind_ind_exp];
        
        InputData input_data;
        GroundTruthData ground_truth_data;
        InputSequence input_sequence;
        InteractiveHelper interactive_helper;
        
        experiments_settings[ind_exp].nb_iterations_interaction = 100;
        experiments_settings[ind_exp].nb_possible_skips = 110;
        loadAerialDataset(input_data,input_sequence,ground_truth_data,interactive_helper,experiments_settings[ind_exp]);
        run_interactive_process(ground_truth_data,input_data,input_sequence,interactive_helper,experiments_settings[ind_exp]);
        
    }
    

    
    for (int rs=66; rs<=67; ++rs)
    {
        std::ostringstream oss_rs;
        oss_rs << rs;
        
        
        for (int ind_ind_exp = 0; ind_ind_exp<0; ++ind_ind_exp)
        {
            int ind_exp = ind_exps[ind_ind_exp];
            
            InputData input_data;
            GroundTruthData ground_truth_data;
            InputSequence input_sequence;
            InteractiveHelper interactive_helper;
            
            
            experiments_settings[ind_exp].nb_iterations_interaction = 50;
            experiments_settings[ind_exp].nb_possible_skips = 100;
            experiments_settings[ind_exp].createVideos = true;
            Settings current_experiment_settings = experiments_settings[ind_exp];
            current_experiment_settings.experiment_identifier = experiments_settings[ind_exp].experiment_identifier + "-RS" + oss_rs.str();
           
            createSyntheticCircleDataset(input_sequence,input_data,ground_truth_data,current_experiment_settings,rs);
            run_interactive_process(ground_truth_data,input_data,input_sequence,interactive_helper,current_experiment_settings);
        
        }
        
        for (int ind_ind_exp = 0; ind_ind_exp<1; ++ind_ind_exp)
        {
            int ind_exp = ind_exps[ind_ind_exp];
            
            InputData input_data;
            GroundTruthData ground_truth_data;
            InputSequence input_sequence;
            InteractiveHelper interactive_helper;
            
            
            experiments_settings[ind_exp].nb_iterations_interaction = 50;
            experiments_settings[ind_exp].nb_possible_skips = 100;
            experiments_settings[ind_exp].createVideos = true;
            Settings current_experiment_settings = experiments_settings[ind_exp];
            current_experiment_settings.experiment_identifier = experiments_settings[ind_exp].experiment_identifier + "-RS" + oss_rs.str();
           
            createSyntheticRasterScan(input_sequence,input_data,ground_truth_data,current_experiment_settings,rs);
            run_interactive_process(ground_truth_data,input_data,input_sequence,interactive_helper,current_experiment_settings);
        
        }
        
//         for (int ind_ind_exp = 3; ind_ind_exp<5; ++ind_ind_exp)
//         {
//             int ind_exp = ind_exps[ind_ind_exp];
//             
//             InputData input_data;
//             GroundTruthData ground_truth_data;
//             InputSequence input_sequence;
//             InteractiveHelper interactive_helper;
//             
//             
//             experiments_settings[ind_exp].nb_iterations_interaction = 50;
//             experiments_settings[ind_exp].nb_possible_skips = 100;
//             Settings current_experiment_settings = experiments_settings[ind_exp];
//             current_experiment_settings.experiment_identifier = experiments_settings[ind_exp].experiment_identifier + "-RS" + oss_rs.str();
//            
//             createSyntheticCircleDataset(input_sequence,input_data,ground_truth_data,current_experiment_settings,rs);
//             run_interactive_procedure_simpler_overlap_models(ground_truth_data,input_data,input_sequence,interactive_helper,current_experiment_settings);
//         
//         }
        
        
   }
    
    
    
    
//     
 //   std::cout << "Prepare dataset" << std::endl;
 //   createSyntheticCircleDataset(input_sequence,input_data,ground_truth_data,settings);
 //   createImmobileDataset(input_data,ground_truth_data,settings);
 //   loadPrecomputedFetoscopyDataset(input_data,input_sequence,ground_truth_data,interactive_helper,settings);
 //   loadAerialDataset(input_data,input_sequence,ground_truth_data,interactive_helper,settings);


}


