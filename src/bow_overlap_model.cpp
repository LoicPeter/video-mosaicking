
#include "bow_overlap_model.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
//#include "stdafx.h"
#include "optimization.h"
#include <opencv2/core.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>








// PCCA

PCCAOverlapModel::PCCAOverlapModel(int dimensionality_BOW, double beta) // we initialise it as a chain
{
    m_weights = std::vector<double>(dimensionality_BOW*dimensionality_BOW,0);
    for (int d=0; d<dimensionality_BOW; ++d)
        m_weights[d*dimensionality_BOW+d] = 1;
    m_beta = beta;
    m_dimensionality_BOW = dimensionality_BOW;
    m_dimensionality_reduced = dimensionality_BOW;
}


PCCAOverlapModel::PCCAOverlapModel(int dimensionality_BOW, int dimensionality_reduced, double beta) // we initialise it as a chain
{
    m_weights = std::vector<double>(dimensionality_BOW*dimensionality_reduced,1/((double)dimensionality_BOW));
    m_beta = beta;
    m_dimensionality_BOW = dimensionality_BOW;
    m_dimensionality_reduced = dimensionality_reduced;
}

double PCCAOverlapModel::_getOverlapProbability(int i, int j, std::vector<double>& bow_appearance_matrix)
{
 //   return getPCCASimilarity(i,j,1,m_dimensionality_BOW,m_dimensionality_reduced,m_beta,bow_appearance_matrix,&(m_weights[0]));

    double squared_norm;
    double* bow_appearance_vector_i_data = &(bow_appearance_matrix[i*m_dimensionality_BOW]);
    double* bow_appearance_vector_j_data = &(bow_appearance_matrix[j*m_dimensionality_BOW]);
    double* weights_data = &(m_weights[0]);
    cv::Mat bow_i_mat = cv::Mat(m_dimensionality_BOW,1,CV_64F,bow_appearance_vector_i_data);
    cv::Mat bow_j_mat = cv::Mat(m_dimensionality_BOW,1,CV_64F,bow_appearance_vector_j_data);
    cv::Mat weights_mat = cv::Mat(m_dimensionality_reduced,m_dimensionality_BOW,CV_64F,weights_data);
    double p = getPCCASimilarity(squared_norm,1,m_beta,bow_i_mat,bow_j_mat,weights_mat);
//    if ((i==1) && (j==50))
//    {
//        std::cout << bow_i_mat << std::endl;
//        std::cout << bow_j_mat << std::endl;
//
//        std::cout << cv::norm(bow_i_mat, cv::NORM_L2SQR) << std::endl;
//        std::cout << cv::norm(bow_j_mat, cv::NORM_L2SQR) << std::endl;
//        std::cout << bow_i_mat.dot(bow_j_mat) << std::endl;
//        std::cout << squared_norm << std::endl;
//        std::cout << p << std::endl;
//        //std::cout << weights_mat << std::endl;
//    }
    return p;
}

void PCCAOverlapModel::_getOverlapProbabilities(cv::Mat& overlap_probabilities, std::vector<double>& bow_appearance_matrix)
{
    CV_Assert(overlap_probabilities.depth() == CV_64F);
    int nb_frames = overlap_probabilities.rows;
    for (int i=0; i<nb_frames; ++i)
    {
        double* row_i = overlap_probabilities.ptr<double>(i);
        for (int j=(i+1); j<nb_frames; ++j)
        {
            row_i[j] = this->_getOverlapProbability(i,j,bow_appearance_matrix);
        }
    }
}

void PCCAOverlapModel::displayWeights() const
{
    int nb_weights = (int)m_weights.size();
    for (int k=0; k<nb_weights; ++k)
        std::cout << m_weights[k] << " ";
    std::cout << std::endl;
}



void PCCAOverlapModel::inferWeights(std::vector<LabeledImagePair>& training_pairs, std::vector<double>& bow_appearance_matrix)
{
    int nb_training_pairs = (int)training_pairs.size();
    std::vector<std::pair<int,int>> training_indices(nb_training_pairs);
    std::vector<double> training_labels(nb_training_pairs);
    for (int k=0; k<nb_training_pairs; ++k)
    {
        training_indices[k] = training_pairs[k].image_indices;
        if (training_pairs[k].this_pair_overlaps)
            training_labels[k] = 1;
        else
            training_labels[k] = 0;
    }
    
    this->inferWeights(training_indices,training_labels,bow_appearance_matrix);
    
}


void PCCAOverlapModel::inferWeights(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, std::vector<double>& bow_appearance_matrix)
{

    // Prepare optimisation data for alglib
    StructureForInferencePCCAWeights struct_inf_weights;

    struct_inf_weights.dimensionality_BOW = m_dimensionality_BOW;
    struct_inf_weights.dimensionality_reduced = m_dimensionality_reduced;
    struct_inf_weights.beta = m_beta;
    struct_inf_weights.training_pairs = &training_pairs;
    struct_inf_weights.training_labels = &training_labels;

    int nb_frames = ((int)bow_appearance_matrix.size())/m_dimensionality_BOW;
    std::vector<cv::Mat> bow_appearance(nb_frames);
    for (int ind_frame=0; ind_frame<nb_frames; ++ind_frame)
        bow_appearance[ind_frame] = cv::Mat(m_dimensionality_BOW,1,CV_64F,&(bow_appearance_matrix[ind_frame*m_dimensionality_BOW]));
    struct_inf_weights.bow_appearance_matrix = &bow_appearance_matrix;
    struct_inf_weights.bow_appearance = &bow_appearance;

//    std::vector<Eigen::SparseVector<double>> bow_appearance(nb_frames);
//    for (int ind_frame=0; ind_frame<nb_frames; ++ind_frame)
//    {
//        bow_appearance[ind_frame] = Eigen::SparseVector<double>(m_dimensionality_BOW);
//        for (int d=0; d<m_dimensionality_BOW; ++d)
//        {
//            double descriptor_value = bow_appearance_matrix[ind_frame*m_dimensionality_BOW+d];
//            if (descriptor_value!=0)
//                bow_appearance[ind_frame].insert(d) = descriptor_value;
//        }
//    }
//    struct_inf_weights.bow_appearance = &bow_appearance;



    // Vector of unknowns (shallow copy)
    alglib::real_1d_array x;
    //x.attach_to_ptr(m_nb_unknowns*m_nb_coordinates,&(m_unknown_coordinatewise_variances[0]));
    x.setcontent(m_dimensionality_BOW*m_dimensionality_reduced,&(m_weights[0]));



    // Count the number of labels from each class
    int nb_positives(0), nb_negatives(0);
    int nb_training_samples = (int)training_labels.size();
    for (int k=0; k<nb_training_samples; ++k)
    {
        if (training_labels[k]==1)
            ++nb_positives;
        else
            ++nb_negatives;
    }
    std::vector<double> balancing_weights(2,1);
    balancing_weights[0] = nb_positives/((double)nb_negatives);
    struct_inf_weights.balancing_weights = balancing_weights;


    // Precompute Cn
    int dimensionality_BOW_squared = m_dimensionality_BOW*m_dimensionality_BOW;
    //std::vector<double> precomputed_Cn(nb_training_samples*dimensionality_BOW_squared);
    std::vector<cv::Mat> precomputed_Cn(nb_training_samples);
    for (int k=0; k<nb_training_samples; ++k)
    {
        //std::cout << k << " ";
        int i = training_pairs[k].first;
        int j = training_pairs[k].second;
        precomputed_Cn[k] = cv::Mat(m_dimensionality_BOW,m_dimensionality_BOW,CV_64F);
        for (int d1=0; d1<m_dimensionality_BOW; ++d1)
        {
            for (int d2=d1; d2<m_dimensionality_BOW; ++d2)
            {
                precomputed_Cn[k].at<double>(d1,d2) = (bow_appearance_matrix[i*m_dimensionality_BOW+d1] - bow_appearance_matrix[j*m_dimensionality_BOW+d1])*(bow_appearance_matrix[i*m_dimensionality_BOW+d2] - bow_appearance_matrix[j*m_dimensionality_BOW+d2]);
                precomputed_Cn[k].at<double>(d2,d1) = precomputed_Cn[k].at<double>(d1,d2);
            }
        }
    }
    struct_inf_weights.precomputed_Cn = &precomputed_Cn;

//    std::vector<Eigen::MatrixXd> precomputed_Cn(nb_training_samples);
//    for (int k=0; k<nb_training_samples; ++k)
//    {
//        //std::cout << k << " ";
//        int i = training_pairs[k].first;
//        int j = training_pairs[k].second;
//        precomputed_Cn[k] = Eigen::MatrixXd(m_dimensionality_BOW,m_dimensionality_BOW);
//        for (int d1=0; d1<m_dimensionality_BOW; ++d1)
//        {
//            for (int d2=d1; d2<m_dimensionality_BOW; ++d2)
//            {
//                precomputed_Cn[k](d1,d2) = (bow_appearance_matrix[i*m_dimensionality_BOW+d1] - bow_appearance_matrix[j*m_dimensionality_BOW+d1])*(bow_appearance_matrix[i*m_dimensionality_BOW+d2] - bow_appearance_matrix[j*m_dimensionality_BOW+d2]);
//                precomputed_Cn[k](d2,d1) = precomputed_Cn[k](d1,d2);
//            }
//        }
//    }
//    struct_inf_weights.precomputed_Cn = &precomputed_Cn;




    void *struct_inf_weights_void_ptr;
    struct_inf_weights_void_ptr = &struct_inf_weights;

    double epsg = 0.0001;
    double epsf = 0.01;
    double epsx = 0;
    double diffstep = 1.0e-6;
    alglib::ae_int_t maxits = 0;

  //  printf("%s\n", x.tostring(1).c_str()); // EXPECTED: [1.500,0.500]

   alglib::minlbfgsstate state;
   alglib::minlbfgsreport rep;
   alglib::minlbfgscreate(1, x, state);
   alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
   alglib::minlbfgsoptimize(state,compute_loss_function_inference_pcca_weights_grad,NULL,struct_inf_weights_void_ptr);
   alglib::minlbfgsresults(state, x, rep);

//    alglib::minbleicstate state;
//    alglib::minbleicreport rep;
//    alglib::minbleiccreate(x, state);
//    alglib::minbleicsetbc(state, bnd_l, bnd_u);
//    alglib::minbleicsetcond(state, epsg, epsf, epsx, maxits);
//    alglib::minbleicoptimize(state,compute_loss_function_inference_variances_grad,NULL,struct_inf_var_void_ptr);
//    alglib::minbleicresults(state, x, rep);

//     alglib::mincgstate state;
//     alglib::mincgreport rep;
//     alglib::mincgcreate(x,state);
//     alglib::mincgsetcond(state, epsg, epsf, epsx, maxits);
//     alglib::mincgoptimize(state,compute_loss_function_inference_pcca_weights_grad,NULL,struct_inf_weights_void_ptr);
//     alglib::mincgresults(state, x, rep);

    printf("%s\n", x.tostring(1).c_str()); // EXPECTED: [1.500,0.500]

    for (int d=0; d<(m_dimensionality_BOW*m_dimensionality_reduced); ++d)
        m_weights[d] = x[d];

//    for (int k=0; k<(m_nb_unknowns*m_nb_coordinates); ++k)
//        m_unknown_coordinatewise_variances[k] = 0.5*(m_min_variance + m_max_variance) + 0.5*(m_max_variance - m_min_variance)*tanh(x[k]);

}

//-------------------------------------
// ExternalOverlapModel
//-------------------------------------


void ExternalOverlapModel::_getOverlapProbabilities(cv::Mat& overlap_probabilities, const Eigen::MatrixXd& bow_appearance_matrix)
{
    CV_Assert(overlap_probabilities.depth() == CV_64F);
    int nb_frames = overlap_probabilities.rows;
    for (int i=0; i<nb_frames; ++i)
    {
        double* row_i = overlap_probabilities.ptr<double>(i);
        for (int j=(i+1); j<nb_frames; ++j)
        {
            row_i[j] = this->_getOverlapProbability(i,j,bow_appearance_matrix);
        }
    }
}

void ExternalOverlapModel::train(std::vector<LabeledImagePair>& training_pairs, Eigen::MatrixXd& bow_appearance_matrix)
{
    int nb_training_pairs = (int)training_pairs.size();
    std::vector<std::pair<int,int>> training_indices(nb_training_pairs);
    std::vector<double> training_labels(nb_training_pairs);
    for (int k=0; k<nb_training_pairs; ++k)
    {
        training_indices[k] = training_pairs[k].image_indices;
        if (training_pairs[k].this_pair_overlaps)
            training_labels[k] = 1;
        else
            training_labels[k] = 0;
    }
    
    this->train(training_indices,training_labels,bow_appearance_matrix);
    
}



// -----------------------------------------------------
// DiagonalPCCAOverlapModel
// -----------------------------------------------------

DiagonalPCCAOverlapModel::DiagonalPCCAOverlapModel(int dimensionality_BOW, double beta, double intercept)
{
    this->m_weights = Eigen::VectorXd::Constant(dimensionality_BOW,1);
    this->m_beta = beta;
    m_preallocated_product = Eigen::VectorXd(dimensionality_BOW);
    this->m_intercept = intercept;
}

double DiagonalPCCAOverlapModel::_getOverlapProbability(int i, int j, const Eigen::MatrixXd& bow_appearance_matrix)
{
    return getPCCASimilarity_Diagonal(i,j,1,this->m_beta,this->m_preallocated_product,this->m_weights,bow_appearance_matrix,this->m_intercept);
}
    
void DiagonalPCCAOverlapModel::train(std::vector<LabeledImagePair>& training_pairs, Eigen::MatrixXd& bow_appearance_matrix)
{
    int nb_training_pairs = (int)training_pairs.size();
    std::vector<std::pair<int,int>> training_indices(nb_training_pairs);
    std::vector<double> training_labels(nb_training_pairs), importance_weights(nb_training_pairs);
    for (int k=0; k<nb_training_pairs; ++k)
    {
        training_indices[k] = training_pairs[k].image_indices;
        if (training_pairs[k].this_pair_overlaps)
            training_labels[k] = 1;
        else
            training_labels[k] = 0;
        importance_weights[k] = training_pairs[k].importance_weight;
    }
    
    this->train(training_indices,training_labels,importance_weights,bow_appearance_matrix);
    
}
    
void DiagonalPCCAOverlapModel::train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix)
{
    int nb_training_samples = (int)training_labels.size();
    std::vector<double> importance_weights(nb_training_samples,1);
    this->train(training_pairs,training_labels,importance_weights,bow_appearance_matrix);
}
    
    
void DiagonalPCCAOverlapModel::train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, std::vector<double>& importance_weights, Eigen::MatrixXd& bow_appearance_matrix)
{
    int dimensionality = this->m_weights.rows();
    
    // Prepare optimisation data for alglib
    StructureForTrainingPCCADiagonal struct_inf_weights;
    struct_inf_weights.dimensionality_BOW = dimensionality;
    struct_inf_weights.beta = this->m_beta;
    struct_inf_weights.training_labels = &training_labels;

    // Count the number of labels from each class
   // std::cout << "Precompute" << std::endl;
    double nb_positives(0), nb_negatives(0);
    int nb_training_samples = (int)training_labels.size();
    struct_inf_weights.importance_weights.resize(nb_training_samples);
    Eigen::MatrixXd precomputed_Delta_n(dimensionality,nb_training_samples);
    Eigen::VectorXd diff;
    for (int k=0; k<nb_training_samples; ++k)
    {
        if (training_labels[k]==1)
            nb_positives += importance_weights[k];
        else
            nb_negatives += importance_weights[k];
        int i = training_pairs[k].first;
        int j = training_pairs[k].second;
        diff = bow_appearance_matrix.block(0,i,dimensionality,1) - bow_appearance_matrix.block(0,j,dimensionality,1);
        precomputed_Delta_n.block(0,k,dimensionality,1) = diff.array().square();
    }
    struct_inf_weights.precomputed_Delta_n = &precomputed_Delta_n;
    for (int k=0; k<nb_training_samples; ++k)
    {
        if (training_labels[k]==1)
             struct_inf_weights.importance_weights[k] = importance_weights[k]/((double)2*nb_positives); 
        else
             struct_inf_weights.importance_weights[k] = importance_weights[k]/((double)2*nb_negatives); 
    }
    
    void *struct_inf_weights_void_ptr;
    struct_inf_weights_void_ptr = &struct_inf_weights;

    
    // Vector of unknowns
    //std::cout << "Prepare optimisation" << std::endl;
    alglib::real_1d_array x;
    Eigen::VectorXd log_weights = Eigen::VectorXd::Zero(dimensionality+1);
   // log_weights.segment(0,dimensionality) = this->m_weights.array().log();
  //  log_weights.segment(0,dimensionality) = this->m_weights;
 //   log_weights(dimensionality) = this->m_intercept;
    x.setcontent(dimensionality+1,&(log_weights[0]));
    double epsg = 0.00001;
    double epsf = 0.00001;
    double epsx = 0;
    double diffstep = 1.0e-6;
    alglib::ae_int_t maxits = 0;
    alglib::minlbfgsstate state;
    alglib::minlbfgsreport rep;
    alglib::minlbfgscreate(1, x, state);
    alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
    alglib::minlbfgsoptimize(state, compute_loss_function_pcca_diagonal,NULL,struct_inf_weights_void_ptr);
    alglib::minlbfgsresults(state, x, rep);
    
    //     alglib::mincgstate state;
//     alglib::mincgreport rep;
//     alglib::mincgcreate(x,state);
//     alglib::mincgsetcond(state, epsg, epsf, epsx, maxits);
//     alglib::mincgoptimize(state,compute_loss_function_inference_pcca_weights_grad,NULL,struct_inf_weights_void_ptr);
//     alglib::mincgresults(state, x, rep);

    for (int d=0; d<dimensionality; ++d)
        this->m_weights[d] = x[d];
        //this->m_weights[d] = std::exp(x[d]);
    this->m_intercept = x[dimensionality];
  //  std::cout << "New weights: " << this->m_weights.transpose() << std::endl;
}



// void compute_loss_function_pcca_diagonal(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr) // x = log-weights
// {
// 
//     bool do_we_want_to_check_gradients = true; // set to true to debug and check the gradients numerically, false for best performance
//     
//     bool check_gradients;
//     if (do_we_want_to_check_gradients)
//         check_gradients = grad[0]!=(-1);
//     else
//         check_gradients = false;
//     
//     // Catch back the data passed as void pointer
//     StructureForTrainingPCCADiagonal* struct_inf_weights = static_cast<StructureForTrainingPCCADiagonal*>(ptr);
// 
//     // Nb training pairs
//   //  std::cout << "Compute loss" << std::endl;
//     int nb_training_pairs = (int)struct_inf_weights->training_labels->size();
//     int dimensionality_BOW = struct_inf_weights->dimensionality_BOW;
//     double beta = struct_inf_weights->beta;
//     Eigen::MatrixXd *precomputed_Delta_n = struct_inf_weights->precomputed_Delta_n;
//     // We precompute Delta(n,d) = squared d-th dimension of the vector x_{i_n} - x_{j_n}
//     
//     // Regularisation
//     double lambda = 0;
//     
//     // Initial loss function
//     func = 0;
//     Eigen::VectorXd squared_weights(dimensionality_BOW+1), grad_eigen(dimensionality_BOW+1);
//     for (int d=0; d<dimensionality_BOW; ++d)
//     {    
//         grad_eigen(d) = 0;
//         squared_weights(d) = std::exp(2*x[d]);
//     }
//     grad_eigen(dimensionality_BOW) = 0;
//     double intercept = x[dimensionality_BOW];
// 
//     for (int k=0; k<nb_training_pairs; ++k)
//     {
//         int label_01 = (*(struct_inf_weights->training_labels))[k];
//         double label = 2*label_01 - 1;
//         double balancing_weight = struct_inf_weights->importance_weights[k];
//         sum_balancing_weights += balancing_weight;
//         double lin_comb = ((squared_weights.transpose())*precomputed_Delta_n->block(0,k,dimensionality_BOW,1)).trace(); // dummy trace, this is a number
//         double score_in_sigmoid = ((double)label)*(intercept - lin_comb);
//         
//         // Loss
//         func -= (balancing_weight/beta)*std::log(sigmoid(score_in_sigmoid,beta));
// 
//         grad_eigen.segment(0,dimensionality_BOW) += 2*label*balancing_weight*sigmoid(-score_in_sigmoid,beta)*(precomputed_Delta_n->block(0,k,dimensionality_BOW,1).cwiseProduct(squared_weights));
//         grad_eigen(dimensionality_BOW) -= label*balancing_weight*sigmoid(-score_in_sigmoid,beta);
//     }
//     
//     for (int d=0; d<dimensionality_BOW; ++d)
//     {    
//         func += lambda*(squared_weights(d) - 2*std::exp(x[d]) + 1);
//         grad_eigen(d) += 2*lambda*(squared_weights(d) - std::exp(x[d]));
//     }
//     
//     
//     for (int d=0; d<(dimensionality_BOW+1); ++d)
//         grad[d] = grad_eigen[d];
// 
//     std::cout << "Loss = " << func << std::endl;
//   //  printf("%s\n",  grad.tostring(1).c_str());
// 
//     // Check gradients empirically
//     if (check_gradients)
//     {
//         double eps = 0.00001;
//         for (int p=dimensionality_BOW; p>=0; --p)
//         {
//             alglib::real_1d_array x_eps, dummy_grad;
//             double func_eps, flag_grad(-1);
//             x_eps.setcontent(dimensionality_BOW+1,&(x[0]));
//             dummy_grad.setcontent(dimensionality_BOW+1,&flag_grad);
//             x_eps[p] = x[p] + eps;
//             compute_loss_function_pcca_diagonal(x_eps,func_eps,dummy_grad,ptr);
//             double empirical_grad_p = (func_eps - func)/eps;
//             std::cout << "Parameter " << p << ": Empirical grad: " << empirical_grad_p << " ; Computed grad: " << grad[p] << std::endl;
//         }    
//     }
// 
//     
// }


void compute_loss_function_pcca_diagonal(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr) // x = log-weights
{

    bool do_we_want_to_check_gradients = false; // set to true to debug and check the gradients numerically, false for best performance
    
    bool check_gradients;
    if (do_we_want_to_check_gradients)
        check_gradients = grad[0]!=(-1);
    else
        check_gradients = false;
    
    // Catch back the data passed as void pointer
    StructureForTrainingPCCADiagonal* struct_inf_weights = static_cast<StructureForTrainingPCCADiagonal*>(ptr);

    // Nb training pairs
  //  std::cout << "Compute loss" << std::endl;
    int nb_training_pairs = (int)struct_inf_weights->training_labels->size();
    int dimensionality_BOW = struct_inf_weights->dimensionality_BOW;
    double beta = struct_inf_weights->beta;
    Eigen::MatrixXd *precomputed_Delta_n = struct_inf_weights->precomputed_Delta_n;
    // We precompute Delta(n,d) = squared d-th dimension of the vector x_{i_n} - x_{j_n}
    
    // Regularisation
    double lambda = 10;
    
    // Initial loss function
    func = 0;
    Eigen::VectorXd weights(dimensionality_BOW+1), grad_eigen(dimensionality_BOW+1);
    for (int d=0; d<dimensionality_BOW; ++d)
    {    
        grad_eigen(d) = 0;
        weights(d) = x[d];
    }
    grad_eigen(dimensionality_BOW) = 0;
    double intercept = x[dimensionality_BOW];

    for (int k=0; k<nb_training_pairs; ++k)
    {
        int label_01 = (*(struct_inf_weights->training_labels))[k];
        double label = 2*label_01 - 1;
        double balancing_weight = struct_inf_weights->importance_weights[k];
        double lin_comb = ((weights.transpose())*precomputed_Delta_n->block(0,k,dimensionality_BOW,1)).trace(); // dummy trace, this is a number
        double score_in_sigmoid = ((double)label)*(intercept - lin_comb);
        
        // Loss
        func -= (balancing_weight/beta)*std::log(sigmoid(score_in_sigmoid,beta));

        grad_eigen.segment(0,dimensionality_BOW) += label*balancing_weight*sigmoid(-score_in_sigmoid,beta)*(precomputed_Delta_n->block(0,k,dimensionality_BOW,1));
        grad_eigen(dimensionality_BOW) -= label*balancing_weight*sigmoid(-score_in_sigmoid,beta);
    }
    
    for (int d=0; d<dimensionality_BOW; ++d)
    {    
        func += lambda*std::pow<double>(weights(d) - 1,2);
        grad_eigen(d) += 2*lambda*(weights(d) - 1);
    }
    
    func += lambda*std::pow<double>(intercept - 1,2);
    grad_eigen(dimensionality_BOW) += 2*lambda*(intercept - 1);
    
    
    for (int d=0; d<(dimensionality_BOW+1); ++d)
       grad[d] = grad_eigen[d];

 //   std::cout << "Loss = " << func << std::endl;
  //  printf("%s\n",  grad.tostring(1).c_str());

    // Check gradients empirically
    if (check_gradients)
    {
        double eps = 0.00001;
        for (int p=dimensionality_BOW; p>=0; --p)
        {
            alglib::real_1d_array x_eps, dummy_grad;
            double func_eps, flag_grad(-1);
            x_eps.setcontent(dimensionality_BOW+1,&(x[0]));
            dummy_grad.setcontent(dimensionality_BOW+1,&flag_grad);
            x_eps[p] = x[p] + eps;
            compute_loss_function_pcca_diagonal(x_eps,func_eps,dummy_grad,ptr);
            double empirical_grad_p = (func_eps - func)/eps;
            std::cout << "Parameter " << p << ": Empirical grad: " << empirical_grad_p << " ; Computed grad: " << grad[p] << std::endl;
        }    
    }

    
}

// The one in the paper so far (12.05.2020)
// -----------------------------------------------------
// NonIncreasingDiagonalPCCAOverlapModel
// -----------------------------------------------------
/*
NonIncreasingDiagonalPCCAOverlapModel::NonIncreasingDiagonalPCCAOverlapModel(int dimensionality_BOW, double beta, double intercept)
{
    this->m_weights = Eigen::VectorXd::Constant(dimensionality_BOW,1);
    this->m_beta = beta;
    m_preallocated_product = Eigen::VectorXd(dimensionality_BOW);
}

double NonIncreasingDiagonalPCCAOverlapModel::_getOverlapProbability(int i, int j, const Eigen::MatrixXd& bow_appearance_matrix)
{
    return getPCCASimilarity_Diagonal(i,j,1,this->m_beta,this->m_preallocated_product,this->m_weights,bow_appearance_matrix,1);
}
    
void NonIncreasingDiagonalPCCAOverlapModel::train(std::vector<LabeledImagePair>& training_pairs, Eigen::MatrixXd& bow_appearance_matrix)
{
    int nb_training_pairs = (int)training_pairs.size();
    std::vector<std::pair<int,int>> training_indices(nb_training_pairs);
    std::vector<double> training_labels(nb_training_pairs), importance_weights(nb_training_pairs);
    for (int k=0; k<nb_training_pairs; ++k)
    {
        training_indices[k] = training_pairs[k].image_indices;
        if (training_pairs[k].this_pair_overlaps)
            training_labels[k] = 1;
        else
            training_labels[k] = 0;
        importance_weights[k] = training_pairs[k].importance_weight;
    }
    
    this->train(training_indices,training_labels,importance_weights,bow_appearance_matrix);
    
}
    
void NonIncreasingDiagonalPCCAOverlapModel::train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix)
{
    int nb_training_samples = (int)training_labels.size();
    std::vector<double> importance_weights(nb_training_samples,1);
    this->train(training_pairs,training_labels,importance_weights,bow_appearance_matrix);
}
    
    
void NonIncreasingDiagonalPCCAOverlapModel::train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, std::vector<double>& importance_weights, Eigen::MatrixXd& bow_appearance_matrix)
{
    int dimensionality = this->m_weights.rows();
    
    // Prepare optimisation data for alglib
    StructureForTrainingPCCADiagonal struct_inf_weights;
    struct_inf_weights.dimensionality_BOW = dimensionality;
    struct_inf_weights.beta = this->m_beta;
    struct_inf_weights.training_labels = &training_labels;
    struct_inf_weights.current_weights = m_weights;
    // Count the number of labels from each class
   // std::cout << "Precompute" << std::endl;
    double nb_positives(0), nb_negatives(0);
    int nb_training_samples = (int)training_labels.size();
    struct_inf_weights.importance_weights.resize(nb_training_samples);
    Eigen::MatrixXd precomputed_Delta_n(dimensionality,nb_training_samples);
    Eigen::VectorXd diff;
    for (int k=0; k<nb_training_samples; ++k)
    {
        if (training_labels[k]==1)
            nb_positives += importance_weights[k];
        else
            nb_negatives += importance_weights[k];
        int i = training_pairs[k].first;
        int j = training_pairs[k].second;
        diff = bow_appearance_matrix.block(0,i,dimensionality,1) - bow_appearance_matrix.block(0,j,dimensionality,1);
        precomputed_Delta_n.block(0,k,dimensionality,1) = diff.array().square();
    }
    struct_inf_weights.precomputed_Delta_n = &precomputed_Delta_n;
    for (int k=0; k<nb_training_samples; ++k)
    {
        if (training_labels[k]==1)
             struct_inf_weights.importance_weights[k] = importance_weights[k]/((double)2*nb_positives); 
        else
             struct_inf_weights.importance_weights[k] = importance_weights[k]/((double)2*nb_negatives); 
    }
    
    void *struct_inf_weights_void_ptr;
    struct_inf_weights_void_ptr = &struct_inf_weights;

    
    // Vector of unknowns
    //std::cout << "Prepare optimisation" << std::endl;
    alglib::real_1d_array x;
    Eigen::VectorXd log_delta = Eigen::VectorXd::Zero(dimensionality,1);
   // log_weights.segment(0,dimensionality) = this->m_weights.array().log();
  //  log_weights.segment(0,dimensionality) = this->m_weights;
 //   log_weights(dimensionality) = this->m_intercept;
    x.setcontent(dimensionality,&(log_delta[0]));
    double epsg = 0.00001;
    double epsf = 0.00001;
    double epsx = 0;
    double diffstep = 1.0e-6;
    alglib::ae_int_t maxits = 0;
    alglib::minlbfgsstate state;
    alglib::minlbfgsreport rep;
    alglib::minlbfgscreate(1, x, state);
    alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
    alglib::minlbfgsoptimize(state, compute_loss_function_pcca_diagonal_non_increasing,NULL,struct_inf_weights_void_ptr);
    alglib::minlbfgsresults(state, x, rep);

    for (int d=0; d<dimensionality; ++d)
        this->m_weights[d] += std::exp(x[d]);
    //std::cout << "New weights: " << this->m_weights.transpose() << std::endl;
}


void compute_loss_function_pcca_diagonal_non_increasing(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr) // x = log-weights
{

    bool do_we_want_to_check_gradients = false; // set to true to debug and check the gradients numerically, false for best performance
    
    bool check_gradients;
    if (do_we_want_to_check_gradients)
        check_gradients = grad[0]!=(-1);
    else
        check_gradients = false;
    
    // Catch back the data passed as void pointer
    StructureForTrainingPCCADiagonal* struct_inf_weights = static_cast<StructureForTrainingPCCADiagonal*>(ptr);

    // Nb training pairs
  //  std::cout << "Compute loss" << std::endl;
    int nb_training_pairs = (int)struct_inf_weights->training_labels->size();
    int dimensionality_BOW = struct_inf_weights->dimensionality_BOW;
    double beta = struct_inf_weights->beta;
    Eigen::MatrixXd *precomputed_Delta_n = struct_inf_weights->precomputed_Delta_n;
    Eigen::VectorXd current_weights = struct_inf_weights->current_weights;
    // We precompute Delta(n,d) = squared d-th dimension of the vector x_{i_n} - x_{j_n}
    
    // Regularisation
    double lambda = 0;
    
    // Initial loss function
    func = 0;
    Eigen::VectorXd weights(dimensionality_BOW), grad_eigen(dimensionality_BOW);
    for (int d=0; d<dimensionality_BOW; ++d)
    {    
        grad_eigen(d) = 0;
        weights(d) = current_weights(d) + std::exp(x[d]);
    }
    double intercept = 1;

    for (int k=0; k<nb_training_pairs; ++k)
    {
        int label_01 = (*(struct_inf_weights->training_labels))[k];
        double label = 2*label_01 - 1;
        double balancing_weight = struct_inf_weights->importance_weights[k];
        double lin_comb = ((weights.transpose())*precomputed_Delta_n->block(0,k,dimensionality_BOW,1)).trace(); // dummy trace, this is a number
        double score_in_sigmoid = ((double)label)*(intercept - lin_comb);
        
        // Loss
        func -= (balancing_weight/beta)*std::log(sigmoid(score_in_sigmoid,beta));

        for (int d=0; d<dimensionality_BOW; ++d)
            grad_eigen(d) += label*balancing_weight*sigmoid(-score_in_sigmoid,beta)*std::exp(x[d])*(precomputed_Delta_n->operator()(d,k));
    }
    
    for (int d=0; d<dimensionality_BOW; ++d)
    {    
        func += lambda*std::pow<double>(weights(d),2);
        grad_eigen(d) += 2*lambda*weights(d)*std::exp(x[d]);
    }
    
    
    for (int d=0; d<dimensionality_BOW; ++d)
        grad[d] = grad_eigen[d];

  //  std::cout << "Loss = " << func << std::endl;
  //  printf("%s\n",  grad.tostring(1).c_str());

    // Check gradients empirically
    if (check_gradients)
    {
        double eps = 0.00001;
        for (int p=(dimensionality_BOW-1); p>=0; --p)
        {
            alglib::real_1d_array x_eps, dummy_grad;
            double func_eps, flag_grad(-1);
            x_eps.setcontent(dimensionality_BOW,&(x[0]));
            dummy_grad.setcontent(dimensionality_BOW,&flag_grad);
            x_eps[p] = x[p] + eps;
            compute_loss_function_pcca_diagonal_non_increasing(x_eps,func_eps,dummy_grad,ptr);
            double empirical_grad_p = (func_eps - func)/eps;
            std::cout << "Parameter " << p << ": Empirical grad: " << empirical_grad_p << " ; Computed grad: " << grad[p] << std::endl;
        }    
    }

    
}*/


// -----------------------------------------------------
// NonIncreasingDiagonalPCCAOverlapModel
// -----------------------------------------------------

NonIncreasingDiagonalPCCAOverlapModel::NonIncreasingDiagonalPCCAOverlapModel(int dimensionality_BOW, double beta, double intercept)
{
    this->m_weights = Eigen::VectorXd::Constant(dimensionality_BOW+1,1);
    this->m_weights(dimensionality_BOW) = intercept;
    this->m_beta = beta;
    m_preallocated_product = Eigen::VectorXd(dimensionality_BOW);
}

double NonIncreasingDiagonalPCCAOverlapModel::_getOverlapProbability(int i, int j, const Eigen::MatrixXd& bow_appearance_matrix)
{
    int dimensionality_BOW = m_preallocated_product.rows();
    return getPCCASimilarity_Diagonal(i,j,1,this->m_beta,this->m_preallocated_product,this->m_weights.segment(0,dimensionality_BOW),bow_appearance_matrix,this->m_weights(dimensionality_BOW));
}
    
void NonIncreasingDiagonalPCCAOverlapModel::train(std::vector<LabeledImagePair>& training_pairs, Eigen::MatrixXd& bow_appearance_matrix)
{
    int nb_training_pairs = (int)training_pairs.size();
    std::vector<std::pair<int,int>> training_indices(nb_training_pairs);
    std::vector<double> training_labels(nb_training_pairs), importance_weights(nb_training_pairs);
    for (int k=0; k<nb_training_pairs; ++k)
    {
        training_indices[k] = training_pairs[k].image_indices;
        if (training_pairs[k].this_pair_overlaps)
            training_labels[k] = 1;
        else
            training_labels[k] = 0;
        importance_weights[k] = training_pairs[k].importance_weight;
    }
    
    this->train(training_indices,training_labels,importance_weights,bow_appearance_matrix);
    
}
    
void NonIncreasingDiagonalPCCAOverlapModel::train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix)
{
    int nb_training_samples = (int)training_labels.size();
    std::vector<double> importance_weights(nb_training_samples,1);
    this->train(training_pairs,training_labels,importance_weights,bow_appearance_matrix);
}
    
    
void NonIncreasingDiagonalPCCAOverlapModel::train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, std::vector<double>& importance_weights, Eigen::MatrixXd& bow_appearance_matrix)
{
    int dimensionality = this->m_weights.rows() - 1;
    
    // Prepare optimisation data for alglib
    StructureForTrainingPCCADiagonal struct_inf_weights;
    struct_inf_weights.dimensionality_BOW = dimensionality;
    struct_inf_weights.beta = this->m_beta;
    struct_inf_weights.training_labels = &training_labels;
    struct_inf_weights.current_weights = m_weights;
    // Count the number of labels from each class
   // std::cout << "Precompute" << std::endl;
    double nb_positives(0), nb_negatives(0);
    int nb_training_samples = (int)training_labels.size();
    struct_inf_weights.importance_weights.resize(nb_training_samples);
    Eigen::MatrixXd precomputed_Delta_n(dimensionality,nb_training_samples);
    Eigen::VectorXd diff;
    for (int k=0; k<nb_training_samples; ++k)
    {
        if (training_labels[k]==1)
            nb_positives += importance_weights[k];
        else
            nb_negatives += importance_weights[k];
        int i = training_pairs[k].first;
        int j = training_pairs[k].second;
        diff = bow_appearance_matrix.block(0,i,dimensionality,1) - bow_appearance_matrix.block(0,j,dimensionality,1);
        precomputed_Delta_n.block(0,k,dimensionality,1) = diff.array().square();
    }
    struct_inf_weights.precomputed_Delta_n = &precomputed_Delta_n;
    for (int k=0; k<nb_training_samples; ++k)
    {
        if (training_labels[k]==1)
             struct_inf_weights.importance_weights[k] = importance_weights[k]/((double)2*nb_positives); 
        else
             struct_inf_weights.importance_weights[k] = importance_weights[k]/((double)2*nb_negatives); 
    }
    
    void *struct_inf_weights_void_ptr;
    struct_inf_weights_void_ptr = &struct_inf_weights;

    
    // Vector of unknowns
    //std::cout << "Prepare optimisation" << std::endl;
    alglib::real_1d_array x;
    Eigen::VectorXd log_delta = Eigen::VectorXd::Zero(dimensionality+1);
   // log_weights.segment(0,dimensionality) = this->m_weights.array().log();
  //  log_weights.segment(0,dimensionality) = this->m_weights;
 //   log_weights(dimensionality) = this->m_intercept;
    x.setcontent(dimensionality+1,&(log_delta[0]));
    double epsg = 0.0000001;
    double epsf = 0.0000001;
    double epsx = 0;
    double diffstep = 1.0e-6;
    alglib::ae_int_t maxits = 0;
//     alglib::minlbfgsstate state;
//     alglib::minlbfgsreport rep;
//     alglib::minlbfgscreate(1, x, state);
//     alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
//     alglib::minlbfgsoptimize(state, compute_loss_function_pcca_diagonal_non_increasing,NULL,struct_inf_weights_void_ptr);
//     alglib::minlbfgsresults(state, x, rep);
//     
    alglib::mincgstate state;
    alglib::mincgreport rep;
    alglib::mincgcreate(x,state);
    alglib::mincgsetcond(state, epsg, epsf, epsx, maxits);
    alglib::mincgoptimize(state,compute_loss_function_pcca_diagonal_non_increasing,NULL,struct_inf_weights_void_ptr);
    alglib::mincgresults(state, x, rep);

    
    
    for (int d=0; d<dimensionality; ++d)
        this->m_weights(d)= 1 + std::exp(x[d]);
    this->m_weights(dimensionality)= 1 - std::exp(x[dimensionality]);
  
}


void compute_loss_function_pcca_diagonal_non_increasing(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr) // x = log-weights
{

    bool do_we_want_to_check_gradients = false; // set to true to debug and check the gradients numerically, false for best performance
    
    bool check_gradients;
    if (do_we_want_to_check_gradients)
        check_gradients = grad[0]!=(-1);
    else
        check_gradients = false;
    
    // Catch back the data passed as void pointer
    StructureForTrainingPCCADiagonal* struct_inf_weights = static_cast<StructureForTrainingPCCADiagonal*>(ptr);

    // Nb training pairs
  //  std::cout << "Compute loss" << std::endl;
    int nb_training_pairs = (int)struct_inf_weights->training_labels->size();
    int dimensionality_BOW = struct_inf_weights->dimensionality_BOW;
    double beta = struct_inf_weights->beta;
    Eigen::MatrixXd *precomputed_Delta_n = struct_inf_weights->precomputed_Delta_n;
    Eigen::VectorXd current_weights = struct_inf_weights->current_weights;
    // We precompute Delta(n,d) = squared d-th dimension of the vector x_{i_n} - x_{j_n}
    
    // Regularisation
    double lambda = 0;
    
    // Initial loss function
    func = 0;
    Eigen::VectorXd weights(dimensionality_BOW), grad_eigen(dimensionality_BOW+1);
    for (int d=0; d<dimensionality_BOW; ++d)
    {    
        grad_eigen(d) = 0;
        weights(d) = 1 + std::exp(x[d]);
    }
    grad_eigen(dimensionality_BOW) = 0;
    double intercept = 1 - std::exp(x[dimensionality_BOW]);
    //double intercept = 1;

    for (int k=0; k<nb_training_pairs; ++k)
    {
        int label_01 = (*(struct_inf_weights->training_labels))[k];
        double label = 2*label_01 - 1;
        double balancing_weight = struct_inf_weights->importance_weights[k];
        double lin_comb = ((weights.transpose())*precomputed_Delta_n->block(0,k,dimensionality_BOW,1)).trace(); // dummy trace, this is a number
        double score_in_sigmoid = ((double)label)*(intercept - lin_comb);
        
        // Loss
        func -= (balancing_weight/beta)*std::log(sigmoid(score_in_sigmoid,beta));

        for (int d=0; d<dimensionality_BOW; ++d)
            grad_eigen(d) += label*balancing_weight*sigmoid(-score_in_sigmoid,beta)*std::exp(x[d])*(precomputed_Delta_n->operator()(d,k));
        grad_eigen(dimensionality_BOW) += label*balancing_weight*sigmoid(-score_in_sigmoid,beta)*std::exp(x[dimensionality_BOW]);
    }
    
    for (int d=0; d<dimensionality_BOW; ++d)
    {    
        func += lambda*std::pow<double>(weights(d),2);
        grad_eigen(d) += 2*lambda*weights(d)*std::exp(x[d]);
    }
    func += lambda*std::pow<double>(intercept,2);
    grad_eigen(dimensionality_BOW) += 2*lambda*intercept*std::exp(x[dimensionality_BOW]);
    
    for (int d=0; d<(dimensionality_BOW+1); ++d)
        grad[d] = grad_eigen[d];

  //  std::cout << "Loss = " << func << std::endl;
  //  printf("%s\n",  grad.tostring(1).c_str());

    // Check gradients empirically
    if (check_gradients)
    {
        double eps = 0.00001;
        for (int p=(dimensionality_BOW-1); p>=0; --p)
        {
            alglib::real_1d_array x_eps, dummy_grad;
            double func_eps, flag_grad(-1);
            x_eps.setcontent(dimensionality_BOW+1,&(x[0]));
            dummy_grad.setcontent(dimensionality_BOW+1,&flag_grad);
            x_eps[p] = x[p] + eps;
            compute_loss_function_pcca_diagonal_non_increasing(x_eps,func_eps,dummy_grad,ptr);
            double empirical_grad_p = (func_eps - func)/eps;
            std::cout << "Parameter " << p << ": Empirical grad: " << empirical_grad_p << " ; Computed grad: " << grad[p] << std::endl;
        }    
    }

    
}



// -----------------------------------------------------
// LinearPCCAOverlapModel
// -----------------------------------------------------

LinearPCCAOverlapModel::LinearPCCAOverlapModel(int dimensionality_BOW, double beta)
{
    this->m_weights = Eigen::VectorXd::Constant(dimensionality_BOW+1,1);
    this->m_beta = beta;
    m_preallocated_product = Eigen::VectorXd(dimensionality_BOW+1);
}

double LinearPCCAOverlapModel::_getOverlapProbability(int i, int j, const Eigen::MatrixXd& bow_appearance_matrix)
{
    int dimensionality = this->m_weights.rows();
    return getPCCASimilarity_Linear(i,j,1,this->m_beta,this->m_preallocated_product,this->m_weights,bow_appearance_matrix,this->m_weights(dimensionality-1));
}
    

void LinearPCCAOverlapModel::train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix)
{
    int dimensionality = this->m_weights.rows();
    
    // Prepare optimisation data for alglib
    StructureForTrainingPCCADiagonal struct_inf_weights;
    struct_inf_weights.dimensionality_BOW = dimensionality;
    struct_inf_weights.beta = this->m_beta;
    struct_inf_weights.training_labels = &training_labels;

    // Count the number of labels from each class
   // std::cout << "Precompute" << std::endl;
    int nb_positives(0), nb_negatives(0);
    int nb_training_samples = (int)training_labels.size();
    Eigen::MatrixXd precomputed_Delta_n(dimensionality,nb_training_samples);
    Eigen::VectorXd diff;
    for (int k=0; k<nb_training_samples; ++k)
    {
        if (training_labels[k]==1)
            ++nb_positives;
        else
            ++nb_negatives;
        int i = training_pairs[k].first;
        int j = training_pairs[k].second;
        diff = bow_appearance_matrix.block(0,i,dimensionality-1,1) - bow_appearance_matrix.block(0,j,dimensionality-1,1);
        precomputed_Delta_n.block(0,k,dimensionality-1,1) = diff.array().square();
        precomputed_Delta_n(dimensionality-1,k) = -1;
    }
    struct_inf_weights.precomputed_Delta_n = &precomputed_Delta_n;
    std::vector<double> balancing_weights(2,1);
    balancing_weights[0] = 1/((double)2*nb_negatives);
    balancing_weights[1] = 1/((double)2*nb_positives); // this weighting includes a normalization with respect to the number of samples: the weights sum to 1
    struct_inf_weights.balancing_weights = balancing_weights;
   // std::cout << "Balancing weight: " << balancing_weights[0] << " " << balancing_weights[1] << std::endl;
    
    void *struct_inf_weights_void_ptr;
    struct_inf_weights_void_ptr = &struct_inf_weights;

    
    // Vector of unknowns
    //std::cout << "Prepare optimisation" << std::endl;
    alglib::real_1d_array x;
    x.setcontent(dimensionality,&(this->m_weights(0)));
    double epsg = 0.0001;
    double epsf = 0.01;
    double epsx = 0;
    double diffstep = 1.0e-6;
    alglib::ae_int_t maxits = 0;
    alglib::minlbfgsstate state;
    alglib::minlbfgsreport rep;
    alglib::minlbfgscreate(1, x, state);
    alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
    alglib::minlbfgsoptimize(state, compute_loss_function_pcca_linear,NULL,struct_inf_weights_void_ptr);
    alglib::minlbfgsresults(state, x, rep);

    for (int d=0; d<dimensionality; ++d)
        this->m_weights[d] = x[d];
    
   // std::cout << "New weights: " << this->m_weights.transpose() << std::endl;
}



void compute_loss_function_pcca_linear(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr) // x = log-weights
{
    bool do_we_want_to_check_gradients = false; // set to true to debug and check the gradients numerically, false for best performance
    
    bool check_gradients;
    if (do_we_want_to_check_gradients)
        check_gradients = grad[0]!=(-1);
    else
        check_gradients = false;
    
    
    // Catch back the data passed as void pointer
    StructureForTrainingPCCADiagonal* struct_inf_weights = static_cast<StructureForTrainingPCCADiagonal*>(ptr);

    // Nb training pairs
  //  std::cout << "Compute loss" << std::endl;
    int nb_training_pairs = (int)struct_inf_weights->training_labels->size();
    int dimensionality_BOW = struct_inf_weights->dimensionality_BOW;
    double beta = struct_inf_weights->beta;
    Eigen::MatrixXd *precomputed_Delta_n = struct_inf_weights->precomputed_Delta_n;
    // We precompute Delta(n,d) = squared d-th dimension of the vector x_{i_n} - x_{j_n}
    
    // Regularisation
    double lambda = 0.001;
    
    // Initial loss function
    func = 0;
    Eigen::VectorXd weights_eigen(dimensionality_BOW), grad_eigen(dimensionality_BOW);
    for (int d=0; d<dimensionality_BOW; ++d)  
    {
        grad_eigen(d) = 0;
        weights_eigen(d) = x[d];
    }
    
    for (int k=0; k<nb_training_pairs; ++k)
    {
        int label_01 = (*(struct_inf_weights->training_labels))[k];
        double label = 2*label_01 - 1;
        double balancing_weight = struct_inf_weights->balancing_weights[label_01];
        double lin_comb = ((weights_eigen.transpose())*precomputed_Delta_n->block(0,k,dimensionality_BOW,1)).trace(); // dummy trace, this is a number
        double score_in_sigmoid = ((double)label)*(-lin_comb);
        
        // Loss
        func -= (balancing_weight/beta)*std::log(sigmoid(score_in_sigmoid,beta));

        grad_eigen += label*balancing_weight*sigmoid(-score_in_sigmoid,beta)*(precomputed_Delta_n->block(0,k,dimensionality_BOW,1));

    }
    
//     for (int d=0; d<(dimensionality_BOW-1); ++d) // we do not regularize the intercept
//     {    
//         func += lambda*(weights_eigen(d) - 1)*(weights_eigen(d) - 1);
//         grad_eigen(d) += 2*lambda*(weights_eigen(d) - 1);
//     }
    for (int d=0; d<(dimensionality_BOW-1); ++d) // we do not regularize the intercept
    {    
        func += lambda*weights_eigen(d)*weights_eigen(d);
        grad_eigen(d) += 2*lambda*weights_eigen(d);
    }
    
    for (int d=0; d<dimensionality_BOW; ++d)
        grad[d] = grad_eigen[d];

   // std::cout << "Loss = " << func << std::endl;
    //printf("%s\n",  grad.tostring(1).c_str());
    
    // Check gradients empirically
    if (check_gradients)
    {
        double eps = 0.00001;
        for (int p=0; p<(dimensionality_BOW); ++p)
        {
            alglib::real_1d_array x_eps, dummy_grad;
            double func_eps, flag_grad(-1);
            x_eps.setcontent(dimensionality_BOW,&(x[0]));
            dummy_grad.setcontent(dimensionality_BOW,&flag_grad);
            x_eps[p] = x[p] + eps;
            compute_loss_function_pcca_linear(x_eps,func_eps,dummy_grad,ptr);
            double empirical_grad_p = (func_eps - func)/eps;
            std::cout << "Parameter " << p << ": Empirical grad: " << empirical_grad_p << " ; Computed grad: " << grad[p] << std::endl;
        }    
    }

}


// -----------------------------------------------------
// FullPCCAOverlapModel
// -----------------------------------------------------

// FullPCCAOverlapModel::FullPCCAOverlapModel()
// {
//     
// }
// 
// FullPCCAOverlapModel::FullPCCAOverlapModel(int dimensionality_BOW, double beta)
// {
//     this->m_weights = Eigen::MatrixXd(dimensionality_BOW,dimensionality_BOW);
//     this->m_weights.setIdentity();
//     this->m_beta = beta;
//     this->m_preallocated_product = Eigen::VectorXd(dimensionality_BOW);
// }
// 
// double FullPCCAOverlapModel::_getOverlapProbability(int i, int j, const Eigen::MatrixXd& bow_appearance_matrix)
// {
//     return getPCCASimilarity(i,j,1,this->m_beta,this->m_preallocated_product,this->m_weights,bow_appearance_matrix);
// }
//     
// 
// void FullPCCAOverlapModel::train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix)
// {
//     int dimensionality = this->m_weights.rows();
//     
//     // Prepare optimisation data for alglib
//     StructureForTrainingPCCAFull struct_inf_weights;
//     struct_inf_weights.dimensionality_BOW = dimensionality;
//     struct_inf_weights.beta = this->m_beta;
//     struct_inf_weights.training_labels = &training_labels;
// 
//     // Count the number of labels from each class
//    // std::cout << "Precompute" << std::endl;
//     int nb_positives(0), nb_negatives(0);
//     int nb_training_samples = (int)training_labels.size();
//     Eigen::MatrixXd precomputed_Delta_n(dimensionality,nb_training_samples);
//     for (int k=0; k<nb_training_samples; ++k)
//     {
//         if (training_labels[k]==1)
//             ++nb_positives;
//         else
//             ++nb_negatives;
//         int i = training_pairs[k].first;
//         int j = training_pairs[k].second;
//         precomputed_Delta_n.block(0,k,dimensionality,1) = bow_appearance_matrix.block(0,i,dimensionality,1) - bow_appearance_matrix.block(0,j,dimensionality,1);
//     }
//     struct_inf_weights.precomputed_Delta_n = &precomputed_Delta_n;
//     Eigen::MatrixXd precomputed_Delta_n_tr = precomputed_Delta_n.transpose();
//     struct_inf_weights.precomputed_Delta_n_tr = &precomputed_Delta_n_tr;
//     
//     std::vector<double> balancing_weights(2,1);
//     balancing_weights[0] = nb_positives/((double)nb_negatives);
//     struct_inf_weights.balancing_weights = balancing_weights;
//     
//     // Prepare the canonical basis
//     std::vector<Eigen::SparseMatrix<double>> B_d;
//     create_canonical_basis(B_d,dimensionality,dimensionality);
//     struct_inf_weights.B_d = &B_d;
//     
//     void *struct_inf_weights_void_ptr;
//     struct_inf_weights_void_ptr = &struct_inf_weights;
// 
//     
//     // Vector of unknowns
//   //  std::cout << "Prepare optimisation" << std::endl;
//     alglib::real_1d_array x;
//     x.setcontent(dimensionality*dimensionality,(double*)this->m_weights.data());
//     double epsg = 0.0001;
//     double epsf = 0.01;
//     double epsx = 0;
//     double diffstep = 1.0e-6;
//     alglib::ae_int_t maxits = 0;
//     alglib::minlbfgsstate state;
//     alglib::minlbfgsreport rep;
//     alglib::minlbfgscreate(1, x, state);
//     alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
//     alglib::minlbfgsoptimize(state, compute_loss_function_pcca_full,NULL,struct_inf_weights_void_ptr);
//     alglib::minlbfgsresults(state, x, rep);
//  
//     Eigen::Map<Eigen::MatrixXd> W(&(x[0]),dimensionality,dimensionality);
//     this->m_weights = W;
//     
// }
// 
// void create_canonical_basis(std::vector<Eigen::SparseMatrix<double>>& B_d, int nb_rows, int nb_cols)
// {
//     B_d.resize(nb_rows*nb_cols);
//     std::vector<Eigen::Triplet<double>> triplet_list;
//     int d=0;
//     for (int j=0; j<nb_cols; ++j)
//     {
//         for (int i=0; i<nb_rows; ++i)
//         {
//             triplet_list.clear();
//             triplet_list.push_back(Eigen::Triplet<double>(i,j,1));
//             B_d[d] = Eigen::SparseMatrix<double>(nb_rows,nb_cols);
//             B_d[d].setFromTriplets(triplet_list.begin(),triplet_list.end());
//             ++d;
//         }
//     }
// }
// 
// void compute_loss_function_pcca_full(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr) // x = weights
// {
// 
//     
//     bool check_gradients = false;
// //    bool check_gradients = grad[0]!=(-1);   // uncomment to check the gradients numerically
//     
//     // Catch back the data passed as void pointer
//     StructureForTrainingPCCAFull* struct_inf_weights = static_cast<StructureForTrainingPCCAFull*>(ptr);
// 
//     // Nb training pairs
//  //   std::cout << "Compute loss" << std::endl;
//     int nb_training_pairs = (int)struct_inf_weights->training_labels->size();
//     int dimensionality_BOW = struct_inf_weights->dimensionality_BOW;
//     double beta = struct_inf_weights->beta;
//     Eigen::MatrixXd *precomputed_Delta_n = struct_inf_weights->precomputed_Delta_n;
//     Eigen::MatrixXd *precomputed_Delta_n_tr = struct_inf_weights->precomputed_Delta_n_tr;
//     // We precompute Delta(n,d) =  d-th dimension of the vector x_{i_n} - x_{j_n}
//     int dim_total = dimensionality_BOW*dimensionality_BOW;
//     
//     // Initial loss function
//     func = 0;
//     for (int d=0; d<dim_total; ++d)
//         grad[d] = 0;
//     
//     Eigen::Map<Eigen::MatrixXd> W((double*)&(x[0]),dimensionality_BOW,dimensionality_BOW);
//     Eigen::VectorXd W_times_Delta_n, B_d_times_Delta_n; 
//     for (int k=0; k<nb_training_pairs; ++k)
//     {
//   //      std::cout << "Pair " << k << std::endl;
//         int label_01 = (*(struct_inf_weights->training_labels))[k];
//         double label = 2*label_01 - 1;
//         double balancing_weight = struct_inf_weights->balancing_weights[label_01];
//         W_times_Delta_n = W*(precomputed_Delta_n->block(0,k,dimensionality_BOW,1));
//         double lin_comb = W_times_Delta_n.squaredNorm();
//         double score_in_sigmoid = ((double)label)*(1 - lin_comb);
//         
//         // Loss
//         func -= (balancing_weight/beta)*std::log(sigmoid(score_in_sigmoid,beta));
//         
//         double grad_factor = 2*label*balancing_weight*sigmoid(-score_in_sigmoid,beta);
//         for (int d=0; d<dim_total; ++d)
//         {
//             B_d_times_Delta_n = (struct_inf_weights->B_d->at(d))*(precomputed_Delta_n->block(0,k,dimensionality_BOW,1));
//             double deriv_term = (B_d_times_Delta_n.transpose())*W_times_Delta_n;
//             grad[d] += grad_factor*deriv_term;
//         }
//     }
// 
//  //   std::cout << "Loss = " << func << std::endl;
//  //   printf("%s\n",  grad.tostring(1).c_str());
//     
//     // Check gradients empirically
//     if (check_gradients)
//     {
//         double eps = 0.01;
//         for (int p=0; p<(dim_total); ++p)
//         {
//             alglib::real_1d_array x_eps, dummy_grad;
//             double func_eps, flag_grad(-1);
//             x_eps.setcontent(dim_total,&(x[0]));
//             dummy_grad.setcontent(dim_total,&flag_grad);
//             x_eps[p] = x[p] + eps;
//             compute_loss_function_pcca_full(x_eps,func_eps,dummy_grad,ptr);
//             double empirical_grad_p = (func_eps - func)/eps;
//             std::cout << "Parameter " << p << ": Empirical grad: " << empirical_grad_p << " ; Computed grad: " << grad[p] << std::endl;
//         }    
//     }
// 
// }

// -----------------------------------------------------
// FullPCCAOverlapModel  // not necessarily square
// -----------------------------------------------------

FullPCCAOverlapModel::FullPCCAOverlapModel()
{
    
}

FullPCCAOverlapModel::FullPCCAOverlapModel(int dimensionality_reduced, int dimensionality_BOW, double beta)
{
    this->m_weights = Eigen::MatrixXd::Constant(dimensionality_reduced,dimensionality_BOW,0.1);
    //this->m_weights.block(0,0,1,dimensionality_BOW) = Eigen::MatrixXd::Constant(1,dimensionality_BOW,1);
    this->m_beta = beta;
    this->m_preallocated_product = Eigen::VectorXd(dimensionality_reduced);
}

double FullPCCAOverlapModel::_getOverlapProbability(int i, int j, const Eigen::MatrixXd& bow_appearance_matrix)
{
    return getPCCASimilarity(i,j,1,this->m_beta,this->m_preallocated_product,this->m_weights,bow_appearance_matrix);
}
    

void FullPCCAOverlapModel::train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix)
{
    int dimensionality_reduced = this->m_weights.rows();
    int dimensionality_BOW = this->m_weights.cols();
    
    // Prepare optimisation data for alglib
    StructureForTrainingPCCAFull struct_inf_weights;
    struct_inf_weights.dimensionality_BOW = dimensionality_BOW;
    struct_inf_weights.dimensionality_reduced = dimensionality_reduced;
    struct_inf_weights.beta = this->m_beta;
    struct_inf_weights.training_labels = &training_labels;

    // Count the number of labels from each class
   // std::cout << "Precompute" << std::endl;
    int nb_positives(0), nb_negatives(0);
    int nb_training_samples = (int)training_labels.size();
    Eigen::MatrixXd precomputed_Delta_n(dimensionality_BOW,nb_training_samples);
    for (int k=0; k<nb_training_samples; ++k)
    {
        if (training_labels[k]==1)
            ++nb_positives;
        else
            ++nb_negatives;
        int i = training_pairs[k].first;
        int j = training_pairs[k].second;
        precomputed_Delta_n.block(0,k,dimensionality_BOW,1) = bow_appearance_matrix.block(0,i,dimensionality_BOW,1) - bow_appearance_matrix.block(0,j,dimensionality_BOW,1);
    }
    struct_inf_weights.precomputed_Delta_n = &precomputed_Delta_n;
    Eigen::MatrixXd precomputed_Delta_n_tr = precomputed_Delta_n.transpose();
    struct_inf_weights.precomputed_Delta_n_tr = &precomputed_Delta_n_tr;
    
    std::vector<double> balancing_weights(2,1);
    balancing_weights[0] = 1/((double)2*nb_negatives);
    balancing_weights[1] = 1/((double)2*nb_positives); // this weighting includes a normalization with respect to the number of samples: the weights sum to 1
    struct_inf_weights.balancing_weights = balancing_weights;
    
    // Prepare the canonical basis
    std::vector<Eigen::SparseMatrix<double>> B_d;
    create_canonical_basis(B_d,dimensionality_reduced,dimensionality_BOW);
    struct_inf_weights.B_d = &B_d;
    
    void *struct_inf_weights_void_ptr;
    struct_inf_weights_void_ptr = &struct_inf_weights;

    
    // Vector of unknowns
  //  std::cout << "Prepare optimisation" << std::endl;
    alglib::real_1d_array x;
    x.setcontent(dimensionality_reduced*dimensionality_BOW,(double*)this->m_weights.data());
    double epsg = 0.0001;
    double epsf = 0.01;
    double epsx = 0;
    double diffstep = 1.0e-6;
    alglib::ae_int_t maxits = 0;
    alglib::minlbfgsstate state;
    alglib::minlbfgsreport rep;
    alglib::minlbfgscreate(1, x, state);
    alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
    alglib::minlbfgsoptimize(state, compute_loss_function_pcca_full,NULL,struct_inf_weights_void_ptr);
    alglib::minlbfgsresults(state, x, rep);
 
    Eigen::Map<Eigen::MatrixXd> W(&(x[0]),dimensionality_reduced,dimensionality_BOW);
    this->m_weights = W;
    
}

void create_canonical_basis(std::vector<Eigen::SparseMatrix<double>>& B_d, int nb_rows, int nb_cols)
{
    B_d.resize(nb_rows*nb_cols);
    std::vector<Eigen::Triplet<double>> triplet_list;
    int d=0;
    for (int j=0; j<nb_cols; ++j)
    {
        for (int i=0; i<nb_rows; ++i)
        {
            triplet_list.clear();
            triplet_list.push_back(Eigen::Triplet<double>(i,j,1));
            B_d[d] = Eigen::SparseMatrix<double>(nb_rows,nb_cols);
            B_d[d].setFromTriplets(triplet_list.begin(),triplet_list.end());
            ++d;
        }
    }
}

void compute_loss_function_pcca_full(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr) // x = weights
{

    
    bool check_gradients = false;
//    bool check_gradients = grad[0]!=(-1);   // uncomment to check the gradients numerically
    
    // Catch back the data passed as void pointer
    StructureForTrainingPCCAFull* struct_inf_weights = static_cast<StructureForTrainingPCCAFull*>(ptr);

    // Nb training pairs
 //   std::cout << "Compute loss" << std::endl;
    int nb_training_pairs = (int)struct_inf_weights->training_labels->size();
    int dimensionality_BOW = struct_inf_weights->dimensionality_BOW;
    int dimensionality_reduced = struct_inf_weights->dimensionality_reduced;
    double beta = struct_inf_weights->beta;
    Eigen::MatrixXd *precomputed_Delta_n = struct_inf_weights->precomputed_Delta_n;
    Eigen::MatrixXd *precomputed_Delta_n_tr = struct_inf_weights->precomputed_Delta_n_tr;
    // We precompute Delta(n,d) =  d-th dimension of the vector x_{i_n} - x_{j_n}
    int dim_total = dimensionality_reduced*dimensionality_BOW;
    
    // Initial loss function
    func = 0;
    for (int d=0; d<dim_total; ++d)
        grad[d] = 0;
    
    Eigen::Map<Eigen::MatrixXd> W((double*)&(x[0]),dimensionality_reduced,dimensionality_BOW);
    Eigen::VectorXd W_times_Delta_n, B_d_times_Delta_n; 
    for (int k=0; k<nb_training_pairs; ++k)
    {
       // std::cout << "Pair " << k << std::endl;
        int label_01 = (*(struct_inf_weights->training_labels))[k];
        double label = 2*label_01 - 1;
        double balancing_weight = struct_inf_weights->balancing_weights[label_01];
        W_times_Delta_n = W*(precomputed_Delta_n->block(0,k,dimensionality_BOW,1));
        double lin_comb = W_times_Delta_n.squaredNorm();
        double score_in_sigmoid = ((double)label)*(1 - lin_comb);
        
        // Loss
        func -= (balancing_weight/beta)*std::log(sigmoid(score_in_sigmoid,beta));
        
        double grad_factor = 2*label*balancing_weight*sigmoid(-score_in_sigmoid,beta);
        for (int d=0; d<dim_total; ++d)
        {
            B_d_times_Delta_n = (struct_inf_weights->B_d->at(d))*(precomputed_Delta_n->block(0,k,dimensionality_BOW,1));
            double deriv_term = (B_d_times_Delta_n.transpose())*W_times_Delta_n;
            grad[d] += grad_factor*deriv_term;
        }
    }

    double lambda = 0;
    for (int d=0; d<dim_total; ++d)
    {    
        func += lambda*x[d]*x[d];
        grad[d] += 2*lambda*x[d];
    }
    std::cout << "Loss = " << func << std::endl;
 //   printf("%s\n",  grad.tostring(1).c_str());
    
    // Check gradients empirically
    if (check_gradients)
    {
        double eps = 0.0001;
        for (int p=0; p<(dim_total); ++p)
        {
            alglib::real_1d_array x_eps, dummy_grad;
            double func_eps, flag_grad(-1);
            x_eps.setcontent(dim_total,&(x[0]));
            dummy_grad.setcontent(dim_total,&flag_grad);
            x_eps[p] = x[p] + eps;
            compute_loss_function_pcca_full(x_eps,func_eps,dummy_grad,ptr);
            double empirical_grad_p = (func_eps - func)/eps;
            std::cout << "Parameter " << p << ": Empirical grad: " << empirical_grad_p << " ; Computed grad: " << grad[p] << std::endl;
        }    
    }

}

// -----------------------------------------------------
// PositiveDefinitePCCAOverlapModel
// -----------------------------------------------------

PositiveDefinitePCCAOverlapModel::PositiveDefinitePCCAOverlapModel(int dimensionality_BOW, double beta, double intercept)
{
 //   this->m_weights = Eigen::MatrixXd(dimensionality_BOW,dimensionality_BOW);
 //   this->m_weights.setIdentity();
    this->m_L = Eigen::MatrixXd(dimensionality_BOW,dimensionality_BOW);
    this->m_L.setIdentity();
    this->m_beta = beta;
    this->m_preallocated_product = Eigen::VectorXd(dimensionality_BOW);
    this->m_intercept = intercept;
}

void setLMatrix(Eigen::MatrixXd& L, const double* parameters)
{
    int d = L.cols();
    for (int k=0; k<d; ++k)
        L(k,k) = std::exp(parameters[k]); // first we set the diagonal
    int k(d);
    for (int j=0; j<d; ++j) // now the (strict) lower-triangular part
    {
        for (int i=(j+1); i<d; ++i)
        {
            L(i,j) = parameters[k];
            ++k;
        }
    }
}

double PositiveDefinitePCCAOverlapModel::_getOverlapProbability(int i, int j, const Eigen::MatrixXd& bow_appearance_matrix)
{
    getPCCASimilarity(i,j,1,m_beta,this->m_preallocated_product,this->m_L,bow_appearance_matrix,this->m_intercept,1);
}

void PositiveDefinitePCCAOverlapModel::setWeightMatrix(const double* parameters)
{
    setLMatrix(this->m_L,parameters);
 //   m_weight = (this->m_L.triangularView<Eigen::Lower>())*(this->m_L.triangularView<Eigen::Lower>().transpose());
 //   m_weights = (this->m_L)*(this->m_L.transpose());
}

void PositiveDefinitePCCAOverlapModel::getParameters(double* parameters) const
{
    int d = this->m_L.cols();
    for (int k=0; k<d; ++k)
        parameters[k] = std::log(this->m_L(k,k));
    int k(d);
    for (int j=0; j<d; ++j) // now the (strict) lower-triangular part
    {
        for (int i=(j+1); i<d; ++i)
        {
            parameters[k] = this->m_L(i,j);
            ++k;
        }
    }
    
}

void create_low_triangular_canonical_basis(std::vector<Eigen::SparseMatrix<double>>& B_d, int d)
{
    
    B_d.resize(d*(d+1)/2);
    std::vector<Eigen::Triplet<double>> triplet_list;
    for (int k=0; k<d; ++k) // Diagonal part
    {
        triplet_list.clear();
        triplet_list.push_back(Eigen::Triplet<double>(k,k,1));
        B_d[k] = Eigen::SparseMatrix<double>(d,d);
        B_d[k].setFromTriplets(triplet_list.begin(),triplet_list.end());
    }
    int k(d);
    for (int j=0; j<d; ++j) // Strict lower triangluar part
    {
        for (int i=(j+1); i<d; ++i)
        {
            triplet_list.clear();
            triplet_list.push_back(Eigen::Triplet<double>(i,j,1));
            B_d[k] = Eigen::SparseMatrix<double>(d,d);
            B_d[k].setFromTriplets(triplet_list.begin(),triplet_list.end());
            ++k;
        }
    }
}

// void PositiveDefinitePCCAOverlapModel::train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix)
// {
//     int dimensionality = this->m_L.rows();
//     // Prepare optimisation data for alglib
//     StructureForTrainingPCCAFull struct_inf_weights;
//     struct_inf_weights.dimensionality_BOW = dimensionality;
//     struct_inf_weights.beta = this->m_beta;
//     struct_inf_weights.training_labels = &training_labels;
// 
//     // Count the number of labels from each class
//    // std::cout << "Precompute" << std::endl;
//     int nb_positives(0), nb_negatives(0);
//     int nb_training_samples = (int)training_labels.size();
//     Eigen::MatrixXd precomputed_Delta_n(dimensionality,nb_training_samples), precomputed_L(dimensionality,dimensionality);
//     for (int k=0; k<nb_training_samples; ++k)
//     {
//         if (training_labels[k]==1)
//             ++nb_positives;
//         else
//             ++nb_negatives;
//         int i = training_pairs[k].first;
//         int j = training_pairs[k].second;
//         precomputed_Delta_n.block(0,k,dimensionality,1) = bow_appearance_matrix.block(0,i,dimensionality,1) - bow_appearance_matrix.block(0,j,dimensionality,1);
//     }
//     struct_inf_weights.precomputed_Delta_n = &precomputed_Delta_n;
//     Eigen::MatrixXd precomputed_Delta_n_tr = precomputed_Delta_n.transpose();
//     struct_inf_weights.precomputed_Delta_n_tr = &precomputed_Delta_n_tr;
//     std::vector<double> balancing_weights(2,1);
//    // balancing_weights[0] = 1/((double)2*nb_negatives);
//     balancing_weights[0] = 1/((double)2*nb_negatives);
//     balancing_weights[1] = 1/((double)2*nb_positives); // this weighting includes a normalization with respect to the number of samples: the weights sum to 1
//     struct_inf_weights.balancing_weights = balancing_weights;
//     
//     // Prepare the canonical basis
//     std::vector<Eigen::SparseMatrix<double>> B_d;
//     create_low_triangular_canonical_basis(B_d,dimensionality);
//     struct_inf_weights.B_d = &B_d;
//     
//     void *struct_inf_weights_void_ptr;
//     struct_inf_weights_void_ptr = &struct_inf_weights;
// 
//     
//     alglib::real_1d_array x;
//     int nb_parameters = dimensionality*(dimensionality+1)/2;
//     
//     double *p = new double[nb_parameters];
//     this->getParameters(p);
//     x.setcontent(nb_parameters,p);
//     
// 
//     double epsg = 0.0001;
//     double epsf = 0.01;
//     double epsx = 0;
//     double diffstep = 1.0e-6;
//     alglib::ae_int_t maxits = 0;
//     alglib::minlbfgsstate state;
//     alglib::minlbfgsreport rep;
//     alglib::minlbfgscreate(1, x, state);
//     alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
//     alglib::minlbfgsoptimize(state, compute_loss_function_pcca_positive_definite,NULL,struct_inf_weights_void_ptr);
//     alglib::minlbfgsresults(state, x, rep);
//  
//     
//     this->setWeightMatrix(&(x[0]));
//     
//     
//   //  std::cout << "New weight matrix: " << this->m_weights << std::endl;
//     
//     delete[] p;
//     
// }


void PositiveDefinitePCCAOverlapModel::train(std::vector<LabeledImagePair>& training_pairs, Eigen::MatrixXd& bow_appearance_matrix)
{
    int nb_training_pairs = (int)training_pairs.size();
    std::vector<std::pair<int,int>> training_indices(nb_training_pairs);
    std::vector<double> training_labels(nb_training_pairs), importance_weights(nb_training_pairs);
    for (int k=0; k<nb_training_pairs; ++k)
    {
        training_indices[k] = training_pairs[k].image_indices;
        if (training_pairs[k].this_pair_overlaps)
            training_labels[k] = 1;
        else
            training_labels[k] = 0;
        importance_weights[k] = training_pairs[k].importance_weight;
    }
    
    this->train(training_indices,training_labels,importance_weights,bow_appearance_matrix);
    
}

// Same as above but we add the intercept
void PositiveDefinitePCCAOverlapModel::train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix)
{
    int nb_training_samples = (int)training_labels.size();
    std::vector<double> importance_weights(nb_training_samples,1);
    this->train(training_pairs,training_labels,importance_weights,bow_appearance_matrix);
}



// Same as above but we add the intercept
void PositiveDefinitePCCAOverlapModel::train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, std::vector<double>& importance_weights, Eigen::MatrixXd& bow_appearance_matrix)
{
    int dimensionality = this->m_L.rows();
    // Prepare optimisation data for alglib
    StructureForTrainingPCCAFull struct_inf_weights;
    struct_inf_weights.dimensionality_BOW = dimensionality;
    struct_inf_weights.beta = this->m_beta;
    struct_inf_weights.training_labels = &training_labels;

    
    // Count the number of labels from each class
   // std::cout << "Precompute" << std::endl;
    double nb_positives(0), nb_negatives(0);
    int nb_training_samples = (int)training_labels.size();
    struct_inf_weights.importance_weights.resize(nb_training_samples);
    Eigen::MatrixXd precomputed_Delta_n(dimensionality,nb_training_samples), precomputed_L(dimensionality,dimensionality);
    for (int k=0; k<nb_training_samples; ++k)
    {
        if (training_labels[k]==1)
            nb_positives += importance_weights[k];
        else
            nb_negatives += importance_weights[k];
        int i = training_pairs[k].first;
        int j = training_pairs[k].second;
        precomputed_Delta_n.block(0,k,dimensionality,1) = bow_appearance_matrix.block(0,i,dimensionality,1) - bow_appearance_matrix.block(0,j,dimensionality,1);
    }
    struct_inf_weights.precomputed_Delta_n = &precomputed_Delta_n;
    Eigen::MatrixXd precomputed_Delta_n_tr = precomputed_Delta_n.transpose();
    struct_inf_weights.precomputed_Delta_n_tr = &precomputed_Delta_n_tr;
    
    for (int k=0; k<nb_training_samples; ++k)
    {
        if (training_labels[k]==1)
             struct_inf_weights.importance_weights[k] = importance_weights[k]/((double)2*nb_positives); 
        else
             struct_inf_weights.importance_weights[k] = importance_weights[k]/((double)2*nb_negatives); 
    }
    
    
    // Prepare the canonical basis
    std::vector<Eigen::SparseMatrix<double>> B_d;
    create_low_triangular_canonical_basis(B_d,dimensionality);
    struct_inf_weights.B_d = &B_d;
    
    void *struct_inf_weights_void_ptr;
    struct_inf_weights_void_ptr = &struct_inf_weights;

    
    alglib::real_1d_array x;
    int nb_parameters = dimensionality*(dimensionality+1)/2;
    
    double *p = new double[nb_parameters+1];
    this->getParameters(p);
    p[nb_parameters] = this->m_intercept;
    x.setcontent(nb_parameters+1,p);
    

    double epsg = 0.0001;
    double epsf = 0.01;
    double epsx = 0;
    double diffstep = 1.0e-6;
    alglib::ae_int_t maxits = 0;
    alglib::minlbfgsstate state;
    alglib::minlbfgsreport rep;
    alglib::minlbfgscreate(1, x, state);
    alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
    alglib::minlbfgsoptimize(state, compute_loss_function_pcca_positive_definite,NULL,struct_inf_weights_void_ptr);
    alglib::minlbfgsresults(state, x, rep);
 
    
    this->setWeightMatrix(&(x[0]));
    this->m_intercept = x[nb_parameters];
    
  //  std::cout << "New weight matrix: " << this->m_weights << std::endl;
    
    delete[] p;
    
}






// Sort of PCCA inspired
void compute_loss_function_pcca_positive_definite(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr) // x = weights
{
    bool do_we_want_to_check_gradients = false; // set to true to debug and check the gradients numerically, false for best performance    
    
    bool check_gradients;
    if (do_we_want_to_check_gradients)
        check_gradients = grad[0]!=(-1);
    else
        check_gradients = false;
    
    // Catch back the data passed as void pointer
    StructureForTrainingPCCAFull* struct_inf_weights = static_cast<StructureForTrainingPCCAFull*>(ptr);

    // Nb training pairs
 //   std::cout << "Compute loss" << std::endl;
    int nb_training_pairs = (int)struct_inf_weights->training_labels->size();
    int dimensionality_BOW = struct_inf_weights->dimensionality_BOW;
    double beta = struct_inf_weights->beta;
    Eigen::MatrixXd *precomputed_Delta_n = struct_inf_weights->precomputed_Delta_n;
    Eigen::MatrixXd *precomputed_Delta_n_tr = struct_inf_weights->precomputed_Delta_n_tr;
    // We precompute Delta(n,d) =  d-th dimension of the vector x_{i_n} - x_{j_n}
    int dim_total = dimensionality_BOW*(dimensionality_BOW+1)/2;
    
    double intercept = x[dim_total];
    
    
    // Initial loss function
    func = 0;
    for (int d=0; d<(dim_total+1); ++d)
        grad[d] = 0;
    
    Eigen::VectorXd W_times_Delta_n, dW_dx_times_Delta_n; 
    
  //  Eigen::TriangularView<Eigen::MatrixXd,Eigen::Lower> L = precomputed_L->triangularView<Eigen::Lower>();
 //   Eigen::TriangularView<Eigen::MatrixXd,Eigen::Upper> L_tr = L.transpose();
    

    
    // We convert the current vector of parameters
  //  setLMatrix(*(struct_inf_weights->precomputed_L),&(x[0]));
  //  Eigen::Map<Eigen::MatrixXd> L(struct_inf_weights->precomputed_L->data(),dimensionality_BOW,dimensionality_BOW);
    //std::cout << *(struct_inf_weights->precomputed_L) << std::endl;
    
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(dimensionality_BOW,dimensionality_BOW);
    setLMatrix(L,&(x[0]));
    Eigen::MatrixXd L_tr = L.transpose();

    
    for (int k=0; k<nb_training_pairs; ++k)
    {
        //std::cout << "Pair " << k << std::endl;
        int label_01 = (*(struct_inf_weights->training_labels))[k];
        double label = 2*label_01 - 1;
        double balancing_weight = struct_inf_weights->importance_weights[k];
        W_times_Delta_n = L_tr*(precomputed_Delta_n->block(0,k,dimensionality_BOW,1));
        double lin_comb = W_times_Delta_n.squaredNorm();
        double score_in_sigmoid = ((double)label)*(intercept - lin_comb);
        
        // Loss
  //      std::cout << "Score: " << score_in_sigmoid << std::endl;
        func -= (balancing_weight/beta)*std::log(sigmoid(score_in_sigmoid,beta));
        
        double grad_factor = label*balancing_weight*sigmoid(-score_in_sigmoid,beta);
        
        grad[dim_total] -= grad_factor;
        
        // Diagonal part
        for (int d=0; d<dim_total; ++d)
        {
            if (d<dimensionality_BOW) // if diagonal term
                dW_dx_times_Delta_n = L(d,d)*((struct_inf_weights->B_d->at(d))*L_tr + L*(struct_inf_weights->B_d->at(d)))*(precomputed_Delta_n->block(0,k,dimensionality_BOW,1));
            else   // if strict lower triangular term
                dW_dx_times_Delta_n = ((struct_inf_weights->B_d->at(d))*L_tr + L*(struct_inf_weights->B_d->at(d)))*(precomputed_Delta_n->block(0,k,dimensionality_BOW,1));
            grad[d] += grad_factor*(dW_dx_times_Delta_n.transpose())*W_times_Delta_n;
        }
    }

    std::cout << "Loss = " << func << std::endl;
   // printf("%s\n",  grad.tostring(1).c_str());
    
    // Check gradients empirically
    if (check_gradients)
    {
        double eps = 0.00001;
        for (int p=dim_total; p>=0; --p)
        {
            alglib::real_1d_array x_eps, dummy_grad;
            double func_eps, flag_grad(-1);
            x_eps.setcontent(dim_total+1,&(x[0]));
            dummy_grad.setcontent(dim_total+1,&flag_grad);
            x_eps[p] = x[p] + eps;
            compute_loss_function_pcca_positive_definite(x_eps,func_eps,dummy_grad,ptr);
            double empirical_grad_p = (func_eps - func)/eps;
            std::cout << "Parameter " << p << ": Empirical grad: " << empirical_grad_p << " ; Computed grad: " << grad[p] << std::endl;
        }    
    }

}

// Another loss function
// void compute_loss_function_pcca_positive_definite(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr) // x = weights
// {
//     bool do_we_want_to_check_gradients = false; // set to true to debug and check the gradients numerically, false for best performance
//     
//     double intercept = 0; // default: 1
//     
//     
//     bool check_gradients;
//     if (do_we_want_to_check_gradients)
//         check_gradients = grad[0]!=(-1);
//     else
//         check_gradients = false;
//     
//     // Catch back the data passed as void pointer
//     StructureForTrainingPCCAFull* struct_inf_weights = static_cast<StructureForTrainingPCCAFull*>(ptr);
// 
//     // Nb training pairs
//  //   std::cout << "Compute loss" << std::endl;
//     int nb_training_pairs = (int)struct_inf_weights->training_labels->size();
//     int dimensionality_BOW = struct_inf_weights->dimensionality_BOW;
//     double beta = struct_inf_weights->beta;
//     Eigen::MatrixXd *precomputed_Delta_n = struct_inf_weights->precomputed_Delta_n;
//     Eigen::MatrixXd *precomputed_Delta_n_tr = struct_inf_weights->precomputed_Delta_n_tr;
//     // We precompute Delta(n,d) =  d-th dimension of the vector x_{i_n} - x_{j_n}
//     int dim_total = dimensionality_BOW*(dimensionality_BOW+1)/2;
//     
//     
//     
//     // Initial loss function
//     func = 0;
//     for (int d=0; d<dim_total; ++d)
//         grad[d] = 0;
//     
//     Eigen::VectorXd W_times_Delta_n, dW_dx_times_Delta_n; 
//     
//   //  Eigen::TriangularView<Eigen::MatrixXd,Eigen::Lower> L = precomputed_L->triangularView<Eigen::Lower>();
//  //   Eigen::TriangularView<Eigen::MatrixXd,Eigen::Upper> L_tr = L.transpose();
//     
// 
//     
//     // We convert the current vector of parameters
//   //  setLMatrix(*(struct_inf_weights->precomputed_L),&(x[0]));
//   //  Eigen::Map<Eigen::MatrixXd> L(struct_inf_weights->precomputed_L->data(),dimensionality_BOW,dimensionality_BOW);
//     //std::cout << *(struct_inf_weights->precomputed_L) << std::endl;
//     
//     Eigen::MatrixXd L = Eigen::MatrixXd::Zero(dimensionality_BOW,dimensionality_BOW);
//     setLMatrix(L,&(x[0]));
//     Eigen::MatrixXd L_tr = L.transpose();
//     
//     
//     for (int k=0; k<nb_training_pairs; ++k)
//     {
//         //std::cout << "Pair " << k << std::endl;
//         int label_01 = (*(struct_inf_weights->training_labels))[k];
//         double label = 2*label_01 - 1;
//         double balancing_weight = struct_inf_weights->balancing_weights[label_01];
//         W_times_Delta_n = L_tr*(precomputed_Delta_n->block(0,k,dimensionality_BOW,1));
//         double lin_comb = W_times_Delta_n.squaredNorm();
//         double score_in_sigmoid = -lin_comb;
//         
//         // Loss
//   //      std::cout << "Score: " << score_in_sigmoid << std::endl;
//         
//         double grad_factor;
//         if (label_01==1)
//         {
//             func -= (balancing_weight/beta)*std::log(2*sigmoid(score_in_sigmoid,beta));
//             grad_factor = balancing_weight*sigmoid(-score_in_sigmoid,beta);
//         }
//         else
//         {
//             func -= (balancing_weight/beta)*std::log(1 - 2*sigmoid(score_in_sigmoid,beta));
//             grad_factor = balancing_weight*(-2*sigmoid(score_in_sigmoid,beta)*(1 - sigmoid(score_in_sigmoid,beta)))/(1 - 2*sigmoid(score_in_sigmoid,beta));
//         }
//         
//         // Derivative of the score in the sigmoid
//         for (int d=0; d<dim_total; ++d)
//         {
//             if (d<dimensionality_BOW) // if diagonal term
//                 dW_dx_times_Delta_n = L(d,d)*((struct_inf_weights->B_d->at(d))*L_tr + L*(struct_inf_weights->B_d->at(d)))*(precomputed_Delta_n->block(0,k,dimensionality_BOW,1));
//             else   // if strict lower triangular term
//                 dW_dx_times_Delta_n = ((struct_inf_weights->B_d->at(d))*L_tr + L*(struct_inf_weights->B_d->at(d)))*(precomputed_Delta_n->block(0,k,dimensionality_BOW,1));
//             grad[d] += grad_factor*(dW_dx_times_Delta_n.transpose())*W_times_Delta_n;
//         }
//     }
// 
//     std::cout << "Loss = " << func << std::endl;
//    // printf("%s\n",  grad.tostring(1).c_str());
//     
//     // Check gradients empirically
//     if (check_gradients)
//     {
//         double eps = 0.00001;
//         for (int p=0; p<(dim_total); ++p)
//         {
//             alglib::real_1d_array x_eps, dummy_grad;
//             double func_eps, flag_grad(-1);
//             x_eps.setcontent(dim_total,&(x[0]));
//             dummy_grad.setcontent(dim_total,&flag_grad);
//             x_eps[p] = x[p] + eps;
//             compute_loss_function_pcca_positive_definite(x_eps,func_eps,dummy_grad,ptr);
//             double empirical_grad_p = (func_eps - func)/eps;
//             std::cout << "Parameter " << p << ": Empirical grad: " << empirical_grad_p << " ; Computed grad: " << grad[p] << std::endl;
//         }    
//     }
// 
// }



DifferenceBOWOverlapModel::DifferenceBOWOverlapModel(int dimensionality)
{
   m_weights = Eigen::VectorXd::Zero(dimensionality+1);
}

double DifferenceBOWOverlapModel::_getOverlapProbability(int i, int j, const Eigen::MatrixXd& bow_appearance_matrix)
{
    double score_in_sigmoid;
    return getDifferenceBOWSimilarity(score_in_sigmoid,i,j,bow_appearance_matrix,m_weights);
}


double getDifferenceBOWSimilarity(double& score_in_sigmoid, int i, int j, const Eigen::MatrixXd& bow_appearance_matrix, Eigen::VectorXd& weights)
{
    int dimensionality_BOW = weights.rows() - 1; // includes the final weight
    Eigen::VectorXd diff_vector = bow_appearance_matrix.block(0,i,dimensionality_BOW,1) - bow_appearance_matrix.block(0,j,dimensionality_BOW,1);
    Eigen::VectorXd abs_diff_vector = diff_vector.array().abs();
    score_in_sigmoid = diff_vector.dot(weights.segment(0,dimensionality_BOW)) + weights(dimensionality_BOW);
    return sigmoid(score_in_sigmoid);
}


//void BOWOverlapModel::displayWeights() const
//{
//    int nb_weights = (int)m_weights.size();
//    for (int k=0; k<nb_weights; ++k)
//        std::cout << m_weights[k] << " ";
//    std::cout << std::endl;
//}
//
void DifferenceBOWOverlapModel::train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix)
{
   int dimensionality = (int)m_weights.size() - 1;

   
   double lambda = 0.0001; // regularisation
   
   
   // Prepare optimisation data for alglib
   StructureForInferenceDifferenceBOWWeights struct_inf_weights;
   struct_inf_weights.dimensionality = dimensionality;
   struct_inf_weights.training_pairs = &training_pairs;
   struct_inf_weights.training_labels = &training_labels;
   struct_inf_weights.bow_appearance_matrix = &bow_appearance_matrix;
   struct_inf_weights.lambda = lambda;

   // Vector of unknowns (shallow copy)
   alglib::real_1d_array x;
   //x.attach_to_ptr(m_nb_unknowns*m_nb_coordinates,&(m_unknown_coordinatewise_variances[0]));
   x.setcontent(dimensionality+1,&(m_weights(0)));

   void *struct_inf_weights_void_ptr;
   struct_inf_weights_void_ptr = &struct_inf_weights;

   // Count the number of labels from each class
   int nb_positives(0), nb_negatives(0);
   int nb_training_samples = (int)training_labels.size();

   for (int k=0; k<nb_training_samples; ++k)
   {
       if (training_labels[k]==1)
           ++nb_positives;
       else
           ++nb_negatives;
   }
   std::vector<double> balancing_weights(2,1);
   balancing_weights[0] = nb_positives/((double)nb_negatives);
   struct_inf_weights.balancing_weights = balancing_weights;



   double epsg = 0.0001;
   double epsf = 0.01;
   double epsx = 0;
   double diffstep = 1.0e-6;
   alglib::ae_int_t maxits = 0;

 //  printf("%s\n", x.tostring(1).c_str()); // EXPECTED: [1.500,0.500]

//    alglib::minlbfgsstate state;
//    alglib::minlbfgsreport rep;
//    alglib::minlbfgscreate(1, x, state);
//    alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
//    alglib::minlbfgsoptimize(state,compute_loss_function_inference_difference_bow_weights_grad,NULL,struct_inf_weights_void_ptr);
//    alglib::minlbfgsresults(state, x, rep);

//    alglib::minbleicstate state;
//    alglib::minbleicreport rep;
//    alglib::minbleiccreate(x, state);
//    alglib::minbleicsetbc(state, bnd_l, bnd_u);
//    alglib::minbleicsetcond(state, epsg, epsf, epsx, maxits);
//    alglib::minbleicoptimize(state,compute_loss_function_inference_variances_grad,NULL,struct_inf_var_void_ptr);
//    alglib::minbleicresults(state, x, rep);

   alglib::mincgstate state;
   alglib::mincgreport rep;
   alglib::mincgcreate(x,state);
   alglib::mincgsetcond(state, epsg, epsf, epsx, maxits);
   alglib::mincgoptimize(state,compute_loss_function_inference_difference_bow_weights_grad,NULL,struct_inf_weights_void_ptr);
   alglib::mincgresults(state, x, rep);

  // printf("%s\n", x.tostring(1).c_str()); // EXPECTED: [1.500,0.500]

   for (int d=0; d<=dimensionality; ++d)
       m_weights(d) = x[d];

//    for (int k=0; k<(m_nb_unknowns*m_nb_coordinates); ++k)
//        m_unknown_coordinatewise_variances[k] = 0.5*(m_min_variance + m_max_variance) + 0.5*(m_max_variance - m_min_variance)*tanh(x[k]);

}


void compute_loss_function_inference_difference_bow_weights_grad(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr) // x = weights
{
    bool check_gradients = false;
//    bool check_gradients = grad[0]!=(-1);   // uncomment to check the gradients numerically
    
    
   // Catch back the data passed as void pointer
   StructureForInferenceDifferenceBOWWeights* struct_inf_weights = static_cast<StructureForInferenceDifferenceBOWWeights*>(ptr);

   // Nb training pairs
   int nb_training_pairs = (int)struct_inf_weights->training_pairs->size();
   int dimensionality = struct_inf_weights->dimensionality;
 //  double *bow_appearance_matrix = &((*(struct_inf_weights->bow_appearance_matrix))[0]);
   double lambda = struct_inf_weights->lambda;

   Eigen::VectorXd x_eigen(dimensionality+1);
   
   
   // Initial loss function
   func = 0;
   for (int ind=0; ind<(dimensionality+1); ++ind)
   {
       grad[ind] = 0;
       x_eigen(ind) = x[ind];
   }
//   double alpha = 0.95;
//   double eps=0.05;

    std::cout << "Nb training pairs: " << nb_training_pairs << std::endl;

   for (int k=0; k<nb_training_pairs; ++k)
   {
       // Current pair
       int i = (*(struct_inf_weights->training_pairs))[k].first;
       int j = (*(struct_inf_weights->training_pairs))[k].second;
        double label = (*(struct_inf_weights->training_labels))[k];
       double score;
       double overlap_probability = getDifferenceBOWSimilarity(score,i,j,*(struct_inf_weights->bow_appearance_matrix),x_eigen);

     //  std::cout << i << " " << j << " " << label <<std::endl;
     //  std::cout << overlap_probability << std::endl;
       

       func -= struct_inf_weights->balancing_weights[label]*(label*log(overlap_probability) + (1 - label)*log(1 - overlap_probability));

       double common_factor_gradient = struct_inf_weights->balancing_weights[label]*((1-label)*overlap_probability - label*(1-overlap_probability));
       grad[dimensionality] += common_factor_gradient;
       for (int d=0; d<dimensionality; ++d)
       {
        //   std::cout << "Difference " << d << ": " << std::abs<double>((*(struct_inf_weights->bow_appearance_matrix))(d,i) - (*(struct_inf_weights->bow_appearance_matrix))(d,j)) << std::endl;
           //grad[d] += common_factor_gradient*std::abs<double>(bow_appearance_matrix[i*dimensionality+d] - bow_appearance_matrix[j*dimensionality+d]);
           grad[d] += common_factor_gradient*std::abs<double>((*(struct_inf_weights->bow_appearance_matrix))(d,i) - (*(struct_inf_weights->bow_appearance_matrix))(d,j));
       }
   }



   // Regularisation
   for (int d=0; d<=dimensionality; ++d)
   {
       func += lambda*nb_training_pairs*std::pow<double>(x[d],2)/(dimensionality+1);
       grad[d] += 2*lambda*nb_training_pairs*x[d]/(dimensionality+1);
   }

   std::cout << "Total loss = " << func << std::endl;
   printf("%s\n",  x.tostring(1).c_str());
   
    // Check gradients empirically
    if (check_gradients)
    {
        double eps = 0.001;
        for (int p=0; p<(dimensionality+1); ++p)
        {
            alglib::real_1d_array x_eps, dummy_grad;
            double func_eps, flag_grad(-1);
            x_eps.setcontent(dimensionality+1,&(x[0]));
            dummy_grad.setcontent(dimensionality+1,&flag_grad);
            x_eps[p] = x[p] + eps;
            compute_loss_function_inference_difference_bow_weights_grad(x_eps,func_eps,dummy_grad,ptr);
            double empirical_grad_p = (func_eps - func)/eps;
            std::cout << "Parameter " << p << ": Empirical grad: " << empirical_grad_p << " ; Computed grad: " << grad[p] << std::endl;
        }    
    }

}

void compute_loss_function_inference_difference_bow_weights(const alglib::real_1d_array &x, double &func, void *ptr) // x = weights
{
   // Catch back the data passed as void pointer
   StructureForInferenceDifferenceBOWWeights* struct_inf_weights = static_cast<StructureForInferenceDifferenceBOWWeights*>(ptr);

   // Nb training pairs
   int nb_training_pairs = (int)struct_inf_weights->training_pairs->size();
   int dimensionality = struct_inf_weights->dimensionality;
 //  double *bow_appearance_matrix = &((*(struct_inf_weights->bow_appearance_matrix))[0]);
   double lambda = struct_inf_weights->lambda;

   Eigen::VectorXd x_eigen(dimensionality+1);
   
   
   // Initial loss function
   func = 0;
   for (int ind=0; ind<(dimensionality+1); ++ind)
   {
       x_eigen(ind) = x[ind];
   }
//   double alpha = 0.95;
//   double eps=0.05;

    std::cout << "Nb training pairs: " << nb_training_pairs << std::endl;

   for (int k=0; k<nb_training_pairs; ++k)
   {
       // Current pair
       int i = (*(struct_inf_weights->training_pairs))[k].first;
       int j = (*(struct_inf_weights->training_pairs))[k].second;
        double label = (*(struct_inf_weights->training_labels))[k];
       double score;
       double overlap_probability = getDifferenceBOWSimilarity(score,i,j,*(struct_inf_weights->bow_appearance_matrix),x_eigen);

       func -= struct_inf_weights->balancing_weights[label]*(label*log(overlap_probability) + (1 - label)*log(1 - overlap_probability));
       std::cout << func << std::endl;
   }



   // Regularisation
   for (int d=0; d<=dimensionality; ++d)
   {
       func += lambda*nb_training_pairs*std::pow<double>(x[d],2)/(dimensionality+1);
   }

   std::cout << "Total loss = " << func << std::endl;
   printf("%s\n",  x.tostring(1).c_str());
   

}
// -----------------------------------------------------
// Other functions
// -----------------------------------------------------



void compute_loss_function_inference_pcca_weights_grad(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr) // x = weights
{

    // Catch back the data passed as void pointer
    StructureForInferencePCCAWeights* struct_inf_weights = static_cast<StructureForInferencePCCAWeights*>(ptr);

    // Nb training pairs
    int nb_training_pairs = (int)struct_inf_weights->training_pairs->size();
    int dimensionality_BOW = struct_inf_weights->dimensionality_BOW;
    int dimensionality_BOW_squared = dimensionality_BOW*dimensionality_BOW;
    int dimensionality_reduced = struct_inf_weights->dimensionality_reduced;
    int dimensionality_total = dimensionality_BOW*dimensionality_reduced;
    //double *bow_appearance_matrix = &((*(struct_inf_weights->bow_appearance_matrix))[0]);
    double beta = struct_inf_weights->beta;
    std::vector<cv::Mat> *precomputed_Cn = struct_inf_weights->precomputed_Cn; // precomputed_Cn[k*dimensionality_BOW_squared+(m*dimensionality_BOW)+n] = entry at the roow m, column n of the matrix Cn used in gradient
    std::vector<cv::Mat> *bow_appearance = struct_inf_weights->bow_appearance;
//    std::vector<Eigen::MatrixXd>  *precomputed_Cn = struct_inf_weights->precomputed_Cn; // precomputed_Cn[k*dimensionality_BOW_squared+(m*dimensionality_BOW)+n] = entry at the roow m, column n of the matrix Cn used in gradient
//    std::vector<Eigen::SparseVector<double>> *bow_appearance = struct_inf_weights->bow_appearance;
    // Initial loss function
    func = 0;
    for (int d=0; d<dimensionality_total; ++d)
        grad[d] = 0;

 //   double alpha = 0.95;
 //   double eps=0.05;
    double *grad_data = grad.getcontent();
    cv::Mat x_mat(dimensionality_reduced,dimensionality_BOW,CV_64F); // wrapper for L
    for (int d_r=0; d_r<dimensionality_reduced; ++d_r)
    {
        for (int d=0; d<dimensionality_BOW; ++d)
            x_mat.at<double>(d_r,d) = x[d_r*dimensionality_BOW + d];
    }
    cv::Mat grad_mat(dimensionality_reduced,dimensionality_BOW,CV_64F,grad_data); // wrapper for L

//    Eigen::SparseMatrix<double,Eigen::RowMajor> x_mat_sparse(dimensionality_reduced,dimensionality_BOW);
//    std::vector<Eigen::Triplet<double>> list_values;
//    for (int d_r=0; d_r<dimensionality_reduced; ++d_r)
//    {
//        for (int d=0; d<dimensionality_BOW; ++d)
//        {
//            if (x[d_r*dimensionality_BOW + d]!=0)
//                list_values.push_back(Eigen::Triplet<double>(d_r,d,x[d_r*dimensionality_BOW + d]));
//        }
//    }
//    x_mat_sparse.setFromTriplets(list_values.begin(),list_values.end());
//    Eigen::MatrixXd grad_mat(dimensionality_reduced,dimensionality_BOW);
//    for (int d_r=0; d_r<dimensionality_reduced; ++d_r)
//    {
//        for (int d=0; d<dimensionality_BOW; ++d)
//            grad_mat(d_r,d) = 0;
//    }



    for (int k=0; k<nb_training_pairs; ++k)
    {
        //std::cout << k << " ";
        // Current pair

        int i = (*(struct_inf_weights->training_pairs))[k].first;
        int j = (*(struct_inf_weights->training_pairs))[k].second;

        int label_01 = (*(struct_inf_weights->training_labels))[k];
        double label = 2*label_01 - 1;
        double balancing_weight = struct_inf_weights->balancing_weights[label_01];
        double squared_norm;
        //double sigmoid_part = getPCCASimilarity(squared_norm,i,j,label,dimensionality_BOW,dimensionality_reduced,beta,*(struct_inf_weights->bow_appearance_matrix),x.getcontent());
        double sigmoid_part = getPCCASimilarity(squared_norm,label,beta,(*bow_appearance)[i],(*bow_appearance)[j],x_mat);
        double common_factor_gradient = 2*balancing_weight*((double)label)*(1-sigmoid_part);

        // Loss
        func += (balancing_weight/beta)*log(1 + exp(beta*(((double)label)*(squared_norm-1))));

        grad_mat += common_factor_gradient*x_mat*((*precomputed_Cn)[k]);

        // Gradient (we might want to optimise these matrix multiplications with OpenCV or Eigen...)
//        for (int d_r=0; d_r<dimensionality_reduced; ++d_r)
//        {
//
//            for (int d=0; d<dimensionality_BOW; ++d)
//            {
//                double L_times_Cn_entry(0);
//                //double fixed_difference = (*(struct_inf_weights->bow_appearance_matrix))[i*dimensionality_BOW+d] - (*(struct_inf_weights->bow_appearance_matrix))[j*dimensionality_BOW+d];
//                for (int m=0; m<dimensionality_BOW; ++m)
//                {
//                    //L_times_Cn_entry += x[d_r*dimensionality_BOW + m]*((*(struct_inf_weights->bow_appearance_matrix))[i*dimensionality_BOW+m] - (*(struct_inf_weights->bow_appearance_matrix))[j*dimensionality_BOW+m]);
//                    L_times_Cn_entry += x[d_r*dimensionality_BOW + m]*precomputed_Cn[k*dimensionality_BOW_squared+(d*dimensionality_BOW)+m];
//                }
//               // L_times_Cn_entry *= fixed_difference;
//                grad[d_r*dimensionality_BOW + d] += common_factor_gradient*L_times_Cn_entry;
//            }
//        }
    }

//    for (int d_r=0; d_r<dimensionality_reduced; ++d_r)
//    {
//        for (int d=0; d<dimensionality_BOW; ++d)
//            grad[d_r*dimensionality_BOW + d] = grad_mat(d_r,d);
//    }

    std::cout << "Total loss = " << func << std::endl;
    //printf("%s\n",  grad.tostring(1).c_str());

}

double sigmoid(double x, double beta)
{
	return 1/(1 + exp(-beta*x));
}


double getPCCASimilarity(int i, int j, int y, int dimensionality_BOW, int dimensionality_reduced, double beta, const std::vector<double>& bow_appearance_matrix, const double *weights)
{
    double squared_norm;
    return getPCCASimilarity(squared_norm,i,j,y,dimensionality_BOW,dimensionality_reduced,beta,bow_appearance_matrix,weights);
}

double getPCCASimilarity(double &squared_norm, int i, int j, int y, int dimensionality_BOW, int dimensionality_reduced, double beta, const std::vector<double>& bow_appearance_matrix, const double *weights) // y is 1 or -1
{
    squared_norm = 0;
    for (int d_r=0; d_r<dimensionality_reduced; ++d_r)
    {
        double coordinate=0;
        for (int d=0; d<dimensionality_BOW; ++d)
        {
            coordinate += weights[d_r*dimensionality_BOW + d]*(bow_appearance_matrix[i*dimensionality_BOW+d] - bow_appearance_matrix[j*dimensionality_BOW+d]);
        }
        squared_norm += std::pow<double>(coordinate,2);
    }

    return sigmoid(((double)y)*(1-squared_norm),beta);
}

double getPCCASimilarity(double &squared_norm, int y, double beta, cv::Mat& bow_appearance_vector_i, cv::Mat& bow_appearance_vector_j, cv::Mat& weights) // y is 1 or -1
{
    squared_norm = cv::norm(weights*(bow_appearance_vector_i - bow_appearance_vector_j), cv::NORM_L2SQR);
    return sigmoid(((double)y)*(1-squared_norm),beta);
}

double getPCCASimilarity(double &squared_norm, int y, double beta, Eigen::SparseVector<double>& bow_appearance_vector_i, Eigen::SparseVector<double>& bow_appearance_vector_j, Eigen::SparseMatrix<double,Eigen::RowMajor>& weights) // y is 1 or -1
{
    Eigen::VectorXd product = weights*(bow_appearance_vector_i - bow_appearance_vector_j);
    squared_norm = product.squaredNorm();
    return sigmoid(((double)y)*(1-squared_norm),beta);
}







// New one based on Eigen

double getPCCASimilarity_Linear(int i, int j, int y, double beta, Eigen::VectorXd& preallocated_product, const Eigen::VectorXd& diagonal_weights, const Eigen::MatrixXd& bow_appearance_matrix, double intercept)
{
    int d = diagonal_weights.rows() - 1; // we remove the intercept
    preallocated_product = bow_appearance_matrix.block(0,i,d,1) - bow_appearance_matrix.block(0,j,d,1);
    double linear_combination = preallocated_product.transpose()*(diagonal_weights.segment(0,d).asDiagonal())*preallocated_product;
    return sigmoid(((double)y)*(intercept-linear_combination),beta);
}

double getPCCASimilarity_Diagonal(int i, int j, int y, double beta, Eigen::VectorXd& preallocated_product, const Eigen::VectorXd& diagonal_weights, const Eigen::MatrixXd& bow_appearance_matrix, double intercept)
{
    int d = diagonal_weights.rows();
    preallocated_product = bow_appearance_matrix.block(0,i,d,1) - bow_appearance_matrix.block(0,j,d,1);
    double score = (preallocated_product.transpose())*(diagonal_weights.asDiagonal())*preallocated_product;
    return sigmoid(((double)y)*(intercept-score),beta);
}

double getPCCASimilarity(int i, int j, int y, double beta, Eigen::VectorXd& preallocated_product, const Eigen::MatrixXd& W, const Eigen::MatrixXd& bow_appearance_matrix, double intercept, double coefficient)
{
    int d = W.rows();
    preallocated_product = W*bow_appearance_matrix.block(0,i,d,1) - W*bow_appearance_matrix.block(0,j,d,1);
    double squared_norm = preallocated_product.squaredNorm();
    return coefficient*sigmoid(((double)y)*(intercept-squared_norm),beta);
}





double countNbCommonPoints(const Eigen::VectorXd& x_i, const Eigen::VectorXd& x_j)
{
    int nb_descriptors = x_i.rows();
    double count(0);
    for (int d=0; d<nb_descriptors; ++d)
        count += std::min<double>(x_i(d),x_j(d));
    return (count/((double)100));
}

double computeCosineSimilarity(const Eigen::VectorXd& x_i, const Eigen::VectorXd& x_j)
{
    return x_i.dot(x_j);
}



