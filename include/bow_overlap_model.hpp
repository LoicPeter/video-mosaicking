#ifndef BOW_OVERLAP_MODEL_INCLUDED
#define BOW_OVERLAP_MODEL_INCLUDED
#include "graph.hpp"
//#include "stdafx.h"
#include "optimization.h"
#include <opencv2/core.hpp>
#include "structures_for_interactive_mosaicking.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>





class PCCAOverlapModel
{
    public:
    PCCAOverlapModel(int dimensionality_BOW, int dimensionality_reduced, double beta);
    PCCAOverlapModel(int dimensionality_BOW, double beta);
    virtual double _getOverlapProbability(int i, int j, std::vector<double>& bow_appearance_matrix);
    void _getOverlapProbabilities(cv::Mat& overlap_probabilities, std::vector<double>& bow_appearance_matrix);
    void inferWeights(std::vector<LabeledImagePair>& training_pairs, std::vector<double>& bow_appearance_matrix);
    virtual void inferWeights(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, std::vector<double>& bow_appearance_matrix);
    void displayWeights() const;

    private:
    std::vector<double> m_weights;
    double m_beta;
    int m_dimensionality_BOW;
    int m_dimensionality_reduced;

};

class ExternalOverlapModel
{
    public:
     virtual double _getOverlapProbability(int i, int j, const Eigen::MatrixXd& bow_appearance_matrix) = 0;
     void _getOverlapProbabilities(cv::Mat& overlap_probabilities, const Eigen::MatrixXd& bow_appearance_matrix);
     virtual void train(std::vector<LabeledImagePair>& training_pairs, Eigen::MatrixXd& bow_appearance_matrix);
     virtual void train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix) = 0;
};

class TrivialExternalModel : public ExternalOverlapModel
{
    public:
    double _getOverlapProbability(int i, int j, const Eigen::MatrixXd& bow_appearance_matrix) {return 1;};
    void train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix) {};
};

class DiagonalPCCAOverlapModel :  public ExternalOverlapModel
{
    public:
    DiagonalPCCAOverlapModel(int dimensionality_BOW, double beta, double intercept = 1);
    double _getOverlapProbability(int i, int j, const Eigen::MatrixXd& bow_appearance_matrix);
    void train(std::vector<LabeledImagePair>& training_pairs, Eigen::MatrixXd& bow_appearance_matrix);
    void train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix);
    void train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, std::vector<double>& importance_weights, Eigen::MatrixXd& bow_appearance_matrix);
    
    private:
    Eigen::VectorXd m_weights;
    double m_beta;
    Eigen::VectorXd m_preallocated_product;
    double m_intercept;
};

class NonIncreasingDiagonalPCCAOverlapModel :  public ExternalOverlapModel
{
    public:
    NonIncreasingDiagonalPCCAOverlapModel(int dimensionality_BOW, double beta, double intercept = 1);
    double _getOverlapProbability(int i, int j, const Eigen::MatrixXd& bow_appearance_matrix);
    void train(std::vector<LabeledImagePair>& training_pairs, Eigen::MatrixXd& bow_appearance_matrix);
    void train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix);
    void train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, std::vector<double>& importance_weights, Eigen::MatrixXd& bow_appearance_matrix);
    
    private:
    Eigen::VectorXd m_weights;
    double m_beta;
    Eigen::VectorXd m_preallocated_product;
};

class LinearPCCAOverlapModel :  public ExternalOverlapModel
{
    public:
    LinearPCCAOverlapModel(int dimensionality_BOW, double beta);
    double _getOverlapProbability(int i, int j, const Eigen::MatrixXd& bow_appearance_matrix);
    void train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix);
    
    private:
    Eigen::VectorXd m_weights;
    double m_beta;
    Eigen::VectorXd m_preallocated_product;
};

class FullPCCAOverlapModel :  public ExternalOverlapModel
{
    public:
    FullPCCAOverlapModel();
    FullPCCAOverlapModel(int dimensionality_reduced, int dimensionality_BOW, double beta);
    double _getOverlapProbability(int i, int j, const Eigen::MatrixXd& bow_appearance_matrix);
    virtual void train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix);
    
    protected:
    Eigen::MatrixXd m_weights;
    double m_beta;
    Eigen::VectorXd m_preallocated_product;
};

class PositiveDefinitePCCAOverlapModel :  public ExternalOverlapModel
{
    public:
    PositiveDefinitePCCAOverlapModel(int dimensionality_BOW, double beta, double intercept = 1);
    double _getOverlapProbability(int i, int j, const Eigen::MatrixXd& bow_appearance_matrix);
    void train(std::vector<LabeledImagePair>& training_pairs, Eigen::MatrixXd& bow_appearance_matrix);
    void train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix);
    void train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, std::vector<double>& importance_weights, Eigen::MatrixXd& bow_appearance_matrix);
    void setWeightMatrix(const double* parameters);
    void getParameters(double* parameters) const;
    
    private:
    Eigen::MatrixXd m_L;
    double m_beta;
    Eigen::VectorXd m_preallocated_product;
    double m_intercept;
};

class LowRankPCCAOverlapModel :  public FullPCCAOverlapModel
{
    public:
    LowRankPCCAOverlapModel(int dimensionality_reduced, int dimensionality_BOW, double beta);
    void train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix);
    void setWeightMatrix(const double* parameters);
    void getParameters(double* parameters) const;
    
    private:
    Eigen::MatrixXd m_L; // L is of size dimensionality_reduced x dimensionality_BOW
};

void setLMatrix(Eigen::MatrixXd& L, const double* parameters);

void create_low_triangular_canonical_basis(std::vector<Eigen::SparseMatrix<double>>& B_d, int d);

void create_canonical_basis_col_major(std::vector<Eigen::SparseMatrix<double>>& B_d, int rows, int cols);

void compute_loss_function_pcca_positive_definite(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr);


class DifferenceBOWOverlapModel : public ExternalOverlapModel
{
   public:
   DifferenceBOWOverlapModel(int dimensionality);
   double _getOverlapProbability(int i, int j, const Eigen::MatrixXd& bow_appearance_matrix);
   void train(std::vector<std::pair<int,int>>& training_pairs, std::vector<double>& training_labels, Eigen::MatrixXd& bow_appearance_matrix);

   private:
   Eigen::VectorXd m_weights;

};



void compute_loss_function_inference_difference_bow_weights_grad(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr);

void compute_loss_function_inference_difference_bow_weights(const alglib::real_1d_array &x, double &func, void *ptr); // x = weights

double getDifferenceBOWSimilarity(double& score_in_signoid, int i, int j, const Eigen::MatrixXd& bow_appearance_matrix, Eigen::VectorXd& weights);

struct StructureForInferencePCCAWeights
{
    int dimensionality_BOW;
    int dimensionality_reduced;
    double beta;
    std::vector<std::pair<int,int>> *training_pairs;
    std::vector<double> *training_labels;
    //double *bow_appearance_matrix = &((*(struct_inf_weights->bow_appearance_matrix))[0]);
  //  double* precomputed_Cn; // precomputed_Cn[k*dimensionality_BOW_squared+(m*dimensionality_BOW)+n] = entry at the roow m, column n of the matrix Cn used in gradient

  // With OpenCV
    std::vector<cv::Mat> *precomputed_Cn;
    std::vector<cv::Mat> *bow_appearance;

  // With Eigen
   // std::vector<Eigen::MatrixXd> *precomputed_Cn;
  //  std::vector<Eigen::SparseVector<double>> *bow_appearance;

    std::vector<double> *bow_appearance_matrix;
    std::vector<double> balancing_weights;
};

struct StructureForInferenceDifferenceBOWWeights
{
   int dimensionality;
   std::vector<std::pair<int,int>> *training_pairs;
   std::vector<double> *training_labels;
   Eigen::MatrixXd *bow_appearance_matrix;
   std::vector<double> balancing_weights;
   double lambda;
};

struct StructureForTrainingPCCADiagonal
{
    int dimensionality_BOW;
    double beta;
    std::vector<double> *training_labels;
    Eigen::MatrixXd *precomputed_Delta_n;
    std::vector<double> balancing_weights;
    std::vector<double> importance_weights;
    Eigen::VectorXd current_weights; // for the non increasing strategy
};

struct StructureForTrainingPCCAFull
{
    int dimensionality_BOW;
    int dimensionality_reduced;
    double beta;
    std::vector<double> *training_labels;
    Eigen::MatrixXd *precomputed_Delta_n;
    Eigen::MatrixXd *precomputed_Delta_n_tr;
    std::vector<double> balancing_weights;
    std::vector<double> importance_weights;
    std::vector<Eigen::SparseMatrix<double>> *B_d;
    Eigen::MatrixXd *precomputed_L;
};

void compute_loss_function_pcca_linear(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr);

void compute_loss_function_pcca_diagonal(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr);

void compute_loss_function_pcca_diagonal_non_increasing(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr);

void compute_loss_function_pcca_full(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr) ;

void compute_loss_function_inference_pcca_weights_grad(const alglib::real_1d_array &x, double &func,  alglib::real_1d_array &grad, void *ptr);

double getPCCASimilarity(int i, int j, int y, int dimensionality_BOW, int dimensionality_reduced, double beta, const std::vector<double>& bow_appearance_matrix, const double *weights);

double getPCCASimilarity(double &squared_norm, int i, int j, int y, int dimensionality_BOW, int dimensionality_reduced, double beta, const std::vector<double>& bow_appearance_matrix, const double *weights); // y is 1 or -1

double getPCCASimilarity(double &squared_norm, int y, double beta, cv::Mat& bow_appearance_vector_i, cv::Mat& bow_appearance_vector_j, cv::Mat& weights);

double getPCCASimilarity(double &squared_norm, int y, double beta, Eigen::SparseVector<double>& bow_appearance_vector_i, Eigen::SparseVector<double>& bow_appearance_vector_j, Eigen::SparseMatrix<double,Eigen::RowMajor>& weights); // y is 1 or -1

double sigmoid(double x, double beta = 1);

double getPCCASimilarity_Linear(int i, int j, int y, double beta, Eigen::VectorXd& preallocated_product, const Eigen::VectorXd& diagonal_weights, const Eigen::MatrixXd& bow_appearance_matrix, double intercept = 1);

double getPCCASimilarity_Diagonal(int i, int j, int y, double beta, Eigen::VectorXd& preallocated_product, const Eigen::VectorXd& diagonal_weights, const Eigen::MatrixXd& bow_appearance_matrix, double intercept = 1);

double getPCCASimilarity(int i, int j, int y, double beta, Eigen::VectorXd& preallocated_product, const Eigen::MatrixXd& W, const Eigen::MatrixXd& bow_appearance_matrix, double intercept = 1, double coefficient = 1);

void create_canonical_basis(std::vector<Eigen::SparseMatrix<double>>& B_d, int nb_rows, int nb_cols);



// Functions to test different types of similarity matrices

double countNbCommonPoints(const Eigen::VectorXd& x_i, const Eigen::VectorXd& x_j);

double computeCosineSimilarity(const Eigen::VectorXd& x_i, const Eigen::VectorXd& x_j);







#endif // GRAPH_INCLUDED
