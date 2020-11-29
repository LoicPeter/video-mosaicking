#ifndef UPDATABLE_CHOLESKY_H_INCLUDED
#define UPDATABLE_CHOLESKY_H_INCLUDED

#include <Eigen/Cholesky>

class UpdatableCholesky : public Eigen::LLT<Eigen::MatrixXd>
{
    public:
    UpdatableCholesky();
    UpdatableCholesky(const Eigen::MatrixXd& P);
    void update(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& S);
    void update(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& S, const Eigen::MatrixXd& P);
};



#endif
