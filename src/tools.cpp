#include <iostream>
#include "tools.h"
#include <math.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    
    if (estimations.size() != ground_truth.size()) {
        cout << "Estimations and ground_truth must be of the same size!" << endl;
    }
    
    if (estimations.size() == 0) {
        cout << "There must be at least 1 element in estimations and ground_truth " << endl;
    }
    
    VectorXd diff;
    VectorXd diff2;
    //accumulate squared residuals
    for(int i=0; i < estimations.size(); ++i){
        diff = estimations[i] - ground_truth[i];
        diff2 = (diff.array() * diff.array());
        rmse += diff2;
    }
    
    // ... your code here
    rmse = rmse.array() / estimations.size();
    
    //calculate the squared root
    rmse = rmse.array().sqrt();
    
    //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    
    MatrixXd Hj(3,4);
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);
    float r2 = px * px + py * py;
    
    
    //check division by zero
    if (fabs(r2) < 0.001) {
        cout << "division by zero!" << endl;
        Hj << 0, 0, 0, 0,
              0, 0, 0, 0,
              0, 0, 0, 0;
        return Hj;
    } else {
        //compute the Jacobian matrix
        float r = sqrt(r2);
        float r32 = r * r2;
        float vp = vx * py - vy * px;
        float vpa = px * vy - py * vx;
        
        float el00 = px / r;
        float el01 = py / r;
        Hj <<       el00,       el01,    0,    0,
                -(py/r2),    (px/r2),    0,    0,
               py*vp/r32, px*vpa/r32, el00, el01;
    }
    
    
    return Hj;
}
