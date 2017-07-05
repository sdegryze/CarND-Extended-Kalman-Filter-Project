#include "kalman_filter.h"
#include <iostream>
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
  long x_size = x_.size();
  I_ = MatrixXd::Identity(x_size, x_size);
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;
  
  //new estimate
  x_ = x_ + (K * y);
  P_ = (I_ - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  
  //pre-compute a set of terms to avoid repeated calculation
  float c1 = px * px + py * py;
  
  VectorXd h_ = VectorXd(3);
  if (fabs(c1) < 0.001) {
    std::cout << "division by zero!" << std::endl;
    h_ << 0, 0, 0;
  } else {
    float c2 = sqrt(c1);
    
    // note that atain2 automatically returns a value between -PI and +PI
    float phi = atan2(py, px);
    
    h_ << c2, phi, (px * vx + py * vy) / c2;
  }
  
  // measurement residual
  VectorXd y = z - h_;
  
  // ensure y(1) (aka rho) is between -PI and +PI by adding/subtracting 2 * PI
  while (y(1) > M_PI) y(1) -= 2 * M_PI;
  while (y(1) <= -M_PI) y(1) += 2 * M_PI;
  
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;
  
  //new estimate
  x_ = x_ + (K * y);
  P_ = (I_ - K * H_) * P_;
}
