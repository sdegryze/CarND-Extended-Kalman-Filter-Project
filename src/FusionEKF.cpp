#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;
  
  // Initialize and set the measurement function matrix for laser
  // This is the relation between a laser measurement and the state vector
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // Initialize and set measurement covariance matrix for laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0,      0.0225;

  // Initialize and set measurement covariance matrix for radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0,      0,
              0,    0.0009, 0,
              0,    0,      0.09;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    
    previous_timestamp_ = measurement_pack.timestamp_;

    // Vector to hold state (position and velocity)
    VectorXd x_in = VectorXd(4);
    
    // Define and set initial state covariance matrix
    MatrixXd P_in = MatrixXd(4, 4);
    P_in << 1, 0, 0,    0,
            0, 1, 0,    0,
            0, 0, 1000, 0,
            0, 0, 0,    1000;
    
    // Initialize and set the state transition function
    // This matrix represents simple laws of linear motion
    MatrixXd F_in = MatrixXd(4, 4);
    F_in << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;
    
    MatrixXd H_in; // measurement function matrix
    MatrixXd R_in; // covariance matrix for measurement noise
    MatrixXd Q_in = MatrixXd(4, 4); // covariance matrix for process noise

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state
      // assume 0 velocity
      float rho = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];

      float px = rho * cos(phi);
      float py = rho * sin(phi);
      
      x_in << px, py, 0, 0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state with position of first laser measurement but 0 velocity
      x_in << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }
      
    ekf_.Init(x_in, P_in, F_in, H_in, R_in, Q_in);
    cout << "EKF initialized with x_ = " << ekf_.x_(0) << " " << ekf_.x_(1) << " " << ekf_.x_(2) << " " << ekf_.x_(3) << " " << endl;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
    
  // compute the time elapsed between the current and previous measurements
  // note that dt must be expressed in seconds, hence the division by 10e6 from microseconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  
  //1. Modify the F matrix so that the new elapsed time is integrated
  ekf_.F_ << 1, 0, dt,  0,
             0, 1,  0, dt,
             0, 0,  1,  0,
             0, 0,  0,  1;
  //2. Set the process covariance matrix Q

  float dt2 = dt * dt;
  float dt3 = dt2 * dt / 2;
  float dt4 = dt2 * dt2 / 4;
  
  // Update the noise covariance matrix. Note that noise_ax = 9 and noise_ay = 9
  ekf_.Q_ << dt4 * noise_ax,              0, dt3 * noise_ax,              0,
             0             , dt4 * noise_ay,              0, dt3 * noise_ay,
             dt3 * noise_ax,              0, dt2 * noise_ax,              0,
             0             , dt3 * noise_ay,              0, dt2 * noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    cout << "RADAR with measurements: " << measurement_pack.raw_measurements_(0) << " " <<
      measurement_pack.raw_measurements_(1) << " " <<
      measurement_pack.raw_measurements_(2) << " " << endl;
    
    ekf_.R_ = R_radar_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else { // LASER
    cout << "LASER  with measurements: " << measurement_pack.raw_measurements_(0) << " " <<
      measurement_pack.raw_measurements_(1) << " " << endl;
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
