#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.0;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false; //only set after invoking ProcessMeasurement
  n_x_ = 5;                //set state dimension
  n_aug_ = 7;              //set augmented dimension
  lambda_ = 3 - n_aug_;    //define spreading parameter
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);
  //create augmented covariance matrix
  P_aug_ = MatrixXd(n_aug_, n_aug_);
  //create sigma point matrix
  Xsig_aug_ = MatrixXd(n_aug_, 2*n_aug_+1);

  //set vector for weights
  weights_ = VectorXd(2*n_aug_ + 1);
  weights_(0) = lambda_/(lambda_ + n_aug_);
  for (int i = 1; i < 2*n_aug_+1; ++i)
  {
      weights_(i) = 0.5/(n_aug_ + lambda_);
  }
  NIS_radar_ = 0.0;
  NIS_lidar_ = 0.0;
}

UKF::~UKF() {}

bool UKF::InitializeFromMeasurement(const MeasurementPackage& meas_package)
{
    bool result = false;
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        double px = meas_package.raw_measurements_(0);
	double py = meas_package.raw_measurements_(1);

	x_ << px, py, 0, 0, 0;

	P_ << std_laspx_ * std_laspx_,                       0, 0, 0, 0,
	                            0, std_laspy_ * std_laspy_, 0, 0, 0,
				    0,                       0, 1, 0, 0,
				    0,                       0, 0, 1, 0,
				    0,                       0, 0, 0, 1;
        result = true;
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
       float rho     = meas_package.raw_measurements_(0);
       float phi     = meas_package.raw_measurements_(1);
       float rho_dot = meas_package.raw_measurements_(2);

       float px = rho*cos(phi); //pos x
       float py = rho*sin(phi); //pos y
       float vx = rho_dot * cos(phi);
       float vy = rho_dot * sin(phi);
       float v = sqrt(vx*vx + vy*vy);

       x_ << px, py, v, 0, 0;

       P_ << std_radr_*std_radr_, 0, 0, 0, 0,
             0, std_radr_*std_radr_, 0, 0, 0,
             0, 0, std_radrd_*std_radrd_, 0, 0,
	     0, 0, 0, std_radphi_*std_radphi_, 0,
	     0,0,0,0,1;
       result = true;
    }

    return result;
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_)
  {
      is_initialized_ = InitializeFromMeasurement(meas_package);
      time_us_ = meas_package.timestamp_;
      return;
  }

  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  //prediction step
  Prediction(dt);

  //update step
  if (meas_package.sensor_type_ == MeasurementPackage::LASER)
      UpdateLidar(meas_package);
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
      UpdateRadar(meas_package);
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.setZero(n_aug_);
  x_aug.head(5) = x_;

  //create augmented covariance matrix
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(5, 5) = P_;
  P_aug_(5,5) = std_a_ * std_a_;
  P_aug_(6,6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug_.llt().matrixL();

  //create sigma point matrix
  Xsig_aug_.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i)
  {
      Xsig_aug_.col(i+1)        = x_aug + sqrt(lambda_ + n_aug_)*L.col(i);
      Xsig_aug_.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_)*L.col(i);
  }
  //predict sigma points
  for (int i = 0; i < 2*n_aug_+1; ++i)
  {
      //extract values for better readability
      double p_x      = Xsig_aug_(0, i);
      double p_y      = Xsig_aug_(1, i);
      double v        = Xsig_aug_(2, i);
      double yaw      = Xsig_aug_(3, i);
      double yawd     = Xsig_aug_(4, i);
      double nu_a     = Xsig_aug_(5, i);
      double nu_yawdd = Xsig_aug_(6, i);
      //predicted state values
      double px_p, py_p;

      //avoid division by zero
      if (fabs(yawd) > 1e-6)
      {
          px_p = p_x + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
	  py_p = p_y + v/yawd * (cos(yaw) - cos(yaw+yawd*delta_t));
      } else {
         px_p = p_x + v*delta_t*cos(yaw);
	 py_p = p_y + v*delta_t*sin(yaw);
      }
      double v_p = v;
      double yaw_p = yaw + yawd*delta_t;
      double yawd_p = yawd;
      //add noise
      px_p += 0.5*nu_a*delta_t*delta_t*cos(yaw);
      py_p += 0.5*nu_a*delta_t*delta_t*sin(yaw);
      v_p += nu_a*delta_t;

      yaw_p += 0.5*nu_yawdd*delta_t*delta_t;
      yawd_p += nu_yawdd*delta_t;

      Xsig_pred_(0, i) = px_p;
      Xsig_pred_(1, i) = py_p;
      Xsig_pred_(2, i) = v_p;
      Xsig_pred_(3, i) = yaw_p;
      Xsig_pred_(4, i) = yawd_p;
  }
  //pedict mean and covariance

  //predict mean
  x_.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; ++i)
  {  //iterate over sigma points
     x_ += weights_(i)*Xsig_pred_.col(i);
  }

  //predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1;++i)
  { //iterate over sigma points
      VectorXd x_diff = Xsig_pred_.col(i) - x_;
      //angle normalization
      while (x_diff(3) > M_PI) x_diff(3) -= 2*M_PI;
      while (x_diff(3) < -M_PI) x_diff(3) += 2*M_PI;

      P_ += weights_(i)*x_diff*x_diff.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  //use standard kalman filter as lidar are linear
  int n_z = 2;

  MatrixXd H = MatrixXd(n_z, n_x_);
  H.setZero(n_z, n_x_);
  H(0,0) = H(1,1) = 1;

  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_*std_laspx_, 0,
       0,                     std_laspy_*std_laspy_;

  VectorXd z_pred = H * x_; 
  VectorXd y = meas_package.raw_measurements_ - z_pred;

  MatrixXd S = H * P_ * H.transpose() + R;
  MatrixXd K = P_ * H.transpose() * S.inverse();

  //new estimate
  x_ = x_ + (K*y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K*H)*P_;

  NIS_lidar_ = y.transpose()*S.inverse()*y;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
   //create matrix for sigma points in measurement space
   //
   int n_z = 3; //set measurement dimension, radar can measure, r, phi and r_dot
   MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);
   VectorXd z_pred = VectorXd(n_z); //mean predicted measurement
   MatrixXd S = MatrixXd(n_z, n_z); //measurement covariance matrix S

   //transform sigma points into measurement space
   for (int i=0; i < 2*n_aug_+1;++i)
   {   //2n+1 sigma points
       //extract value for better readability
       double p_x = Xsig_pred_(0, i);
       double p_y = Xsig_pred_(1, i);
       double v   = Xsig_pred_(2, i);
       double yaw = Xsig_pred_(3, i);

       double v1 = cos(yaw)*v;
       double v2 = sin(yaw)*v;

       //measurement model
       Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                   //r
       Zsig(1, i) = atan2(p_y, p_x);                           //phi
       Zsig(2, i) = (p_x*v1 + p_y*v2)/sqrt(p_x*p_x + p_y*p_y); //r_dot
   }

   //predict mean
   z_pred.fill(0.0);
   for (int i = 0; i < 2*n_aug_+1; ++i)
   {
       z_pred += weights_(i)*Zsig.col(i);
   }
   //innovation covariance matrix S
   S.fill(0.0);
   for (int i = 0; i < 2*n_aug_+1; ++i)
   {
       //residual
       VectorXd zdiff = Zsig.col(i) - z_pred;
       //angle normalization
       while (zdiff(1) > M_PI)  zdiff(1) -= 2*M_PI;
       while (zdiff(1) < -M_PI) zdiff(1) += 2*M_PI;

       S = S + weights_(i) * zdiff * zdiff.transpose();
   }

   //add measurement noise covariance matrix
   MatrixXd R = MatrixXd(n_z, n_z);
   R << std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
	0, 0, std_radrd_*std_radrd_;
   S = S + R;

   //create matrix for cross correlation Tc
   MatrixXd Tc = MatrixXd(n_x_, n_z);
   Tc.fill(0.0);
   for (int i = 0; i < 2*n_aug_+1; ++i)
   {   //2n+1 sigma points

       //residual
       VectorXd z_diff = Zsig.col(i) - z_pred;
       //angle normailization
       while (z_diff(1) > M_PI)  z_diff(1) -=2*M_PI;
       while (z_diff(1) < -M_PI) z_diff(1) += 2*M_PI;

       //state difference
       VectorXd x_diff = Xsig_pred_.col(i) - x_;
       while (x_diff(3) > M_PI)  x_diff(3) -= 2*M_PI;
       while (x_diff(3) < -M_PI) x_diff(3) += 2*M_PI;

       Tc = Tc + weights_(i)*x_diff*z_diff.transpose();
   }
   //kalman gain, K
   MatrixXd K = Tc * S.inverse();
   //residual
   VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
   //angle normalization
   while (z_diff(1) > M_PI)  z_diff(1) -= 2*M_PI;
   while (z_diff(1) < -M_PI) z_diff(1) += 2*M_PI;

   //update state mean and covariance matrix
   x_ += K * z_diff;
   P_ -= K*S*K.transpose();

   //calculate radar NIS
   NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
