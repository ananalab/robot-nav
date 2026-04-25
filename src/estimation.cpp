#include "estimation.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

#include "core.hpp"

EkfConfig loadEkfConfig(const ConfigNode& root, const RobotConfig& robot) {
  EkfConfig cfg;
  cfg.tau_v = robot.tau;
  cfg.tau_omega = robot.tau;
  cfg.cmd_delay = root.getDouble("cmd_delay", 0.0, 1.0);
  const ConfigNode q = root.child("process_noise");
  cfg.process_noise << q.getDouble("xy", 0.0, 1.0), q.getDouble("xy", 0.0, 1.0),
      q.getDouble("theta", 0.0, 10.0), q.getDouble("v", 0.0, 10.0), q.getDouble("omega", 0.0, 100.0);
  const ConfigNode s = root.child("initial_std");
  cfg.initial_std << s.getDouble("xy", 1e-4, 1.0), s.getDouble("xy", 1e-4, 1.0),
      degToRad(s.getDouble("theta_deg", 0.01, 180.0)), s.getDouble("v", 1e-4, 2.0),
      s.getDouble("omega", 1e-4, 20.0);
  cfg.gate_chi2 = root.getDouble("gate_chi2", 0.1, 1000.0);
  cfg.buffer_duration = root.getDouble("buffer_duration", 0.1, 10.0);
  cfg.lost_timeout = root.getDouble("lost_timeout", 0.05, 10.0);
  cfg.latency_compensation = root.getBool("latency_compensation");
  cfg.max_rejections_before_reset = root.getInt("max_rejections_before_reset", 1, 1000);
  return cfg;
}

const char* toString(EkfHealth h) {
  switch (h) {
    case EkfHealth::Uninitialized: return "INIT";
    case EkfHealth::Ok: return "OK";
    case EkfHealth::Lost: return "LOST";
  }
  return "?";
}

Ekf::Ekf(const EkfConfig& config) : config_(config) {}

void Ekf::addInput(Time t, const Twist2D& cmd) {
  inputs_.push_back({t + config_.cmd_delay, cmd});
}

Twist2D Ekf::inputAt(Time t) const {
  Twist2D u;
  for (const Input& in : inputs_) {
    if (in.t_effective > t + 1e-12) break;
    u = in.cmd;
  }
  return u;
}

void Ekf::propagate(Time t) {
  while (time_ < t - 1e-12) {
    const double dt = std::min(config_.max_step, t - time_);
    const Twist2D u = inputAt(time_);
    const Matrix5d F = motionJacobian(x_, dt, config_.tau_v, config_.tau_omega);
    x_ = motionModel(x_, u, dt, config_.tau_v, config_.tau_omega);
    // P = F P F^T + Q dt  (bruit de processus proportionnel au temps écoulé)
    P_ = F * P_ * F.transpose();
    P_.diagonal() += config_.process_noise * dt;
    time_ += dt;
  }
}

void Ekf::predict(Time t) {
  if (!initialized_) {
    time_ = t;
    return;
  }
  propagate(t);
  history_.push_back({time_, x_, P_});
  trimBuffers();
}

void Ekf::initialize(const PoseMeasurement& z) {
  x_ << z.pose.x, z.pose.y, z.pose.theta, 0.0, 0.0;
  P_ = config_.initial_std.cwiseProduct(config_.initial_std).asDiagonal();
  time_ = z.stamp;
  history_.clear();
  initialized_ = true;
  consecutive_rejections_ = 0;
}

bool Ekf::correct(const PoseMeasurement& z) {
  // Innovation y = z - H x, avec H = [I3 0] ; l'angle est renormalisé.
  Eigen::Vector3d y(z.pose.x - x_(0), z.pose.y - x_(1), wrapAngle(z.pose.theta - x_(2)));
  const Eigen::Matrix3d S = P_.topLeftCorner<3, 3>() + z.cov;
  const Eigen::Matrix3d S_inv = S.inverse();
  // Gating de Mahalanobis : une mesure trop improbable est rejetée.
  if (y.transpose() * S_inv * y > config_.gate_chi2) {
    return false;
  }
  // Gain K = P H^T S^-1, puis mise à jour en forme de Joseph (P reste symétrique positive).
  const Eigen::Matrix<double, 5, 3> K = P_.leftCols<3>() * S_inv;
  x_ += K * y;
  x_(2) = wrapAngle(x_(2));
  Eigen::Matrix<double, 5, 5> I_KH = Matrix5d::Identity();
  I_KH.leftCols<3>() -= K;
  P_ = I_KH * P_ * I_KH.transpose() + K * z.cov * K.transpose();
  return true;
}

bool Ekf::update(const PoseMeasurement& measurement, Time now) {
