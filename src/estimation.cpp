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
  PoseMeasurement z = measurement;
  if (!config_.latency_compensation) {
    z.stamp = std::max(time_, now);  // ablation : on fait comme si la mesure était actuelle
  }
  if (!initialized_ || health(now) == EkfHealth::Lost) {
    initialize(z);
    propagate(now);
    history_.push_back({time_, x_, P_});
    last_accept_time_ = now;
    return true;
  }

  bool ok = false;
  if (z.stamp >= time_) {
    propagate(z.stamp);
    ok = correct(z);
    history_.push_back({time_, x_, P_});
  } else {
    // Mesure du passé : on revient au dernier instantané avant z.stamp...
    auto it = std::find_if(history_.rbegin(), history_.rend(),
                           [&](const Snapshot& s) { return s.t <= z.stamp; });
    if (it == history_.rend()) {
      ++rejected_;  // plus ancienne que le tampon
      return false;
    }
    const Time end_time = time_;
    std::vector<Time> replay_times;
    for (auto later = it.base(); later != history_.end(); ++later) replay_times.push_back(later->t);
    x_ = it->x;
    P_ = it->P;
    time_ = it->t;
    history_.erase(it.base(), history_.end());
    // ... on corrige à l'instant de la mesure, puis on rejoue les entrées jusqu'à maintenant.
    propagate(z.stamp);
    ok = correct(z);
    for (Time t : replay_times) {
      propagate(t);
      history_.push_back({time_, x_, P_});
    }
    propagate(end_time);
  }

  if (ok) {
    last_accept_time_ = now;
    consecutive_rejections_ = 0;
  } else {
    ++rejected_;
    // Beaucoup de rejets d'affilée : le filtre s'est sans doute perdu, on repart de la mesure.
    if (++consecutive_rejections_ >= config_.max_rejections_before_reset) {
      initialized_ = false;
      return update(measurement, now);
    }
  }
  return ok;
}

EkfHealth Ekf::health(Time now) const {
  if (!initialized_) return EkfHealth::Uninitialized;
  return now - last_accept_time_ > config_.lost_timeout ? EkfHealth::Lost : EkfHealth::Ok;
}

void Ekf::trimBuffers() {
  const Time limit = time_ - config_.buffer_duration;
  while (history_.size() > 1 && history_.front().t < limit) history_.pop_front();
  // On garde la dernière entrée antérieure à la limite : elle est encore active.
  while (inputs_.size() > 1 && inputs_[1].t_effective < limit) inputs_.pop_front();
}

BatteryMonitorConfig loadBatteryMonitorConfig(const ConfigNode& section) {
  BatteryMonitorConfig cfg;
  cfg.ratio = section.getDouble("ratio", 0.1, 1.0);
  cfg.stall_ratio = section.getDouble("stall_ratio", 0.0, cfg.ratio);
  cfg.duration = section.getDouble("duration", 0.1, 60.0);
  cfg.window = section.getDouble("window", 0.1, 5.0);
  cfg.min_speed = section.getDouble("min_speed", 0.0, 1.0);
  cfg.max_turn_rate = section.getDouble("max_turn_rate", 0.0, 10.0);
  return cfg;
}

BatteryMonitor::BatteryMonitor(const BatteryMonitorConfig& config) : config_(config) {}

void BatteryMonitor::update(Time t, const Pose2D& estimate,
                            const Twist2D& applied) {
  const double dt = samples_.empty() ? 0.0 : t - samples_.back().t;
  samples_.push_back({t, estimate.x, estimate.y, applied});
  while (samples_.size() > 2 && samples_[1].t <= t - config_.window) samples_.pop_front();
  const Sample& first = samples_.front();
  const double span = t - first.t;
  if (span < config_.window * 0.9) return;

  // La fenêtre ne compte que si toutes les consignes y sont "ligne droite, assez vite".
  double expected = 0.0;
  for (const Sample& s : samples_) {
    if (s.cmd.v < config_.min_speed || std::abs(s.cmd.omega) > config_.max_turn_rate) return;
    expected += s.cmd.v / samples_.size();
  }
  const double measured = std::hypot(estimate.x - first.x, estimate.y - first.y) / span;
  if (measured < config_.stall_ratio * expected) {
    return;  // bloqué (collision) : ni preuve de pile faible, ni de pile bonne
  }
  if (measured < config_.ratio * expected) {
    low_time_ += dt;
  } else {
    low_time_ = 0.0;
  }
}
