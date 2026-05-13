#include "control.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core.hpp"

const char* toString(ControlState state) {
  switch (state) {
    case ControlState::Rotating: return "ROTATING";
    case ControlState::Tracking: return "TRACKING";
    case ControlState::Arrived: return "ARRIVED";
    case ControlState::Blocked: return "BLOCKED";
    case ControlState::Stopped: return "STOPPED";
  }
  return "?";
}

ControlLimits loadControlLimits(const ConfigNode& section) {
  ControlLimits l;
  l.v_max = section.getDouble("v_max", 0.01, 1.0);
  l.omega_max = section.getDouble("omega_max", 0.1, 20.0);
  l.accel_max = section.getDouble("accel_max", 0.01, 10.0);
  l.alpha_max = section.getDouble("alpha_max", 0.1, 100.0);
  return l;
}

RotationConfig loadRotationConfig(const ConfigNode& root) {
  RotationConfig r;
  const ConfigNode rot = root.child("rotation");
  r.start_threshold = degToRad(rot.getDouble("start_threshold_deg", 5.0, 180.0));
  r.stop_threshold = degToRad(rot.getDouble("stop_threshold_deg", 1.0, 90.0));
  r.gain = rot.getDouble("gain", 0.1, 20.0);
  r.omega_max = rot.getDouble("omega_max", 0.1, 20.0);
  r.lag = rot.getDouble("lag", 0.0, 1.0);
  const ConfigNode goal = root.child("goal");
  r.goal_tolerance = goal.getDouble("position_tolerance", 0.002, 0.2);
  r.heading_tolerance = degToRad(goal.getDouble("heading_tolerance_deg", 0.5, 45.0));
  r.hold_gain = goal.getDouble("hold_gain", 0.0, 50.0);
  return r;
}

Twist2D rotateTowards(double target_heading, const Pose2D& est,
                            const Twist2D& vel, const RotationConfig& rotation) {
  const double predicted = est.theta + vel.omega * rotation.lag;
  const double error = wrapAngle(target_heading - predicted);
  return {0.0, std::clamp(rotation.gain * error, -rotation.omega_max, rotation.omega_max)};
}

Twist2D finalAlignment(const Pose2D& est, const Twist2D& vel,
                             const Path& path, const RotationConfig& rotation,
                             const ControlLimits& limits, bool& arrived) {
  arrived = false;
  if (!path.goal_heading) {
    arrived = true;
    return {};
  }
  const double predicted = est.theta + vel.omega * rotation.lag;
  // On coupe à la moitié de la tolérance : le robot continue un peu sur son élan.
  if (std::abs(wrapAngle(*path.goal_heading - predicted)) < rotation.heading_tolerance / 2.0) {
    arrived = true;
    return {};
  }
  Twist2D cmd = rotateTowards(*path.goal_heading, est, vel, rotation);
  // Maintien de position : deux moteurs jamais identiques font dériver la rotation
  // sur place. On corrige la composante de l'erreur selon l'axe du robot :
  //   v = hold_gain * ((but - position) . (cos theta, sin theta))
  const PathPoint& goal = path.points.back();
  const double forward_error =
      (goal.x - est.x) * std::cos(est.theta) + (goal.y - est.y) * std::sin(est.theta);
  cmd.v = std::clamp(rotation.hold_gain * forward_error, -limits.v_max, limits.v_max);
  return cmd;
}

double rateLimit(double target, double previous, double rate, double dt) {
  const double max_step = rate * dt;
  return std::clamp(target, previous - max_step, previous + max_step);
}

std::size_t closestPointForward(const Path& path, const Pose2D& pose,
                                std::size_t from, double window) {
  std::size_t best = from;
  double best_d2 = 1e18;
  const double s_limit = path.points[from].s + window;
  for (std::size_t i = from; i < path.points.size() && path.points[i].s <= s_limit; ++i) {
    const double dx = path.points[i].x - pose.x;
    const double dy = path.points[i].y - pose.y;
    const double d2 = dx * dx + dy * dy;
    if (d2 < best_d2) {
      best_d2 = d2;
      best = i;
    }
  }
  return best;
}

double lateralError(const Path& path, std::size_t i, const Pose2D& pose) {
  // Projection du vecteur (point du chemin -> robot) sur la normale gauche du chemin.
  const PathPoint& p = path.points[i];
  return -std::sin(p.theta) * (pose.x - p.x) + std::cos(p.theta) * (pose.y - p.y);
}

WheelMapperConfig loadWheelMapperConfig(const RobotConfig& robot,
                                        const ConfigNode& section) {
  WheelMapperConfig cfg;
  cfg.wheel_base = robot.wheel_base;
  cfg.left = robot.left;
  cfg.right = robot.right;
  cfg.min_wheel_speed = section.getDouble("min_wheel_speed", 0.0, 0.3);
  cfg.start_speed = section.getDouble("start_speed", 0.0, cfg.min_wheel_speed);
  cfg.stop_speed = section.getDouble("stop_speed", 0.0, cfg.start_speed);
  cfg.min_command = section.getDoubleOr("min_command", 0.0, 0.0, 1.0);
  cfg.kick_command = section.getDoubleOr("kick_command", 0.0, 0.0, 1.0);
  cfg.kick_cycles = static_cast<int>(section.getDoubleOr("kick_cycles", 0.0, 0.0, 100.0));
  return cfg;
}

WheelMapper::WheelMapper(const WheelMapperConfig& config) : config_(config) {}

double WheelMapper::applyDeadzone(double speed, bool& active) const {
  // Sous la zone morte, la roue ne bouge pas : une petite consigne non nulle
  // laisserait le robot bloqué juste avant le but. On l'arrondit donc soit à 0,
  // soit à la vitesse minimale, avec deux seuils différents (hystérésis) pour
  // ne pas osciller entre "arrêt" et "marche" d'un cycle à l'autre.
  const double magnitude = std::abs(speed);
  const double threshold = active ? config_.stop_speed : config_.start_speed;
  if (magnitude < threshold || magnitude == 0.0) {
    active = false;
    return 0.0;
  }
  active = true;
  const double raised = std::max(magnitude, config_.min_wheel_speed);
  return speed > 0.0 ? raised : -raised;
}

double WheelMapper::applyFriction(double u, bool started, int& kick) const {
  if (u == 0.0) {
    kick = 0;
    return 0.0;
  }
  if (started) kick = config_.kick_cycles;
  double floor = config_.min_command;
  if (kick > 0) {
    floor = std::max(floor, config_.kick_command);
    --kick;
  }
  return u > 0.0 ? std::max(u, floor) : std::min(u, -floor);
}

WheelCmd WheelMapper::map(const Twist2D& twist) {
  // Cinématique inverse : vL = v - omega * b / 2,  vR = v + omega * b / 2
  double left = twist.v - twist.omega * config_.wheel_base / 2.0;
  double right = twist.v + twist.omega * config_.wheel_base / 2.0;

  // Saturation : si une roue dépasse sa vitesse max, on réduit les deux du même
  // facteur, ce qui garde la courbure omega / v (donc la trajectoire).
  const double ratio = std::max(std::abs(left) / maxWheelSpeed(config_.left),
                                std::abs(right) / maxWheelSpeed(config_.right));
  if (ratio > 1.0) {
    left /= ratio;
    right /= ratio;
  }

  const bool left_was_active = left_active_, right_was_active = right_active_;
  left = applyDeadzone(left, left_active_);
  right = applyDeadzone(right, right_active_);
  last_left_ = left;
  last_right_ = right;

  // Modèle moteur inverse : u = sign(v) * (u0 + |v| / k), puis frottements du vrai robot.
  return {applyFriction(commandFromWheelSpeed(config_.left, left), left_active_ && !left_was_active, left_kick_),
          applyFriction(commandFromWheelSpeed(config_.right, right), right_active_ && !right_was_active, right_kick_)};
}

PurePursuitConfig loadPurePursuitConfig(const ConfigNode& section) {
  PurePursuitConfig c;
  c.lookahead_base = section.getDouble("lookahead_base", 0.0, 1.0);
  c.lookahead_gain = section.getDouble("lookahead_gain", 0.0, 5.0);
  c.lookahead_min = section.getDouble("lookahead_min", 0.01, 1.0);
  c.lookahead_max = section.getDouble("lookahead_max", c.lookahead_min, 2.0);
  c.lateral_accel_max = section.getDouble("lateral_accel_max", 0.01, 10.0);
  c.decel = section.getDouble("decel", 0.01, 10.0);
  c.approach_gain = section.getDouble("approach_gain", 0.1, 50.0);
  c.stop_distance = section.getDouble("stop_distance", 0.001, 0.1);
  c.search_window = section.getDouble("search_window", 0.05, 5.0);
  return c;
}

PurePursuit::PurePursuit(const PurePursuitConfig& config, const ControlLimits& limits,
                         const RotationConfig& rotation)
    : config_(config), limits_(limits), rotation_(rotation) {}

void PurePursuit::reset() {
  state_ = ControlState::Rotating;
  index_ = 0;
  started_ = false;
  final_phase_ = false;
  debug_ = {};
}

Twist2D PurePursuit::compute(const Pose2D& est, const Twist2D& vel,
                                   const Path& path, Time t) {
  if (path.empty()) {
    state_ = ControlState::Stopped;
    last_cmd_ = {};
    return {};
  }
