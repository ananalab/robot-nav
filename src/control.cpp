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
  if (state_ == ControlState::Arrived || state_ == ControlState::Stopped) {
    last_cmd_ = {};
    return {};
  }
  index_ = closestPointForward(path, est, index_, config_.search_window);
  debug_.path_index = index_;
  debug_.lateral_error = lateralError(path, index_, est);

  const PathPoint& goal = path.points.back();
  const double distance_to_goal = std::hypot(goal.x - est.x, goal.y - est.y);
  if (distance_to_goal < config_.stop_distance || final_phase_) {
    final_phase_ = true;  // une fois arrivé en position, on ne repart plus en suivi
    bool arrived = false;
    last_cmd_ = finalAlignment(est, vel, path, rotation_, limits_, arrived);
    state_ = arrived ? ControlState::Arrived : ControlState::Rotating;
    last_time_ = t;
    return last_cmd_;
  }

  // Pré-rotation : si le robot regarde trop loin de la tangente du chemin, il
  // tourne d'abord sur place (sauf tout près du but, où l'on veut juste finir).
  const double heading_error = wrapAngle(path.points[index_].theta - est.theta);
  const bool near_goal = distance_to_goal < config_.lookahead_min;
  if (!started_) {
    state_ = std::abs(heading_error) > rotation_.start_threshold ? ControlState::Rotating
                                                                  : ControlState::Tracking;
    started_ = true;
  } else if (state_ == ControlState::Tracking && !near_goal &&
             std::abs(heading_error) > rotation_.start_threshold) {
    state_ = ControlState::Rotating;
  } else if (state_ == ControlState::Rotating && std::abs(heading_error) < rotation_.stop_threshold) {
    state_ = ControlState::Tracking;
  }

  Twist2D cmd = state_ == ControlState::Rotating
                          ? rotateTowards(path.points[index_].theta, est, vel, rotation_)
                          : track(est, vel, path, distance_to_goal);
  last_cmd_ = applyLimits(cmd, t);
  return last_cmd_;
}

Twist2D PurePursuit::track(const Pose2D& est, const Twist2D& vel,
                                 const Path& path, double distance_to_goal) {
  // Distance d'anticipation : plus on va vite, plus on regarde loin.
  const double lookahead = std::clamp(config_.lookahead_base + config_.lookahead_gain * std::abs(vel.v),
                                      config_.lookahead_min, config_.lookahead_max);
  std::size_t target = index_;
  const double s_target = path.points[index_].s + lookahead;
  while (target + 1 < path.points.size() && path.points[target].s < s_target) {
    ++target;
  }
  const PathPoint& p = path.points[target];
  debug_.lookahead = {p.x, p.y};

  // Point visé exprimé dans le repère du robot.
  const Pose2D local = compose(inverse(est), Pose2D{p.x, p.y, 0.0});
  const double d = std::max(std::hypot(local.x, local.y), 1e-6);

  // But dépassé (il est derrière nous, tout près) : on recule au lieu de faire demi-tour.
  const bool reverse = local.x < 0.0 && distance_to_goal < config_.lookahead_min &&
                       target + 1 == path.points.size();
  const double alpha = reverse ? std::atan2(-local.y, -local.x) : std::atan2(local.y, local.x);
  const double kappa = 2.0 * std::sin(alpha) / d;

  // Vitesse : limite globale, accélération latérale dans les virages, freinage avant le but.
  double speed = std::min(limits_.v_max, std::sqrt(2.0 * config_.decel * distance_to_goal));
  speed = std::min(speed, config_.approach_gain * distance_to_goal);
  if (std::abs(kappa) > 1e-6) {
    speed = std::min(speed, std::sqrt(config_.lateral_accel_max / std::abs(kappa)));
  }
  return {reverse ? -speed : speed, speed * kappa};
}

Twist2D PurePursuit::applyLimits(Twist2D cmd, Time t) {
  // omega trop grand : on réduit v et omega ensemble (même courbure).
  if (std::abs(cmd.omega) > limits_.omega_max) {
    const double scale = limits_.omega_max / std::abs(cmd.omega);
    cmd.v *= scale;
    cmd.omega *= scale;
  }
  // Accélérations limitées ; les décélérations restent immédiates (sécurité).
  const double dt = last_time_ < 0.0 ? 0.0 : std::max(0.0, t - last_time_);
  last_time_ = t;
  if (std::abs(cmd.v) > std::abs(last_cmd_.v)) {
    cmd.v = rateLimit(cmd.v, last_cmd_.v, limits_.accel_max, dt);
  }
  if (std::abs(cmd.omega) > std::abs(last_cmd_.omega)) {
    cmd.omega = rateLimit(cmd.omega, last_cmd_.omega, limits_.alpha_max, dt);
  }
  return cmd;
}

namespace {
constexpr double kLethal = 254.0;
constexpr double kAbandoned = 1e6;  // coût d'un échantillon qui touche un obstacle
constexpr int kDrawnSamples = 20;   // trajectoires gardées pour l'affichage
constexpr double kLethalStepPenalty = 100.0;  // pas létal quand le départ l'est déjà
}  // namespace

MppiConfig loadMppiConfig(const ConfigNode& s, double tau, unsigned seed) {
  MppiConfig c;
  c.horizon = s.getDouble("horizon", 0.1, 10.0);
  c.dt = s.getDouble("dt", 0.005, 0.5);
  c.samples = s.getInt("samples", 8, 100000);
  c.sigma_v = s.getDouble("sigma_v", 0.0, 1.0);
  c.sigma_omega = s.getDouble("sigma_omega", 0.0, 10.0);
  c.lambda = s.getDouble("lambda", 1e-4, 1e4);
  c.w_path = s.getDouble("w_path", 0.0, 1e6);
  c.w_obstacle = s.getDouble("w_obstacle", 0.0, 1e6);
  c.w_dcmd = s.getDouble("w_dcmd", 0.0, 1e6);
  c.w_heading = s.getDouble("w_heading", 0.0, 1e6);
  c.w_speed = s.getDouble("w_speed", 0.0, 1e6);
  c.w_terminal = s.getDouble("w_terminal", 0.0, 1e6);
  c.v_min = s.getDouble("v_min", -1.0, 0.0);
  c.handover_distance = s.getDouble("handover_distance", 0.0, 1.0);
  c.tau_v = tau;
  c.tau_omega = tau;
  c.seed = seed;
  return c;
}

Mppi::Mppi(const MppiConfig& config, const PurePursuitConfig& speed_profile,
           const ControlLimits& limits, const RotationConfig& rotation, const CostMap* costmap)
    : config_(config), speed_(speed_profile), limits_(limits), rotation_(rotation),
      costmap_(costmap), rng_(config.seed),
      finisher_(speed_profile, limits, rotation) {}

void Mppi::reset() {
  state_ = ControlState::Rotating;
  index_ = 0;
  started_ = false;
  handed_over_ = false;
  finisher_.reset();
  nominal_.clear();
  debug_ = {};
}

double Mppi::obstacleCost(double x, double y) const {
  const GridIndex c = costmap_->grid().worldToCell(x, y);
  return costmap_->lethal(c) ? kLethal : costmap_->cost(c);
}

double Mppi::referenceSpeed(double distance_to_goal) const {
  // Même profil de vitesse que Pure Pursuit : freinage avant le but, approche lente.
  return std::min({limits_.v_max, std::sqrt(2.0 * speed_.decel * distance_to_goal),
                   speed_.approach_gain * distance_to_goal});
}

Twist2D Mppi::compute(const Pose2D& est, const Twist2D& vel,
                            const Path& path, Time t) {
  if (path.empty()) {
    state_ = ControlState::Stopped;
    return {};
  }
  if (state_ == ControlState::Arrived || state_ == ControlState::Stopped) return {};
  const PathPoint& goal = path.points.back();
  const double distance_to_goal = std::hypot(goal.x - est.x, goal.y - est.y);
  if (distance_to_goal < config_.handover_distance || handed_over_) {
    // Approche finale par Pure Pursuit (une fois commencée, on ne revient pas à MPPI).
    handed_over_ = true;
    last_cmd_ = finisher_.compute(est, vel, path, t);
    state_ = finisher_.state();
    debug_ = finisher_.debug();
    return last_cmd_;
  }
  index_ = closestPointForward(path, est, index_, speed_.search_window);
  debug_.path_index = index_;
  debug_.lateral_error = lateralError(path, index_, est);

  // Pré-rotation sur place, comme Pure Pursuit (MPPI gère mal les demi-tours sur place).
  const double heading_error = wrapAngle(path.points[index_].theta - est.theta);
  if (!started_) {
    state_ = std::abs(heading_error) > rotation_.start_threshold ? ControlState::Rotating
                                                                  : ControlState::Tracking;
    started_ = true;
  } else if (state_ != ControlState::Rotating && std::abs(heading_error) > rotation_.start_threshold) {
    state_ = ControlState::Rotating;
  } else if (state_ == ControlState::Rotating && std::abs(heading_error) < rotation_.stop_threshold) {
    state_ = ControlState::Tracking;
    nominal_.clear();
  }
  last_cmd_ = state_ == ControlState::Rotating
                  ? rotateTowards(path.points[index_].theta, est, vel, rotation_)
                  : optimize(est, vel, path, t);
  return last_cmd_;
}

std::vector<Eigen::Vector2d> Mppi::rollout(const Eigen::Matrix<double, 5, 1>& x0,
                                           const std::vector<Twist2D>& u) const {
  std::vector<Eigen::Vector2d> points;
  Vector5d x = x0;
  for (const Twist2D& cmd : u) {
    x = motionModel(x, cmd, config_.dt, config_.tau_v, config_.tau_omega);
    points.emplace_back(x(0), x(1));
  }
  return points;
}

Twist2D Mppi::optimize(const Pose2D& est, const Twist2D& vel, const Path& path, Time t) {
  const int steps = static_cast<int>(std::lround(config_.horizon / config_.dt));
  const int samples = config_.samples;
  // Démarrage à chaud : la séquence du cycle précédent, décalée d'un pas.
  if (static_cast<int>(nominal_.size()) != steps) {
    nominal_.assign(steps, {std::max(0.0, vel.v), vel.omega});
  } else {
    nominal_.erase(nominal_.begin());
    nominal_.push_back(nominal_.back());
  }
