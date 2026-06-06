#include "sim.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "core.hpp"

namespace {
MotorParams loadMotor(const ConfigNode& node) {
  return {node.getDouble("gain", 0.05, 3.0), node.getDouble("deadzone", 0.0, 0.9)};
}
}  // namespace

std::vector<Box> loadBoxes(const ConfigNode& node, double default_height) {
  std::vector<Box> boxes;
  if (!node.has("obstacles")) {
    return boxes;
  }
  for (const ConfigNode& b : node.list("obstacles")) {
    Box box;
    box.x = b.getDouble("x", -5.0, 5.0);
    box.y = b.getDouble("y", -5.0, 5.0);
    box.length = b.getDouble("length", 0.01, 2.0);
    box.width = b.getDouble("width", 0.01, 2.0);
    box.height = b.getDoubleOr("height", default_height, 0.0, 1.0);
    box.yaw = degToRad(b.getDoubleOr("yaw", 0.0, -360.0, 360.0));
    box.color = b.has("color") ? b.getString("color") : "red";
    boxes.push_back(box);
  }
  return boxes;
}

SimWorldConfig loadSimWorldConfig(const ConfigNode& sim_root,
                                  const RobotConfig& robot_geometry,
                                  const ArenaConfig& arena) {
  SimWorldConfig cfg;
  cfg.arena = arena;
  cfg.robot = robot_geometry;
  cfg.seed = static_cast<unsigned>(sim_root.getInt("seed", 0, 1000000000));

  const ConfigNode physics = sim_root.child("physics");
  cfg.dt = physics.getDouble("dt", 1e-4, 0.01);
  cfg.slip_sigma = physics.getDouble("slip_sigma", 0.0, 0.5);
  cfg.slip_period = physics.getDouble("slip_period", 0.001, 1.0);
  cfg.battery_sag = physics.getDouble("battery_sag", 0.0, 0.9);
  cfg.battery_duration = physics.getDouble("battery_duration", 1.0, 1e5);

  // Les vrais moteurs de la sim diffèrent volontairement de robot.yaml
  // (ce que le contrôleur croit savoir) pour tester la robustesse.
  const ConfigNode motors = sim_root.child("true_motors");
  cfg.robot.left = loadMotor(motors.child("left"));
  cfg.robot.right = loadMotor(motors.child("right"));
  cfg.robot.tau = motors.getDouble("tau", 0.005, 1.0);
  cfg.robot.delay = motors.getDouble("delay", 0.0, 0.5);

  const ConfigNode world = sim_root.child("world");
  const auto start = world.getDoubleList("start", 3);
  cfg.start = {start[0], start[1], degToRad(start[2])};
  cfg.obstacles = loadBoxes(world, arena.obstacle_height);
  return cfg;
}

double distanceToBox(double x, double y, const Box& box) {
  // On exprime le point dans le repère de la boîte, puis on cherche le point de
  // la boîte le plus proche (bornage de chaque coordonnée).
  const double c = std::cos(box.yaw);
  const double s = std::sin(box.yaw);
  const double dx = x - box.x;
  const double dy = y - box.y;
  const double local_x = c * dx + s * dy;
  const double local_y = -s * dx + c * dy;
  const double ex = local_x - std::clamp(local_x, -box.length / 2, box.length / 2);
  const double ey = local_y - std::clamp(local_y, -box.width / 2, box.width / 2);
  return std::hypot(ex, ey);
}

bool discTouchesBox(double cx, double cy, double radius, const Box& box) {
  return distanceToBox(cx, cy, box) < radius;
}

SimWorld::SimWorld(const SimWorldConfig& config)
    : config_(config), rng_(config.seed), slip_noise_(0.0, config.slip_sigma),
      pose_(config.start) {}

void SimWorld::resetPose(const Pose2D& pose) {
  pose_ = pose;
  v_left_ = v_right_ = 0.0;
  active_cmd_ = {};
  pending_.clear();
  in_contact_ = false;
}

void SimWorld::setCommand(const WheelCmd& cmd) {
  WheelCmd clamped{std::clamp(cmd.left, -1.0, 1.0), std::clamp(cmd.right, -1.0, 1.0)};
  pending_.push_back({time_ + config_.robot.delay, clamped});
}

Twist2D SimWorld::groundTruthTwist() const {
  return {(v_left_ + v_right_) / 2.0, (v_right_ - v_left_) / config_.robot.wheel_base};
}

void SimWorld::advanceTo(Time t) {
  // Petite tolérance pour ne pas faire un pas de trop à cause des arrondis.
  while (time_ + config_.dt <= t + 1e-9) {
    step(config_.dt);
  }
}

double SimWorld::wheelTarget(const MotorParams& motor, double u) const {
  // Baisse de batterie : k(t) = k * (1 - sag * t / T)
  MotorParams aged = motor;
  const double fraction = std::min(1.0, time_ / config_.battery_duration);
  aged.gain = motor.gain * (1.0 - config_.battery_sag * fraction);
  return wheelSpeedFromCommand(aged, u);
}

int SimWorld::poseContact(const Pose2D& pose) const {
  const double r = config_.robot.robot_radius;
  const auto& a = config_.arena;
  // Le bord de l'arène compte comme un mur.
  if (pose.x - r < a.x_min || pose.x + r > a.x_max || pose.y - r < a.y_min ||
      pose.y + r > a.y_max) {
    return 2;
  }
  int contact = 0;
  for (const Box& box : config_.obstacles) {
    if (discTouchesBox(pose.x, pose.y, r, box)) {
      contact = std::max(contact, box.height > 0.0 ? 2 : 1);  // feuille à plat : 1
    }
  }
  return contact;
}

void SimWorld::step(double dt) {
  // 1) Les commandes dont le retard est écoulé deviennent actives.
  while (!pending_.empty() && pending_.front().effective_time <= time_ + 1e-12) {
    active_cmd_ = pending_.front().cmd;
