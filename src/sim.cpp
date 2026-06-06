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
    pending_.pop_front();
  }

  // 2) Dynamique du 1er ordre de chaque roue : tau * dv/dt = v_ss - v
  const double tau = config_.robot.tau;
  v_left_ += dt / tau * (wheelTarget(config_.robot.left, active_cmd_.left) - v_left_);
  v_right_ += dt / tau * (wheelTarget(config_.robot.right, active_cmd_.right) - v_right_);

  // 3) Glissement multiplicatif v_eff = v * (1 + eps), tiré toutes les slip_period.
  if (config_.slip_sigma > 0.0 && time_ >= next_slip_draw_) {
    slip_left_ = slip_noise_(rng_);
    slip_right_ = slip_noise_(rng_);
    next_slip_draw_ = time_ + config_.slip_period;
  }
  const double vl = v_left_ * (1.0 + slip_left_);
  const double vr = v_right_ * (1.0 + slip_right_);

  // 4) Cinématique unicycle : v = (vL + vR) / 2,  omega = (vR - vL) / b.
  //    Intégration au point milieu de l'angle (plus précise qu'Euler simple).
  const double v = (vl + vr) / 2.0;
  const double omega = (vr - vl) / config_.robot.wheel_base;
  const double mid_theta = pose_.theta + omega * dt / 2.0;
  Pose2D next = pose_;
  next.x += v * std::cos(mid_theta) * dt;
  next.y += v * std::sin(mid_theta) * dt;
  next.theta = wrapAngle(pose_.theta + omega * dt);

  // 5) Collision : un contact = un événement. Contre un mur ou une boîte, le robot reste
  //    bloqué là où il est ; sur une feuille posée à plat, il roule dessus.
  const bool moving = std::abs(v) > 1e-9 || std::abs(omega) > 1e-9;
  const int contact = moving ? poseContact(next) : 0;
  if (contact > 0) {
    if (!in_contact_) {
      ++collision_count_;
    }
    in_contact_ = true;
    if (contact == 1) pose_ = next;
  } else {
    pose_ = next;
    in_contact_ = moving ? false : in_contact_;
  }

  if (active_cmd_.left != 0.0 || active_cmd_.right != 0.0) {
    motor_on_time_ += dt;
  }
  time_ += dt;
}

std::vector<Scenario> loadScenarios(const std::string& path, double obstacle_height) {
  const ConfigNode root = ConfigNode::loadFile(path);
  std::vector<Scenario> scenarios;
  for (const ConfigNode& node : root.list("scenarios")) {
    Scenario sc;
    sc.name = node.getString("name");
    const auto start = node.getDoubleList("start", 3);
    sc.start = {start[0], start[1], degToRad(start[2])};
    const auto goal = node.getDoubleList("goal");
    if (goal.size() != 2 && goal.size() != 3) {
      throw ConfigError(path + ": " + sc.name + ".goal : [x, y] ou [x, y, cap_deg] attendu");
    }
    sc.goal = {goal[0], goal[1]};
    if (goal.size() == 3) {
      sc.goal_heading = degToRad(goal[2]);
    }
    sc.obstacles = loadBoxes(node, obstacle_height);
    scenarios.push_back(sc);
  }
  return scenarios;
}

#include "fw_protocol.h"  // firmware/robot_fw : le code de la vraie carte

SimFirmware::SimFirmware(SimWorld& world, const IClock& clock, double watchdog, bool ack)
    : world_(world), clock_(clock), watchdog_(watchdog), ack_(ack) {
  world_.setCommand({0.0, 0.0});
  reply("B");  // comme la carte au démarrage
}

void SimFirmware::reply(const char* body) {
  char out[FW_LINE_MAX];
  fwFrame(body, out);
  outbox_.emplace_back(out);
}

bool SimFirmware::writeLine(std::string_view line) {
  checkWatchdog();
  // Même contrainte que la carte : buffer de 32 octets.
  if (line.size() >= FW_LINE_MAX) {
    ++invalid_;
    return true;
  }
  const std::string text(line);
  FwCommand cmd;
  if (!fwParseLine(text.c_str(), cmd)) {
    ++invalid_;
    return true;  // la carte ignore les trames invalides
  }
  ++valid_;
  last_valid_ = clock_.now();
  watchdog_fired_ = false;
  char body[12];
  switch (cmd.type) {
    case FW_MOTOR:
      world_.setCommand({cmd.left / 255.0, cmd.right / 255.0});
      if (ack_) {
        std::snprintf(body, sizeof(body), "A,%d", cmd.seq);
        reply(body);
      }
      break;
    case FW_STOP:
      world_.setCommand({0.0, 0.0});
      break;
    case FW_PING:
      if (ack_) {
        std::snprintf(body, sizeof(body), "A,%d", cmd.seq);
        reply(body);
      }
      break;
    default:
      break;
  }
  return true;
}

void SimFirmware::checkWatchdog() {
  if (!watchdog_fired_ && clock_.now() - last_valid_ > watchdog_) {
    world_.setCommand({0.0, 0.0});
    reply("W");
    watchdog_fired_ = true;
  }
}

std::optional<std::string> SimFirmware::readLine() {
  checkWatchdog();
  if (outbox_.empty()) return std::nullopt;
  std::string line = outbox_.front();
  outbox_.pop_front();
  return line;
}
