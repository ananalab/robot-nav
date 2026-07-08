#include "app.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <optional>

#include "control.hpp"
#include "core.hpp"
#include "hw.hpp"
#include "sim.hpp"

NavigatorConfig loadNavigatorConfig(const std::string& config_dir) {
  const RobotConfig robot =
      loadRobotConfig(ConfigNode::loadFile(config_dir + "/robot.yaml").child("robot"));
  const ConfigNode control = ConfigNode::loadFile(config_dir + "/params.yaml").child("control");
  NavigatorConfig cfg;
  cfg.control_rate = control.getDouble("rate", 1.0, 200.0);
  cfg.planner = loadPlannerConfig(ConfigNode::loadFile(config_dir + "/params.yaml").child("planning"),
                                            robot.robot_radius);
  cfg.limits = loadControlLimits(control.child("limits"));
  cfg.rotation = loadRotationConfig(control);
  cfg.pure_pursuit = loadPurePursuitConfig(control.child("pure_pursuit"));
  cfg.mppi = loadMppiConfig(control.child("mppi"), robot.tau,
                                     static_cast<unsigned>(control.child("mppi").getInt("seed", 0, 1000000)));
  cfg.wheel_mapper = loadWheelMapperConfig(robot, control.child("wheel_mapper"));
  return cfg;
}

Navigator::Navigator(const NavigatorConfig& config)
    : config_(config),
      controller_(std::make_unique<PurePursuit>(config.pure_pursuit, config.limits,
                                                         config.rotation)),
      mapper_(config.wheel_mapper) {}

Navigator::~Navigator() = default;

void Navigator::setMap(const OccupancyGrid& grid) {
  costmap_ = CostMap(grid, config_.planner.inflation);
  has_map_ = true;
}

void Navigator::setGoal(const Eigen::Vector2d& goal, std::optional<double> heading) {
  goal_ = goal;
  goal_heading_ = heading;
  replan_requested_ = true;
}

void Navigator::setController(ControllerKind kind) {
  kind_ = kind;
  if (kind == ControllerKind::Mppi) {
    controller_ = std::make_unique<Mppi>(config_.mppi, config_.pure_pursuit, config_.limits,
                                         config_.rotation, &costmap_);
  } else {
    controller_ = std::make_unique<PurePursuit>(config_.pure_pursuit, config_.limits,
                                                         config_.rotation);
  }
  events_.push_back({"CONTROLLER", controller_->name()});
}

const Mppi* Navigator::mppi() const {
  return kind_ == ControllerKind::Mppi ? static_cast<const Mppi*>(controller_.get())
                                       : nullptr;
}

ControlState Navigator::controlState() const {
  if (blocked_) return ControlState::Blocked;
  if (emergency_stop_ || !goal_) return ControlState::Stopped;
  return controller_->state();
}

bool Navigator::needsReplan(const Pose2D& est, std::string& reason) const {
  if (replan_requested_) {
    reason = "demande (nouveau but ou touche r)";
    return true;
  }
  if (path_.empty() || controller_->state() == ControlState::Arrived) {
    return false;
  }
  if (pathBlocked(costmap_, path_, controller_->debug().path_index)) {
    reason = "chemin bloque";
    return true;
  }
  if (distanceToPath(path_, est.x, est.y) > config_.planner.replan_deviation) {
    reason = "ecart au chemin";
    return true;
  }
  return false;
}

void Navigator::replan(const Pose2D& est, const std::string& reason) {
  replan_requested_ = false;
  last_plan_ = planPath(costmap_, est, *goal_, goal_heading_, config_.planner);
  if (!last_plan_.ok()) {
    events_.push_back({"PLAN_FAILED", toString(last_plan_.status)});
    path_ = {};
    blocked_ = true;
    return;
  }
  blocked_ = false;
  path_ = last_plan_.path;
  controller_->reset();
  events_.push_back({"REPLAN", reason + ", longueur " + std::to_string(path_.length())});
  if (last_plan_.start_escaped) {
    events_.push_back({"START_ESCAPED", "depart en zone letale, sortie par BFS"});
  }
}

WheelCmd Navigator::step(const Pose2D& est, const Twist2D& vel,
                               bool pose_valid, Time t) {
  last_twist_ = {};
  if (emergency_stop_ || !pose_valid || !goal_ || !has_map_) {
    mapper_.map({});  // remet l'hystérésis de zone morte à zéro
    return {};
  }
  std::string reason;
  if (needsReplan(est, reason)) {
    replan(est, reason);
  }
  if (blocked_ || path_.empty()) {
    mapper_.map({});
    return {};
  }
  last_twist_ = controller_->compute(est, vel, path_, t);
  return mapper_.map(last_twist_);
}

Twist2D Navigator::appliedTwist() const {
  const double l = mapper_.lastLeftSpeed();
  const double r = mapper_.lastRightSpeed();
  return {(l + r) / 2.0, (r - l) / config_.wheel_mapper.wheel_base};
}

std::vector<NavEvent> Navigator::takeEvents() {
  std::vector<NavEvent> out;
  out.swap(events_);
  return out;
}

SimSetup loadSimSetup(const std::string& config_dir) {
  SimSetup s;
  s.config_dir = config_dir;
  s.robot = loadRobotConfig(ConfigNode::loadFile(config_dir + "/robot.yaml").child("robot"));
  s.arena = loadArenaConfig(ConfigNode::loadFile(config_dir + "/robot.yaml").child("arena"));
  s.world = loadSimWorldConfig(ConfigNode::loadFile(config_dir + "/params.yaml").child("sim"), s.robot,
                                    s.arena);
  s.nav = loadNavigatorConfig(config_dir);
  return s;
}

OccupancyGrid groundTruthGrid(const ArenaConfig& arena,
                                        const std::vector<Box>& boxes, double resolution) {
  OccupancyGrid grid = makeArenaGrid(arena, resolution);
  for (const Box& b : boxes) {
    grid.fillPolygon(rectangleCorners(b.x, b.y, b.length, b.width, b.yaw));
  }
  return grid;
}

GroundTruthSource::GroundTruthSource(const SimWorld& world, double resolution)
    : world_(world), resolution_(resolution) {}

StateEstimate GroundTruthSource::estimate(Time /*t*/) {
  return {world_.groundTruthPose(), world_.groundTruthTwist(), true};
}

OccupancyGrid GroundTruthSource::buildMap() {
