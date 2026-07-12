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
  return groundTruthGrid(world_.config().arena, world_.obstacles(), resolution_);
}

#ifdef RN_WITH_OPENCV
// Défini dans vision_source.cpp (mode full : caméra simulée + perception + EKF).
std::unique_ptr<IStateSource> makeVisionSource(const SimSetup& setup, const SimWorld& world,
                                               unsigned seed, EventLog* events);
#endif

namespace {

double clearance(const SimWorld& world) {
  const Pose2D p = world.groundTruthPose();
  double best = 1e9;
  for (const Box& b : world.obstacles()) {
    best = std::min(best, distanceToBox(p.x, p.y, b) - world.config().robot.robot_radius);
  }
  return best;
}

std::unique_ptr<IStateSource> makeSource(const SimSetup& setup, const SimWorld& world,
                                         const EpisodeOptions& options, EventLog* events) {
  if (options.mode == PoseMode::Full) {
#ifdef RN_WITH_OPENCV
    return makeVisionSource(setup, world, options.seed, events);
#else
    (void)events;
    throw std::runtime_error("mode full indisponible : compilé sans OpenCV");
#endif
  }
  return std::make_unique<GroundTruthSource>(world, setup.nav.planner.resolution);
}

}  // namespace

void writePathCsv(const Path& path, const std::string& file) {
  CsvLogger out(file, {"x", "y", "theta", "curvature", "s"});
  for (const PathPoint& p : path.points) {
    out.log({p.x, p.y, p.theta, p.curvature, p.s});
  }
}

EpisodeMetrics runEpisode(const SimSetup& setup, const Scenario& scenario,
                          const EpisodeOptions& options) {
  SimWorldConfig world_cfg = setup.world;
  world_cfg.start = scenario.start;
  world_cfg.obstacles = scenario.obstacles;
  world_cfg.seed = setup.world.seed + 7919u * options.seed;  // une graine différente par essai
  SimWorld world(world_cfg);
  // Même chemin que sur le vrai robot : protocole série -> firmware (simulé) -> moteurs.
  SimClock clock;
  SimFirmware firmware(world, clock);
  ProtocolRobot robot(firmware, clock, ProtocolRobotConfig{});
  robot.waitReady();

  std::optional<CsvLogger> log;
  std::optional<EventLog> events;
  if (!options.log_dir.empty()) {
    std::filesystem::create_directories(options.log_dir);
    log.emplace(options.log_dir + "/log.csv",
                std::vector<std::string>{"t", "gt_x", "gt_y", "gt_theta", "est_x", "est_y",
                                         "est_theta", "est_valid", "v_cmd", "omega_cmd", "u_left",
                                         "u_right", "path_index", "lateral_error", "compute_ms"});
    events.emplace(options.log_dir + "/events.csv");
  }

  std::unique_ptr<IStateSource> source =
      makeSource(setup, world, options, events ? &*events : nullptr);
  Navigator nav(setup.nav);
  nav.setController(options.controller);
  nav.setMap(source->buildMap());
  nav.setGoal(scenario.goal, scenario.goal_heading);

  EpisodeMetrics m;
  const double dt = 1.0 / setup.nav.control_rate;
  double lateral_sum2 = 0.0;
  int lateral_count = 0;
  Pose2D previous = world.groundTruthPose();
  std::optional<double> arrival_time;
  int last_collisions = 0;

  for (Time t = 0.0; t < options.timeout; t += dt) {
    clock.set(t);
    world.advanceTo(t);
    const StateEstimate est = source->estimate(t);

    const auto t0 = std::chrono::steady_clock::now();
    const WheelCmd cmd = nav.step(est.pose, est.vel, est.valid, t);
    m.compute_ms.push_back(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    robot.send(cmd);
    robot.poll();
    source->commandSent(nav.appliedTwist(), t);

    // Métriques sur la vérité terrain.
    const Pose2D gt = world.groundTruthPose();
    m.traveled_length += std::hypot(gt.x - previous.x, gt.y - previous.y);
    previous = gt;
    m.min_clearance = std::min(m.min_clearance, clearance(world));
    if (nav.controlState() == ControlState::Tracking) {
      lateral_sum2 += nav.controller().debug().lateral_error * nav.controller().debug().lateral_error;
      ++lateral_count;
    }
    if (m.planned_length == 0.0 && !nav.path().empty()) {
      m.planned_length = nav.path().length();
    }
    if (events) {
      for (const NavEvent& e : nav.takeEvents()) {
        events->log(t, e.type, e.message);
        if (e.type == "REPLAN") writePathCsv(nav.path(), options.log_dir + "/path.csv");
      }
      if (world.collisionCount() > last_collisions) events->log(t, "COLLISION", "contact");
    }
    last_collisions = world.collisionCount();
    if (log) {
      log->log({t, gt.x, gt.y, gt.theta, est.pose.x, est.pose.y, est.pose.theta,
                est.valid ? 1.0 : 0.0, nav.lastTwist().v, nav.lastTwist().omega, cmd.left,
                cmd.right, static_cast<double>(nav.controller().debug().path_index),
                nav.controller().debug().lateral_error, m.compute_ms.back()});
    }

    if (source->lowBattery() && !m.low_battery_alert) {
      m.low_battery_alert = true;
      if (events) events->log(t, "LOW_BATTERY", "Pile faible : arret du robot");
    }
    const ControlState state = nav.controlState();
    if (m.low_battery_alert) {
      m.failure = "pile faible";
      break;
    }
    if (state == ControlState::Blocked) {
      m.failure = "planification impossible (" + std::string(toString(nav.lastPlan().status)) + ")";
      break;
    }
    if (state == ControlState::Arrived && !arrival_time) {
      arrival_time = t;
    }
    if (arrival_time && t - *arrival_time >= options.settle_time) {
      break;
    }
  }

  const Pose2D final_pose = world.groundTruthPose();
  m.arrived = arrival_time.has_value();
  m.time_to_goal = arrival_time.value_or(options.timeout);
  m.final_position_error =
      std::hypot(final_pose.x - scenario.goal.x(), final_pose.y - scenario.goal.y());
  if (scenario.goal_heading) {
    m.final_heading_error = std::abs(wrapAngle(final_pose.theta - *scenario.goal_heading));
  }
  m.collisions = world.collisionCount();
  m.pose_rejections = source->rejections();
  m.lateral_rms = lateral_count > 0 ? std::sqrt(lateral_sum2 / lateral_count) : 0.0;

  const double pos_tol = setup.nav.rotation.goal_tolerance;
  const double head_tol = setup.nav.rotation.heading_tolerance;
  if (m.failure.empty()) {
    if (!m.arrived) m.failure = "temps depasse";
    else if (m.collisions > 0) m.failure = "collision";
    else if (m.final_position_error > pos_tol) m.failure = "erreur de position finale";
    else if (m.final_heading_error > head_tol) m.failure = "erreur de cap finale";
  }
  m.success = m.failure.empty();
  if (events) {
    events->log(m.time_to_goal, m.success ? "SUCCESS" : "FAILURE", m.failure);
  }
  return m;
}
