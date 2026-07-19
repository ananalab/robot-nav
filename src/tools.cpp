#include "tools.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "app.hpp"
#include "control.hpp"
#include "core.hpp"
#include "hw.hpp"
#include "sim.hpp"
#include "vision.hpp"

namespace {
ConfigNode load(const std::string& dir, const std::string& section) {
  return ConfigNode::loadFile(dir + "/params.yaml").child(section);
}
EkfConfig ekfConfig(const std::string& dir, const RobotConfig& robot,
                                bool latency_compensation) {
  EkfConfig cfg = loadEkfConfig(load(dir, "estimation"), robot);
  cfg.latency_compensation = cfg.latency_compensation && latency_compensation;
  return cfg;
}
}  // namespace

StateEstimator::StateEstimator(const std::string& config_dir, const RobotConfig& robot,
                               const ArenaConfig& arena, double grid_resolution,
                               bool latency_compensation)
    : robot_(robot),
      localizer_(loadLocalizerConfig(load(config_dir, "perception").child("localizer"),
                                                 robot, arena.robot_tag_id)),
      obstacles_(loadObstacleDetectorConfig(
                     load(config_dir, "perception").child("obstacles")),
                 arena, grid_resolution),
      ekf_(ekfConfig(config_dir, robot, latency_compensation)),
      battery_(loadBatteryMonitorConfig(load(config_dir, "estimation").child("battery"))) {}

const Localization& StateEstimator::processFrame(const Frame& frame,
                                                             Time now) {
  detections_ = detector_.detect(frame.image);
  last_loc_ = localizer_.localize(detections_, model_, frame.stamp);
  if (last_loc_.ok) {
    ekf_.update(last_loc_.measurement, now);
  }
  return last_loc_;
}

StateEstimate StateEstimator::estimate(Time now) {
  ekf_.predict(now);
  battery_.update(now, ekf_.pose(), last_cmd_);
  const bool ok = ekf_.health(now) == EkfHealth::Ok;
  return {ekf_.pose(), ekf_.velocity(), ok && !battery_.alert()};
}

void StateEstimator::commandSent(const Twist2D& cmd, Time t) {
  ekf_.addInput(t, cmd);
  last_cmd_ = cmd;
}

void StateEstimator::addObstacleFrame(const Frame& frame) {
  // Le robot est masqué s'il est localisé sur cette image.
  const auto loc = localizer_.localize(detector_.detect(frame.image), model_, frame.stamp);
  std::optional<Pose2D> robot;
  if (loc.ok) robot = loc.measurement.pose;
  obstacles_.addFrame(frame.image, model_, robot, robot_.marker_height);
}


// Mode "full" de la simulation : caméra simulée -> StateEstimator (détection ArUco,
// localisation, EKF). C'est exactement la chaîne du vrai robot, seule la caméra est simulée.

namespace {

SimCameraConfig cameraConfig(const SimSetup& s) {
  SimCameraConfig cfg = loadSimCameraConfig(
      ConfigNode::loadFile(s.config_dir + "/params.yaml").child("sim").child("camera"));
  if (s.camera_latency >= 0.0) cfg.latency = s.camera_latency;
  return cfg;
}

class VisionSource : public IStateSource {
 public:
  VisionSource(const SimSetup& setup, const SimWorld& world, unsigned seed,
               EventLog* events)
      : events_(events),
        camera_(cameraConfig(setup), setup.world.robot, setup.arena, world, clock_, seed + 101),
        estimator_(setup.config_dir, setup.robot, setup.arena, setup.nav.planner.resolution,
                   setup.latency_compensation) {
    // Extrinsèques estimés comme en réel : intrinsèques connus, 30 images des tags sol.
    ArenaCalibrator calibrator(camera_.trueModel().intrinsics(), setup.arena);
    for (int i = 0; i < 600 && calibrator.framesUsed() < 30; ++i) {
      if (const auto f = captureNow()) calibrator.addFrame(f->image);
    }
    const ArenaCalibration calib = calibrator.solve();
    if (!calib.ok) throw std::runtime_error("calibration de l'arène impossible (tags sol non vus)");
    estimator_.setCameraModel(
        CameraModel(camera_.trueModel().intrinsics(), calib.rvec, calib.tvec));
  }

  StateEstimate estimate(Time t) override {
    clock_.set(t);
    camera_.capture(t);
    if (const auto frame = camera_.grab()) estimator_.processFrame(*frame, t);
    const StateEstimate est = estimator_.estimate(t);
    const bool ok = estimator_.health(t) == EkfHealth::Ok;
    if (!ok && was_ok_ && events_) events_->log(t, "LOST", "aucune mesure acceptee depuis 0.5 s");
    was_ok_ = ok;
    return est;
  }

  void commandSent(const Twist2D& cmd, Time t) override { estimator_.commandSent(cmd, t); }

  OccupancyGrid buildMap() override {
    // Mode statique : vote sur N images au démarrage, robot immobile.
    estimator_.startObstacleDetection();
    for (int i = 0; !estimator_.obstaclesReady() && i < 1000; ++i) {
      if (const auto frame = captureNow()) estimator_.addObstacleFrame(*frame);
    }
    return estimator_.obstacleGrid();
  }

  bool lowBattery() const override { return estimator_.lowBattery(); }
  int rejections() const override { return estimator_.ekf().rejectedCount(); }

 private:
  // Image rendue tout de suite (préparation avant le départ, sur une horloge à part).
  std::optional<Frame> captureNow() {
    setup_clock_ += 1.0;
    clock_.set(setup_clock_);
    camera_.capture(setup_clock_);
    clock_.set(setup_clock_ + 10.0);
    auto frame = camera_.grab();
    clock_.set(0.0);
    return frame;
  }

  EventLog* events_;
  SimClock clock_;
  double setup_clock_ = -1e6;  // temps fictif des images de préparation
  SimCamera camera_;
  StateEstimator estimator_;
  bool was_ok_ = false;
};

}  // namespace

std::unique_ptr<IStateSource> makeVisionSource(const SimSetup& setup, const SimWorld& world,
                                               unsigned seed, EventLog* events) {
  return std::make_unique<VisionSource>(setup, world, seed, events);
}

TopView::TopView(const OccupancyGrid& grid, double meters_per_pixel)
    : x_min_(grid.xMin()), y_min_(grid.yMin()), mpp_(meters_per_pixel) {
  size_ = cv::Size(static_cast<int>(std::ceil(grid.cols() * grid.resolution() / mpp_)),
                   static_cast<int>(std::ceil(grid.rows() * grid.resolution() / mpp_)));
}

cv::Mat TopView::blankImage() const { return cv::Mat(size_, CV_8UC3, cv::Scalar(255, 255, 255)); }

cv::Point TopView::toPixel(double x, double y) const {
  return {static_cast<int>(std::lround((x - x_min_) / mpp_)),
          size_.height - 1 - static_cast<int>(std::lround((y - y_min_) / mpp_))};
}

cv::Point2d TopView::toWorld(int px, int py) const {
  return {x_min_ + px * mpp_, y_min_ + (size_.height - 1 - py) * mpp_};
}

void TopView::drawCostMap(cv::Mat& img, const CostMap& map) const {
  const OccupancyGrid& grid = map.grid();
  for (int py = 0; py < size_.height; ++py) {
    for (int px = 0; px < size_.width; ++px) {
      const cv::Point2d w = toWorld(px, py);
      const GridIndex c = grid.worldToCell(w.x, w.y);
      if (!grid.inside(c)) continue;
      cv::Vec3b color(255, 255, 255);
      if (grid.at(c) != CellState::Free) {
        color = cv::Vec3b(30, 30, 30);
      } else if (map.lethal(c)) {
        color = cv::Vec3b(110, 110, 200);
      } else if (map.cost(c) > 0) {
        // Plus le coût est fort, plus le bleu est foncé.
        const int shade = 255 - map.cost(c) * 120 / 253;
        color = cv::Vec3b(255, static_cast<uchar>(shade), static_cast<uchar>(shade));
      }
      img.at<cv::Vec3b>(py, px) = color;
    }
  }
}

void TopView::drawPath(cv::Mat& img, const Path& path, const cv::Scalar& color,
                       int thickness) const {
  for (std::size_t i = 1; i < path.points.size(); ++i) {
    cv::line(img, toPixel(path.points[i - 1].x, path.points[i - 1].y),
             toPixel(path.points[i].x, path.points[i].y), color, thickness, cv::LINE_AA);
  }
}

void TopView::drawRobot(cv::Mat& img, const Pose2D& pose, double radius,
                        const cv::Scalar& color) const {
  const cv::Point center = toPixel(pose.x, pose.y);
  cv::circle(img, center, static_cast<int>(radius / mpp_), color, 2, cv::LINE_AA);
  const cv::Point nose =
      toPixel(pose.x + radius * std::cos(pose.theta), pose.y + radius * std::sin(pose.theta));
  cv::line(img, center, nose, color, 2, cv::LINE_AA);
}

void TopView::drawPoint(cv::Mat& img, double x, double y, const cv::Scalar& color,
                        int radius_px) const {
  cv::circle(img, toPixel(x, y), radius_px, color, cv::FILLED, cv::LINE_AA);
}

void drawDetections(cv::Mat& img, const std::vector<MarkerDetection>& detections) {
  for (const auto& d : detections) {
    std::vector<cv::Point> poly;
    for (const auto& c : d.corners) poly.emplace_back(c);
    cv::polylines(img, poly, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    cv::circle(img, poly[0], 4, cv::Scalar(0, 0, 255), cv::FILLED);  // coin haut-gauche
    cv::putText(img, std::to_string(d.id), poly[0] + cv::Point(5, -5), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(0, 255, 0), 2);
  }
}

void drawWorldPolyline(cv::Mat& img, const CameraModel& camera,
                       const std::vector<Eigen::Vector2d>& points, double h, const cv::Scalar& color,
                       int thickness) {
  if (points.size() < 2) return;
  std::vector<cv::Point3f> world;
  for (const auto& p : points) {
    world.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(h));
  }
  const std::vector<cv::Point2f> px = camera.project(world);
  for (std::size_t i = 1; i < px.size(); ++i) {
    cv::line(img, px[i - 1], px[i], color, thickness, cv::LINE_AA);
  }
}

void drawWorldGrid(cv::Mat& img, const CameraModel& camera,
                   const ArenaConfig& arena, double step) {
  const cv::Scalar grid(200, 200, 0);
  for (double x = arena.x_min; x <= arena.x_max + 1e-9; x += step) {
    drawWorldPolyline(img, camera, {{x, arena.y_min}, {x, arena.y_max}}, 0.0, grid, 1);
  }
  for (double y = arena.y_min; y <= arena.y_max + 1e-9; y += step) {
    drawWorldPolyline(img, camera, {{arena.x_min, y}, {arena.x_max, y}}, 0.0, grid, 1);
  }
  drawWorldPolyline(img, camera,
                    {{arena.x_min, arena.y_min}, {arena.x_max, arena.y_min}, {arena.x_max, arena.y_max},
                     {arena.x_min, arena.y_max}, {arena.x_min, arena.y_min}},
                    0.0, cv::Scalar(0, 200, 255), 2);
  drawPoseAxes(img, camera, {0.0, 0.0, 0.0}, 0.0, 0.15, 3);
}

void drawPoseAxes(cv::Mat& img, const CameraModel& camera, const Pose2D& pose,
                  double h, double length, int thickness) {
  const Eigen::Vector2d o(pose.x, pose.y);
  const Eigen::Vector2d x = transformPoint(pose, {length, 0.0});
  const Eigen::Vector2d y = transformPoint(pose, {0.0, length});
  drawWorldPolyline(img, camera, {o, x}, h, cv::Scalar(0, 0, 255), thickness);
  drawWorldPolyline(img, camera, {o, y}, h, cv::Scalar(0, 200, 0), thickness);
}

void drawStatusLine(cv::Mat& img, const std::string& text, int row, const cv::Scalar& color) {
  const cv::Point origin(8, 22 + 22 * row);
  int baseline = 0;
  const cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
  cv::rectangle(img, origin + cv::Point(-4, 6), origin + cv::Point(size.width + 4, -size.height - 4),
                cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(img, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv::LINE_AA);
}

RigOptions parseRigOptions(int argc, char** argv, std::vector<std::string>& rest) {
  RigOptions o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("valeur manquante après " + a);
      return argv[++i];
    };
    if (a == "--sim") o.sim = true;
    else if (a == "--real") o.sim = false;
    else if (a == "--mode") o.sim = next() != "real";
    else if (a == "--config") o.config_dir = next();
    else if (a == "--port") o.port = next();
    else if (a == "--wifi") o.link = "wifi";
    else if (a == "--usb") o.link = "serial";
    else if (a == "--camera") o.camera_source = next();
    else if (a == "--no-camera") o.camera = false;
    else if (a == "--realtime") o.realtime = true;
    else if (a == "--seed") o.seed = static_cast<unsigned>(std::stoul(next()));
    else rest.push_back(a);
  }
  return o;
}

Rig::Rig(const RigOptions& options) : options_(options) {
  setup_ = loadSimSetup(options.config_dir);
  const ConfigNode params = ConfigNode::loadFile(options.config_dir + "/params.yaml");
  const ConfigNode hardware = params.child("hardware");
  extrinsics_path_ = options.config_dir + "/arena_extrinsics.yaml";
  bool wifi = false;

  if (options.sim) {
    auto clock = std::make_unique<SimClock>();
    if (options.empty_arena) setup_.world.obstacles.clear();
    world_ = std::make_unique<SimWorld>(setup_.world);
    transport_ = std::make_unique<SimFirmware>(*world_, *clock);
    const SimCameraConfig cam_cfg = loadSimCameraConfig(params.child("sim").child("camera"));
    auto camera = std::make_unique<SimCamera>(cam_cfg, setup_.world.robot, setup_.arena,
                                                   *world_, *clock, options.seed);
    intrinsics_ = cam_cfg.intrinsics;
    sim_camera_ = camera.get();
    camera_ = std::move(camera);
    clock_ = std::move(clock);
  } else {
    clock_ = std::make_unique<SteadyClock>();
    const std::string link = options.link.empty() ? hardware.getString("link") : options.link;
    if (link == "wifi") {
      wifi = true;
      const std::string ip = hardware.getString("robot_ip");
      std::cout << "Wi-Fi : robot " << ip << "\n";
      transport_ = std::make_unique<UdpTransport>(ip, hardware.getInt("udp_port", 1, 65535));
    } else {
      std::string port = options.port.empty() ? hardware.getString("serial_port") : options.port;
      if (port == "auto") {
        const auto ports = listSerialPorts();
        if (ports.empty()) throw std::runtime_error("aucune carte trouvée : câble USB branché ?");
        port = ports.front();
      }
      std::cout << "Port série : " << port << "\n";
      transport_ = std::make_unique<SerialTransport>(port, hardware.getInt("serial_baud", 9600, 1000000));
    }
    if (options.camera) {
      const std::string intrinsics_file = options.config_dir + "/camera.yaml";
      if (!std::filesystem::exists(intrinsics_file)) {
        throw std::runtime_error(intrinsics_file + " absent : lancez d'abord \"camera_tool calib\"");
      }
      intrinsics_ = loadIntrinsics(intrinsics_file);
      const std::string source = options.camera_source.empty() ? hardware.getString("camera_source") : options.camera_source;
      camera_ = std::make_unique<OpenCvCamera>(
          source, hardware.getInt("camera_width", 16, 10000), hardware.getInt("camera_height", 16, 10000),
          hardware.getDouble("camera_latency", 0.0, 2.0), *clock_);
    }
  }
  robot_ = std::make_unique<ProtocolRobot>(*transport_, *clock_,
                                           loadProtocolRobotConfig(hardware));
  // En Wi-Fi, le robot ne répond pas : on n'attend pas la trame "B".
  if (!wifi && !robot_->waitReady()) {
    std::cout << "Attention : pas de trame de démarrage reçue de la carte (firmware flashé ?)\n";
  }
}

Rig::~Rig() {
  robot_.reset();  // envoie "S" avant de fermer le port
}

void Rig::tick(double dt) {
  if (options_.sim) {
    const Time target = clock_->now() + dt;
    clock_->sleepUntil(target);  // SimClock : saut dans le temps
    world_->advanceTo(target);
    sim_camera_->capture(target);
    if (options_.realtime) wall_clock_.sleepUntil(target);
  } else {
    // Cadence fixe : on attend seulement ce qui reste du tour (le calcul en a déjà pris une partie).
    next_tick_ = std::max(next_tick_ + dt, clock_->now());
    clock_->sleepUntil(next_tick_);
  }
  robot_->poll();
}

CameraModel Rig::cameraModel(bool force) {
  cv::Vec3d rvec, tvec;
  if (!options_.sim && !force && loadExtrinsics(extrinsics_path_, rvec, tvec)) {
    return CameraModel(intrinsics_, rvec, tvec);
  }
  std::cout << "Calibration de l'arène (30 images des tags sol)...\n";
  ArenaCalibrator calibrator(intrinsics_, setup_.arena);
  for (int i = 0; i < 600 && calibrator.framesUsed() < 30; ++i) {
    tick(1.0 / 30.0);
    if (const auto f = camera_->grab()) calibrator.addFrame(f->image);
  }
  const ArenaCalibration calib = calibrator.solve();
  if (!calib.ok) throw std::runtime_error("calibration de l'arène impossible : tags sol non vus");
  std::cout << "Erreur de reprojection : " << calib.reprojection_error << " px, " << calib.tags_seen
            << " tags sol vus\n";
  if (!options_.sim) saveExtrinsics(extrinsics_path_, calib.rvec, calib.tvec, calib.reprojection_error);
  return CameraModel(intrinsics_, calib.rvec, calib.tvec);
}

namespace {

constexpr double kDt = 1.0 / 30.0;  // un cycle = une image

struct Sample {
  Time t;
  Pose2D pose;
};

// Pente d'une droite y = a t + b par moindres carrés.
double slope(const std::vector<double>& t, const std::vector<double>& y) {
  const double n = static_cast<double>(t.size());
  double st = 0, sy = 0, stt = 0, sty = 0;
  for (std::size_t i = 0; i < t.size(); ++i) {
    st += t[i];
    sy += y[i];
    stt += t[i] * t[i];
    sty += t[i] * y[i];
  }
  const double den = n * stt - st * st;
  return std::abs(den) < 1e-12 ? 0.0 : (n * sty - st * sy) / den;
}

class Procedure {
 public:
  Procedure(Rig& rig, const CameraModel& camera, const MotorCalibrationOptions& opt)
      : rig_(rig), opt_(opt),
        estimator_(rig.setup().config_dir, rig.setup().robot, rig.setup().arena,
                   rig.setup().nav.planner.resolution) {
    estimator_.setCameraModel(camera);
  }

  MotorCalibrationResult run();

 private:
  void log(const std::string& msg) const {
    if (opt_.verbose) std::cout << msg << std::endl;
  }
  bool tickAndMeasure(std::vector<Sample>& out);
  bool unsafe(const Pose2D& p) const;
  bool drive(const WheelCmd& cmd, double duration, double settle, std::vector<Sample>& out,
             Time& t_cmd);
  bool steadyState(const std::vector<Sample>& s, Time from, Time to, double& v,
                   double& omega) const;
  bool rampDeadzone(bool left_wheel, double& u0);
  bool rotateTo(double heading);
  double availableDistance(const Pose2D& p, double direction) const;
  void fitDynamics();

  Rig& rig_;
  MotorCalibrationOptions opt_;
  StateEstimator estimator_;
  Time last_seen_ = -1e9;
  Pose2D last_pose_;
  MotorCalibrationResult result_;
  struct Step {
    Time t_cmd;
    double v_ss;
    double direction;  // +1 avant, -1 arrière
    std::vector<Sample> samples;
  };
  std::vector<Step> steps_;
};

bool Procedure::tickAndMeasure(std::vector<Sample>& out) {
  rig_.tick(kDt);
  const Time now = rig_.clock().now();
  if (const auto frame = rig_.camera().grab()) {
    const auto& loc = estimator_.processFrame(*frame, now);
    if (loc.ok) {
      out.push_back({loc.measurement.stamp, loc.measurement.pose});
      last_seen_ = now;
      last_pose_ = loc.measurement.pose;
    }
  }
  return now - last_seen_ < 0.5;  // tag vu récemment ?
}

bool Procedure::unsafe(const Pose2D& p) const {
  const auto& a = rig_.setup().arena;
  const double m = opt_.edge_margin;
  return p.x < a.x_min + m || p.x > a.x_max - m || p.y < a.y_min + m || p.y > a.y_max - m;
}

double Procedure::availableDistance(const Pose2D& p, double direction) const {
  // Distance le long du cap (ou à l'opposé) avant d'entrer dans la marge du bord.
  const auto& a = rig_.setup().arena;
  const double dx = direction * std::cos(p.theta), dy = direction * std::sin(p.theta);
  double d = 10.0;
  if (dx > 1e-6) d = std::min(d, (a.x_max - opt_.edge_margin - p.x) / dx);
  if (dx < -1e-6) d = std::min(d, (a.x_min + opt_.edge_margin - p.x) / dx);
  if (dy > 1e-6) d = std::min(d, (a.y_max - opt_.edge_margin - p.y) / dy);
  if (dy < -1e-6) d = std::min(d, (a.y_min + opt_.edge_margin - p.y) / dy);
  return std::max(0.0, d);
}

bool Procedure::drive(const WheelCmd& cmd, double duration, double settle,
                      std::vector<Sample>& out, Time& t_cmd) {
  ProtocolRobot& robot = rig_.robot();
  t_cmd = rig_.clock().now();
  robot.send(cmd);
  bool safe = true;
  while (rig_.clock().now() < t_cmd + duration) {
    robot.send(cmd);
    if (!tickAndMeasure(out)) {
      log("  Arrêt : tag robot perdu.");
      safe = false;
      break;
    }
    if (unsafe(last_pose_)) {
      log("  Arrêt : trop près du bord de l'arène.");
      safe = false;
      break;
    }
    if (cmd.left != 0.0 || cmd.right != 0.0) result_.motor_time += kDt;
  }
  robot.stop();
  const Time stop_time = rig_.clock().now();
  while (rig_.clock().now() < stop_time + settle) tickAndMeasure(out);
  return safe;
}

bool Procedure::steadyState(const std::vector<Sample>& s, Time from, Time to,
                            double& v, double& omega) const {
  std::vector<double> t, x, y, th;
  for (const Sample& smp : s) {
    if (smp.t < from || smp.t > to) continue;
    // Cap "déroulé" : pas de saut de 2 pi entre deux mesures.
    const double theta = th.empty() ? smp.pose.theta
                                    : th.back() + wrapAngle(smp.pose.theta - th.back());
    t.push_back(smp.t);
    x.push_back(smp.pose.x);
    y.push_back(smp.pose.y);
    th.push_back(theta);
  }
  if (t.size() < 4) return false;
  // Vitesse = pente de la position ; avance = projection sur le cap moyen.
  const double vx = slope(t, x), vy = slope(t, y);
  double mean_theta = 0.0;
  for (double a : th) mean_theta += a / th.size();
  v = vx * std::cos(mean_theta) + vy * std::sin(mean_theta);
  omega = slope(t, th);
  return true;
}

bool Procedure::rampDeadzone(bool left_wheel, double& u0) {
  // Rampe u = 0.15 + 0.25 t sur une seule roue ; on s'arrête au premier mouvement.
  const double u_start = 0.15, rate = 0.25, u_end = 0.8;
  std::vector<Sample> samples;
  for (int i = 0; i < 10; ++i) tickAndMeasure(samples);  // pose de repos
  if (samples.empty()) return false;
  const Pose2D rest = samples.back().pose;
  ProtocolRobot& robot = rig_.robot();
  const Time t0 = rig_.clock().now();
  const double delay_guess = 0.05;
  bool moved = false;
  while (!moved) {
    const double u = u_start + rate * (rig_.clock().now() - t0);
    if (u > u_end) break;
    robot.send(left_wheel ? WheelCmd{u, 0.0} : WheelCmd{0.0, u});
    const std::size_t before = samples.size();
    if (!tickAndMeasure(samples)) break;
    result_.motor_time += kDt;
    for (std::size_t i = before; i < samples.size(); ++i) {
      const Pose2D& p = samples[i].pose;
      if (std::hypot(p.x - rest.x, p.y - rest.y) > 0.006 ||
          std::abs(wrapAngle(p.theta - rest.theta)) > degToRad(2.0)) {
        // Commande en vigueur quand le mouvement a commencé (instant de capture - retard).
        u0 = u_start + rate * (samples[i].t - delay_guess - t0);
        moved = true;
      }
    }
  }
  robot.stop();
  const Time stop = rig_.clock().now();
  while (rig_.clock().now() < stop + 0.4) tickAndMeasure(samples);
  return moved;
}

bool Procedure::rotateTo(double heading) {
  // Rotation sur place avec le modèle actuel de robot.yaml (approximatif, suffisant ici).
  const SimSetup& s = rig_.setup();
  WheelMapper mapper(s.nav.wheel_mapper);
  std::vector<Sample> samples;
  const Time t0 = rig_.clock().now();
  while (rig_.clock().now() < t0 + 6.0) {
    if (!tickAndMeasure(samples)) break;
    const double err = wrapAngle(heading - last_pose_.theta);
    if (std::abs(err) < degToRad(4.0)) break;
    rig_.robot().send(mapper.map({0.0, std::clamp(2.0 * err, -1.5, 1.5)}));
    result_.motor_time += kDt;
  }
  rig_.robot().stop();
  const Time stop = rig_.clock().now();
  while (rig_.clock().now() < stop + 0.4) tickAndMeasure(samples);
  return std::abs(wrapAngle(heading - last_pose_.theta)) < degToRad(8.0);
}

void Procedure::fitDynamics() {
  // On cherche (d, tau) qui expliquent le mieux le déplacement mesuré pendant les échelons :
  //   s(t') = v_ss (t' - d - tau (1 - exp(-(t' - d) / tau)))  pour t' = t - t_cmd > d, 0 sinon.
  double best = 1e18;
  for (double d = 0.0; d <= 0.25; d += 0.001) {
    for (double tau = 0.02; tau <= 0.4; tau += 0.002) {
      double err = 0.0;
      for (const Step& st : steps_) {
        const Pose2D& p0 = st.samples.front().pose;
        const double c = std::cos(p0.theta), s = std::sin(p0.theta);
        for (const Sample& smp : st.samples) {
          const double tp = smp.t - st.t_cmd;
          if (tp > 0.6) break;  // début de la réponse seulement
          const double measured = (smp.pose.x - p0.x) * c + (smp.pose.y - p0.y) * s;
          const double model = tp <= d ? 0.0 : st.v_ss * (tp - d - tau * (1.0 - std::exp(-(tp - d) / tau)));
          err += (measured - model) * (measured - model);
        }
      }
      if (err < best) {
        best = err;
        result_.delay = d;
        result_.tau = tau;
      }
    }
  }
}

MotorCalibrationResult Procedure::run() {
  const double b = rig_.setup().robot.wheel_base;
  log("Rappel : pile 9 V branchée (les moteurs ne tournent pas sur l'USB seul).");
  std::vector<Sample> warmup;
  for (int i = 0; i < 15; ++i) tickAndMeasure(warmup);
  if (warmup.empty()) {
    result_.error = "tag robot invisible : vérifiez la caméra et le tag 10";
    return result_;
  }

  log("1) Rampes de zone morte (une roue à la fois)");
  if (!rampDeadzone(true, result_.ramp_deadzone_left) || !rampDeadzone(false, result_.ramp_deadzone_right)) {
    log("  (rampe incomplète : la zone morte viendra seulement des échelons)");
  }
  log("   gauche ~" + std::to_string(result_.ramp_deadzone_left) + ", droite ~" +
      std::to_string(result_.ramp_deadzone_right));

  log("2) Échelons avant / arrière");
  const std::vector<double> levels = opt_.short_mode ? std::vector<double>{0.5, 0.7, 0.85, 1.0}
                                                     : std::vector<double>{0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
  const auto& arena = rig_.setup().arena;
  for (double u : levels) {
    // Face au grand côté de l'arène, pour avoir de la place.
    const double heading = (arena.x_max - arena.x_min) >= (arena.y_max - arena.y_min) ? 0.0 : kPi / 2;
    rotateTo(heading);
    for (const double dir : {1.0, -1.0}) {
      const double v_guess = 1.2 * wheelSpeedFromCommand(rig_.setup().robot.left, u) + 0.02;
      const double room = availableDistance(last_pose_, dir) - 0.06;
      const double duration = std::clamp(room / v_guess, 0.5, 1.5);
      Step step;
      step.direction = dir;
      if (!drive({dir * u, dir * u}, duration, 0.4, step.samples, step.t_cmd)) continue;
      double v = 0, omega = 0;
      // Régime permanent : après ~4 tau (0.45 s), jusqu'à la fin de l'échelon.
      if (!steadyState(step.samples, step.t_cmd + 0.45, step.t_cmd + duration, v, omega)) continue;
      result_.left_samples.push_back({dir * u, v - omega * b / 2.0});
      result_.right_samples.push_back({dir * u, v + omega * b / 2.0});
      step.v_ss = v;
      if (dir > 0 && duration >= 0.7) steps_.push_back(step);
      log("   u = " + std::to_string(dir * u) + " : v = " + std::to_string(v) + " m/s, omega = " +
          std::to_string(omega) + " rad/s");
    }
  }

  log("3) Rotations sur place");
  for (double u : opt_.short_mode ? std::vector<double>{0.6} : std::vector<double>{0.5, 0.7}) {
    for (const double dir : {1.0, -1.0}) {
      std::vector<Sample> s;
      Time t_cmd = 0;
      if (!drive({-dir * u, dir * u}, 1.2, 0.4, s, t_cmd)) continue;
      double v = 0, omega = 0;
      if (!steadyState(s, t_cmd + 0.45, t_cmd + 1.2, v, omega)) continue;
      result_.left_samples.push_back({-dir * u, v - omega * b / 2.0});
      result_.right_samples.push_back({dir * u, v + omega * b / 2.0});
    }
  }

  log("4) Moindres carrés");
  if (!fitMotor(result_.left_samples, result_.left) || !fitMotor(result_.right_samples, result_.right)) {
    result_.error = "pas assez de mesures pour ajuster le modèle";
    return result_;
  }
  fitDynamics();
  result_.ok = true;
  return result_;
}

}  // namespace

bool fitMotor(const std::vector<WheelSample>& samples, MotorParams& out) {
  // |v| = k |u| - k u0  : droite |v| = a |u| + c, d'où k = a et u0 = -c / a.
  std::vector<double> u, v;
  for (const WheelSample& s : samples) {
    if (std::abs(s.v) > 0.01) {
      u.push_back(std::abs(s.u));
      v.push_back(std::abs(s.v));
    }
  }
  if (u.size() < 3) return false;
  const double a = slope(u, v);
  double mean_u = 0, mean_v = 0;
  for (std::size_t i = 0; i < u.size(); ++i) {
    mean_u += u[i] / u.size();
    mean_v += v[i] / v.size();
  }
  const double c = mean_v - a * mean_u;
  if (a <= 0.0) return false;
  out.gain = a;
  out.deadzone = std::clamp(-c / a, 0.0, 0.9);
  return true;
}

void writeMotorsToRobotYaml(const std::string& path, const MotorCalibrationResult& r) {
  std::ifstream in(path);
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);) lines.push_back(line);
  // Parcours ligne à ligne : on sait dans quelle sous-section on est (motors, left, right).
  std::string section, wheel;
  auto setValue = [](std::string& line, double value) {
    const std::size_t colon = line.find(':');
    const std::size_t comment = line.find('#');
    std::ostringstream v;
    v << value;
    std::string out = line.substr(0, colon + 1) + " " + v.str();
    if (comment != std::string::npos) {
      out.resize(std::max(out.size(), comment), ' ');
      out += line.substr(comment);
    }
    line = out;
  };
  // robot.yaml : robot: (0) > motors: (2) > left/right: (4) > gain/deadzone (6) ; tau/delay (4).
  for (std::string& line : lines) {
    const std::size_t indent = line.find_first_not_of(' ');
    if (indent == std::string::npos || line[indent] == '#') continue;
    const std::string key = line.substr(indent, line.find(':') - indent);
    if (indent <= 2) section = key;
    if (section != "motors") continue;
    if (indent == 4 && (key == "left" || key == "right")) wheel = key;
    if (indent == 4 && key == "tau") setValue(line, r.tau);
    if (indent == 4 && key == "delay") setValue(line, r.delay);
    if (indent == 6 && key == "gain") setValue(line, wheel == "left" ? r.left.gain : r.right.gain);
    if (indent == 6 && key == "deadzone") setValue(line, wheel == "left" ? r.left.deadzone : r.right.deadzone);
  }
  std::ofstream out(path);
  for (const std::string& line : lines) out << line << "\n";
}

MotorCalibrationResult calibrateMotors(Rig& rig, const CameraModel& camera,
                                       const MotorCalibrationOptions& options) {
  Procedure procedure(rig, camera, options);
  return procedure.run();
}
