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
