#include "vision.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d.hpp>  // OpenCV 5 : convexHull et contourArea sont dans "geometry"
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

#include "core.hpp"

namespace {
Eigen::Matrix3d toEigen(const cv::Matx33d& m) {
  Eigen::Matrix3d out;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) out(r, c) = m(r, c);
  return out;
}
}  // namespace

CameraIntrinsics loadIntrinsics(const std::string& path) {
  const ConfigNode root = ConfigNode::loadFile(path);
  CameraIntrinsics intr;
  intr.width = root.getInt("width", 16, 10000);
  intr.height = root.getInt("height", 16, 10000);
  const double fx = root.getDouble("fx", 10.0, 1e5);
  const double fy = root.getDouble("fy", 10.0, 1e5);
  const double cx = root.getDouble("cx", 0.0, intr.width);
  const double cy = root.getDouble("cy", 0.0, intr.height);
  intr.K = cv::Matx33d(fx, 0, cx, 0, fy, cy, 0, 0, 1);
  intr.distortion = root.getDoubleList("distortion", 5);
  return intr;
}

void saveIntrinsics(const std::string& path, const CameraIntrinsics& intr, double rms_error) {
  YAML::Emitter out;
  out << YAML::Comment("Intrinsèques de la caméra, produits par camera_tool calib") << YAML::BeginMap;
  out << YAML::Key << "width" << YAML::Value << intr.width;
  out << YAML::Key << "height" << YAML::Value << intr.height;
  out << YAML::Key << "fx" << YAML::Value << intr.K(0, 0);
  out << YAML::Key << "fy" << YAML::Value << intr.K(1, 1);
  out << YAML::Key << "cx" << YAML::Value << intr.K(0, 2);
  out << YAML::Key << "cy" << YAML::Value << intr.K(1, 2);
  out << YAML::Key << "distortion" << YAML::Value << YAML::Flow << intr.distortion;
  out << YAML::Key << "rms_reprojection_px" << YAML::Value << rms_error;
  out << YAML::EndMap;
  std::ofstream(path) << out.c_str() << "\n";
}

CameraModel::CameraModel(const CameraIntrinsics& intrinsics, const cv::Vec3d& rvec,
                         const cv::Vec3d& tvec)
    : intr_(intrinsics), rvec_(rvec), tvec_(tvec) {
  cv::Matx33d r;
  cv::Rodrigues(rvec, r);
  R_ = toEigen(r);
  // C = -R^T t
  center_ = -R_.transpose() * Eigen::Vector3d(tvec[0], tvec[1], tvec[2]);
}

cv::Point2d CameraModel::project(const Eigen::Vector3d& world) const {
  const std::vector<cv::Point3f> pts{cv::Point3f(static_cast<float>(world.x()),
                                                 static_cast<float>(world.y()),
                                                 static_cast<float>(world.z()))};
  return project(pts)[0];
}

std::vector<cv::Point2f> CameraModel::project(const std::vector<cv::Point3f>& world) const {
  std::vector<cv::Point2f> pixels;
  if (!world.empty()) {
    cv::projectPoints(world, rvec_, tvec_, intr_.K, intr_.distortion, pixels);
  }
  return pixels;
}

Eigen::Vector3d CameraModel::ray(const cv::Point2d& pixel) const {
  // Pixel -> coordonnées normalisées sans distorsion (x_n, y_n), rayon caméra (x_n, y_n, 1),
  // puis passage au repère monde : d_w = R^T d_c.
  std::vector<cv::Point2d> in{pixel}, out;
  cv::undistortPoints(in, out, intr_.K, intr_.distortion);
  return (R_.transpose() * Eigen::Vector3d(out[0].x, out[0].y, 1.0)).normalized();
}

std::optional<Eigen::Vector3d> CameraModel::intersectPlane(const cv::Point2d& pixel,
                                                           double h) const {
  const Eigen::Vector3d d = ray(pixel);
  // Point du rayon : C + s d ; on veut C.z + s d.z = h.
  if (std::abs(d.z()) < 1e-9) return std::nullopt;
  const double s = (h - center_.z()) / d.z();
  if (s <= 0.0) return std::nullopt;
  return center_ + s * d;
}

std::vector<Eigen::Vector2d> CameraModel::intersectPlane(const std::vector<cv::Point2f>& pixels,
                                                         double h) const {
  std::vector<Eigen::Vector2d> points;
  if (pixels.empty()) return points;
  std::vector<cv::Point2f> normalized;
  cv::undistortPoints(pixels, normalized, intr_.K, intr_.distortion);
  for (const cv::Point2f& n : normalized) {
    const Eigen::Vector3d d = R_.transpose() * Eigen::Vector3d(n.x, n.y, 1.0);
    const double s = (h - center_.z()) / d.z();
    const Eigen::Vector3d p = center_ + s * d;
    points.emplace_back(p.x(), p.y());
  }
  return points;
}

void lookDownExtrinsics(const Eigen::Vector3d& position, double tilt, double yaw, cv::Vec3d& rvec,
                        cv::Vec3d& tvec) {
  // Axes de la caméra exprimés dans le monde (colonnes de R_wc) pour une caméra
  // qui regarde vers le bas : x_c = +x_w, y_c = -y_w (le haut de l'image vers +y),
  // z_c = -z_w. Puis inclinaison autour de x_c et rotation autour de z_w.
  Eigen::Matrix3d down;
  down << 1, 0, 0, 0, -1, 0, 0, 0, -1;
  const Eigen::Matrix3d tilt_rot =
      Eigen::AngleAxisd(tilt, Eigen::Vector3d::UnitX()).toRotationMatrix();
  const Eigen::Matrix3d yaw_rot =
      Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Matrix3d r_wc = yaw_rot * down * tilt_rot;
  const Eigen::Matrix3d r_cw = r_wc.transpose();
  const Eigen::Vector3d t = -r_cw * position;
  cv::Matx33d r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r(i, j) = r_cw(i, j);
  cv::Rodrigues(r, rvec);
  tvec = cv::Vec3d(t.x(), t.y(), t.z());
}

bool loadExtrinsics(const std::string& path, cv::Vec3d& rvec, cv::Vec3d& tvec) {
  std::ifstream test(path);
  if (!test) return false;
  const ConfigNode root = ConfigNode::loadFile(path);
  const auto r = root.getDoubleList("rvec", 3);
  const auto t = root.getDoubleList("tvec", 3);
  rvec = cv::Vec3d(r[0], r[1], r[2]);
  tvec = cv::Vec3d(t[0], t[1], t[2]);
  return true;
}

void saveExtrinsics(const std::string& path, const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                    double reprojection_error) {
  YAML::Emitter out;
  out << YAML::Comment("Pose caméra / monde, produite par camera_tool arena") << YAML::BeginMap;
  out << YAML::Key << "rvec" << YAML::Value << YAML::Flow
      << std::vector<double>{rvec[0], rvec[1], rvec[2]};
  out << YAML::Key << "tvec" << YAML::Value << YAML::Flow
      << std::vector<double>{tvec[0], tvec[1], tvec[2]};
  out << YAML::Key << "reprojection_error_px" << YAML::Value << reprojection_error;
  out << YAML::EndMap;
  std::ofstream(path) << out.c_str() << "\n";
}

cv::aruco::Dictionary markerDictionary() {
  return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
}

MarkerDetector::MarkerDetector() {
  cv::aruco::DetectorParameters params;
  params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
  detector_ = cv::aruco::ArucoDetector(markerDictionary(), params);
}

std::vector<MarkerDetection> MarkerDetector::detect(const cv::Mat& image) const {
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<int> ids;
  detector_.detectMarkers(image, corners, ids);
  std::vector<MarkerDetection> out;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    MarkerDetection d;
    d.id = ids[i];
    for (int k = 0; k < 4; ++k) d.corners[k] = corners[i][k];
    out.push_back(d);
  }
  return out;
}

std::array<Eigen::Vector2d, 4> markerCorners(const Pose2D& marker_pose, double size) {
  // Dans M : haut-gauche (+s/2, +s/2), haut-droite (+s/2, -s/2),
  //          bas-droite (-s/2, -s/2), bas-gauche (-s/2, +s/2).
  const double h = size / 2.0;
  const std::array<Eigen::Vector2d, 4> local{Eigen::Vector2d(h, h), Eigen::Vector2d(h, -h),
                                             Eigen::Vector2d(-h, -h), Eigen::Vector2d(-h, h)};
  std::array<Eigen::Vector2d, 4> world;
  for (int k = 0; k < 4; ++k) world[k] = transformPoint(marker_pose, local[k]);
  return world;
}

Pose2D floorTagPose(const FloorTag& tag) { return {tag.x, tag.y, kPi / 2}; }

namespace {
double rmsError(const std::vector<cv::Point3f>& object, const std::vector<cv::Point2f>& image,
                const cv::Vec3d& rvec, const cv::Vec3d& tvec, const CameraIntrinsics& intr) {
  std::vector<cv::Point2f> projected;
  cv::projectPoints(object, rvec, tvec, intr.K, intr.distortion, projected);
  double sum = 0.0;
  for (std::size_t i = 0; i < image.size(); ++i) {
    const cv::Point2f d = projected[i] - image[i];
    sum += d.dot(d);
  }
  return std::sqrt(sum / std::max<std::size_t>(1, image.size()));
}
}  // namespace

ArenaCalibrator::ArenaCalibrator(const CameraIntrinsics& intrinsics,
                                 const ArenaConfig& arena)
    : intr_(intrinsics), arena_(arena) {}

void ArenaCalibrator::matchFloorTags(const std::vector<MarkerDetection>& detections,
                                     std::vector<cv::Point3f>& object,
                                     std::vector<cv::Point2f>& image, std::vector<int>* ids) const {
  for (const MarkerDetection& d : detections) {
    for (const FloorTag& tag : arena_.floor_tags) {
      if (tag.id != d.id) continue;
      const auto corners = markerCorners(floorTagPose(tag), arena_.floor_tag_size);
      for (int k = 0; k < 4; ++k) {
        object.emplace_back(static_cast<float>(corners[k].x()), static_cast<float>(corners[k].y()), 0.0f);
        image.push_back(d.corners[k]);
      }
      if (ids) ids->push_back(tag.id);
    }
  }
}

int ArenaCalibrator::addFrame(const cv::Mat& image) {
  std::vector<cv::Point3f> object;
  std::vector<cv::Point2f> pixels;
  std::vector<int> ids;
  matchFloorTags(detector_.detect(image), object, pixels, &ids);
  const int count = static_cast<int>(ids.size());
  if (count >= 2) {
    object_points_.insert(object_points_.end(), object.begin(), object.end());
    image_points_.insert(image_points_.end(), pixels.begin(), pixels.end());
    for (int id : ids) {
      if (std::find(ids_seen_.begin(), ids_seen_.end(), id) == ids_seen_.end()) ids_seen_.push_back(id);
    }
    ++frames_used_;
  }
  return count;
}

ArenaCalibration ArenaCalibrator::solve() const {
  ArenaCalibration result;
  result.frames_used = frames_used_;
  result.tags_seen = static_cast<int>(ids_seen_.size());
  if (frames_used_ == 0) return result;
  // IPPE : solution exacte pour des points coplanaires (ici le sol, z = 0),
  // puis raffinement non linéaire de l'erreur de reprojection (Levenberg-Marquardt).
  cv::Vec3d rvec, tvec;
  if (!cv::solvePnP(object_points_, image_points_, intr_.K, intr_.distortion, rvec, tvec, false,
                    cv::SOLVEPNP_IPPE)) {
    return result;
  }
  cv::solvePnPRefineLM(object_points_, image_points_, intr_.K, intr_.distortion, rvec, tvec);
  result.ok = true;
  result.rvec = rvec;
  result.tvec = tvec;
  result.reprojection_error = rmsError(object_points_, image_points_, rvec, tvec, intr_);
  return result;
}

double ArenaCalibrator::reprojectionError(const cv::Mat& image, const cv::Vec3d& rvec,
                                          const cv::Vec3d& tvec) const {
  std::vector<cv::Point3f> object;
  std::vector<cv::Point2f> pixels;
  matchFloorTags(detector_.detect(image), object, pixels, nullptr);
  if (object.empty()) return -1.0;
  return rmsError(object, pixels, rvec, tvec, intr_);
}

LocalizerConfig loadLocalizerConfig(const ConfigNode& section,
                                    const RobotConfig& robot, int tag_id) {
  LocalizerConfig cfg;
  cfg.tag_id = tag_id;
  cfg.marker_size = robot.marker_size;
  cfg.marker_height = robot.marker_height;
  cfg.marker_offset = robot.marker_offset;
  const std::string method = section.getString("method");
  if (method != "A" && method != "B") {
    throw ConfigError(section.file() + ": localizer.method : A ou B attendu");
  }
  cfg.method = method == "A" ? LocalizerMethod::RayPlane : LocalizerMethod::Pnp;
  cfg.size_tolerance = section.getDouble("size_tolerance", 0.01, 0.5);
  cfg.sigma_xy = section.getDouble("sigma_xy", 1e-4, 0.1);
  cfg.sigma_theta = degToRad(section.getDouble("sigma_theta_deg", 0.01, 20.0));
  cfg.residual_scale = section.getDouble("residual_scale", 1e-4, 0.1);
  return cfg;
}

double kabsch2D(const std::vector<Eigen::Vector2d>& model, const std::vector<Eigen::Vector2d>& world,
                Pose2D& pose) {
  // 1) On centre les deux nuages de points.
  const std::size_t n = model.size();
  Eigen::Vector2d mean_m = Eigen::Vector2d::Zero(), mean_w = Eigen::Vector2d::Zero();
  for (std::size_t i = 0; i < n; ++i) {
    mean_m += model[i] / static_cast<double>(n);
    mean_w += world[i] / static_cast<double>(n);
  }
  // 2) L'angle optimal est theta = atan2(sum m' x w', sum m' . w').
  double sum_cross = 0.0, sum_dot = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const Eigen::Vector2d m = model[i] - mean_m;
    const Eigen::Vector2d w = world[i] - mean_w;
    sum_cross += m.x() * w.y() - m.y() * w.x();
    sum_dot += m.dot(w);
  }
  const double theta = std::atan2(sum_cross, sum_dot);
  // 3) Translation : t = moyenne_w - R(theta) moyenne_m.
  const Pose2D rotation{0.0, 0.0, theta};
  const Eigen::Vector2d rotated_mean = transformPoint(rotation, mean_m);
  pose = {mean_w.x() - rotated_mean.x(), mean_w.y() - rotated_mean.y(), theta};
  // Résidu RMS.
  double sum2 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum2 += (transformPoint(pose, model[i]) - world[i]).squaredNorm();
  }
  return std::sqrt(sum2 / static_cast<double>(n));
}

RobotLocalizer::RobotLocalizer(const LocalizerConfig& config) : config_(config) {}

bool RobotLocalizer::tagPoseRayPlane(const MarkerDetection& d, const CameraModel& camera,
                                     Localization& out) const {
  const std::vector<cv::Point2f> pixels(d.corners.begin(), d.corners.end());
  const std::vector<Eigen::Vector2d> world = camera.intersectPlane(pixels, config_.marker_height);
  const auto local = markerCorners({0.0, 0.0, 0.0}, config_.marker_size);
  const std::vector<Eigen::Vector2d> model(local.begin(), local.end());
  out.residual = kabsch2D(model, world, out.tag_pose);
  double perimeter = 0.0;
  for (int k = 0; k < 4; ++k) perimeter += (world[(k + 1) % 4] - world[k]).norm();
  out.reconstructed_size = perimeter / 4.0;
  return true;
}

bool RobotLocalizer::tagPosePnp(const MarkerDetection& d, const CameraModel& camera,
                                Localization& out) const {
  // Coins dans le repère OpenCV du tag (x à droite, y en haut), ordre exigé par IPPE_SQUARE.
  const float h = static_cast<float>(config_.marker_size / 2.0);
  const std::vector<cv::Point3f> object{{-h, h, 0}, {h, h, 0}, {h, -h, 0}, {-h, -h, 0}};
  const std::vector<cv::Point2f> pixels(d.corners.begin(), d.corners.end());
  cv::Vec3d rvec_cm, tvec_cm;
  if (!cv::solvePnP(object, pixels, camera.intrinsics().K, camera.intrinsics().distortion, rvec_cm,
                    tvec_cm, false, cv::SOLVEPNP_IPPE_SQUARE)) {
    return false;
  }
  // T_wm = T_wc * T_cm, avec T_wc = inverse(R_cw, t_cw).
  cv::Matx33d r_cw, r_cm;
  cv::Rodrigues(camera.rvec(), r_cw);
  cv::Rodrigues(rvec_cm, r_cm);
  const cv::Matx33d r_wm = r_cw.t() * r_cm;
  const cv::Vec3d t_wm = r_cw.t() * (tvec_cm - camera.tvec());
  // Le x de notre repère M est le +y du repère OpenCV du tag (le haut du tag).
  const cv::Vec3d up = r_wm * cv::Vec3d(0, 1, 0);
  out.tag_pose = {t_wm[0], t_wm[1], std::atan2(up[1], up[0])};
  out.residual = 0.0;
  out.reconstructed_size = config_.marker_size;
  // Contrôle de cohérence : le tag doit être à la bonne hauteur.
  return std::abs(t_wm[2] - config_.marker_height) < 0.03;
}

Localization RobotLocalizer::fromCorners(const MarkerDetection& d, const CameraModel& camera) const {
  Localization out;
  out.corners = d.corners;
  const bool ok = config_.method == LocalizerMethod::RayPlane ? tagPoseRayPlane(d, camera, out)
                                                               : tagPosePnp(d, camera, out);
  if (!ok) {
    out.reject_reason = "pose du tag incohérente";
    return out;
  }
  if (std::abs(out.reconstructed_size - config_.marker_size) > config_.size_tolerance * config_.marker_size) {
    out.reject_reason = "taille du tag incohérente";
    return out;
  }
  // Pose du robot : T_WR = T_WM * inverse(T_RM).
  out.measurement.pose = compose(out.tag_pose, inverse(config_.marker_offset));
  // Plus le résidu est grand, moins on fait confiance à la mesure.
  const double scale = 1.0 + out.residual / config_.residual_scale;
  const double sxy = config_.sigma_xy * scale;
  const double sth = config_.sigma_theta * scale;
  out.measurement.cov = Eigen::Vector3d(sxy * sxy, sxy * sxy, sth * sth).asDiagonal();
  out.ok = true;
  return out;
}

Localization RobotLocalizer::localize(const std::vector<MarkerDetection>& detections,
                                      const CameraModel& camera, Time stamp) const {
  for (const MarkerDetection& d : detections) {
    if (d.id == config_.tag_id) {
      Localization loc = fromCorners(d, camera);
      loc.measurement.stamp = stamp;
      return loc;
    }
  }
  Localization none;
  none.reject_reason = "tag robot non vu";
  return none;
}

ObstacleDetectorConfig loadObstacleDetectorConfig(const ConfigNode& section) {
  ObstacleDetectorConfig cfg;
  cfg.morph_kernel = section.getInt("morph_kernel", 1, 51);
  cfg.min_area_px = section.getDouble("min_area_px", 0.0, 1e6);
  cfg.frames = section.getInt("frames", 1, 1000);
  cfg.robot_mask_radius = section.getDouble("robot_mask_radius", 0.0, 1.0);
  return cfg;
}

ObstacleDetector::ObstacleDetector(const ObstacleDetectorConfig& config,
                                   const ArenaConfig& arena, double resolution)
    : config_(config), arena_(arena), resolution_(resolution) {}

cv::Mat ObstacleDetector::colorMask(const cv::Mat& bgr) const {
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  cv::Mat mask = cv::Mat::zeros(bgr.size(), CV_8UC1);
  for (const ColorRange& c : arena_.obstacle_colors) {
    cv::Mat part;
    if (c.h_min <= c.h_max) {
      cv::inRange(hsv, cv::Scalar(c.h_min, c.s_min, c.v_min), cv::Scalar(c.h_max, c.s_max, c.v_max), part);
    } else {
      // Teinte qui fait le tour (rouge) : [h_min, 179] ou [0, h_max].
      cv::Mat low, high;
      cv::inRange(hsv, cv::Scalar(c.h_min, c.s_min, c.v_min), cv::Scalar(179, c.s_max, c.v_max), high);
      cv::inRange(hsv, cv::Scalar(0, c.s_min, c.v_min), cv::Scalar(c.h_max, c.s_max, c.v_max), low);
      part = low | high;
    }
    mask |= part;
  }
  // Ouverture (enlève les petits points isolés) puis fermeture (bouche les petits trous).
  const cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(config_.morph_kernel, config_.morph_kernel));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
  return mask;
}

void ObstacleDetector::maskRobotAndTags(cv::Mat& mask, const CameraModel& camera,
                                        const std::optional<Pose2D>& robot,
                                        double robot_height) const {
  if (robot) {
    // Disque autour du robot, projeté au sol et à la hauteur du toit : on efface l'enveloppe.
    std::vector<cv::Point3f> pts;
    for (int k = 0; k < 24; ++k) {
      const double a = 2.0 * kPi * k / 24;
      const float x = static_cast<float>(robot->x + config_.robot_mask_radius * std::cos(a));
      const float y = static_cast<float>(robot->y + config_.robot_mask_radius * std::sin(a));
      pts.emplace_back(x, y, 0.0f);
      pts.emplace_back(x, y, static_cast<float>(robot_height));
    }
    std::vector<cv::Point2f> px = camera.project(pts);
    std::vector<cv::Point> hull_in(px.begin(), px.end()), hull;
    cv::convexHull(hull_in, hull);
    cv::fillConvexPoly(mask, hull, cv::Scalar(0));
  }
  for (const FloorTag& tag : arena_.floor_tags) {
    const auto corners = markerCorners(floorTagPose(tag), arena_.floor_tag_size * 8.0 / 6.0);
    std::vector<cv::Point3f> pts;
    for (const auto& c : corners) pts.emplace_back(static_cast<float>(c.x()), static_cast<float>(c.y()), 0.0f);
    const std::vector<cv::Point2f> px = camera.project(pts);
    const std::vector<cv::Point> poly(px.begin(), px.end());
    cv::fillConvexPoly(mask, poly, cv::Scalar(0));
  }
}

OccupancyGrid ObstacleDetector::detect(const cv::Mat& bgr, const CameraModel& camera,
                                       const std::optional<Pose2D>& robot,
                                       double robot_marker_height) const {
  cv::Mat mask = colorMask(bgr);
  maskRobotAndTags(mask, camera, robot, robot_marker_height);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  OccupancyGrid grid = makeArenaGrid(arena_, resolution_);
  for (const auto& contour : contours) {
    if (cv::contourArea(contour) < config_.min_area_px) continue;
    const std::vector<cv::Point2f> pixels(contour.begin(), contour.end());
    // Projections sur le dessus (z = h) et sur le sol (z = 0) ; on garde l'intersection.
    OccupancyGrid top(grid.xMin(), grid.yMin(), grid.cols(), grid.rows(), resolution_);
    OccupancyGrid floor(grid.xMin(), grid.yMin(), grid.cols(), grid.rows(), resolution_);
    top.fillPolygon(camera.intersectPlane(pixels, arena_.obstacle_height));
    floor.fillPolygon(camera.intersectPlane(pixels, 0.0));
    for (int i = 0; i < grid.cols() * grid.rows(); ++i) {
      const GridIndex c = grid.fromLinear(i);
      if (top.at(c) == CellState::Occupied && floor.at(c) == CellState::Occupied) {
        grid.set(c, CellState::Occupied);
      }
    }
  }
  return grid;
}

void ObstacleDetector::startAccumulation() {
  votes_.clear();
  frames_added_ = 0;
}

void ObstacleDetector::addFrame(const cv::Mat& bgr, const CameraModel& camera,
                                const std::optional<Pose2D>& robot,
                                double robot_marker_height) {
  const OccupancyGrid g = detect(bgr, camera, robot, robot_marker_height);
  if (votes_.empty()) votes_.assign(g.cols() * g.rows(), 0);
