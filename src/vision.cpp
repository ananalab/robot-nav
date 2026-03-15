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
