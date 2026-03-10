#pragma once

#include <Eigen/Core>
#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/videoio.hpp>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "core.hpp"
#include "planning.hpp"
#include "sim.hpp"

// Source d'images : caméra simulée ou vraie caméra (iPhone).

struct Frame {
  cv::Mat image;
  Time stamp = 0.0;  // instant de capture (pas de réception)
};

class ICamera {
 public:
  virtual ~ICamera() = default;
  // Dernière image disponible, ou rien si aucune nouvelle image depuis le dernier appel.
  virtual std::optional<Frame> grab() = 0;
};

// Modèle de caméra sténopé + distorsion (5 coefficients OpenCV) et pose dans le monde.
//
// Convention OpenCV : un point monde X_w a pour coordonnées caméra
//   X_c = R * X_w + t        (R, t = extrinsèques "monde -> caméra")
// Le centre optique dans le monde est C = -R^T t.

struct CameraIntrinsics {
  int width = 1280;
  int height = 720;
  cv::Matx33d K = cv::Matx33d::eye();
  std::vector<double> distortion = std::vector<double>(5, 0.0);  // k1 k2 p1 p2 k3
};

// Lit camera.yaml (produit par camera_tool calib).
CameraIntrinsics loadIntrinsics(const std::string& path);
void saveIntrinsics(const std::string& path, const CameraIntrinsics& intr, double rms_error);

class CameraModel {
 public:
  CameraModel() = default;
  CameraModel(const CameraIntrinsics& intrinsics, const cv::Vec3d& rvec, const cv::Vec3d& tvec);

  const CameraIntrinsics& intrinsics() const { return intr_; }
  const cv::Vec3d& rvec() const { return rvec_; }
  const cv::Vec3d& tvec() const { return tvec_; }
  Eigen::Vector3d center() const { return center_; }

  // Pixel (avec distorsion) d'un point du monde.
  cv::Point2d project(const Eigen::Vector3d& world) const;
  std::vector<cv::Point2f> project(const std::vector<cv::Point3f>& world) const;

  // Direction (unitaire, repère monde) du rayon qui passe par ce pixel.
  Eigen::Vector3d ray(const cv::Point2d& pixel) const;
  // Intersection du rayon du pixel avec le plan horizontal z = h (vide si parallèle ou derrière).
  std::optional<Eigen::Vector3d> intersectPlane(const cv::Point2d& pixel, double h) const;
  // Version pour plusieurs pixels (plus rapide : une seule correction de distorsion).
  std::vector<Eigen::Vector2d> intersectPlane(const std::vector<cv::Point2f>& pixels,
                                              double h) const;

 private:
  CameraIntrinsics intr_;
  cv::Vec3d rvec_, tvec_;
  Eigen::Matrix3d R_ = Eigen::Matrix3d::Identity();  // monde -> caméra
  Eigen::Vector3d center_ = Eigen::Vector3d::Zero();
};

// Extrinsèques d'une caméra placée en "position", regardant vers le bas, inclinée
// de tilt (autour de son axe x) et tournée de yaw (autour de z monde). Image : x monde
// vers la droite, y monde vers le haut quand tilt = yaw = 0.
void lookDownExtrinsics(const Eigen::Vector3d& position, double tilt, double yaw, cv::Vec3d& rvec,
                        cv::Vec3d& tvec);

// Lit / écrit arena_extrinsics.yaml (rvec, tvec, erreur de reprojection).
bool loadExtrinsics(const std::string& path, cv::Vec3d& rvec, cv::Vec3d& tvec);
void saveExtrinsics(const std::string& path, const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                    double reprojection_error);

// Détection des tags ArUco et géométrie des tags.
//
// Repère d'un tag M : centre du carré noir, x vers le HAUT du tag imprimé (la flèche
// « AVANT » pour le tag robot), y vers la gauche. Les 4 coins sont donnés dans
// l'ordre d'OpenCV : haut-gauche, haut-droite, bas-droite, bas-gauche (image du tag).
//
// Tags sol : posés le haut vers +y du monde, donc leur repère M est tourné de +90°.

struct MarkerDetection {
  int id = -1;
  std::array<cv::Point2f, 4> corners;  // pixels, ordre OpenCV
};

// Détecteur DICT_4X4_50 avec raffinement sous-pixel des coins.
class MarkerDetector {
 public:
  MarkerDetector();
  std::vector<MarkerDetection> detect(const cv::Mat& image) const;

 private:
  cv::aruco::ArucoDetector detector_;
};

// Dictionnaire des tags sol et robot.
cv::aruco::Dictionary markerDictionary();

// Coins (ordre OpenCV) d'un tag de côté "size" dont le repère M a la pose donnée.
std::array<Eigen::Vector2d, 4> markerCorners(const Pose2D& marker_pose, double size);

// Pose du repère M d'un tag sol (cap +90° : le haut du tag regarde vers +y).
Pose2D floorTagPose(const FloorTag& tag);

// ArenaCalibrator : pose de la caméra dans le monde à partir des tags du sol.
//
// On accumule les coins des tags sol vus sur N images (caméra fixe), puis un seul
// solvePnP sur tous ces points (IPPE, plan z = 0), affiné par Levenberg-Marquardt.

struct ArenaCalibration {
  bool ok = false;
  cv::Vec3d rvec, tvec;
  double reprojection_error = 0.0;  // RMS (px)
  int frames_used = 0;
  int tags_seen = 0;                // nombre de tags sol différents vus
};

class ArenaCalibrator {
 public:
  ArenaCalibrator(const CameraIntrinsics& intrinsics, const ArenaConfig& arena);

  // Ajoute une image ; renvoie le nombre de tags sol vus (>= 2 pour qu'elle serve).
  int addFrame(const cv::Mat& image);
  int framesUsed() const { return frames_used_; }
  ArenaCalibration solve() const;

  // Erreur de reprojection RMS des tags sol d'une image avec des extrinsèques donnés
  // (vérification périodique : la caméra a-t-elle bougé ?). -1 si aucun tag vu.
  double reprojectionError(const cv::Mat& image, const cv::Vec3d& rvec,
                           const cv::Vec3d& tvec) const;

 private:
  void matchFloorTags(const std::vector<MarkerDetection>& detections,
                      std::vector<cv::Point3f>& object, std::vector<cv::Point2f>& image,
                      std::vector<int>* ids) const;

  CameraIntrinsics intr_;
  ArenaConfig arena_;
  MarkerDetector detector_;
  std::vector<cv::Point3f> object_points_;
  std::vector<cv::Point2f> image_points_;
  std::vector<int> ids_seen_;
  int frames_used_ = 0;
