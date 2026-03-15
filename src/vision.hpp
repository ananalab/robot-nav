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
};

// RobotLocalizer : pose du robot à partir du tag 10 posé sur son toit.
//
// Méthode A (par défaut) : coins du tag -> rayons -> intersection avec le plan
// z = h_marker -> 4 points du monde -> ajustement rigide 2D (Kabsch) -> pose du tag
// -> pose du robot (en retirant l'offset du tag). Le résidu de l'ajustement mesure la
// qualité de la détection et augmente la covariance.
// Méthode B : solvePnP IPPE_SQUARE (pose 3D du tag), gardée pour comparaison.

enum class LocalizerMethod { RayPlane, Pnp };  // A, B

struct LocalizerConfig {
  int tag_id = 10;
  double marker_size = 0.09;
  double marker_height = 0.10;
  Pose2D marker_offset;      // pose du tag dans le repère robot
  LocalizerMethod method = LocalizerMethod::RayPlane;
  double size_tolerance = 0.10;    // rejet si la taille reconstruite s'écarte de plus de 10 %
  double sigma_xy = 0.003;         // écart-type de base de la position (m)
  double sigma_theta = 0.017;      // écart-type de base du cap (rad)
  double residual_scale = 0.002;   // résidu (m) qui double l'écart-type
};

LocalizerConfig loadLocalizerConfig(const ConfigNode& section,
                                    const RobotConfig& robot, int tag_id);

struct Localization {
  bool ok = false;
  std::string reject_reason;
  PoseMeasurement measurement;  // pose du ROBOT (pas du tag)
  Pose2D tag_pose;
  double residual = 0.0;              // RMS de l'ajustement (m)
  double reconstructed_size = 0.0;    // côté moyen reconstruit (m)
  std::array<cv::Point2f, 4> corners{};
};

// Ajustement rigide 2D par moindres carrés : (theta, t) minimisant
// sum |R(theta) model_i + t - world_i|^2. Renvoie le résidu RMS.
double kabsch2D(const std::vector<Eigen::Vector2d>& model, const std::vector<Eigen::Vector2d>& world,
                Pose2D& pose);

class RobotLocalizer {
 public:
  explicit RobotLocalizer(const LocalizerConfig& config);

  // Cherche le tag robot parmi les détections. stamp = instant de capture.
  Localization localize(const std::vector<MarkerDetection>& detections, const CameraModel& camera,
                        Time stamp) const;

 private:
  Localization fromCorners(const MarkerDetection& d, const CameraModel& camera) const;
  bool tagPoseRayPlane(const MarkerDetection& d, const CameraModel& camera, Localization& out) const;
  bool tagPosePnp(const MarkerDetection& d, const CameraModel& camera, Localization& out) const;

  LocalizerConfig config_;
};

// ObstacleDetector : boîtes de couleur vive -> grille d'occupation.
//
// 1) seuillage HSV, ouverture puis fermeture morphologiques, contours assez grands ;
// 2) on masque le robot et les tags sol ;
// 3) chaque contour est projeté sur le plan du dessus des boîtes (z = h) ET sur le sol
//    (z = 0). Vu d'en haut en biais, une boîte montre son dessus et ses flancs côté caméra :
//    - projeté sur z = h, le flanc déborde du côté de la caméra ;
//    - projeté sur z = 0, le dessus déborde du côté opposé.
//    L'intersection des deux projections redonne l'empreinte réelle de la boîte.
// 4) mode statique : vote majoritaire sur N images.

struct ObstacleDetectorConfig {
  int morph_kernel = 5;        // taille du noyau morphologique (px)
  double min_area_px = 150.0;  // contours plus petits ignorés (bruit)
  int frames = 20;             // images accumulées en mode statique
  double robot_mask_radius = 0.13;  // disque masqué autour du robot (m)
};

ObstacleDetectorConfig loadObstacleDetectorConfig(const ConfigNode& section);

class ObstacleDetector {
 public:
  ObstacleDetector(const ObstacleDetectorConfig& config, const ArenaConfig& arena,
                   double resolution);

  // Masque binaire (255 = couleur d'obstacle), après morphologie.
  cv::Mat colorMask(const cv::Mat& bgr) const;
  // Grille d'occupation d'une seule image (cellules hors arène occupées).
  OccupancyGrid detect(const cv::Mat& bgr, const CameraModel& camera,
                                 const std::optional<Pose2D>& robot,
                                 double robot_marker_height) const;

  // Mode statique : accumulation puis vote majoritaire.
  void startAccumulation();
  void addFrame(const cv::Mat& bgr, const CameraModel& camera,
                const std::optional<Pose2D>& robot, double robot_marker_height);
  bool ready() const { return frames_added_ >= config_.frames; }
  OccupancyGrid result() const;

 private:
  void maskRobotAndTags(cv::Mat& mask, const CameraModel& camera,
                        const std::optional<Pose2D>& robot, double robot_height) const;

  ObstacleDetectorConfig config_;
  ArenaConfig arena_;
  double resolution_;
  std::vector<int> votes_;
  int frames_added_ = 0;
};

// Intersection sur union des cellules occupées DANS l'arène (comparaison à la vérité terrain).
double occupancyIoU(const OccupancyGrid& a, const OccupancyGrid& b,
                    const ArenaConfig& arena);

// Vraie caméra (iPhone par Caméra de continuité, appli webcam, ou flux réseau MJPEG/RTSP).
//
// Un thread de capture lit les images en continu et ne garde que la DERNIÈRE : sinon les
// images s'accumulent dans le tampon d'OpenCV et la latence augmente sans fin.
// Horodatage : instant de réception - latence mesurée (outil camera_tool latency).

class OpenCvCamera : public ICamera {
 public:
  // source : "0", "1"... (index) ou URL. Lève si la caméra ne s'ouvre pas.
  OpenCvCamera(const std::string& source, int width, int height, double latency,
               const IClock& clock);
  ~OpenCvCamera() override;
  OpenCvCamera(const OpenCvCamera&) = delete;
  OpenCvCamera& operator=(const OpenCvCamera&) = delete;

  std::optional<Frame> grab() override;
  // Images reçues par seconde (moyenne glissante).
  double fps() const { return fps_.load(); }

 private:
  void captureLoop();

  cv::VideoCapture capture_;
  double latency_;
  const IClock& clock_;
  std::thread thread_;
  std::atomic<bool> running_{true};
  std::atomic<double> fps_{0.0};
  std::mutex mutex_;          // protège latest_ et fresh_ (partagés avec le thread)
  Frame latest_;
  bool fresh_ = false;
};

// Calibration des intrinsèques (K + distorsion) avec un plateau ChArUco.
// Un plateau ChArUco = damier dont les cases blanches portent des tags ArUco : chaque coin
// du damier est identifié même si le plateau n'est vu qu'en partie.

struct CharucoSpec {
  int squares_x = 5, squares_y = 7;
  double square = 0.035;  // m
  double marker = 0.026;  // m
};

// Lit la section "charuco" ; square_mm_override > 0 remplace la taille des cases (mode écran).
CharucoSpec loadCharucoSpec(const ConfigNode& section, double square_mm_override = 0.0);
cv::aruco::CharucoBoard makeBoard(const CharucoSpec& spec);

class CameraCalibrator {
 public:
  explicit CameraCalibrator(const CharucoSpec& spec);

  // Coins ChArUco détectés dans l'image (pour l'affichage).
  std::vector<cv::Point2f> detectCorners(const cv::Mat& image) const;
  // Ajoute une vue si assez de coins sont visibles (>= 8). Renvoie true si ajoutée.
  bool addView(const cv::Mat& image);
  int views() const { return static_cast<int>(object_points_.size()); }
  // Calibre ; renvoie l'erreur de reprojection RMS (px), ou -1 si pas assez de vues.
  double calibrate(CameraIntrinsics& out) const;

 private:
  cv::aruco::CharucoBoard board_;
  cv::aruco::CharucoDetector detector_;
  std::vector<std::vector<cv::Point3f>> object_points_;
  std::vector<std::vector<cv::Point2f>> image_points_;
  cv::Size image_size_;
};

// Caméra simulée : rendu synthétique de l'arène vue d'en haut.
//
// Rendu : sol gris texturé + tags sol (une homographie du plan z = 0), boîtes 3D
// (algorithme du peintre), robot (châssis sombre + toit blanc avec le tag 10),
// distorsion k1, flou, bruit, exposition variable. Les images sont livrées après
// "latency" secondes ; certaines sont perdues ou ont le tag robot masqué.

struct SimCameraConfig {
  CameraIntrinsics intrinsics;
  Eigen::Vector3d position{0.5, 0.35, 1.2};  // centre optique dans le monde (m)
  double tilt = 0.0;                         // inclinaison (rad)
  double yaw = 0.0;                          // (rad)
  double fps = 30.0;
  double latency = 0.15;                     // s
  double p_drop = 0.0;                       // probabilité d'image perdue
  double p_occlusion = 0.0;                  // probabilité de tag robot masqué
  double blur_sigma = 0.0;                   // flou gaussien (px)
  double noise_sigma = 0.0;                  // bruit pixel (niveaux de gris)
  double exposure_sigma = 0.0;               // variation relative d'exposition
  double chassis_length = 0.15, chassis_width = 0.11;  // boîte sombre du châssis (m)
  double roof_size = 0.11;                   // carton blanc portant le tag (m)
  std::vector<std::pair<std::string, cv::Scalar>> colors;  // couleur BGR de chaque obstacle
};

SimCameraConfig loadSimCameraConfig(const ConfigNode& camera_section);

class SimCamera : public ICamera {
 public:
  SimCamera(const SimCameraConfig& config, const RobotConfig& robot,
            const ArenaConfig& arena, const SimWorld& world, const IClock& clock,
            unsigned seed);

  // Rend l'état actuel du monde, horodaté t ; l'image sera livrée à t + latency.
  void capture(Time t);
  // Dernière image livrée depuis le dernier appel (selon l'horloge).
  std::optional<Frame> grab() override;

  // Rendu sans dégradation ni latence (tests, affichage).
  cv::Mat render(bool show_robot_tag = true) const;
  // Le vrai modèle de caméra (extrinsèques exactes) : sert de vérité terrain.
  const CameraModel& trueModel() const { return model_; }

 private:
  void buildFloorTexture();
  void buildDistortionMaps();
  void drawBox(cv::Mat& img, const Box& box, const cv::Scalar& color) const;
  void warpPlaneTexture(cv::Mat& img, const cv::Mat& texture, const cv::Matx33d& tex_to_world,
                        double h) const;
  cv::Point2f projectIdeal(const Eigen::Vector3d& p) const;
  cv::Matx33d planeHomography(double h) const;
  cv::Scalar colorOf(const std::string& name) const;
  void degrade(cv::Mat& img);

  SimCameraConfig config_;
  RobotConfig robot_;
  ArenaConfig arena_;
  const SimWorld& world_;
  const IClock& clock_;
  std::mt19937 rng_;
  CameraModel model_;
  Eigen::Matrix3d R_;  // monde -> caméra
  Eigen::Vector3d t_;
  cv::Mat floor_texture_, robot_texture_;
  cv::Matx33d floor_to_world_;  // pixel de texture -> (x, y, 1) monde
  cv::Mat map_x_, map_y_;       // pour appliquer la distorsion
  std::deque<Frame> pending_;
};
