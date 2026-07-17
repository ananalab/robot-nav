#pragma once

#include <Eigen/Core>
#include <memory>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

#include "app.hpp"
#include "hw.hpp"
#include "vision.hpp"
#include "estimation.hpp"

// StateEstimator : image -> tags ArUco -> pose du robot -> EKF (+ alerte pile, obstacles).
// Utilisé tel quel par la simulation "full" (bench) et par le vrai robot (robot_nav, teleop,
// robot_tool calib) : seule la source des images change.

class StateEstimator {
 public:
  StateEstimator(const std::string& config_dir, const RobotConfig& robot,
                 const ArenaConfig& arena, double grid_resolution,
                 bool latency_compensation = true);

  void setCameraModel(const CameraModel& model) { model_ = model; }
  const CameraModel& cameraModel() const { return model_; }

  // Détecte les tags, localise le robot et met à jour l'EKF. Renvoie la localisation brute.
  const Localization& processFrame(const Frame& frame, Time now);
  // Prédit jusqu'à "now" ; valid = EKF en bonne santé et pas d'alerte pile.
  StateEstimate estimate(Time now);
  // Consigne (v, omega) réellement envoyée aux roues.
  void commandSent(const Twist2D& cmd, Time t);

  void startObstacleDetection() { obstacles_.startAccumulation(); }
  void addObstacleFrame(const Frame& frame);
  bool obstaclesReady() const { return obstacles_.ready(); }
  OccupancyGrid obstacleGrid() const { return obstacles_.result(); }

  const std::vector<MarkerDetection>& lastDetections() const { return detections_; }
  const Localization& lastLocalization() const { return last_loc_; }
  const Ekf& ekf() const { return ekf_; }
  EkfHealth health(Time now) const { return ekf_.health(now); }
  bool lowBattery() const { return battery_.alert(); }

 private:
  RobotConfig robot_;
  CameraModel model_;
  MarkerDetector detector_;
  RobotLocalizer localizer_;
  ObstacleDetector obstacles_;
  Ekf ekf_;
  BatteryMonitor battery_;
  std::vector<MarkerDetection> detections_;
  Localization last_loc_;
  Twist2D last_cmd_;
};


// Vue de dessus de la carte (partie droite de la fenêtre, ) : 1 px = 5 mm.
// L'axe y du monde pointe vers le haut de l'image.

class TopView {
 public:
  TopView(const OccupancyGrid& grid, double meters_per_pixel = 0.005);

  // Fond : obstacles (noir), zone létale (rouge sombre), coût d'inflation (dégradé).
  void drawCostMap(cv::Mat& img, const CostMap& map) const;
  void drawPath(cv::Mat& img, const Path& path, const cv::Scalar& color, int thickness = 2) const;
  void drawRobot(cv::Mat& img, const Pose2D& pose, double radius, const cv::Scalar& color) const;
  void drawPoint(cv::Mat& img, double x, double y, const cv::Scalar& color, int radius_px = 4) const;

  cv::Mat blankImage() const;
  cv::Point toPixel(double x, double y) const;
  cv::Point2d toWorld(int px, int py) const;
  cv::Size size() const { return size_; }

 private:
  double x_min_, y_min_, mpp_;
  cv::Size size_;
};

// Surimpressions sur l'image caméra (partie gauche de la fenêtre, ).

// Contours et numéros des tags détectés.
void drawDetections(cv::Mat& img, const std::vector<MarkerDetection>& detections);
// Grille du sol tous les "step" mètres, contour de l'arène et axes du repère monde.
void drawWorldGrid(cv::Mat& img, const CameraModel& camera,
                   const ArenaConfig& arena, double step = 0.1);
// Repère d'une pose (x rouge, y vert) au plan z = h.
void drawPoseAxes(cv::Mat& img, const CameraModel& camera, const Pose2D& pose,
                  double h, double length, int thickness = 2);
// Polyligne du monde (plan z = h) reprojetée dans l'image.
void drawWorldPolyline(cv::Mat& img, const CameraModel& camera,
                       const std::vector<Eigen::Vector2d>& points, double h, const cv::Scalar& color,
                       int thickness = 2);
// Texte lisible (fond sombre) à la ligne "row" en haut à gauche.
void drawStatusLine(cv::Mat& img, const std::string& text, int row,
                    const cv::Scalar& color = cv::Scalar(255, 255, 255));

// Rig : tout le matériel dont un outil a besoin (horloge, robot, caméra), en simulation ou
// en réel. Les outils (teleop, robot_tool calib, camera_tool arena, robot_nav) n'utilisent que cette
// classe : le même code tourne donc contre la sim et contre le vrai robot.
//
//   sim  : SimClock + SimWorld + SimFirmware + SimCamera
//   réel : SteadyClock + SerialTransport (USB) + OpenCvCamera (iPhone)
// Dans les deux cas le robot est un ProtocolRobot.

class SimCamera;
class SimFirmware;

struct RigOptions {
  bool sim = true;
  std::string config_dir = "config";
  std::string port;           // réel : port série ("" = config, "auto" = détection)
  std::string link;           // réel : "serial" ou "wifi" ("" = config)
  std::string camera_source;  // réel : index ou URL ("" = config)
  bool realtime = false;      // sim : ralentir au temps réel (affichage)
  bool camera = true;         // réel : false = sans caméra (--no-camera, téléop seulement)
  bool empty_arena = false;   // sim : sans les obstacles de params.yaml (sim) (calibrations)
  unsigned seed = 1;
};

// Lit les options communes : --sim | --real, --config, --port, --camera, --no-camera,
// --realtime, --seed.
// Les arguments non reconnus sont laissés dans "rest".
RigOptions parseRigOptions(int argc, char** argv, std::vector<std::string>& rest);

class Rig {
 public:
  explicit Rig(const RigOptions& options);
  ~Rig();
  Rig(const Rig&) = delete;
  Rig& operator=(const Rig&) = delete;

  bool isSim() const { return options_.sim; }
  IClock& clock() { return *clock_; }
  ProtocolRobot& robot() { return *robot_; }
  ICamera& camera() { return *camera_; }
  bool hasCamera() const { return camera_ != nullptr; }
  const CameraIntrinsics& intrinsics() const { return intrinsics_; }
  const SimSetup& setup() const { return setup_; }
  // Monde simulé (vérité terrain) ; nullptr en réel.
  SimWorld* world() { return world_.get(); }

  // Avance jusqu'au prochain instant : en sim, physique + une image ; en réel, attente.
  void tick(double dt);

  // Extrinsèques : lus dans le fichier de config (réel), sinon calibrés sur 30 images
  // (et sauvegardés en réel). force = recalibrer quand même.
  CameraModel cameraModel(bool force = false);
  // Chemin du fichier d'extrinsèques (réel).
  std::string extrinsicsPath() const { return extrinsics_path_; }

 private:
  RigOptions options_;
  SimSetup setup_;
  CameraIntrinsics intrinsics_;
  std::string extrinsics_path_;
  std::unique_ptr<IClock> clock_;
  SteadyClock wall_clock_;  // sim en temps réel
  Time next_tick_ = 0.0;    // réel : instant du prochain tour
  std::unique_ptr<SimWorld> world_;
  std::unique_ptr<ITransport> transport_;
  std::unique_ptr<ProtocolRobot> robot_;
  std::unique_ptr<ICamera> camera_;
  SimCamera* sim_camera_ = nullptr;  // = camera_ en sim (non possédé)
};

// Identification automatique du modèle moteur.
//
//  1) rampe lente de u sur chaque roue seule -> premier mouvement vu par la caméra (zone morte) ;
//  2) échelons u = 0.4 ... 1.0, chacun en avant puis en arrière (le robot reste au centre) ;
//     vitesse et rotation en régime permanent -> vR = v + omega b/2, vL = v - omega b/2 ;
//  3) rotations sur place (-u, +u) en complément ;
//  4) moindres carrés |v| = k (|u| - u0) pour chaque roue, puis tau et retard d sur les
//     réponses aux échelons ;
//  5) sécurité : arrêt si le robot approche à moins de 15 cm du bord ou si le tag est perdu.
// Toutes les mesures viennent de la caméra (même code en sim et en réel).

struct MotorCalibrationOptions {
  bool short_mode = false;    // 4 échelons au lieu de 7 (pile faible)
  double edge_margin = 0.15;  // distance minimale au bord de l'arène (m)
  bool verbose = true;
};

struct WheelSample {
  double u = 0.0;  // commande
  double v = 0.0;  // vitesse mesurée de la roue (m/s)
};

struct MotorCalibrationResult {
  bool ok = false;
  std::string error;
  MotorParams left, right;
  double tau = 0.1;
  double delay = 0.05;
  double ramp_deadzone_left = 0.0;   // estimation de la rampe (information)
  double ramp_deadzone_right = 0.0;
  double motor_time = 0.0;           // temps moteurs allumés (s)
  std::vector<WheelSample> left_samples, right_samples;
};

MotorCalibrationResult calibrateMotors(Rig& rig, const CameraModel& camera,
                                       const MotorCalibrationOptions& options);

// Ajuste |v| = k (|u| - u0) par moindres carrés sur les points où la roue tourne.
bool fitMotor(const std::vector<WheelSample>& samples, MotorParams& out);

// Remplace gain / deadzone / tau / delay dans la section "motors" de robot.yaml
// en gardant le reste du fichier (et les commentaires).
void writeMotorsToRobotYaml(const std::string& path, const MotorCalibrationResult& result);
