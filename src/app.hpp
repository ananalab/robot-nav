#pragma once

#include <Eigen/Core>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core.hpp"
#include "planning.hpp"
#include "control.hpp"
#include "sim.hpp"

// Navigator : assemble planification, contrôle et conversion vers les roues.
// C'est le même objet en simulation, en benchmark et sur le vrai robot.
// Il ne dépend ni d'OpenCV ni du matériel.

// Tous les réglages de navigation, lus dans config/.
struct NavigatorConfig {
  double control_rate = 30.0;
  PlannerConfig planner;
  ControlLimits limits;
  RotationConfig rotation;
  PurePursuitConfig pure_pursuit;
  MppiConfig mppi;
  WheelMapperConfig wheel_mapper;
};

// Lit robot.yaml, params.yaml (planning) et params.yaml (control) dans le dossier donné.
NavigatorConfig loadNavigatorConfig(const std::string& config_dir);

enum class ControllerKind { PurePursuit, Mppi };

struct NavEvent {
  std::string type;
  std::string message;
};

class Navigator {
 public:
  explicit Navigator(const NavigatorConfig& config);
  ~Navigator();

  // Nouvelle carte d'obstacles : recalcul de l'inflation ; replanifie si le chemin est bloqué.
  void setMap(const OccupancyGrid& grid);
  void setGoal(const Eigen::Vector2d& goal, std::optional<double> heading);
  void setEmergencyStop(bool on) { emergency_stop_ = on; }
  bool emergencyStop() const { return emergency_stop_; }
  void setController(ControllerKind kind);
  ControllerKind controllerKind() const { return kind_; }
  void requestReplan() { replan_requested_ = true; }

  // Un cycle : (re)planifie si besoin, calcule la consigne et la commande des roues.
  // pose_valid = false (pose perdue) -> arrêt.
  WheelCmd step(const Pose2D& est, const Twist2D& vel, bool pose_valid,
                      Time t);

  const CostMap& costMap() const { return costmap_; }
  const Path& path() const { return path_; }
  const PlanResult& lastPlan() const { return last_plan_; }
  const IController& controller() const { return *controller_; }
  // Le contrôleur MPPI s'il est actif (affichage des trajectoires échantillonnées), sinon nul.
  const Mppi* mppi() const;
  ControlState controlState() const;
  Twist2D lastTwist() const { return last_twist_; }
  // Consigne réellement envoyée aux roues (après saturation et zone morte).
  Twist2D appliedTwist() const;
  const NavigatorConfig& config() const { return config_; }

  // Événements survenus depuis le dernier appel (REPLAN, PLAN_FAILED...), à journaliser.
  std::vector<NavEvent> takeEvents();

 private:
  bool needsReplan(const Pose2D& est, std::string& reason) const;
  void replan(const Pose2D& est, const std::string& reason);

  NavigatorConfig config_;
  CostMap costmap_;
  bool has_map_ = false;
  std::unique_ptr<IController> controller_;
  ControllerKind kind_ = ControllerKind::PurePursuit;
  WheelMapper mapper_;
  std::optional<Eigen::Vector2d> goal_;
  std::optional<double> goal_heading_;
  Path path_;
  PlanResult last_plan_;
  bool replan_requested_ = false;
  bool blocked_ = false;
  bool emergency_stop_ = false;
  Twist2D last_twist_;
  std::vector<NavEvent> events_;
};

// Un épisode de simulation sans affichage : le robot part de "start" et doit
// atteindre "goal". Sert au benchmark (apps/bench) et aux tests de bout en bout.

// Tout ce qu'il faut pour simuler : lu une fois dans config/.
struct SimSetup {
  std::string config_dir;
  RobotConfig robot;   // ce que le contrôleur croit (robot.yaml)
  ArenaConfig arena;
  SimWorldConfig world; // vrais paramètres de la sim (params.yaml (sim))
  NavigatorConfig nav;
  // Réglages forcés par le bench (valeurs négatives = garder la config).
  double camera_latency = -1.0;      // s
  bool latency_compensation = true;  // false = ablation
};

SimSetup loadSimSetup(const std::string& config_dir);

// Estimation d'état fournie au Navigator à chaque cycle.
struct StateEstimate {
  Pose2D pose;
  Twist2D vel;
  bool valid = false;
};

// Source d'estimation : vérité terrain (mode gt_pose) ou caméra + EKF (mode full).
class IStateSource {
 public:
  virtual ~IStateSource() = default;
  // Appelé une fois par cycle, après avoir avancé la physique jusqu'à t.
  virtual StateEstimate estimate(Time t) = 0;
  // Consigne (v, omega) envoyée ce cycle (utile à la prédiction de l'EKF).
  virtual void commandSent(const Twist2D& cmd, Time t) = 0;
  // Carte d'obstacles vue par le système.
  virtual OccupancyGrid buildMap() = 0;
  // Alerte pile faible (mode full uniquement).
  virtual bool lowBattery() const { return false; }
  // Mesures rejetées par le filtre.
  virtual int rejections() const { return 0; }
};

// Carte "vérité terrain" : arène + boîtes réelles de la sim.
OccupancyGrid groundTruthGrid(const ArenaConfig& arena,
                                        const std::vector<Box>& boxes, double resolution);

// Mode gt_pose : renvoie la pose vraie du simulateur, sans vision.
class GroundTruthSource : public IStateSource {
 public:
  GroundTruthSource(const SimWorld& world, double resolution);
  StateEstimate estimate(Time t) override;
  void commandSent(const Twist2D& /*cmd*/, Time /*t*/) override {}
  OccupancyGrid buildMap() override;

 private:
  const SimWorld& world_;
  double resolution_;
};

enum class PoseMode { GroundTruth, Full };

struct EpisodeOptions {
  PoseMode mode = PoseMode::GroundTruth;
  ControllerKind controller = ControllerKind::PurePursuit;
  unsigned seed = 0;
  double timeout = 40.0;      // s (temps simulé)
  double settle_time = 0.5;   // attente après l'arrivée avant de mesurer l'erreur finale (s)
  std::string log_dir;        // vide = pas de log
};

struct EpisodeMetrics {
  bool success = false;
  bool arrived = false;
  std::string failure;           // raison de l'échec (vide si succès)
  double time_to_goal = 0.0;     // s
  double planned_length = 0.0;   // m
  double traveled_length = 0.0;  // m
  double lateral_rms = 0.0;      // m
  double min_clearance = 1e9;    // distance minimale robot (disque) - obstacle (m)
  double final_position_error = 0.0;  // m
  double final_heading_error = 0.0;   // rad (0 si aucun cap demandé)
  int collisions = 0;
  int pose_rejections = 0;       // mesures rejetées par l'EKF (mode full)
  bool low_battery_alert = false;
  std::vector<double> compute_ms;  // temps de calcul par cycle
};

// Écrit le chemin courant (x, y, theta, courbure, s) dans un CSV, pour tools/plot_run.py.
void writePathCsv(const Path& path, const std::string& file);

EpisodeMetrics runEpisode(const SimSetup& setup, const Scenario& scenario,
                          const EpisodeOptions& options);
