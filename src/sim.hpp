#pragma once

#include <Eigen/Core>
#include <deque>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "core.hpp"

// Physique simulée du robot : unicycle à 2 roues, modèle moteur par roue
// (zone morte, gain, 1er ordre, retard), glissement aléatoire, collisions.
//
// La vérité terrain (groundTruthPose, obstacles) ne doit servir qu'aux métriques
// et au mode gt_pose : le pipeline en mode "full" ne la lit jamais.

// Boîte posée au sol (obstacle réel de la simulation).
struct Box {
  double x = 0.0, y = 0.0;          // centre (m)
  double length = 0.1, width = 0.1; // le long de x et de y de la boîte (m)
  double height = 0.06;             // (m)
  double yaw = 0.0;                 // (rad)
  std::string color = "red";
};

struct SimWorldConfig {
  RobotConfig robot;  // géométrie + VRAIS paramètres moteurs de la sim
  ArenaConfig arena;
  double dt = 0.001;            // pas d'intégration (s)
  double slip_sigma = 0.0;      // écart-type du glissement multiplicatif
  double slip_period = 0.05;    // durée pendant laquelle un tirage de glissement est gardé (s)
  double battery_sag = 0.0;     // perte relative de gain au bout de battery_duration
  double battery_duration = 3600.0;
  Pose2D start;
  std::vector<Box> obstacles;
  unsigned seed = 1;
};

// Charge params.yaml (sim) (physique + vrais moteurs + monde). Géométrie et arène sont fournies.
SimWorldConfig loadSimWorldConfig(const ConfigNode& sim_root,
                                  const RobotConfig& robot_geometry,
                                  const ArenaConfig& arena);
// Lit une liste d'obstacles (clé "obstacles") ; utilisé par params.yaml (sim) et les scénarios.
std::vector<Box> loadBoxes(const ConfigNode& node, double default_height);

// Distance du point (x, y) à la boîte (0 si le point est dedans).
double distanceToBox(double x, double y, const Box& box);

// Vrai si le disque (centre, rayon) touche la boîte.
bool discTouchesBox(double cx, double cy, double radius, const Box& box);

class SimWorld {
 public:
  explicit SimWorld(const SimWorldConfig& config);

  // Nouvelle commande reçue à l'instant now() ; elle prend effet après robot.delay.
  void setCommand(const WheelCmd& cmd);
  // Intègre la physique jusqu'à l'instant t (pas fixe de config.dt).
  void advanceTo(Time t);

  Time now() const { return time_; }
  Pose2D groundTruthPose() const { return pose_; }
  Twist2D groundTruthTwist() const;
  const std::vector<Box>& obstacles() const { return config_.obstacles; }
  const SimWorldConfig& config() const { return config_; }

  // Nombre de collisions (fronts montants : un contact prolongé compte une fois).
  int collisionCount() const { return collision_count_; }
  bool inContact() const { return in_contact_; }
  // Temps cumulé avec au moins une roue commandée (budget pile).
  double motorOnTime() const { return motor_on_time_; }

  // Place le robot (début d'un scénario) et remet les vitesses à zéro.
  void resetPose(const Pose2D& pose);

 private:
  struct PendingCmd {
    Time effective_time;
    WheelCmd cmd;
  };
  void step(double dt);
  double wheelTarget(const MotorParams& motor, double u) const;
  int poseContact(const Pose2D& pose) const;  // 0 rien, 1 feuille à plat, 2 mur ou boîte

  SimWorldConfig config_;
  std::mt19937 rng_;
  std::normal_distribution<double> slip_noise_;

  Time time_ = 0.0;
  Pose2D pose_;
  double v_left_ = 0.0, v_right_ = 0.0;  // vitesses des roues (m/s), sans glissement
  double slip_left_ = 0.0, slip_right_ = 0.0;
  Time next_slip_draw_ = 0.0;
  WheelCmd active_cmd_;
  std::deque<PendingCmd> pending_;
  int collision_count_ = 0;
  bool in_contact_ = false;
  double motor_on_time_ = 0.0;
};

// Scénarios de benchmark (config/scenarios/*.yaml) : départ, but, obstacles.

struct Scenario {
  std::string name;
  Pose2D start;
  Eigen::Vector2d goal{0.0, 0.0};
  std::optional<double> goal_heading;  // rad
  std::vector<Box> obstacles;
};

// Charge la liste "scenarios" d'un fichier ; obstacle_height sert de hauteur par défaut.
std::vector<Scenario> loadScenarios(const std::string& path, double obstacle_height);

// SimFirmware : le firmware de la carte, simulé. Il reçoit les mêmes lignes que le
// vrai port série, les analyse avec LE MÊME code que la carte (firmware/robot_fw/fw_protocol.h),
// applique le même watchdog et pilote SimWorld.

class SimFirmware : public ITransport {
 public:
  SimFirmware(SimWorld& world, const IClock& clock, double watchdog = 0.3, bool ack = true);

  bool writeLine(std::string_view line) override;
  std::optional<std::string> readLine() override;

  int validFrames() const { return valid_; }
  int invalidFrames() const { return invalid_; }
  bool watchdogFired() const { return watchdog_fired_; }

 private:
  void checkWatchdog();
  void reply(const char* body);

  SimWorld& world_;
  const IClock& clock_;
  double watchdog_;
  bool ack_;
  Time last_valid_ = 0.0;
  bool watchdog_fired_ = true;
  int valid_ = 0, invalid_ = 0;
  std::deque<std::string> outbox_;
};
