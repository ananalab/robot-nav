#pragma once

#include <Eigen/Core>
#include <random>
#include <vector>

#include "core.hpp"
#include "planning.hpp"

// Interface commune des contrôleurs de suivi de chemin.

enum class ControlState { Rotating, Tracking, Arrived, Blocked, Stopped };

const char* toString(ControlState state);

// Limites communes à tous les contrôleurs (params.yaml (control), section "limits").
struct ControlLimits {
  double v_max = 0.25;      // m/s
  double omega_max = 3.0;   // rad/s
  double accel_max = 0.6;   // m/s^2
  double alpha_max = 8.0;   // rad/s^2
};

ControlLimits loadControlLimits(const ConfigNode& section);

// Réglages de la rotation sur place et de l'arrivée (section "rotation" / "goal").
struct RotationConfig {
  double start_threshold = 0.785;  // écart de cap qui déclenche une pré-rotation (rad, 45°)
  double stop_threshold = 0.17;    // fin de la pré-rotation (rad, ~10°)
  double gain = 3.0;               // omega = gain * erreur de cap (saturé)
  double omega_max = 1.5;          // vitesse de rotation max sur place (rad/s)
  double lag = 0.15;               // retard + inertie des moteurs : on anticipe le cap (s)
  double goal_tolerance = 0.02;    // distance d'arrivée (m)
  double heading_tolerance = 0.087;  // tolérance de cap final (rad, 5°)
  double hold_gain = 5.0;          // maintien de position pendant la rotation finale (1/s)
};

RotationConfig loadRotationConfig(const ConfigNode& root);

// Informations de débogage pour l'affichage et les logs.
struct ControlDebug {
  std::size_t path_index = 0;      // point du chemin le plus proche
  Eigen::Vector2d lookahead{0, 0}; // point d'anticipation (Pure Pursuit)
  double lateral_error = 0.0;      // écart latéral signé (m)
};

class IController {
 public:
  virtual ~IController() = default;
  // Calcule la consigne (v, omega) à partir de la pose et de la vitesse estimées.
  virtual Twist2D compute(const Pose2D& est, const Twist2D& vel,
                                const Path& path, Time t) = 0;
  // À appeler à chaque nouveau chemin.
  virtual void reset() = 0;
  virtual ControlState state() const = 0;
  virtual const ControlDebug& debug() const = 0;
  virtual const char* name() const = 0;
};

// Rotation sur place vers target_heading. Le robot continue de tourner pendant "lag"
// secondes après la commande (retard + inertie) : on corrige l'erreur du cap PRÉDIT
// theta + omega * lag, pour ne pas dépasser.
Twist2D rotateTowards(double target_heading, const Pose2D& est,
                            const Twist2D& vel, const RotationConfig& rotation);

// Phase finale (robot sur le but) : alignement sur le cap demandé s'il y en a un, avec
// maintien de position. arrived = true quand c'est fini (consigne nulle).
Twist2D finalAlignment(const Pose2D& est, const Twist2D& vel,
                             const Path& path, const RotationConfig& rotation,
                             const ControlLimits& limits, bool& arrived);

// Limite la variation d'une consigne : |new - prev| <= rate * dt.
double rateLimit(double target, double previous, double rate, double dt);

// Indice du point le plus proche, cherché seulement en avant de "from" sur
// "window" mètres (l'indice ne recule jamais).
std::size_t closestPointForward(const Path& path, const Pose2D& pose,
                                std::size_t from, double window);

// Écart latéral signé du robot au chemin au point i (positif = robot à gauche).
double lateralError(const Path& path, std::size_t i, const Pose2D& pose);

// WheelMapper : convertit une consigne (v, omega) en commandes de roues u dans [-1, 1]
// en inversant le modèle moteur.

struct WheelMapperConfig {
  double wheel_base = 0.13;
  MotorParams left, right;  // modèle moteur connu (robot.yaml)
  // Gestion de la zone morte, avec hystérésis (vitesses de roue en m/s) :
  double min_wheel_speed = 0.03;  // plus petite vitesse que la roue tient de façon fiable
  double start_speed = 0.015;     // roue à l'arrêt : on démarre si |v| dépasse ce seuil
  double stop_speed = 0.006;      // roue en marche : on s'arrête si |v| passe sous ce seuil
  // Frottements du vrai robot (commandes dans [0, 1]) : une roue qui doit tourner reçoit au
  // moins min_command, et au démarrage kick_command pendant kick_cycles cycles.
  double min_command = 0.0;
  double kick_command = 0.0;
  int kick_cycles = 0;
};

// Construit la config depuis robot.yaml (modèle) et la section wheel_mapper de params.yaml (control).
WheelMapperConfig loadWheelMapperConfig(const RobotConfig& robot,
                                        const ConfigNode& section);

class WheelMapper {
 public:
  explicit WheelMapper(const WheelMapperConfig& config);

  // (v, omega) -> (uL, uR). Garde un état (hystérésis), donc appeler une fois par cycle.
  WheelCmd map(const Twist2D& twist);

  // Vitesses de roue (m/s) calculées au dernier appel, après saturation et zone morte.
  double lastLeftSpeed() const { return last_left_; }
  double lastRightSpeed() const { return last_right_; }

 private:
  double applyDeadzone(double speed, bool& active) const;
  double applyFriction(double u, bool started, int& kick) const;

  WheelMapperConfig config_;
  bool left_active_ = false;
  bool right_active_ = false;
  int left_kick_ = 0, right_kick_ = 0;  // cycles de coup de pouce restants
  double last_left_ = 0.0, last_right_ = 0.0;
};

// Pure Pursuit : le robot vise un point du chemin situé à une distance
