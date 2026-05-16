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
// d'anticipation L devant lui, et suit l'arc de cercle qui y mène.
//
//   L = clamp(L0 + k_v |v|, L_min, L_max)
//   alpha = angle du point visé dans le repère du robot
//   kappa = 2 sin(alpha) / L                       (courbure de l'arc)
//   v = min(v_max, sqrt(a_lat_max / |kappa|), sqrt(2 a_dec d_but), k_approche d_but)
//   omega = v * kappa

struct PurePursuitConfig {
  double lookahead_base = 0.06;   // L0 (m)
  double lookahead_gain = 0.3;    // k_v (s)
  double lookahead_min = 0.05;    // (m)
  double lookahead_max = 0.15;    // (m)
  double lateral_accel_max = 0.3; // a_lat_max (m/s^2)
  double decel = 0.3;             // a_dec (m/s^2)
  double approach_gain = 2.0;     // v <= approach_gain * d_but près du but (1/s)
  double stop_distance = 0.01;    // distance où l'on coupe l'avance (m)
  double search_window = 0.3;     // fenêtre de recherche du point le plus proche (m)
};

PurePursuitConfig loadPurePursuitConfig(const ConfigNode& section);

class PurePursuit : public IController {
 public:
  PurePursuit(const PurePursuitConfig& config, const ControlLimits& limits,
              const RotationConfig& rotation);

  Twist2D compute(const Pose2D& est, const Twist2D& vel,
                        const Path& path, Time t) override;
  void reset() override;
  ControlState state() const override { return state_; }
  const ControlDebug& debug() const override { return debug_; }
  const char* name() const override { return "PurePursuit"; }

 private:
  Twist2D track(const Pose2D& est, const Twist2D& vel, const Path& path,
                      double distance_to_goal);
  Twist2D applyLimits(Twist2D cmd, Time t);

  PurePursuitConfig config_;
  ControlLimits limits_;
  RotationConfig rotation_;
  ControlState state_ = ControlState::Rotating;
  ControlDebug debug_;
  std::size_t index_ = 0;
  bool started_ = false;
  bool final_phase_ = false;
  Twist2D last_cmd_;
  Time last_time_ = -1.0;
};

// MPPI (Model Predictive Path Integral) : commande prédictive par échantillonnage.
//
// À chaque cycle :
//  1) on part d'une séquence nominale de N commandes (v, omega) (celle du cycle précédent,
//     décalée d'un pas : "démarrage à chaud") ;
//  2) on tire K séquences bruitées autour d'elle et on simule chacune avec le modèle du robot
//     (unicycle + 1er ordre, le même que l'EKF) ;
//  3) chaque trajectoire reçoit un coût S_k (écart au chemin, obstacles, à-coups, cap, vitesse,
//     point visé à la fin de l'horizon) ;
//  4) nouvelle séquence = moyenne des séquences pondérée par w_k = exp(-(S_k - S_min) / lambda) ;
//  5) on applique la première commande.
// Les derniers centimètres (approche fine du but, alignement final) sont confiés à Pure
// Pursuit : l'échantillonnage converge mal à 1 cm près.

struct MppiConfig {
  double horizon = 1.5;     // s
  double dt = 0.05;         // pas de l'horizon (s) -> 30 pas
  int samples = 512;        // K
  double sigma_v = 0.08;    // écart-type du bruit sur v (m/s)
  double sigma_omega = 1.0; // écart-type du bruit sur omega (rad/s)
  double lambda = 1.0;      // température : petite = on suit surtout le meilleur échantillon
  double w_path = 300.0;    // distance au chemin au carré
  double w_obstacle = 3.0;  // coût d'inflation (normalisé sur 253)
  double w_dcmd = 0.5;      // variation de commande au carré (normalisée par les sigmas)
  double w_heading = 1.0;   // erreur de cap par rapport au chemin au carré
  double w_speed = 50.0;    // dépassement de la vitesse de référence au carré
  double w_terminal = 100.0;  // distance au point visé en fin d'horizon au carré
  double v_min = -0.05;     // petite marche arrière permise (m/s)
  double handover_distance = 0.08;  // sous cette distance au but, Pure Pursuit termine (m)
  double tau_v = 0.1, tau_omega = 0.1;  // modèle moteur (robot.yaml)
  unsigned seed = 1;
};

MppiConfig loadMppiConfig(const ConfigNode& section, double tau, unsigned seed);

class Mppi : public IController {
 public:
  Mppi(const MppiConfig& config, const PurePursuitConfig& speed_profile, const ControlLimits& limits,
       const RotationConfig& rotation, const CostMap* costmap);

  Twist2D compute(const Pose2D& est, const Twist2D& vel,
                        const Path& path, Time t) override;
  void reset() override;
  ControlState state() const override { return state_; }
  const ControlDebug& debug() const override { return debug_; }
  const char* name() const override { return "MPPI"; }

  // Quelques trajectoires échantillonnées et la trajectoire retenue (affichage).
  const std::vector<std::vector<Eigen::Vector2d>>& sampledTrajectories() const { return sampled_; }
  const std::vector<Eigen::Vector2d>& optimalTrajectory() const { return optimal_; }

 private:
  Twist2D optimize(const Pose2D& est, const Twist2D& vel, const Path& path, Time t);
  double referenceSpeed(double distance_to_goal) const;
  double obstacleCost(double x, double y) const;  // 0..253, 254 = létal
  std::vector<Eigen::Vector2d> rollout(const Eigen::Matrix<double, 5, 1>& x0,
                                       const std::vector<Twist2D>& u) const;

  MppiConfig config_;
  PurePursuitConfig speed_;
  ControlLimits limits_;
  RotationConfig rotation_;
  const CostMap* costmap_;  // carte d'inflation du Navigator
  std::mt19937 rng_;
  std::vector<Twist2D> nominal_;
  Twist2D last_cmd_;
  ControlState state_ = ControlState::Rotating;
  ControlDebug debug_;
  std::size_t index_ = 0;
  bool started_ = false;
  bool handed_over_ = false;
  PurePursuit finisher_;  // approche finale
  std::vector<std::vector<Eigen::Vector2d>> sampled_;
  std::vector<Eigen::Vector2d> optimal_;
};
