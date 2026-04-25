#pragma once

#include <Eigen/Core>
#include <deque>

#include "core.hpp"

// Filtre de Kalman étendu avec compensation de latence.
//
// État x = [x, y, theta, v, omega]. Entrée u = consigne (v_c, omega_c), appliquée
// par le robot après un retard d_cmd. Modèle : motionModel() dans core.hpp (le même que MPPI).
// Mesure z = [x, y, theta] (caméra), horodatée à l'instant de capture de l'image,
// donc dans le passé : on revient à cet instant, on corrige, et on re-prédit jusqu'à
// maintenant en rejouant les entrées (tampon circulaire de 2 s).

struct EkfConfig {
  double tau_v = 0.1, tau_omega = 0.1;  // constantes de temps du modèle (s)
  double cmd_delay = 0.05;              // retard de commande (s)
  Vector5d process_noise = (Vector5d() << 1e-5, 1e-5, 1e-4, 0.05, 0.5).finished();  // var/s
  Vector5d initial_std = (Vector5d() << 0.01, 0.01, 0.087, 0.05, 0.2).finished();
  double gate_chi2 = 11.34;             // chi2 à 3 ddl, 99 %
  double buffer_duration = 2.0;         // s
  double lost_timeout = 0.5;            // s sans mesure acceptée -> LOST
  bool latency_compensation = true;
  int max_rejections_before_reset = 15; // rejets consécutifs avant réinitialisation
  double max_step = 0.01;               // pas de prédiction maximal (s)
};

EkfConfig loadEkfConfig(const ConfigNode& root, const RobotConfig& robot);

enum class EkfHealth { Uninitialized, Ok, Lost };
const char* toString(EkfHealth h);

class Ekf {
 public:
  explicit Ekf(const EkfConfig& config);

  // Consigne envoyée à l'instant t (elle agira à t + cmd_delay).
  void addInput(Time t, const Twist2D& cmd);
  // Prédit jusqu'à l'instant t.
  void predict(Time t);
  // Intègre une mesure ; renvoie false si rejetée (gating, trop ancienne...).
  bool update(const PoseMeasurement& z, Time now);

  EkfHealth health(Time now) const;
  Pose2D pose() const { return {x_(0), x_(1), x_(2)}; }
  Twist2D velocity() const { return {x_(3), x_(4)}; }
  const Vector5d& state() const { return x_; }
  const Matrix5d& covariance() const { return P_; }
  int rejectedCount() const { return rejected_; }

 private:
  struct Snapshot {
    Time t;
    Vector5d x;
    Matrix5d P;
  };
  struct Input {
    Time t_effective;
    Twist2D cmd;
  };
  Twist2D inputAt(Time t) const;
  void propagate(Time t);  // prédit sans enregistrer d'instantané
  bool correct(const PoseMeasurement& z);
  void initialize(const PoseMeasurement& z);
  void trimBuffers();

  EkfConfig config_;
  Vector5d x_ = Vector5d::Zero();
  Matrix5d P_ = Matrix5d::Identity();
  Time time_ = 0.0;
  bool initialized_ = false;
  Time last_accept_time_ = -1e9;
  int consecutive_rejections_ = 0;
  int rejected_ = 0;
  std::deque<Snapshot> history_;
  std::deque<Input> inputs_;
};

// Alerte pile faible. Le firmware ne peut pas mesurer la pile : on compare la
// vitesse réellement mesurée (déplacement de la pose estimée sur une fenêtre) à la
// vitesse prévue par le modèle moteur, en ligne droite et au-dessus de la zone morte.
// Sous 70 % pendant plus de 3 s cumulées -> alerte. Presque à l'arrêt (< 20 %), le robot
// est plutôt bloqué par un obstacle : ces moments ne comptent pas.

struct BatteryMonitorConfig {
  double ratio = 0.7;          // vitesse mesurée / prévue en dessous de laquelle on s'inquiète
  double stall_ratio = 0.2;    // en dessous : robot calé contre un obstacle, pas la pile
  double duration = 3.0;       // durée cumulée avant l'alerte (s)
  double window = 0.5;         // fenêtre de mesure de la vitesse (s)
  double min_speed = 0.08;     // consigne minimale pour juger (m/s)
  double max_turn_rate = 0.6;  // au-delà, ce n'est pas une ligne droite (rad/s)
};

BatteryMonitorConfig loadBatteryMonitorConfig(const ConfigNode& section);

class BatteryMonitor {
 public:
  explicit BatteryMonitor(const BatteryMonitorConfig& config);
  // À chaque cycle : pose estimée et consigne réellement appliquée aux roues.
  void update(Time t, const Pose2D& estimate, const Twist2D& applied);
  bool alert() const { return low_time_ >= config_.duration; }

 private:
  struct Sample {
    Time t;
    double x, y;
    Twist2D cmd;
  };
  BatteryMonitorConfig config_;
  std::deque<Sample> samples_;
  double low_time_ = 0.0;
};
