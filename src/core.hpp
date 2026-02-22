#pragma once

#include <Eigen/Core>
#include <chrono>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <yaml-cpp/yaml.h>

// Types de base partagés par tous les modules.
// Unités SI : mètres, radians, secondes.

// Temps en secondes, toujours fourni par un IClock.
using Time = double;

// Pose dans le plan : position (x, y) et cap theta dans (-pi, pi].
struct Pose2D {
  double x{};
  double y{};
  double theta{};
};

// Vitesse du robot : linéaire v (m/s) et angulaire omega (rad/s).
struct Twist2D {
  double v{};
  double omega{};
};

// Commande des roues, normalisée dans [-1, 1].
struct WheelCmd {
  double left{};
  double right{};
};

// Mesure de pose issue de la caméra, horodatée à l'instant de capture.
struct PoseMeasurement {
  Time stamp{};
  Pose2D pose;
  Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();
};

// Fonctions sur les angles.

constexpr double kPi = 3.14159265358979323846;

// Ramène un angle quelconque dans l'intervalle (-pi, pi].
// Exemple : wrapAngle(3*pi/2) = -pi/2, wrapAngle(-pi) = pi.
double wrapAngle(double angle);

// Conversions degrés <-> radians.
constexpr double degToRad(double deg) { return deg * kPi / 180.0; }
constexpr double radToDeg(double rad) { return rad * 180.0 / kPi; }

// Transformations rigides du plan, SE(2).
//
// Une Pose2D "a" peut se lire comme la transformation qui envoie un point
// exprimé dans le repère A vers le repère parent :
//   p_parent = R(a.theta) * p_A + (a.x, a.y)

// Composition a ∘ b : si b est la pose de C dans A, et a la pose de A dans W,
// le résultat est la pose de C dans W.
Pose2D compose(const Pose2D& a, const Pose2D& b);

// Inverse : si a est la pose de A dans W, le résultat est la pose de W dans A.
Pose2D inverse(const Pose2D& a);

// Applique la pose à un point : R(theta) * p + t.
Eigen::Vector2d transformPoint(const Pose2D& pose, const Eigen::Vector2d& p);


// Horloges. Les algorithmes ne lisent jamais l'heure directement :
// ils passent par IClock, ce qui permet de rejouer la simulation plus vite
// que le temps réel, de façon déterministe.

class IClock {
 public:
  virtual ~IClock() = default;
  // Temps courant en secondes.
  virtual Time now() const = 0;
  // Attend jusqu'à l'instant t (ne fait rien si t est déjà passé).
  virtual void sleepUntil(Time t) = 0;
};

// Horloge réelle, monotone ; t = 0 à la construction.
class SteadyClock : public IClock {
 public:
  SteadyClock();
  Time now() const override;
  void sleepUntil(Time t) override;

 private:
  std::chrono::steady_clock::time_point start_;
};

// Horloge de simulation : le temps n'avance que quand la boucle de sim l'avance.
class SimClock : public IClock {
 public:
  Time now() const override { return now_; }
  // En simulation, "dormir" revient à sauter dans le temps.
  void sleepUntil(Time t) override;
  void advance(double dt) { now_ += dt; }
  void set(Time t) { now_ = t; }

 private:
  Time now_ = 0.0;
};

// Lecture de configuration YAML avec validation.
//
// Toute clé manquante ou valeur hors bornes lève une ConfigError avec un
// message qui donne le fichier et le chemin complet de la clé, par exemple :
//   "config/robot.yaml: motors.left.gain = 5 hors bornes [0.05, 3]"
// Le programme s'arrête donc dès le démarrage plutôt que de rouler avec
// une valeur absurde.

class ConfigError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Un nœud YAML qui se souvient de son fichier et de son chemin (pour les erreurs).
class ConfigNode {
 public:
  // Charge un fichier ; lève ConfigError s'il est absent ou mal formé.
  static ConfigNode loadFile(const std::string& path);
  // Construit depuis un texte YAML (utile dans les tests).
  static ConfigNode fromString(const std::string& yaml_text, const std::string& name = "<texte>");

  bool has(const std::string& key) const;
  // Sous-section obligatoire.
  ConfigNode child(const std::string& key) const;
  // Éléments d'une liste obligatoire (ex. liste d'obstacles).
  std::vector<ConfigNode> list(const std::string& key) const;

  double getDouble(const std::string& key, double min_value, double max_value) const;
  int getInt(const std::string& key, int min_value, int max_value) const;
  bool getBool(const std::string& key) const;
  std::string getString(const std::string& key) const;
  // Liste de nombres ; expected_size = 0 accepte n'importe quelle taille.
  std::vector<double> getDoubleList(const std::string& key, std::size_t expected_size = 0) const;

  // Variantes avec valeur par défaut si la clé est absente (bornes vérifiées sinon).
  double getDoubleOr(const std::string& key, double fallback, double min_value,
                     double max_value) const;

