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

  const std::string& file() const { return file_; }

 private:
  ConfigNode(YAML::Node node, std::string file, std::string path);
  YAML::Node require(const std::string& key) const;
  std::string where(const std::string& key) const;

  YAML::Node node_;
  std::string file_;
  std::string path_;  // chemin depuis la racine du fichier, ex. "motors.left"
};

// Paramètres du robot (config/robot.yaml), partagés par sim, control et estimation.

// Modèle d'un moteur (une roue) : vitesse permanente
//   v_ss(u) = sign(u) * gain * max(0, |u| - deadzone)
struct MotorParams {
  double gain = 0.6;      // m/s par unité de commande au-delà de la zone morte
  double deadzone = 0.35; // u0 : en dessous, la roue ne tourne pas
};

struct RobotConfig {
  double wheel_base = 0.13;      // b, entraxe des roues motrices (m)
  double wheel_diameter = 0.065; // m
  double robot_radius = 0.10;    // rayon du disque englobant (m)
  Pose2D marker_offset;          // pose du tag M dans le repère robot R
  double marker_height = 0.10;   // hauteur du tag au-dessus du sol (m)
  double marker_size = 0.09;     // côté du tag noir (m)
  MotorParams left;
  MotorParams right;
  double tau = 0.1;              // constante de temps du 1er ordre (s)
  double delay = 0.05;           // retard de commande d (s)
};

RobotConfig loadRobotConfig(const ConfigNode& root);

// Vitesse maximale atteignable par une roue (|u| = 1).
double maxWheelSpeed(const MotorParams& motor);

// Modèle direct (régime permanent) : v = sign(u) * gain * max(0, |u| - deadzone).
double wheelSpeedFromCommand(const MotorParams& motor, double u);

// Modèle inverse : u = sign(v) * (deadzone + |v| / gain), saturé à [-1, 1] ; v = 0 donne u = 0.
double commandFromWheelSpeed(const MotorParams& motor, double v);

// Description de l'arène (config/robot.yaml, section arena) : zone de jeu, tags du sol,
// couleurs et hauteur des obstacles.

// Tag ArUco collé au sol ; (x, y) = centre du tag dans le repère monde W.
struct FloorTag {
  int id = 0;
  double x = 0.0;
  double y = 0.0;
};

// Plage HSV d'une couleur d'obstacle (conventions OpenCV : H dans [0, 179]).
// Si h_min > h_max, la plage fait le tour (cas du rouge : 170..10).
struct ColorRange {
  std::string name;
  int h_min = 0, h_max = 179;
  int s_min = 0, s_max = 255;
  int v_min = 0, v_max = 255;
};

struct ArenaConfig {
  double x_min = 0.0, y_min = 0.0, x_max = 1.0, y_max = 0.7;  // zone utilisable (m)
  double floor_tag_size = 0.12;  // côté du carré noir des tags sol (m)
  std::vector<FloorTag> floor_tags;
  int robot_tag_id = 10;
  double obstacle_height = 0.06;  // toutes les boîtes ont la même hauteur (m)
  std::vector<ColorRange> obstacle_colors;
};

ArenaConfig loadArenaConfig(const ConfigNode& root);

// Modèle de mouvement du robot, partagé par l'EKF (estimation) et MPPI (control).
//
// État x = [x, y, theta, v, omega], entrée u = consigne (v_c, omega_c) :
//   x     += v cos(theta) dt          y += v sin(theta) dt       theta += omega dt
//   v     += dt / tau_v (v_c - v)     omega += dt / tau_w (omega_c - omega)

using Vector5d = Eigen::Matrix<double, 5, 1>;
using Matrix5d = Eigen::Matrix<double, 5, 5>;

// Une étape du modèle.
Vector5d motionModel(const Vector5d& x, const Twist2D& u, double dt, double tau_v,
                     double tau_omega);
// Jacobienne du modèle par rapport à l'état (pour l'EKF).
Matrix5d motionJacobian(const Vector5d& x, double dt, double tau_v, double tau_omega);

// Chemin planifié, échantillonné tous les ~1 cm, partagé par planning et control.

struct PathPoint {
  double x = 0.0;
  double y = 0.0;
  double theta = 0.0;      // cap de la tangente (rad)
  double curvature = 0.0;  // dtheta / ds (1/m), positif = virage à gauche
  double s = 0.0;          // abscisse curviligne depuis le début (m)
};

struct Path {
  std::vector<PathPoint> points;
  std::optional<double> goal_heading;  // cap demandé au but (cliquer-glisser), sinon vide

  bool empty() const { return points.empty(); }
  double length() const { return points.empty() ? 0.0 : points.back().s; }
};

// Interfaces matérielles indépendantes d'OpenCV.
// Seules leurs implémentations changent entre simulation et réel.

// Canal ligne à ligne vers le robot (série USB, UDP, ou firmware simulé).
class ITransport {
 public:
  virtual ~ITransport() = default;
  // Envoie une ligne (le '\n' final est ajouté par l'implémentation).
  virtual bool writeLine(std::string_view line) = 0;
  // Renvoie une ligne complète reçue, sans le '\n', ou rien. Non bloquant.
  virtual std::optional<std::string> readLine() = 0;
};


// Journaux d'exécution :
//  - CsvLogger : une ligne de nombres par cycle de contrôle (log.csv) ;
//  - EventLog  : événements ponctuels (COLLISION, LOST, REPLAN...) dans events.csv.

class CsvLogger {
 public:
  // Ouvre (écrase) le fichier et écrit la ligne d'en-tête.
  CsvLogger(const std::string& path, const std::vector<std::string>& columns);
  // Écrit une ligne ; lève std::invalid_argument si le nombre de valeurs est faux.
  void log(const std::vector<double>& values);

 private:
  std::ofstream out_;
  std::size_t num_columns_;
};

class EventLog {
 public:
  explicit EventLog(const std::string& path);
  void log(Time t, const std::string& type, const std::string& message);

 private:
  std::ofstream out_;
};

// Crée et renvoie le dossier <base>/AAAAMMJJ_HHMMSS (heure locale).
std::string makeRunDirectory(const std::string& base = "runs");
