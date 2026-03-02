#include "core.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

double wrapAngle(double angle) {
  // On décale de pi, on prend le reste modulo 2*pi dans (0, 2*pi], puis on
  // re-décale : le résultat tombe dans (-pi, pi].
  double a = std::fmod(angle + kPi, 2.0 * kPi);
  if (a <= 0.0) {
    a += 2.0 * kPi;
  }
  return a - kPi;
}

Pose2D compose(const Pose2D& a, const Pose2D& b) {
  // (x, y) = R(a.theta) * (b.x, b.y) + (a.x, a.y),  theta = a.theta + b.theta
  const double c = std::cos(a.theta);
  const double s = std::sin(a.theta);
  Pose2D out;
  out.x = a.x + c * b.x - s * b.y;
  out.y = a.y + s * b.x + c * b.y;
  out.theta = wrapAngle(a.theta + b.theta);
  return out;
}

Pose2D inverse(const Pose2D& a) {
  // inverse(R, t) = (R^T, -R^T t)
  const double c = std::cos(a.theta);
  const double s = std::sin(a.theta);
  Pose2D out;
  out.x = -(c * a.x + s * a.y);
  out.y = -(-s * a.x + c * a.y);
  out.theta = wrapAngle(-a.theta);
  return out;
}

Eigen::Vector2d transformPoint(const Pose2D& pose, const Eigen::Vector2d& p) {
  const double c = std::cos(pose.theta);
  const double s = std::sin(pose.theta);
  return {pose.x + c * p.x() - s * p.y(), pose.y + s * p.x() + c * p.y()};
}

SteadyClock::SteadyClock() : start_(std::chrono::steady_clock::now()) {}

Time SteadyClock::now() const {
  const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start_;
  return elapsed.count();
}

void SteadyClock::sleepUntil(Time t) {
  const auto target = start_ + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                   std::chrono::duration<double>(t));
  std::this_thread::sleep_until(target);
}

void SimClock::sleepUntil(Time t) {
  if (t > now_) {
    now_ = t;
  }
}

ConfigNode::ConfigNode(YAML::Node node, std::string file, std::string path)
    : node_(std::move(node)), file_(std::move(file)), path_(std::move(path)) {}

ConfigNode ConfigNode::loadFile(const std::string& path) {
  try {
    return ConfigNode(YAML::LoadFile(path), path, "");
  } catch (const YAML::Exception& e) {
    throw ConfigError(path + ": impossible de lire le fichier (" + e.what() + ")");
  }
}

ConfigNode ConfigNode::fromString(const std::string& yaml_text, const std::string& name) {
  try {
    return ConfigNode(YAML::Load(yaml_text), name, "");
  } catch (const YAML::Exception& e) {
    throw ConfigError(name + ": YAML invalide (" + e.what() + ")");
  }
}

std::string ConfigNode::where(const std::string& key) const {
  return file_ + ": " + (path_.empty() ? key : path_ + "." + key);
}

bool ConfigNode::has(const std::string& key) const {
  return node_.IsMap() && node_[key].IsDefined() && !node_[key].IsNull();
}

YAML::Node ConfigNode::require(const std::string& key) const {
  if (!has(key)) {
    throw ConfigError(where(key) + " : clé manquante");
  }
  return node_[key];
}

ConfigNode ConfigNode::child(const std::string& key) const {
  const YAML::Node sub = require(key);
  if (!sub.IsMap()) {
    throw ConfigError(where(key) + " : une section (clé: valeur) est attendue");
  }
  return ConfigNode(sub, file_, path_.empty() ? key : path_ + "." + key);
}

std::vector<ConfigNode> ConfigNode::list(const std::string& key) const {
  const YAML::Node seq = require(key);
  if (!seq.IsSequence()) {
    throw ConfigError(where(key) + " : une liste est attendue");
  }
  std::vector<ConfigNode> items;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    const std::string item_path = (path_.empty() ? key : path_ + "." + key) + "[" +
                                  std::to_string(i) + "]";
    items.push_back(ConfigNode(seq[i], file_, item_path));
  }
  return items;
}

double ConfigNode::getDouble(const std::string& key, double min_value, double max_value) const {
  double value = 0.0;
  try {
    value = require(key).as<double>();
  } catch (const YAML::Exception&) {
    throw ConfigError(where(key) + " : un nombre est attendu");
  }
  if (value < min_value || value > max_value) {
    std::ostringstream msg;
    msg << where(key) << " = " << value << " hors bornes [" << min_value << ", " << max_value
        << "]";
    throw ConfigError(msg.str());
  }
  return value;
}

int ConfigNode::getInt(const std::string& key, int min_value, int max_value) const {
  int value = 0;
  try {
    value = require(key).as<int>();
  } catch (const YAML::Exception&) {
    throw ConfigError(where(key) + " : un entier est attendu");
  }
  if (value < min_value || value > max_value) {
    std::ostringstream msg;
    msg << where(key) << " = " << value << " hors bornes [" << min_value << ", " << max_value
        << "]";
    throw ConfigError(msg.str());
  }
  return value;
}

bool ConfigNode::getBool(const std::string& key) const {
  try {
    return require(key).as<bool>();
  } catch (const YAML::Exception&) {
    throw ConfigError(where(key) + " : true ou false attendu");
  }
}

std::string ConfigNode::getString(const std::string& key) const {
  try {
    return require(key).as<std::string>();
  } catch (const YAML::Exception&) {
    throw ConfigError(where(key) + " : un texte est attendu");
  }
}

std::vector<double> ConfigNode::getDoubleList(const std::string& key,
                                              std::size_t expected_size) const {
  const YAML::Node seq = require(key);
  if (!seq.IsSequence()) {
    throw ConfigError(where(key) + " : une liste de nombres est attendue");
  }
  if (expected_size != 0 && seq.size() != expected_size) {
    throw ConfigError(where(key) + " : " + std::to_string(expected_size) +
                      " valeurs attendues, " + std::to_string(seq.size()) + " trouvées");
  }
  std::vector<double> values;
  try {
    for (const auto& item : seq) {
      values.push_back(item.as<double>());
    }
  } catch (const YAML::Exception&) {
    throw ConfigError(where(key) + " : la liste ne doit contenir que des nombres");
  }
  return values;
}

double ConfigNode::getDoubleOr(const std::string& key, double fallback, double min_value,
                               double max_value) const {
  return has(key) ? getDouble(key, min_value, max_value) : fallback;
}

namespace {
MotorParams loadMotor(const ConfigNode& node) {
  MotorParams m;
  m.gain = node.getDouble("gain", 0.05, 3.0);
  m.deadzone = node.getDouble("deadzone", 0.0, 0.9);
  return m;
}
}  // namespace

RobotConfig loadRobotConfig(const ConfigNode& root) {
  RobotConfig cfg;
  const ConfigNode geo = root.child("geometry");
  cfg.wheel_base = geo.getDouble("wheel_base", 0.05, 0.5);
  cfg.wheel_diameter = geo.getDouble("wheel_diameter", 0.02, 0.2);
  cfg.robot_radius = geo.getDouble("robot_radius", 0.03, 0.5);
  const auto offset = geo.getDoubleList("marker_offset", 3);
  cfg.marker_offset = {offset[0], offset[1], degToRad(offset[2])};
  cfg.marker_height = geo.getDouble("marker_height", 0.0, 0.5);
  cfg.marker_size = geo.getDouble("marker_size", 0.02, 0.3);

  const ConfigNode motors = root.child("motors");
  cfg.left = loadMotor(motors.child("left"));
  cfg.right = loadMotor(motors.child("right"));
  cfg.tau = motors.getDouble("tau", 0.005, 1.0);
  cfg.delay = motors.getDouble("delay", 0.0, 0.5);
  return cfg;
}

double maxWheelSpeed(const MotorParams& motor) { return motor.gain * (1.0 - motor.deadzone); }

double wheelSpeedFromCommand(const MotorParams& motor, double u) {
  const double magnitude = motor.gain * std::max(0.0, std::abs(u) - motor.deadzone);
  return u >= 0.0 ? magnitude : -magnitude;
}

double commandFromWheelSpeed(const MotorParams& motor, double v) {
  if (v == 0.0) {
    return 0.0;
  }
  const double magnitude = std::min(1.0, motor.deadzone + std::abs(v) / motor.gain);
  return v > 0.0 ? magnitude : -magnitude;
}

ArenaConfig loadArenaConfig(const ConfigNode& root) {
  ArenaConfig cfg;
  const auto bounds = root.getDoubleList("bounds", 4);
  cfg.x_min = bounds[0];
  cfg.y_min = bounds[1];
  cfg.x_max = bounds[2];
  cfg.y_max = bounds[3];
  if (cfg.x_max - cfg.x_min < 0.2 || cfg.y_max - cfg.y_min < 0.2) {
    throw ConfigError(root.file() + ": bounds : la zone doit faire au moins 20 cm de côté");
  }
  cfg.floor_tag_size = root.getDouble("floor_tag_size", 0.02, 0.5);
  for (const ConfigNode& tag : root.list("floor_tags")) {
    cfg.floor_tags.push_back({tag.getInt("id", 0, 49), tag.getDouble("x", -10.0, 10.0),
                              tag.getDouble("y", -10.0, 10.0)});
  }
  if (cfg.floor_tags.size() < 2) {
    throw ConfigError(root.file() + ": floor_tags : au moins 2 tags sol sont nécessaires");
  }
  cfg.robot_tag_id = root.getInt("robot_tag_id", 0, 49);
  cfg.obstacle_height = root.getDouble("obstacle_height", 0.0, 0.5);
  for (const ConfigNode& c : root.list("obstacle_colors")) {
    ColorRange range;
    range.name = c.getString("name");
    const auto h = c.getDoubleList("h", 2);
    const auto s = c.getDoubleList("s", 2);
    const auto v = c.getDoubleList("v", 2);
    range.h_min = static_cast<int>(h[0]);
    range.h_max = static_cast<int>(h[1]);
    range.s_min = static_cast<int>(s[0]);
    range.s_max = static_cast<int>(s[1]);
    range.v_min = static_cast<int>(v[0]);
    range.v_max = static_cast<int>(v[1]);
    cfg.obstacle_colors.push_back(range);
  }
  return cfg;
}

Vector5d motionModel(const Vector5d& x, const Twist2D& u, double dt, double tau_v,
                     double tau_omega) {
  Vector5d n = x;
  n(0) += x(3) * std::cos(x(2)) * dt;
  n(1) += x(3) * std::sin(x(2)) * dt;
  n(2) = wrapAngle(x(2) + x(4) * dt);
  n(3) += dt / tau_v * (u.v - x(3));
  n(4) += dt / tau_omega * (u.omega - x(4));
  return n;
}

Matrix5d motionJacobian(const Vector5d& x, double dt, double tau_v, double tau_omega) {
  // Dérivées partielles du modèle par rapport à l'état.
  Matrix5d F = Matrix5d::Identity();
  F(0, 2) = -x(3) * std::sin(x(2)) * dt;
  F(0, 3) = std::cos(x(2)) * dt;
  F(1, 2) = x(3) * std::cos(x(2)) * dt;
  F(1, 3) = std::sin(x(2)) * dt;
  F(2, 4) = dt;
  F(3, 3) = 1.0 - dt / tau_v;
  F(4, 4) = 1.0 - dt / tau_omega;
  return F;
}

CsvLogger::CsvLogger(const std::string& path, const std::vector<std::string>& columns)
    : out_(path), num_columns_(columns.size()) {
  if (!out_) {
    throw std::runtime_error("impossible de créer " + path);
  }
  for (std::size_t i = 0; i < columns.size(); ++i) {
    out_ << (i == 0 ? "" : ",") << columns[i];
  }
  out_ << '\n';
  out_ << std::setprecision(9);
}

void CsvLogger::log(const std::vector<double>& values) {
  if (values.size() != num_columns_) {
    throw std::invalid_argument("CsvLogger : " + std::to_string(values.size()) +
                                " valeurs pour " + std::to_string(num_columns_) + " colonnes");
  }
  for (std::size_t i = 0; i < values.size(); ++i) {
    out_ << (i == 0 ? "" : ",") << values[i];
  }
  out_ << '\n';
}

EventLog::EventLog(const std::string& path) : out_(path) {
  if (!out_) {
    throw std::runtime_error("impossible de créer " + path);
  }
  out_ << "t,type,message\n";
}

void EventLog::log(Time t, const std::string& type, const std::string& message) {
  out_ << t << ',' << type << ",\"" << message << "\"\n";
  out_.flush();  // les événements sont rares : on les veut sur disque même en cas de crash
}

std::string makeRunDirectory(const std::string& base) {
  // Nom du dossier = date et heure locales : c'est un nom de fichier, pas une
  // donnée d'algorithme, donc l'horloge murale est permise ici.
  const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm local{};
#ifdef _WIN32
  localtime_s(&local, &now);
#else
  localtime_r(&now, &local);
#endif
  std::ostringstream name;
  name << std::put_time(&local, "%Y%m%d_%H%M%S");
  std::filesystem::path dir = std::filesystem::path(base) / name.str();
  // Deux exécutions dans la même seconde : on ajoute un suffixe.
  for (int i = 1; std::filesystem::exists(dir); ++i) {
    dir = std::filesystem::path(base) / (name.str() + "_" + std::to_string(i));
  }
  std::filesystem::create_directories(dir);
  return dir.string();
}
