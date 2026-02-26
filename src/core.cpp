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
