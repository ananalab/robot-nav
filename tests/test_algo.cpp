#include <Eigen/Geometry>
#include <algorithm>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <opencv2/calib3d.hpp>  // DIST_L2 est dans le module geometry en OpenCV 5
#include <opencv2/imgproc.hpp>
#include <random>
#include <string>

#include "app.hpp"
#include "control.hpp"
#include "core.hpp"
#include "estimation.hpp"
#include "fw_protocol.h"
#include "hw.hpp"
#include "planning.hpp"
#include "sim.hpp"

// ------------------------------------------------------------ wrapAngle
TEST(WrapAngle, KeepsValuesAlreadyInRange) {
  EXPECT_DOUBLE_EQ(wrapAngle(0.0), 0.0);
  EXPECT_DOUBLE_EQ(wrapAngle(1.0), 1.0);
  EXPECT_DOUBLE_EQ(wrapAngle(-1.0), -1.0);
}

TEST(WrapAngle, BoundariesFollowHalfOpenInterval) {
  // L'intervalle est (-pi, pi] : pi reste pi, -pi devient pi.
  EXPECT_NEAR(wrapAngle(kPi), kPi, 1e-12);
  EXPECT_NEAR(wrapAngle(-kPi), kPi, 1e-12);
}

TEST(WrapAngle, WrapsLargeAngles) {
  EXPECT_NEAR(wrapAngle(1.5 * kPi), -0.5 * kPi, 1e-12);
  EXPECT_NEAR(wrapAngle(-1.5 * kPi), 0.5 * kPi, 1e-12);
  EXPECT_NEAR(wrapAngle(7.0 * kPi + 0.1), -kPi + 0.1, 1e-9);
  EXPECT_NEAR(wrapAngle(-20.0 * kPi + 0.3), 0.3, 1e-9);
}

TEST(WrapAngle, ResultAlwaysInRange) {
  for (double a = -50.0; a <= 50.0; a += 0.037) {
    const double w = wrapAngle(a);
    EXPECT_GT(w, -kPi);
    EXPECT_LE(w, kPi);
    // Même angle modulo 2*pi : sinus et cosinus identiques.
    EXPECT_NEAR(std::sin(w), std::sin(a), 1e-9);
    EXPECT_NEAR(std::cos(w), std::cos(a), 1e-9);
  }
}

// ------------------------------------------------------------ SE(2)
TEST(SE2, ComposeWithInverseIsIdentity) {
  const Pose2D a{1.0, -2.0, 0.7};
  const Pose2D id = compose(a, inverse(a));
  EXPECT_NEAR(id.x, 0.0, 1e-12);
  EXPECT_NEAR(id.y, 0.0, 1e-12);
  EXPECT_NEAR(id.theta, 0.0, 1e-12);
  const Pose2D id2 = compose(inverse(a), a);
  EXPECT_NEAR(id2.x, 0.0, 1e-12);
  EXPECT_NEAR(id2.y, 0.0, 1e-12);
}

TEST(SE2, ComposeKnownValue) {
  // Robot en (1, 0) tourné de 90° ; un point 1 m devant lui est en (1, 1).
  const Pose2D robot{1.0, 0.0, kPi / 2};
  const Pose2D ahead{1.0, 0.0, 0.0};
  const Pose2D out = compose(robot, ahead);
  EXPECT_NEAR(out.x, 1.0, 1e-12);
  EXPECT_NEAR(out.y, 1.0, 1e-12);
  EXPECT_NEAR(out.theta, kPi / 2, 1e-12);
  const Eigen::Vector2d p = transformPoint(robot, {1.0, 0.0});
  EXPECT_NEAR(p.x(), 1.0, 1e-12);
  EXPECT_NEAR(p.y(), 1.0, 1e-12);
}

TEST(SE2, ComposeIsAssociative) {
  const Pose2D a{0.3, 0.1, 0.4}, b{-1.0, 2.0, -2.5}, c{0.5, 0.5, 3.0};
  const Pose2D left = compose(compose(a, b), c);
  const Pose2D right = compose(a, compose(b, c));
  EXPECT_NEAR(left.x, right.x, 1e-12);
  EXPECT_NEAR(left.y, right.y, 1e-12);
  EXPECT_NEAR(wrapAngle(left.theta - right.theta), 0.0, 1e-12);
}

// ------------------------------------------------------------ horloges
TEST(Clock, SimClockOnlyMovesWhenAsked) {
  SimClock clock;
  EXPECT_DOUBLE_EQ(clock.now(), 0.0);
  clock.advance(0.5);
  EXPECT_DOUBLE_EQ(clock.now(), 0.5);
  clock.sleepUntil(2.0);
  EXPECT_DOUBLE_EQ(clock.now(), 2.0);
  clock.sleepUntil(1.0);  // le passé ne fait pas reculer l'horloge
  EXPECT_DOUBLE_EQ(clock.now(), 2.0);
}

TEST(Clock, SteadyClockIsMonotonic) {
  SteadyClock clock;
  const Time t0 = clock.now();
  clock.sleepUntil(t0 + 0.01);
  EXPECT_GE(clock.now(), t0 + 0.01);
}

// ------------------------------------------------------------ config
TEST(Config, ReadsValidValues) {
  const auto cfg = ConfigNode::fromString("a: 1.5\nsub:\n  n: 3\n  ok: true\n  list: [1, 2, 3]\n");
  EXPECT_DOUBLE_EQ(cfg.getDouble("a", 0.0, 2.0), 1.5);
  EXPECT_EQ(cfg.child("sub").getInt("n", 0, 10), 3);
  EXPECT_TRUE(cfg.child("sub").getBool("ok"));
  EXPECT_EQ(cfg.child("sub").getDoubleList("list", 3).size(), 3u);
  EXPECT_DOUBLE_EQ(cfg.getDoubleOr("absent", 4.0, 0.0, 10.0), 4.0);
}

TEST(Config, MissingKeyGivesExplicitError) {
  const auto cfg = ConfigNode::fromString("sub:\n  x: 1\n", "test.yaml");
  try {
    cfg.child("sub").getDouble("y", 0.0, 1.0);
    FAIL() << "une ConfigError était attendue";
  } catch (const ConfigError& e) {
    EXPECT_NE(std::string(e.what()).find("test.yaml: sub.y"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("manquante"), std::string::npos) << e.what();
  }
}

TEST(Config, OutOfBoundsGivesExplicitError) {
  const auto cfg = ConfigNode::fromString("gain: 5\n", "robot.yaml");
  try {
    cfg.getDouble("gain", 0.05, 3.0);
    FAIL() << "une ConfigError était attendue";
  } catch (const ConfigError& e) {
    EXPECT_NE(std::string(e.what()).find("hors bornes"), std::string::npos) << e.what();
  }
}

TEST(Config, WrongTypeAndWrongListSizeAreRejected) {
  const auto cfg = ConfigNode::fromString("a: bonjour\nl: [1, 2]\n");
  EXPECT_THROW(cfg.getDouble("a", 0.0, 1.0), ConfigError);
  EXPECT_THROW(cfg.getDoubleList("l", 3), ConfigError);
  EXPECT_THROW(cfg.child("a"), ConfigError);
}

TEST(Config, ProjectRobotYamlLoads) {
  const auto root = ConfigNode::loadFile(std::string(RN_SOURCE_DIR) + "/config/robot.yaml").child("robot");
  const RobotConfig robot = loadRobotConfig(root);
  EXPECT_GT(robot.wheel_base, 0.0);
  EXPECT_GT(maxWheelSpeed(robot.left), 0.0);
}

TEST(Config, MissingFileIsAnError) {
  EXPECT_THROW(ConfigNode::loadFile("/pas/de/fichier.yaml"), ConfigError);
}

// ------------------------------------------------------------ logger
TEST(Logger, WritesHeaderAndRows) {
  const auto dir = std::filesystem::temp_directory_path() / "rn_test_logger";
  std::filesystem::create_directories(dir);
  const std::string path = (dir / "log.csv").string();
  {
    CsvLogger log(path, {"t", "x"});
    log.log({0.0, 1.5});
    EXPECT_THROW(log.log({1.0}), std::invalid_argument);
  }
  std::ifstream in(path);
  std::string header, row;
  std::getline(in, header);
  std::getline(in, row);
  EXPECT_EQ(header, "t,x");
  EXPECT_EQ(row, "0,1.5");
}

TEST(Logger, RunDirectoriesAreUnique) {
  const auto base = (std::filesystem::temp_directory_path() / "rn_test_runs").string();
  const std::string a = makeRunDirectory(base);
  const std::string b = makeRunDirectory(base);
  EXPECT_NE(a, b);
  EXPECT_TRUE(std::filesystem::is_directory(a));
  std::filesystem::remove_all(base);
}


namespace {
// Monde vide et grand, moteurs identiques, sans bruit.
SimWorldConfig idealWorld() {
  SimWorldConfig cfg;
  cfg.arena.x_min = -10.0;
  cfg.arena.y_min = -10.0;
  cfg.arena.x_max = 10.0;
  cfg.arena.y_max = 10.0;
  cfg.robot.left = {0.6, 0.35};
  cfg.robot.right = {0.6, 0.35};
  cfg.robot.tau = 0.1;
  cfg.robot.delay = 0.05;
  cfg.robot.wheel_base = 0.13;
  cfg.robot.robot_radius = 0.1;
  return cfg;
}
}  // namespace

TEST(SimWorld, StraightLineWhenGainsAreEqual) {
  SimWorld world(idealWorld());
  world.setCommand({0.8, 0.8});
  world.advanceTo(3.0);
  const Pose2D p = world.groundTruthPose();
  EXPECT_NEAR(p.y, 0.0, 1e-9);
  EXPECT_NEAR(p.theta, 0.0, 1e-9);
  // Vitesse permanente v = 0.6 * (0.8 - 0.35) = 0.27 m/s, pendant (3 - retard) s,
  // moins le temps de montée du 1er ordre (environ tau).
  const double v = 0.6 * (0.8 - 0.35);
  EXPECT_NEAR(p.x, v * (3.0 - 0.05 - 0.1), 0.01);
}

TEST(SimWorld, RotationInPlaceMatchesAnalyticRate) {
  SimWorld world(idealWorld());
  world.setCommand({-0.7, 0.7});
  world.advanceTo(2.0);  // 19 tau après le retard : régime permanent atteint
