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
  // omega = (vR - vL) / b avec vR = -vL = 0.6 * (0.7 - 0.35)
  const double wheel = 0.6 * (0.7 - 0.35);
  const double expected_omega = 2.0 * wheel / 0.13;
  EXPECT_NEAR(world.groundTruthTwist().omega, expected_omega, 1e-6);
  EXPECT_NEAR(world.groundTruthTwist().v, 0.0, 1e-12);
  const Pose2D p = world.groundTruthPose();
  EXPECT_NEAR(p.x, 0.0, 1e-9);
  EXPECT_NEAR(p.y, 0.0, 1e-9);
}

TEST(SimWorld, CommandDelayIsRespected) {
  SimWorld world(idealWorld());
  world.setCommand({1.0, 1.0});
  world.advanceTo(0.049);
  EXPECT_DOUBLE_EQ(world.groundTruthPose().x, 0.0);
  world.advanceTo(0.1);
  EXPECT_GT(world.groundTruthPose().x, 0.0);
}

TEST(SimWorld, DeadzoneKeepsRobotStill) {
  SimWorld world(idealWorld());
  world.setCommand({0.3, 0.3});  // sous u0 = 0.35
  world.advanceTo(2.0);
  EXPECT_DOUBLE_EQ(world.groundTruthPose().x, 0.0);
  EXPECT_DOUBLE_EQ(world.motorOnTime() > 0.0, true);
}

TEST(SimWorld, DeterministicWithSameSeed) {
  SimWorldConfig cfg = idealWorld();
  cfg.slip_sigma = 0.1;
  cfg.seed = 7;
  SimWorld a(cfg), b(cfg);
  cfg.seed = 8;
  SimWorld c(cfg);
  for (SimWorld* w : {&a, &b, &c}) {
    w->setCommand({0.9, 0.6});
    w->advanceTo(2.0);
  }
  EXPECT_DOUBLE_EQ(a.groundTruthPose().x, b.groundTruthPose().x);
  EXPECT_DOUBLE_EQ(a.groundTruthPose().y, b.groundTruthPose().y);
  EXPECT_DOUBLE_EQ(a.groundTruthPose().theta, b.groundTruthPose().theta);
  EXPECT_NE(a.groundTruthPose().x, c.groundTruthPose().x);
}

TEST(SimWorld, CollisionBlocksRobotAndCountsOnce) {
  SimWorldConfig cfg = idealWorld();
  cfg.obstacles.push_back(Box{0.5, 0.0, 0.1, 0.4, 0.06, 0.0, "red"});
  SimWorld world(cfg);
  world.setCommand({1.0, 1.0});
  world.advanceTo(5.0);
  // Le robot (rayon 0.1) s'arrête contre la face x = 0.45 de la boîte.
  EXPECT_NEAR(world.groundTruthPose().x, 0.35, 0.005);
  EXPECT_EQ(world.collisionCount(), 1);
  EXPECT_TRUE(world.inContact());
}

TEST(SimWorld, BatterySagReducesSpeed) {
  SimWorldConfig cfg = idealWorld();
  cfg.battery_sag = 0.5;
  cfg.battery_duration = 2.0;
  SimWorld world(cfg);
  world.setCommand({1.0, 1.0});
  world.advanceTo(2.0);
  // En fin de batterie, la vitesse est divisée par 2.
  EXPECT_NEAR(world.groundTruthTwist().v, 0.5 * 0.6 * 0.65, 0.01);
}

TEST(DiscTouchesBox, RotatedBox) {
  const Box box{0.0, 0.0, 0.2, 0.02, 0.06, kPi / 4, "red"};  // fine planche à 45°
  EXPECT_TRUE(discTouchesBox(0.05, 0.05, 0.01, box));    // sur la diagonale
  EXPECT_FALSE(discTouchesBox(0.05, -0.05, 0.01, box));  // de l'autre côté
}

TEST(SimWorld, ProjectConfigLoads) {
  const std::string dir = std::string(RN_SOURCE_DIR) + "/config/";
  const auto robot = loadRobotConfig(ConfigNode::loadFile(dir + "robot.yaml").child("robot"));
  const auto arena = loadArenaConfig(ConfigNode::loadFile(dir + "robot.yaml").child("arena"));
  const auto cfg = loadSimWorldConfig(ConfigNode::loadFile(dir + "params.yaml").child("sim"), robot, arena);
  EXPECT_EQ(cfg.arena.floor_tags.size(), 4u);
  EXPECT_FALSE(cfg.obstacles.empty());
  // Les vrais moteurs de la sim ne sont pas ceux de robot.yaml.
  EXPECT_NE(cfg.robot.left.gain, robot.left.gain);
}


namespace {
WheelMapperConfig mapperConfig() {
  WheelMapperConfig cfg;
  cfg.wheel_base = 0.13;
  cfg.left = {0.6, 0.35};
  cfg.right = {0.66, 0.30};
  cfg.min_wheel_speed = 0.03;
  cfg.start_speed = 0.015;
  cfg.stop_speed = 0.006;
  return cfg;
}
}  // namespace

// ------------------------------------------------------------ modèle moteur
TEST(MotorModel, InverseOfForwardGivesBackCommand) {
  const MotorParams motor{0.6, 0.35};
  for (double u = 0.36; u <= 1.0; u += 0.01) {
    for (double sign : {1.0, -1.0}) {
      const double v = wheelSpeedFromCommand(motor, sign * u);
      const double u_back = commandFromWheelSpeed(motor, v);
      EXPECT_NEAR(u_back, sign * u, 0.01 * u) << "u = " << sign * u;
    }
  }
}

TEST(MotorModel, DeadzoneAndSaturation) {
  const MotorParams motor{0.6, 0.35};
  EXPECT_DOUBLE_EQ(wheelSpeedFromCommand(motor, 0.2), 0.0);
  EXPECT_DOUBLE_EQ(commandFromWheelSpeed(motor, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(commandFromWheelSpeed(motor, 10.0), 1.0);
  EXPECT_DOUBLE_EQ(commandFromWheelSpeed(motor, -10.0), -1.0);
}

// ------------------------------------------------------------ WheelMapper
TEST(WheelMapper, StraightAndRotation) {
  WheelMapper mapper(mapperConfig());
  mapper.map({0.2, 0.0});
  EXPECT_DOUBLE_EQ(mapper.lastLeftSpeed(), 0.2);
  EXPECT_DOUBLE_EQ(mapper.lastRightSpeed(), 0.2);
  mapper.map({0.0, 2.0});
  EXPECT_NEAR(mapper.lastLeftSpeed(), -0.13, 1e-12);
  EXPECT_NEAR(mapper.lastRightSpeed(), 0.13, 1e-12);
}

TEST(WheelMapper, SaturationKeepsCurvature) {
  WheelMapper mapper(mapperConfig());
  const Twist2D wanted{1.0, 3.0};  // bien trop rapide
  const WheelCmd cmd = mapper.map(wanted);
  const double vl = mapper.lastLeftSpeed();
  const double vr = mapper.lastRightSpeed();
  const double v = (vl + vr) / 2.0;
  const double omega = (vr - vl) / 0.13;
  EXPECT_NEAR(omega / v, wanted.omega / wanted.v, 1e-9);  // même courbure
  EXPECT_LE(std::abs(cmd.left), 1.0);
  EXPECT_LE(std::abs(cmd.right), 1.0);
  EXPECT_NEAR(std::max(std::abs(cmd.left), std::abs(cmd.right)), 1.0, 1e-9);
}

TEST(WheelMapper, CommandRoundTripsThroughModel) {
  WheelMapper mapper(mapperConfig());
  const WheelCmd cmd = mapper.map({0.15, 0.5});
  EXPECT_NEAR(wheelSpeedFromCommand(mapperConfig().left, cmd.left), 0.15 - 0.5 * 0.065,
              1e-9);
  EXPECT_NEAR(wheelSpeedFromCommand(mapperConfig().right, cmd.right), 0.15 + 0.5 * 0.065,
              1e-9);
}

TEST(WheelMapper, DeadzoneHysteresis) {
  WheelMapper mapper(mapperConfig());
  // Roue arrêtée : 1 cm/s est sous le seuil de démarrage (1.5 cm/s) -> reste à 0.
  mapper.map({0.01, 0.0});
  EXPECT_DOUBLE_EQ(mapper.lastLeftSpeed(), 0.0);
  // 2 cm/s : démarre, et la vitesse est portée au minimum fiable (3 cm/s).
  mapper.map({0.02, 0.0});
  EXPECT_DOUBLE_EQ(mapper.lastLeftSpeed(), 0.03);
  // En marche, 1 cm/s reste au minimum (pas d'arrêt avant le seuil d'arrêt).
  mapper.map({0.01, 0.0});
  EXPECT_DOUBLE_EQ(mapper.lastLeftSpeed(), 0.03);
  // 0.5 cm/s : sous le seuil d'arrêt -> arrêt.
  mapper.map({0.005, 0.0});
  EXPECT_DOUBLE_EQ(mapper.lastLeftSpeed(), 0.0);
  // Consigne nulle -> commande exactement nulle.
  const WheelCmd cmd = mapper.map({0.0, 0.0});
  EXPECT_DOUBLE_EQ(cmd.left, 0.0);
  EXPECT_DOUBLE_EQ(cmd.right, 0.0);
}

TEST(WheelMapper, ProjectConfigLoads) {
  const std::string dir = std::string(RN_SOURCE_DIR) + "/config/";
  const auto robot = loadRobotConfig(ConfigNode::loadFile(dir + "robot.yaml").child("robot"));
  const auto cfg = loadWheelMapperConfig(
      robot, ConfigNode::loadFile(dir + "params.yaml").child("control").child("wheel_mapper"));
  EXPECT_LE(cfg.stop_speed, cfg.start_speed);
  EXPECT_LE(cfg.start_speed, cfg.min_wheel_speed);
}

// ------------------------------------------------------------ Pure Pursuit

namespace {
Path straightPath(double length) {
  Path path;
  for (int i = 0; i <= static_cast<int>(length * 100); ++i) {
    PathPoint p;
    p.x = i * 0.01;
    p.s = p.x;
    path.points.push_back(p);
  }
  return path;
}
PurePursuit makePurePursuit() {
  PurePursuitConfig cfg;
  cfg.lookahead_gain = 0.0;  // L = L0 = 0.1 m pour vérifier la formule
  cfg.lookahead_base = 0.1;
  cfg.lookahead_min = 0.05;
  cfg.lookahead_max = 0.2;
  return PurePursuit(cfg, ControlLimits{}, RotationConfig{});
}
}  // namespace

TEST(PurePursuit, CurvatureFormulaOnStraightPath) {
  auto pp = makePurePursuit();
  const Path path = straightPath(1.0);
  // Robot à 2 cm à gauche du chemin, aligné : le point visé est à 10 cm devant
  // sur le chemin, alpha = atan2(-0.02, 0.1) environ, kappa = 2 sin(alpha) / d.
  const Pose2D est{0.0, 0.02, 0.0};
  pp.compute(est, {}, path, 0.0);                    // 1er appel : initialise l'état
  const Twist2D cmd = pp.compute(est, {0.1, 0.0}, path, 1.0);
  EXPECT_EQ(pp.state(), ControlState::Tracking);
  const double d = std::hypot(0.1, 0.02);
  const double kappa = 2.0 * std::sin(std::atan2(-0.02, 0.1)) / d;
  ASSERT_GT(cmd.v, 0.0);
  EXPECT_NEAR(cmd.omega / cmd.v, kappa, 0.05 * std::abs(kappa));
  EXPECT_LT(cmd.omega, 0.0);  // tourne vers la droite, vers le chemin
}

TEST(PurePursuit, RotatesFirstWhenFacingAway) {
  auto pp = makePurePursuit();
  const Twist2D cmd = pp.compute({0.0, 0.0, kPi / 2}, {}, straightPath(1.0), 0.0);
  EXPECT_EQ(pp.state(), ControlState::Rotating);
  EXPECT_DOUBLE_EQ(cmd.v, 0.0);
}

TEST(PurePursuit, ArrivesAndAlignsHeading) {
  auto pp = makePurePursuit();
  Path path = straightPath(0.5);
  path.goal_heading = kPi / 2;
  // Sur le but mais cap à 0 : rotation sur place vers +90°.
  Twist2D cmd = pp.compute({0.5, 0.0, 0.0}, {}, path, 0.0);
  EXPECT_EQ(pp.state(), ControlState::Rotating);
  EXPECT_GT(cmd.omega, 0.0);
  cmd = pp.compute({0.5, 0.0, kPi / 2}, {}, path, 0.1);
  EXPECT_EQ(pp.state(), ControlState::Arrived);
  EXPECT_DOUBLE_EQ(cmd.omega, 0.0);
}

// ------------------------------------------------------------ MPPI

namespace {
// Carte sans obstacle (coût nul partout) de -1 à 2 m en x, -1 à 1 m en y.
const CostMap& emptyCostMap() {
  static const CostMap map(OccupancyGrid(-1.0, -1.0, 300, 200, 0.01), InflationConfig{});
  return map;
}
// Mur létal au-dessus de y = 0,03 m (aucune inflation).
const CostMap& wallCostMap() {
  static CostMap map;
  if (map.grid().cols() == 0) {
    OccupancyGrid grid(-1.0, -1.0, 300, 200, 0.01);
    grid.fillPolygon({{-1.0, 0.03}, {2.0, 0.03}, {2.0, 1.0}, {-1.0, 1.0}});
    InflationConfig none;
    none.robot_radius = 0.0;
    none.margin = 0.0;
    none.d_max = 0.01;
    map = CostMap(grid, none);
  }
  return map;
}
Mppi makeMppi(unsigned seed) {
  MppiConfig cfg;
  cfg.seed = seed;
  return Mppi(cfg, PurePursuitConfig{}, ControlLimits{}, RotationConfig{}, &emptyCostMap());
}
}  // namespace

TEST(Mppi, DrivesForwardAlongStraightPath) {
  auto mppi = makeMppi(1);
  const Path path = straightPath(1.0);
  Twist2D cmd;
  for (int i = 0; i < 20; ++i) cmd = mppi.compute({0.0, 0.0, 0.0}, cmd, path, i * 0.033);
  EXPECT_EQ(mppi.state(), ControlState::Tracking);
  EXPECT_GT(cmd.v, 0.05);
  EXPECT_LT(std::abs(cmd.omega), 0.5);
}

TEST(Mppi, DeterministicWithFixedSeed) {
  auto a = makeMppi(3), b = makeMppi(3);
  const Path path = straightPath(1.0);
  const Pose2D pose{0.0, 0.02, 0.1};
  for (int i = 0; i < 5; ++i) {
    const Twist2D ca = a.compute(pose, {0.1, 0.0}, path, i * 0.033);
    const Twist2D cb = b.compute(pose, {0.1, 0.0}, path, i * 0.033);
    EXPECT_DOUBLE_EQ(ca.v, cb.v);
    EXPECT_DOUBLE_EQ(ca.omega, cb.omega);
  }
}

TEST(Mppi, AvoidsLethalSamples) {
  // Mur létal à y > 0.03 : la trajectoire retenue ne doit jamais y entrer.
  MppiConfig cfg;
  Mppi mppi(cfg, PurePursuitConfig{}, ControlLimits{}, RotationConfig{}, &wallCostMap());
  Twist2D cmd;
  for (int i = 0; i < 10; ++i) cmd = mppi.compute({0.0, 0.0, 0.0}, cmd, straightPath(1.0), i * 0.033);
  for (const auto& p : mppi.optimalTrajectory()) EXPECT_LE(p.y(), 0.03);
}

#ifdef NDEBUG
TEST(Mppi, P99ComputeTimeUnder20ms) {
  auto mppi = makeMppi(5);
  const Path path = straightPath(1.0);
  std::vector<double> ms;
  Twist2D cmd;
  for (int i = 0; i < 100; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    cmd = mppi.compute({0.0, 0.01, 0.05}, cmd, path, i * 0.033);
    ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  std::cout << "MPPI : p50 " << ms[50] << " ms, p99 " << ms[98] << " ms\n";
  EXPECT_LT(ms[98], 20.0);
}
#endif

#ifdef RN_WITH_OPENCV
#endif

namespace {
InflationConfig smallRobot() {
  InflationConfig cfg;
  cfg.robot_radius = 0.04;
  cfg.margin = 0.01;
  cfg.alpha = 20.0;
  cfg.d_max = 0.2;
  return cfg;
}

// Cellule libre (non létale) tirée au hasard.
GridIndex randomFreeCell(const CostMap& map, std::mt19937& rng) {
  std::uniform_int_distribution<int> uc(0, map.grid().cols() - 1), ur(0, map.grid().rows() - 1);
  for (int tries = 0; tries < 10000; ++tries) {
    const GridIndex c{uc(rng), ur(rng)};
    if (!map.lethal(c)) return c;
  }
  return {-1, -1};
}
}  // namespace

// ------------------------------------------------------------ grille
TEST(OccupancyGrid, WorldCellRoundTrip) {
  const OccupancyGrid grid(-0.5, 0.2, 100, 50, 0.01);
  const GridIndex c = grid.worldToCell(0.123, 0.456);
  EXPECT_EQ(c.col, 62);
  EXPECT_EQ(c.row, 25);
  const Eigen::Vector2d center = grid.cellToWorld(c);
  EXPECT_NEAR(center.x(), 0.125, 1e-12);
  EXPECT_NEAR(center.y(), 0.455, 1e-12);
  EXPECT_FALSE(grid.inside(grid.worldToCell(-0.51, 0.3)));
}

TEST(OccupancyGrid, FillRotatedRectangle) {
  OccupancyGrid grid(0.0, 0.0, 100, 100, 0.01);
  grid.fillPolygon(rectangleCorners(0.5, 0.5, 0.2, 0.1, kPi / 2));
  EXPECT_EQ(grid.at(grid.worldToCell(0.5, 0.58)), CellState::Occupied);  // le long de y
  EXPECT_EQ(grid.at(grid.worldToCell(0.58, 0.5)), CellState::Free);      // trop loin en x
}

TEST(OccupancyGrid, ArenaBorderIsOccupied) {
  ArenaConfig arena;  // 1.0 x 0.7 par défaut
  const OccupancyGrid grid = makeArenaGrid(arena, 0.01);
  EXPECT_EQ(grid.at(grid.worldToCell(0.5, 0.35)), CellState::Free);
  EXPECT_EQ(grid.at(grid.worldToCell(-0.015, 0.35)), CellState::Occupied);
  EXPECT_EQ(grid.at(grid.worldToCell(0.5, 0.715)), CellState::Occupied);
}

// ------------------------------------------------------------ distance + inflation
TEST(DistanceTransform, OneDimensionalKnownValues) {
  const double far = 1e12;
  std::vector<double> d;
  squaredDistance1D({far, 0.0, far, far, far, 0.0}, d);
  const std::vector<double> expected{1, 0, 1, 4, 1, 0};
  for (std::size_t i = 0; i < d.size(); ++i) EXPECT_DOUBLE_EQ(d[i], expected[i]) << i;
}

TEST(DistanceTransform, MatchesBruteForce) {
  std::mt19937 rng(3);
  const OccupancyGrid grid = randomBoxMap(60, 40, 0.01, 4, rng);
  const std::vector<double> edt = euclideanDistanceTransform(grid);
  for (int i = 0; i < grid.cols() * grid.rows(); ++i) {
    const GridIndex c = grid.fromLinear(i);
    double best = 1e9;
    for (int j = 0; j < grid.cols() * grid.rows(); ++j) {
      const GridIndex o = grid.fromLinear(j);
      if (grid.at(o) != CellState::Free) {
        best = std::min(best, std::hypot(c.col - o.col, c.row - o.row));
      }
    }
    ASSERT_NEAR(edt[i], best, 1e-9) << "cellule " << c.col << "," << c.row;
  }
}

#ifdef RN_WITH_OPENCV
TEST(DistanceTransform, MatchesOpenCv) {
  std::mt19937 rng(5);
  const OccupancyGrid grid = randomBoxMap(200, 120, 0.01, 10, rng);
  const std::vector<double> edt = euclideanDistanceTransform(grid);
  // OpenCV calcule la distance au pixel nul le plus proche : obstacles = 0.
  cv::Mat free_mask(grid.rows(), grid.cols(), CV_8UC1);
  for (int r = 0; r < grid.rows(); ++r)
    for (int c = 0; c < grid.cols(); ++c)
      free_mask.at<uchar>(r, c) = grid.at({c, r}) == CellState::Free ? 255 : 0;
  cv::Mat dist;
  cv::distanceTransform(free_mask, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
  for (int r = 0; r < grid.rows(); ++r)
    for (int c = 0; c < grid.cols(); ++c)
      ASSERT_NEAR(edt[grid.linear({c, r})], dist.at<float>(r, c), 1e-3);
}
#endif

TEST(CostMap, LethalInsideRadiusZeroBeyondDmax) {
  OccupancyGrid grid(0.0, 0.0, 100, 10, 0.01);
  grid.set({0, 5}, CellState::Occupied);
  for (int r = 0; r < 10; ++r) grid.set({0, r}, CellState::Occupied);  // mur à gauche
  const CostMap map(grid, smallRobot());  // létal jusqu'à 5 cm, nul après 20 cm
  EXPECT_TRUE(map.lethal({5, 5}));        // 5 cellules = 5 cm
  EXPECT_FALSE(map.lethal({6, 5}));
  EXPECT_GT(map.cost({6, 5}), 200);
  EXPECT_GT(map.cost({6, 5}), map.cost({10, 5}));  // décroissant
  EXPECT_EQ(map.cost({20, 5}), 0);
  EXPECT_TRUE(map.lethal({-1, 5}));  // hors grille
}

// ------------------------------------------------------------ A*
TEST(AStar, SameCostAsDijkstraOn200RandomMaps) {
  std::mt19937 rng(42);
  int with_path = 0;
  for (int trial = 0; trial < 200; ++trial) {
    const OccupancyGrid grid = randomBoxMap(80, 50, 0.01, 6, rng);
    const CostMap map(grid, smallRobot());
    const GridIndex start = randomFreeCell(map, rng);
    const GridIndex goal = randomFreeCell(map, rng);
    if (start.col < 0 || goal.col < 0) continue;
    const auto astar = searchGrid(map, start, goal, 2.0, true);
    const auto dijkstra = searchGrid(map, start, goal, 2.0, false);
    ASSERT_EQ(astar.found, dijkstra.found) << "essai " << trial;
    if (astar.found) {
      ++with_path;
      ASSERT_NEAR(astar.cost, dijkstra.cost, 1e-9) << "essai " << trial;
      EXPECT_LE(astar.expanded, dijkstra.expanded);
    }
  }
  EXPECT_GT(with_path, 100);  // le test n'a de sens que si la plupart des cartes ont un chemin
}

TEST(AStar, NoCornerCutting) {
  OccupancyGrid grid(0.0, 0.0, 3, 3, 1.0);
  InflationConfig none;
  none.robot_radius = 0.0;
  none.margin = 0.0;
  none.d_max = 0.5;
  // Deux obstacles en diagonale : on ne doit pas passer entre eux.
  grid.set({1, 0}, CellState::Occupied);
  grid.set({0, 1}, CellState::Occupied);
  const CostMap map(grid, none);
  EXPECT_FALSE(searchGrid(map, {0, 0}, {1, 1}, 0.0).found);
}

TEST(AStar, NoPathThroughWall) {
  OccupancyGrid grid(0.0, 0.0, 100, 60, 0.01);
  for (int r = 0; r < 60; ++r) grid.set({50, r}, CellState::Occupied);
  const CostMap map(grid, smallRobot());
  PlannerConfig cfg;
  const PlanResult res = planPath(map, {0.1, 0.3, 0.0}, {0.9, 0.3}, std::nullopt, cfg);
  EXPECT_EQ(res.status, PlanStatus::NoPath);
}

TEST(Planner, StartInLethalZoneEscapes) {
  OccupancyGrid grid(0.0, 0.0, 100, 60, 0.01);
  grid.fillPolygon(rectangleCorners(0.3, 0.3, 0.1, 0.1, 0.0));
  const CostMap map(grid, smallRobot());
  PlannerConfig cfg;
  // Départ à 2 cm du bord de la boîte, donc en zone létale (rayon 5 cm).
  const PlanResult res = planPath(map, {0.37, 0.3, 0.0}, {0.9, 0.5}, std::nullopt, cfg);
  ASSERT_EQ(res.status, PlanStatus::Ok);
  EXPECT_TRUE(res.start_escaped);
  EXPECT_FALSE(map.lethal(res.raw_cells.front()));
}

TEST(Planner, GoalInObstacleIsRejected) {
  OccupancyGrid grid(0.0, 0.0, 100, 60, 0.01);
  grid.fillPolygon(rectangleCorners(0.5, 0.3, 0.1, 0.1, 0.0));
  const CostMap map(grid, smallRobot());
  PlannerConfig cfg;
  EXPECT_EQ(planPath(map, {0.1, 0.1, 0}, {0.5, 0.3}, std::nullopt, cfg).status,
            PlanStatus::GoalInObstacle);
  EXPECT_EQ(planPath(map, {0.1, 0.1, 0}, {5.0, 0.3}, std::nullopt, cfg).status,
            PlanStatus::GoalOutsideMap);
}

// ------------------------------------------------------------ lissage
TEST(Smoother, ResampleSpacing) {
  const auto pts = resamplePolyline({{0, 0}, {0.1, 0}, {0.1, 0.055}}, 0.01);
  for (std::size_t i = 1; i < pts.size(); ++i) {
    EXPECT_LE((pts[i] - pts[i - 1]).norm(), 0.01 + 1e-9);
  }
  EXPECT_NEAR((pts.back() - Eigen::Vector2d(0.1, 0.055)).norm(), 0.0, 1e-12);
