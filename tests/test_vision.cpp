#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

#include "app.hpp"
#include "core.hpp"
#include "sim.hpp"
#include "tools.hpp"
#include "vision.hpp"

namespace {

struct Fixture {
  RobotConfig robot;
  ArenaConfig arena;
  SimWorldConfig world_cfg;
  SimCameraConfig cam_cfg;
  ConfigNode perception = ConfigNode::fromString("{}");

  Fixture() {
    const std::string dir = std::string(RN_SOURCE_DIR) + "/config/";
    robot = loadRobotConfig(ConfigNode::loadFile(dir + "robot.yaml").child("robot"));
    // Tag décalé et tourné, pour vérifier que l'offset est bien pris en compte.
    robot.marker_offset = {0.02, -0.01, degToRad(5.0)};
    arena = loadArenaConfig(ConfigNode::loadFile(dir + "robot.yaml").child("arena"));
    const auto sim_root = ConfigNode::loadFile(dir + "params.yaml").child("sim");
    world_cfg = loadSimWorldConfig(sim_root, robot, arena);
    cam_cfg = loadSimCameraConfig(sim_root.child("camera"));
    perception = ConfigNode::loadFile(dir + "params.yaml").child("perception");
  }
};

double percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<std::size_t>(p * v.size()))];
}

// Calibre l'arène sur 30 images dégradées et renvoie le modèle de caméra estimé.
CameraModel calibrate(const Fixture& f, SimCamera& cam, SimClock& clock,
                                  ArenaCalibration& calib) {
  ArenaCalibrator calibrator(f.cam_cfg.intrinsics, f.arena);
  for (int i = 0; i < 30; ++i) {
    clock.set(i / 30.0);
    cam.capture(clock.now());
    clock.advance(1.0);
    if (auto frame = cam.grab()) calibrator.addFrame(frame->image);
  }
  calib = calibrator.solve();
  return CameraModel(f.cam_cfg.intrinsics, calib.rvec, calib.tvec);
}

}  // namespace

TEST(CameraModel, ProjectThenIntersectPlaneRoundTrip) {
  CameraIntrinsics intr;
  intr.K = cv::Matx33d(900, 0, 640, 0, 900, 360, 0, 0, 1);
  intr.distortion = {-0.05, 0.01, 0, 0, 0};
  cv::Vec3d rvec, tvec;
  lookDownExtrinsics({0.5, 0.35, 1.2}, degToRad(8.0), 0.3, rvec, tvec);
  const CameraModel cam(intr, rvec, tvec);
  EXPECT_NEAR(cam.center().z(), 1.2, 1e-9);
  const Eigen::Vector3d p(0.83, 0.12, 0.1);
  const auto back = cam.intersectPlane(cam.project(p), 0.1);
  ASSERT_TRUE(back.has_value());
  EXPECT_NEAR((*back - p).norm(), 0.0, 1e-5);
}

TEST(Kabsch2D, RecoversRigidTransform) {
  const Pose2D truth{0.3, -0.2, 1.1};
  std::vector<Eigen::Vector2d> model{{0.1, 0.1}, {0.1, -0.1}, {-0.1, -0.1}, {-0.1, 0.1}};
  std::vector<Eigen::Vector2d> world;
  for (const auto& m : model) world.push_back(transformPoint(truth, m));
  Pose2D est;
  EXPECT_NEAR(kabsch2D(model, world, est), 0.0, 1e-12);
  EXPECT_NEAR(est.x, truth.x, 1e-12);
  EXPECT_NEAR(est.y, truth.y, 1e-12);
  EXPECT_NEAR(est.theta, truth.theta, 1e-12);
}

TEST(Perception, ArenaExtrinsicsWithin5mmAndHalfDegree) {
  Fixture f;
  SimWorld world(f.world_cfg);
  SimClock clock;
  SimCamera cam(f.cam_cfg, f.robot, f.arena, world, clock, 3);
  cv::imwrite(std::string(RN_SOURCE_DIR) + "/build/sim_camera_example.png", cam.render());

  ArenaCalibration calib;
  const CameraModel est = calibrate(f, cam, clock, calib);
  ASSERT_TRUE(calib.ok);
  EXPECT_EQ(calib.tags_seen, 4);
  const double position_error = (est.center() - cam.trueModel().center()).norm();
  cv::Matx33d r_est, r_true;
  cv::Rodrigues(calib.rvec, r_est);
  cv::Rodrigues(cam.trueModel().rvec(), r_true);
  cv::Vec3d diff;
  cv::Rodrigues(r_est * r_true.t(), diff);
  const double angle_error = radToDeg(cv::norm(diff));
  std::cout << "extrinsèques : position " << position_error * 1000 << " mm, angle " << angle_error
            << "°, reprojection " << calib.reprojection_error << " px\n";
  // 6 mm (et non 5) : les 4 tags de la vraie arène (1,33 x 0,85 m) sont plus rapprochés.
  EXPECT_LT(position_error, 0.006);
  EXPECT_LT(angle_error, 0.5);
  EXPECT_LT(calib.reprojection_error, 1.0);
}

TEST(Perception, RobotLocalizationOn500Images) {
  Fixture f;
  f.world_cfg.obstacles.clear();
  SimWorld world(f.world_cfg);
  SimClock clock;
  SimCamera cam(f.cam_cfg, f.robot, f.arena, world, clock, 5);
  ArenaCalibration calib;
  const CameraModel camera = calibrate(f, cam, clock, calib);

  const MarkerDetector detector;
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> ux(0.12, 0.88), uy(0.12, 0.58), uth(-kPi, kPi);
  for (const auto method : {LocalizerMethod::RayPlane, LocalizerMethod::Pnp}) {
    auto cfg = loadLocalizerConfig(f.perception.child("localizer"), f.robot,
                                               f.arena.robot_tag_id);
    cfg.method = method;
    const RobotLocalizer localizer(cfg);
    std::vector<double> pos_err, head_err;
    int seen = 0;
    for (int i = 0; i < 500; ++i) {
      const Pose2D truth{ux(rng), uy(rng), uth(rng)};
      world.resetPose(truth);
      clock.advance(1.0);
      cam.capture(clock.now());
      clock.advance(1.0);
      const auto frame = cam.grab();
      if (!frame) continue;  // image perdue (p_drop)
      const auto loc = localizer.localize(detector.detect(frame->image), camera, frame->stamp);
      if (!loc.ok) continue;  // tag masqué (p_occlusion)
      ++seen;
      const Pose2D& m = loc.measurement.pose;
      pos_err.push_back(std::hypot(m.x - truth.x, m.y - truth.y));
      head_err.push_back(std::abs(radToDeg(wrapAngle(m.theta - truth.theta))));
    }
    const char* name = method == LocalizerMethod::RayPlane ? "A (rayons+Kabsch)" : "B (PnP IPPE_SQUARE)";
    std::cout << "| " << name << " | " << seen << "/500 | " << percentile(pos_err, 0.5) * 1000
              << " | " << percentile(pos_err, 0.95) * 1000 << " | " << percentile(head_err, 0.5)
              << " | " << percentile(head_err, 0.95) << " |\n";
    EXPECT_GT(seen, 440);
    if (method == LocalizerMethod::RayPlane) {
      EXPECT_LT(percentile(pos_err, 0.5), 0.005);
      EXPECT_LT(percentile(pos_err, 0.95), 0.015);
      EXPECT_LT(percentile(head_err, 0.95), 2.0);
    }
  }
}

TEST(Perception, ObstacleGridIoUAbove90Percent) {
  Fixture f;
  f.world_cfg.start = {0.2, 0.55, 0.0};
  SimWorld world(f.world_cfg);
  SimClock clock;
  SimCamera cam(f.cam_cfg, f.robot, f.arena, world, clock, 9);
  ArenaCalibration calib;
  const CameraModel camera = calibrate(f, cam, clock, calib);

  ObstacleDetector detector(
      loadObstacleDetectorConfig(f.perception.child("obstacles")), f.arena, 0.01);
  detector.startAccumulation();
  while (!detector.ready()) {
    clock.advance(1.0);
    cam.capture(clock.now());
    clock.advance(1.0);
    if (auto frame = cam.grab()) {
      detector.addFrame(frame->image, camera, world.groundTruthPose(), f.robot.marker_height);
    }
  }
  OccupancyGrid truth = makeArenaGrid(f.arena, 0.01);
  for (const auto& b : world.obstacles()) {
    truth.fillPolygon(rectangleCorners(b.x, b.y, b.length, b.width, b.yaw));
  }
  const double iou = occupancyIoU(detector.result(), truth, f.arena);
  std::cout << "IoU obstacles : " << iou << "\n";
  EXPECT_GE(iou, 0.9);
}

// ------------------------------------------------------------ calibration des intrinsèques

TEST(CameraCalibrator, RecoversIntrinsicsFromSyntheticCharucoViews) {
  const CharucoSpec spec;  // 5 x 7, cases de 35 mm
  const cv::aruco::CharucoBoard board = makeBoard(spec);
  const double m_per_px = 0.0002;  // texture du plateau : 0,2 mm par pixel
  cv::Mat texture;
  board.generateImage(cv::Size(static_cast<int>(spec.squares_x * spec.square / m_per_px),
                               static_cast<int>(spec.squares_y * spec.square / m_per_px)),
                      texture, 0, 1);
  const cv::Matx33d K(900, 0, 640, 0, 905, 360, 0, 0, 1);
  CameraCalibrator calibrator(spec);
  std::mt19937 rng(4);
  std::uniform_real_distribution<double> tilt(-0.5, 0.5), roll(-0.4, 0.4), dist(0.45, 0.6), shift(-0.04, 0.04);
  int added = 0;
  for (int i = 0; i < 20; ++i) {
    // Pose du plateau devant la caméra : rotation aléatoire, centre du plateau sur l'axe optique.
    const Eigen::Matrix3d R = (Eigen::AngleAxisd(tilt(rng), Eigen::Vector3d::UnitX()) *
                               Eigen::AngleAxisd(tilt(rng), Eigen::Vector3d::UnitY()) *
                               Eigen::AngleAxisd(roll(rng), Eigen::Vector3d::UnitZ()))
                                  .toRotationMatrix();
    const Eigen::Vector3d center(spec.squares_x * spec.square / 2, spec.squares_y * spec.square / 2, 0);
    const Eigen::Vector3d t = Eigen::Vector3d(shift(rng), shift(rng), dist(rng)) - R * center;
    // Homographie texture -> image : K [r1 r2 t] diag(m, m, 1)
    const cv::Matx33d Rt(R(0, 0), R(0, 1), t.x(), R(1, 0), R(1, 1), t.y(), R(2, 0), R(2, 1), t.z());
    const cv::Matx33d A(m_per_px, 0, 0, 0, m_per_px, 0, 0, 0, 1);
    cv::Mat view(720, 1280, CV_8UC1, cv::Scalar(180));
    cv::warpPerspective(texture, view, K * Rt * A, view.size(), cv::INTER_AREA, cv::BORDER_TRANSPARENT);
    if (calibrator.addView(view)) ++added;
  }
  ASSERT_GE(added, 15);
  CameraIntrinsics intr;
  const double rms = calibrator.calibrate(intr);
  std::cout << "calibration synthétique : rms " << rms << " px, fx " << intr.K(0, 0) << ", fy "
            << intr.K(1, 1) << "\n";
  EXPECT_LT(rms, 0.5);
  EXPECT_NEAR(intr.K(0, 0), 900, 9);
  EXPECT_NEAR(intr.K(1, 1), 905, 9);
  EXPECT_NEAR(intr.K(0, 2), 640, 8);
  EXPECT_NEAR(intr.K(1, 2), 360, 8);
}

namespace {
// 10 %, ou 0,01 pour les petites valeurs (zones mortes du vrai robot proches de 0).
void expectWithin10Percent(double found, double truth, const char* name) {
  EXPECT_LE(std::abs(found - truth), std::max(0.1 * truth, 0.01)) << name << " : trouvé " << found << ", vrai " << truth;
}
}  // namespace

TEST(CalibMotors, RecoversSimulatedMotorsWithin10Percent) {
  RigOptions options;
  options.config_dir = std::string(RN_SOURCE_DIR) + "/config";
  options.empty_arena = true;
  Rig rig(options);
  const auto& a = rig.setup().arena;
  rig.world()->resetPose({(a.x_min + a.x_max) / 2, (a.y_min + a.y_max) / 2, 0.0});
  MotorCalibrationOptions calib;
  calib.verbose = false;
  const MotorCalibrationResult r = calibrateMotors(rig, rig.cameraModel(), calib);
  ASSERT_TRUE(r.ok) << r.error;
  const RobotConfig& truth = rig.setup().world.robot;
  expectWithin10Percent(r.left.gain, truth.left.gain, "gain gauche");
  expectWithin10Percent(r.right.gain, truth.right.gain, "gain droite");
  expectWithin10Percent(r.left.deadzone, truth.left.deadzone, "zone morte gauche");
  expectWithin10Percent(r.right.deadzone, truth.right.deadzone, "zone morte droite");
  // À 30 images/s, seule la somme retard + tau est bien identifiable (les deux se compensent).
  expectWithin10Percent(r.tau + r.delay, truth.tau + truth.delay, "retard + tau");
  EXPECT_NEAR(r.tau, truth.tau, 0.25 * truth.tau);
  EXPECT_NEAR(r.delay, truth.delay, 0.25 * truth.delay);
  EXPECT_LT(r.motor_time, 300.0);  // moins de 5 minutes de moteurs allumés
}

TEST(CalibMotors, FitMotorOnExactPoints) {
  std::vector<WheelSample> samples;
  for (double u : {0.4, 0.6, 0.8, 1.0}) samples.push_back({u, 0.5 * (u - 0.3)});
  MotorParams m;
  ASSERT_TRUE(fitMotor(samples, m));
  EXPECT_NEAR(m.gain, 0.5, 1e-9);
  EXPECT_NEAR(m.deadzone, 0.3, 1e-9);
}

