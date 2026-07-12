// robot_nav : l'application de navigation.
//
//   robot_nav --mode sim [--gt-pose] [--realtime]      simulation
//   robot_nav --mode real                               vrai robot (iPhone + carte + pile)
//   options : --config <dossier>, --controller pp|mppi, --record (vidéo), --seed N
//   sans fenêtre : --headless --goal x,y[,cap_deg] [--timeout s] [--screenshot image.png] [--record]
//   avec fenêtre, fin au but : --auto-quit --goal x,y (mêmes options)
//
// Fenêtre : image caméra annotée à gauche, carte vue de dessus à droite.
//   clic = but ; cliquer-glisser = but + cap ; ESPACE = arrêt d'urgence ; c = Pure Pursuit / MPPI ;
//   o = re-détecter les obstacles ; r = replanifier ; q ou ÉCHAP = quitter.

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <optional>
#include <sstream>

#include "app.hpp"
#include "core.hpp"
#include "tools.hpp"

namespace {

// Seule variable globale du projet : un gestionnaire de signal (Ctrl-C) ne peut rien
// recevoir d'autre. La boucle principale la lit et s'arrête proprement (le robot reçoit "S").
std::atomic<bool> g_interrupted{false};
void onSigint(int) { g_interrupted = true; }

double msSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::string fmt(double x, int decimals = 1) {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(decimals);
  os << x;
  return os.str();
}

struct Options {
  bool gt_pose = false;
  bool record = false;
  bool headless = false;
  bool auto_quit = false;  // --auto-quit : fenêtre affichée, mais fin au but comme --headless
  ControllerKind controller = ControllerKind::PurePursuit;
  std::optional<Eigen::Vector2d> goal;
  std::optional<double> goal_heading;
  double timeout = 60.0;
  std::string screenshot;  // mode sans fenêtre : image de la dernière vue
};

Options parseOptions(const std::vector<std::string>& args) {
  Options o;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= args.size()) throw std::runtime_error("valeur manquante après " + a);
      return args[++i];
    };
    if (a == "--gt-pose") o.gt_pose = true;
    else if (a == "--record") o.record = true;
    else if (a == "--headless") o.headless = true;
    else if (a == "--auto-quit") o.auto_quit = true;
    else if (a == "--controller") o.controller = next() == "mppi" ? ControllerKind::Mppi : ControllerKind::PurePursuit;
    else if (a == "--timeout") o.timeout = std::stod(next());
    else if (a == "--screenshot") o.screenshot = next();
    else if (a == "--goal") {
      std::stringstream ss(next());
      std::vector<double> v;
      for (std::string part; std::getline(ss, part, ',');) v.push_back(std::stod(part));
      if (v.size() < 2) throw std::runtime_error("--goal x,y[,cap_deg]");
      o.goal = Eigen::Vector2d(v[0], v[1]);
      if (v.size() > 2) o.goal_heading = degToRad(v[2]);
    } else throw std::runtime_error("option inconnue : " + a);
  }
  return o;
}

// Ce que la souris a demandé (rempli par le callback, lu par la boucle).
struct MouseState {
  const TopView* top = nullptr;
  int top_offset_x = 0;              // la carte est à droite de l'image caméra
  double camera_scale = 1.0;         // image caméra affichée réduite
  double display_scale = 1.0;        // fenêtre entière affichée réduite (0,7)
  const CameraModel* camera = nullptr;
  std::optional<Eigen::Vector2d> press;
  std::optional<Eigen::Vector2d> goal;
  std::optional<double> heading;
  bool new_goal = false;
};

// Pixel de la fenêtre -> point du sol (carte de dessus ou image caméra).
std::optional<Eigen::Vector2d> windowToWorld(const MouseState& m, int x, int y) {
  if (x >= m.top_offset_x) {
    const cv::Point2d w = m.top->toWorld(x - m.top_offset_x, y);
    return Eigen::Vector2d(w.x, w.y);
  }
  const auto p = m.camera->intersectPlane(cv::Point2d(x / m.camera_scale, y / m.camera_scale), 0.0);
  if (!p) return std::nullopt;
  return Eigen::Vector2d(p->x(), p->y());
}

void onMouse(int event, int x, int y, int, void* userdata) {
  auto* m = static_cast<MouseState*>(userdata);
  x = static_cast<int>(x / m->display_scale);
  y = static_cast<int>(y / m->display_scale);
  if (event == cv::EVENT_LBUTTONDOWN) {
    m->press = windowToWorld(*m, x, y);
  } else if (event == cv::EVENT_LBUTTONUP && m->press) {
    const auto release = windowToWorld(*m, x, y);
    m->goal = m->press;
    m->heading.reset();
    // Cliquer-glisser de plus de 3 cm : le glissement donne le cap demandé au but.
    if (release && (*release - *m->press).norm() > 0.03) {
      const Eigen::Vector2d d = *release - *m->press;
      m->heading = std::atan2(d.y(), d.x());
    }
    m->press.reset();
    m->new_goal = true;
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, onSigint);
  try {
    std::vector<std::string> rest;
    const RigOptions rig_options = parseRigOptions(argc, argv, rest);
    const Options opt = parseOptions(rest);
    if (opt.gt_pose && !rig_options.sim) throw std::runtime_error("--gt-pose n'existe qu'en simulation");
    if ((opt.headless || opt.auto_quit) && !opt.goal) throw std::runtime_error("--headless et --auto-quit demandent --goal x,y");

    Rig rig(rig_options);
    const SimSetup& setup = rig.setup();
    const CameraModel camera_model = rig.cameraModel();
    StateEstimator estimator(rig_options.config_dir, setup.robot, setup.arena, setup.nav.planner.resolution);
    estimator.setCameraModel(camera_model);
    Navigator nav(setup.nav);
    nav.setController(opt.controller);

    // Dossier de l'exécution : copie de la config, logs, vidéo.
    const std::string run_dir = makeRunDirectory("runs");
    std::filesystem::copy(rig_options.config_dir, run_dir + "/config", std::filesystem::copy_options::recursive);
    CsvLogger log(run_dir + "/log.csv",
                        {"t", "meas_x", "meas_y", "meas_theta", "est_x", "est_y", "est_theta", "cov_xx",
                         "cov_yy", "cov_tt", "v_cmd", "omega_cmd", "pwm_left", "pwm_right", "path_index",
                         "lateral_error", "nav_ms", "vision_ms", "gt_x", "gt_y", "gt_theta"});
    EventLog events(run_dir + "/events.csv");
    std::cout << "Exécution : " << run_dir << "\n";

    // Obstacles : vote sur 20 images au démarrage (ou vérité terrain en mode --gt-pose).
    auto detectObstacles = [&]() {
      if (opt.gt_pose) {
        nav.setMap(groundTruthGrid(setup.arena, rig.world()->obstacles(), setup.nav.planner.resolution));
        return;
      }
      // Vraie caméra : on jette d'abord 1 s d'images (juste après l'ouverture, l'exposition
      // s'ajuste et les carrés rouges peuvent ne pas être vus).
      for (int seen = 0, i = 0; !rig.isSim() && seen < 30 && i < 300; ++i) {
        rig.tick(1.0 / 30.0);
        if (rig.camera().grab()) ++seen;
      }
      estimator.startObstacleDetection();
      for (int i = 0; i < 600 && !estimator.obstaclesReady(); ++i) {
        rig.tick(1.0 / 30.0);
        if (const auto f = rig.camera().grab()) estimator.addObstacleFrame(*f);
      }
      nav.setMap(estimator.obstacleGrid());
      events.log(rig.clock().now(), "OBSTACLES", "carte des obstacles mise a jour");
    };
    detectObstacles();
    if (opt.goal) nav.setGoal(*opt.goal, opt.goal_heading);

    // Image caméra réduite à 800 px de large ; carte de dessus à la même hauteur (lisible et
    // facile à cliquer : environ 1,7 mm par pixel pour l'arène de 1 x 0,7 m).
    const double camera_scale = 800.0 / camera_model.intrinsics().width;
    const double panel_height = camera_model.intrinsics().height * camera_scale;
    const OccupancyGrid& grid = nav.costMap().grid();
    const TopView top(grid, grid.rows() * grid.resolution() / panel_height);
    MouseState mouse;
    mouse.top = &top;
    mouse.camera = &camera_model;
    mouse.camera_scale = camera_scale;
    mouse.top_offset_x = 800;
    mouse.display_scale = 0.7;
    mouse.goal = opt.goal;  // but donné en ligne de commande : affiché sur la carte
    const std::string window = "robot_nav";
    if (!opt.headless) {
      cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
      cv::setMouseCallback(window, onMouse, &mouse);
    }
    std::optional<cv::VideoWriter> video;

    const double dt = 1.0 / setup.nav.control_rate;
    double motor_time = 0.0, nav_ms = 0.0, vision_ms = 0.0;
    bool battery_warned = false;
    std::optional<Time> arrived_at;
    cv::Mat last_image;
    const Time start = rig.clock().now();

    // Image de la fenêtre : caméra annotée à gauche, carte de dessus à droite.
    auto render = [&](Time now, const StateEstimate& est) {
      cv::Mat cam = last_image.clone();
      drawDetections(cam, estimator.lastDetections());
      std::vector<Eigen::Vector2d> path_pts;
      for (const auto& p : nav.path().points) path_pts.emplace_back(p.x, p.y);
      drawWorldPolyline(cam, camera_model, path_pts, 0.0, cv::Scalar(0, 200, 0), 2);
      if (const Mppi* mppi = nav.mppi()) {
        for (const auto& traj : mppi->sampledTrajectories()) drawWorldPolyline(cam, camera_model, traj, 0.0, cv::Scalar(200, 200, 200), 1);
        drawWorldPolyline(cam, camera_model, mppi->optimalTrajectory(), 0.0, cv::Scalar(255, 0, 255), 2);
      }
      drawPoseAxes(cam, camera_model, est.pose, 0.0, 0.1, 3);
      const Eigen::Vector2d la = nav.controller().debug().lookahead;
      drawWorldPolyline(cam, camera_model, {la, la + Eigen::Vector2d(0.005, 0)}, 0.0, cv::Scalar(0, 255, 255), 8);
      cv::resize(cam, cam, cv::Size(), camera_scale, camera_scale);

      cv::Mat map = top.blankImage();
      top.drawCostMap(map, nav.costMap());
      top.drawPath(map, nav.path(), cv::Scalar(0, 150, 0), 2);
      if (rig.world()) top.drawRobot(map, rig.world()->groundTruthPose(), setup.robot.robot_radius, cv::Scalar(180, 180, 180));
      top.drawRobot(map, est.pose, setup.robot.robot_radius, est.valid ? cv::Scalar(200, 0, 0) : cv::Scalar(0, 0, 255));
      if (mouse.goal) top.drawPoint(map, mouse.goal->x(), mouse.goal->y(), cv::Scalar(0, 0, 255), 6);
      if (map.rows < cam.rows) cv::copyMakeBorder(map, map, 0, cam.rows - map.rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
      if (cam.rows < map.rows) cv::copyMakeBorder(cam, cam, 0, map.rows - cam.rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
      cv::Mat images, screen;
      cv::hconcat(cam, map, images);
      // Bandeau d'état sous les images (il ne cache rien).
      cv::copyMakeBorder(images, screen, 0, 75, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(30, 30, 30));
      cv::Mat band = screen(cv::Rect(0, images.rows, screen.cols, 75));

      const std::string health = opt.gt_pose ? "VERITE" : toString(estimator.health(now));
      drawStatusLine(band, "EKF " + health + " | " + toString(nav.controlState()) + " (" +
                                      nav.controller().name() + ") | nav " + fmt(nav_ms, 1) + " ms | vision " +
                                      fmt(vision_ms, 1) + " ms | aller-retour " +
                                      fmt(rig.robot().lastRoundTrip() * 1000, 0) + " ms | moteurs " + fmt(motor_time, 0) + " s", 0);
      drawStatusLine(band, "clic = but, glisser = cap | ESPACE arret d'urgence | c controleur | o obstacles | r replanifier | q quitter", 1,
                          cv::Scalar(0, 255, 255));
      if (nav.emergencyStop()) drawStatusLine(band, battery_warned ? "PILE FAIBLE : robot arrete" : "ARRET D'URGENCE (ESPACE pour reprendre)", 2, cv::Scalar(0, 0, 255));
      return screen;
    };

    int display_count = 0;
    while (!g_interrupted) {
      rig.tick(dt);
      const Time now = rig.clock().now();

      // 1) Perception
      if (const auto frame = rig.camera().grab()) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!opt.gt_pose) estimator.processFrame(*frame, now);
        vision_ms = msSince(t0);
        last_image = frame->image;
      }
      StateEstimate est;
      if (opt.gt_pose) {
        est = {rig.world()->groundTruthPose(), rig.world()->groundTruthTwist(), true};
      } else {
        est = estimator.estimate(now);
      }

      // 2) Navigation
      if (mouse.new_goal && mouse.goal) {
        nav.setGoal(*mouse.goal, mouse.heading);
        events.log(now, "GOAL", fmt(mouse.goal->x(), 3) + "," + fmt(mouse.goal->y(), 3));
        mouse.new_goal = false;
        arrived_at.reset();
      }
      const auto t0 = std::chrono::steady_clock::now();
      const WheelCmd cmd = nav.step(est.pose, est.vel, est.valid, now);
      nav_ms = msSince(t0);
      rig.robot().send(cmd);
      estimator.commandSent(nav.appliedTwist(), now);
      if (cmd.left != 0.0 || cmd.right != 0.0) motor_time += dt;
      for (const auto& e : nav.takeEvents()) events.log(now, e.type, e.message);
      if (estimator.lowBattery() && !battery_warned) {
        battery_warned = true;
        nav.setEmergencyStop(true);
        events.log(now, "LOW_BATTERY", "Pile faible : robot arrete");
        std::cout << "PILE FAIBLE : robot arrêté. Seule une pile neuve permet de continuer (puis refaire robot_tool calib).\n";
      }

      // 3) Journal
      const auto& m = estimator.lastLocalization();
      const Eigen::Matrix3d cov = estimator.ekf().covariance().topLeftCorner<3, 3>();
      const Pose2D gt = rig.world() ? rig.world()->groundTruthPose() : Pose2D{NAN, NAN, NAN};
      log.log({now, m.ok ? m.measurement.pose.x : NAN, m.ok ? m.measurement.pose.y : NAN,
               m.ok ? m.measurement.pose.theta : NAN, est.pose.x, est.pose.y, est.pose.theta, cov(0, 0),
               cov(1, 1), cov(2, 2), nav.lastTwist().v, nav.lastTwist().omega, cmd.left * 255, cmd.right * 255,
               static_cast<double>(nav.controller().debug().path_index), nav.controller().debug().lateral_error,
               nav_ms, vision_ms, gt.x, gt.y, gt.theta});

      // 4) Mode sans fenêtre (ou --auto-quit) : fin au but (ou délai dépassé)
      if (opt.headless || opt.auto_quit) {
        if (opt.headless && opt.record && !last_image.empty()) {  // vidéo aussi sans fenêtre (démos, rapports)
          const cv::Mat screen = render(now, est);
          if (!video) video.emplace(run_dir + "/video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), setup.nav.control_rate, screen.size());
          video->write(screen);
        }
        if (nav.controlState() == ControlState::Arrived && !arrived_at) arrived_at = now;
        const bool done = arrived_at && now - *arrived_at > 0.5;
        if (done || now - start > opt.timeout || nav.controlState() == ControlState::Blocked) {
          std::cout << "État final : " << toString(nav.controlState()) << "\n";
          std::cout << "Pose estimée : " << fmt(est.pose.x, 3) << " " << fmt(est.pose.y, 3) << " "
                    << fmt(radToDeg(est.pose.theta), 1) << " deg\n";
          if (!opt.screenshot.empty() && !last_image.empty()) cv::imwrite(opt.screenshot, render(now, est));
          if (rig.world()) {
            std::cout << "Erreur finale (vérité sim) : "
                      << fmt(1000 * std::hypot(gt.x - opt.goal->x(), gt.y - opt.goal->y()), 1) << " mm\n";
          }
          break;
        }
        if (opt.headless) continue;
      }

      // 5) Affichage : image caméra annotée | carte de dessus
      if (last_image.empty()) continue;
      const cv::Mat screen = render(now, est);
      // Fenêtre et vidéo à 70 % : afficher et encoder une grande image ralentit la boucle.
      cv::Mat small;
      cv::resize(screen, small, cv::Size(), mouse.display_scale, mouse.display_scale, cv::INTER_AREA);
      if (opt.record) {
        if (!video) video.emplace(run_dir + "/video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), setup.nav.control_rate, small.size());
        video->write(small);
      }
      // Sur Mac, rafraîchir la fenêtre (imshow + waitKey) prend environ 30 ms : on ne le fait
      // qu'un tour sur trois, pour que la boucle de contrôle garde ses 30 tours par seconde.
      if (++display_count % 3 != 0) continue;
      cv::imshow(window, small);

      const int key = cv::waitKey(1);
      if (key == 'q' || key == 27) break;
      if (key == ' ') {
        nav.setEmergencyStop(!nav.emergencyStop());
        rig.robot().stop();  // effet immédiat, sans attendre le prochain cycle
        events.log(now, "EMERGENCY_STOP", nav.emergencyStop() ? "on" : "off");
      }
      if (key == 'c') {
        nav.setController(nav.controllerKind() == ControllerKind::Mppi ? ControllerKind::PurePursuit
                                                                            : ControllerKind::Mppi);
        nav.requestReplan();
      }
      if (key == 'o') {
        rig.robot().stop();
        detectObstacles();
        nav.requestReplan();
      }
      if (key == 'r') nav.requestReplan();
    }
    rig.robot().stop();
    std::cout << "Fin. Journal : " << run_dir << " (moteurs allumés " << fmt(motor_time, 0) << " s)\n";
  } catch (const std::exception& e) {
    std::cerr << "Erreur : " << e.what() << "\n";
    return 1;
  }
  return 0;
}
