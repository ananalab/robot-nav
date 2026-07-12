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

