// Outils pour le robot (--sim pour tester contre la simulation, --real sinon).
//   robot_tool check [--port P] [--wheels] [--wifi]   liaison avec la carte ; --wheels : sens des roues
//                                                     (roues en l'air) ; --wifi : par le Wi-Fi
//   robot_tool teleop --sim|--real [--no-camera] [--record] [--demo]  pilotage au clavier : Z/S avancer/reculer, Q/D tourner, ESPACE stop
//   robot_tool calib --sim|--real [--short]  identification du modèle moteur -> config/robot.yaml

#include <cmath>
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <sstream>
#include <string>

#include "core.hpp"
#include "hw.hpp"
#include "sim.hpp"
#include "tools.hpp"

// liaison série et watchdog, sans caméra
int cmdCheck(int argc, char** argv) {
  try {
    std::string port = "auto";
    bool wheels = false;
    bool wifi = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--port" && i + 1 < argc) port = argv[++i];
      else if (a == "--wheels") wheels = true;
      else if (a == "--wifi") wifi = true;
      else throw std::runtime_error("option inconnue : " + a);
    }
    SteadyClock clock;
    std::unique_ptr<ITransport> link;
    if (wifi) {
      const ConfigNode hw = ConfigNode::loadFile("config/params.yaml").child("hardware");
      const std::string ip = hw.getString("robot_ip");
      std::cout << "Wi-Fi : robot " << ip << " (le robot ne répond pas : seul le test des roues a un sens)\n";
      link = std::make_unique<UdpTransport>(ip, hw.getInt("udp_port", 1, 65535));
    } else {
      if (port == "auto") {
        const auto ports = listSerialPorts();
        if (ports.empty()) throw std::runtime_error("aucune carte trouvée : câble USB branché ?");
        port = ports.front();
      }
      std::cout << "Port : " << port << "\n";
      link = std::make_unique<SerialTransport>(port, 115200);
    }
    ProtocolRobot robot(*link, clock, loadProtocolRobotConfig(ConfigNode::loadFile("config/params.yaml").child("hardware")));
    if (!wifi) {
      std::cout << "1) Trame de démarrage \"B\" : " << (robot.waitReady() ? "reçue" : "NON reçue (firmware ?)") << "\n";

      // 2) Commandes nulles pendant 1 s : on mesure l'aller-retour grâce aux acquittements.
      for (int i = 0; i < 25; ++i) {
        robot.send({0.0, 0.0});
        clock.sleepUntil(clock.now() + 0.04);
        robot.poll();
      }
      std::cout << "2) Aller-retour (commande -> acquittement) : "
                << (robot.lastRoundTrip() >= 0 ? std::to_string(static_cast<int>(robot.lastRoundTrip() * 1000)) + " ms"
                                               : std::string("aucun acquittement"))
                << "\n";

      // 3) On se tait 0,5 s : la carte doit déclencher son watchdog et envoyer "W".
      const Time t0 = clock.now();
      while (clock.now() < t0 + 0.5) {
        while (const auto line = link->readLine()) {
          const auto msg = parseRobotLine(*line);
          if (msg && msg->type == RobotMessage::Type::Watchdog) {
            std::cout << "3) Watchdog : trame \"W\" reçue après "
                      << static_cast<int>((clock.now() - t0) * 1000) << " ms de silence (attendu ~300 ms)\n";
          }
        }
        clock.sleepUntil(clock.now() + 0.01);
      }
    }

    if (wheels) {
      std::cout << "4) Sens des roues (ROUES EN L'AIR) : roue GAUCHE en avant 1 s...\n";
      const char* names[2] = {"GAUCHE", "DROITE"};
      const WheelCmd cmds[2] = {{0.6, 0.0}, {0.0, 0.6}};
      for (int w = 0; w < 2; ++w) {
        const char* name = names[w];
        const WheelCmd cmd = cmds[w];
        std::cout << "   roue " << name << " en avant\n";
        const Time start = clock.now();
        while (clock.now() < start + 1.0) {
          robot.send(cmd);
          clock.sleepUntil(clock.now() + 0.04);
          robot.poll();
        }
        robot.stop();
        clock.sleepUntil(clock.now() + 0.7);
      }
      std::cout << "   Une roue a tourné vers l'arrière ? INVERT_LEFT / INVERT_RIGHT = 1 dans firmware/robot_fw/config.h\n"
                << "   L'autre roue a tourné que celle annoncée ? SWAP_SIDES = 1. Puis scripts/flash_firmware.\n";
    }
    robot.stop();
    std::cout << "Terminé.\n";
  } catch (const std::exception& e) {
    std::cerr << "Erreur : " << e.what() << "\n";
    return 1;
  }
  return 0;
}


namespace {

std::optional<WheelCmd> keyToCommand(int key, double u) {
  switch (key) {
    case 'z': case 'w': return WheelCmd{u, u};
    case 's': return WheelCmd{-u, -u};
    case 'q': case 'a': return WheelCmd{-u, u};
    case 'd': return WheelCmd{u, -u};
    case ' ': return WheelCmd{0.0, 0.0};
    default: return std::nullopt;
  }
}

std::string fmt(double x, int decimals = 2) {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(decimals);
  os << x;
  return os.str();
}

}  // namespace

// pilotage au clavier
int cmdTeleop(int argc, char** argv) {
  try {
    std::vector<std::string> rest;
    const RigOptions options = parseRigOptions(argc, argv, rest);
    const bool demo = std::find(rest.begin(), rest.end(), "--demo") != rest.end();
    const bool record = std::find(rest.begin(), rest.end(), "--record") != rest.end();
    Rig rig(options);
    std::cout << "Rappel : les moteurs ne tournent pas sur l'USB seul : branchez la pile 9 V.\n"
              << "Premier essai : robot posé sur une boîte, roues en l'air.\n";

    StateEstimator estimator(options.config_dir, rig.setup().robot, rig.setup().arena,
                                  rig.setup().nav.planner.resolution);
    if (rig.hasCamera()) estimator.setCameraModel(rig.cameraModel());

    const std::string run_dir = makeRunDirectory("runs");
    std::optional<cv::VideoWriter> video;  // --record : vidéo de la fenêtre dans le dossier de l'essai
    CsvLogger log(run_dir + "/teleop.csv",
                        {"t", "u_left", "u_right", "meas_x", "meas_y", "meas_theta", "rtt", "watchdogs"});

    double speed = 0.7;
    WheelCmd cmd;
    Time hold_until = 0.0;
    double motor_time = 0.0;
    const double dt = 1.0 / 30.0;
    // Séquence de démonstration (--demo) : touche, durée (s).
    const std::vector<std::pair<int, double>> script{{'z', 1.0}, {'q', 0.5}, {'z', 0.5}, {'s', 0.5}, {' ', 0.5}};
    std::size_t script_index = 0;
    Time script_next = 0.0;

    while (true) {
      rig.tick(dt);
      const Time now = rig.clock().now();
      std::optional<Frame> frame;
      if (rig.hasCamera()) frame = rig.camera().grab();
      if (frame) estimator.processFrame(*frame, now);
      const Localization& loc = estimator.lastLocalization();

      int key = -1;
      if (demo) {
        if (script_index >= script.size() && now >= script_next) break;
        if (script_index < script.size() && now >= script_next) {
          key = script[script_index].first;
          hold_until = now + script[script_index].second;
          script_next = hold_until;
          ++script_index;
        }
      } else {
        if (frame || !rig.hasCamera()) {
          // Sans caméra : fenêtre noire, juste pour lire le clavier.
          cv::Mat view = frame ? frame->image.clone() : cv::Mat(cv::Size(800, 120), CV_8UC3, cv::Scalar(0, 0, 0));
          drawDetections(view, estimator.lastDetections());
          if (loc.ok) drawPoseAxes(view, estimator.cameraModel(), loc.measurement.pose, rig.setup().robot.marker_height, 0.1);
          drawStatusLine(view, "PWM G " + fmt(cmd.left * 255, 0) + "  D " + fmt(cmd.right * 255, 0) +
                                        "   vitesse " + fmt(speed), 0);
          drawStatusLine(view, loc.ok ? "pose x " + fmt(loc.measurement.pose.x, 3) + " y " +
                                                 fmt(loc.measurement.pose.y, 3) + " cap " +
                                                 fmt(loc.measurement.pose.theta * 180 / 3.14159, 1) + " deg"
                                           : "tag robot non vu", 1);
          drawStatusLine(view, "aller-retour " + fmt(rig.robot().lastRoundTrip() * 1000, 0) +
                                        " ms   watchdog " + std::to_string(rig.robot().watchdogCount()) +
                                        "   moteurs " + fmt(motor_time, 0) + " s", 2);
          drawStatusLine(view, "Z/S avancer/reculer  Q/D tourner  ESPACE stop  +/- vitesse  ECHAP quitter", 3,
                              cv::Scalar(0, 255, 255));
          cv::resize(view, view, cv::Size(), 0.75, 0.75);
          cv::imshow("teleop", view);
          if (record) {
            if (!video) video.emplace(run_dir + "/video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, view.size());
            video->write(view);
          }
        }
        key = cv::waitKey(1);
        if (key == 27) break;
        if (key == '+' || key == '=') speed = std::min(1.0, speed + 0.05);
        if (key == '-') speed = std::max(0.3, speed - 0.05);
        // 0,6 s : plus long que le délai de répétition des touches du Mac (sinon le robot hoquette).
        if (keyToCommand(key, speed)) hold_until = now + 0.6;
      }
      if (const auto c = keyToCommand(key, speed)) cmd = *c;
      if (now > hold_until) cmd = {};  // touche relâchée
      rig.robot().send(cmd);
      if (cmd.left != 0.0 || cmd.right != 0.0) motor_time += dt;
      log.log({now, cmd.left, cmd.right, loc.ok ? loc.measurement.pose.x : NAN,
               loc.ok ? loc.measurement.pose.y : NAN, loc.ok ? loc.measurement.pose.theta : NAN,
               rig.robot().lastRoundTrip(), static_cast<double>(rig.robot().watchdogCount())});
    }
    rig.robot().stop();
    const Localization& loc = estimator.lastLocalization();
    std::cout << "Pose finale mesurée : "
              << (loc.ok ? fmt(loc.measurement.pose.x, 3) + " " + fmt(loc.measurement.pose.y, 3) + " " +
                               fmt(loc.measurement.pose.theta, 3)
                         : std::string("tag non vu"))
              << "\nJournal : " << run_dir << "/teleop.csv\n";
    if (rig.world()) {
      const Pose2D gt = rig.world()->groundTruthPose();
      std::cout << "Pose vraie (sim)    : " << fmt(gt.x, 3) << " " << fmt(gt.y, 3) << " " << fmt(gt.theta, 3) << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Erreur : " << e.what() << "\n";
    return 1;
  }
  return 0;
}


namespace {
void compare(const char* name, double found, double truth) {
  const double err = 100.0 * (found - truth) / truth;
  std::cout << "  " << name << " : trouvé " << found << ", vrai " << truth << " (" << err << " %)"
            << (std::abs(err) <= 10.0 ? "  OK" : "  HORS ±10 %") << "\n";
}
}  // namespace

// calibration des moteurs
int cmdCalib(int argc, char** argv) {
  try {
    std::vector<std::string> rest;
    RigOptions rig_options = parseRigOptions(argc, argv, rest);
    rig_options.empty_arena = true;  // la calibration se fait arène vide
    MotorCalibrationOptions options;
    bool write = true;
    for (const std::string& a : rest) {
      if (a == "--short") options.short_mode = true;
      else if (a == "--no-write") write = false;
      else throw std::runtime_error("option inconnue : " + a);
    }
    Rig rig(rig_options);
    if (rig.isSim()) {
      // Robot au centre de l'arène, face à +x, comme on le posera en vrai.
      const auto& a = rig.setup().arena;
      rig.world()->resetPose({(a.x_min + a.x_max) / 2, (a.y_min + a.y_max) / 2, 0.0});
    }
    const CameraModel camera = rig.cameraModel();
    const MotorCalibrationResult r = calibrateMotors(rig, camera, options);
    if (!r.ok) {
      std::cerr << "Échec : " << r.error << "\n";
      return 1;
    }
    std::cout << "\nRésultat (moteurs allumés " << r.motor_time << " s) :\n"
              << "  gauche : gain " << r.left.gain << " m/s, zone morte " << r.left.deadzone << "\n"
              << "  droite : gain " << r.right.gain << " m/s, zone morte " << r.right.deadzone << "\n"
              << "  tau " << r.tau << " s, retard " << r.delay << " s\n";
    if (rig.isSim()) {
      const RobotConfig& truth = rig.setup().world.robot;
      std::cout << "\nComparaison avec les vrais paramètres de la simulation :\n";
      compare("gain gauche", r.left.gain, truth.left.gain);
      compare("gain droite", r.right.gain, truth.right.gain);
      compare("zone morte gauche", r.left.deadzone, truth.left.deadzone);
      compare("zone morte droite", r.right.deadzone, truth.right.deadzone);
      compare("tau", r.tau, truth.tau);
      compare("retard", r.delay, truth.delay);
    }
    if (write && !rig.isSim()) {
      writeMotorsToRobotYaml(rig_options.config_dir + "/robot.yaml", r);
      std::cout << "\n" << rig_options.config_dir << "/robot.yaml mis à jour.\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Erreur : " << e.what() << "\n";
    return 1;
  }
  return 0;
}

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "check") return cmdCheck(argc - 1, argv + 1);
  if (cmd == "teleop") return cmdTeleop(argc - 1, argv + 1);
  if (cmd == "calib") return cmdCalib(argc - 1, argv + 1);
  std::cerr << "Outils pour le robot (--sim pour tester contre la simulation, --real sinon).\n  robot_tool check [--port P] [--wheels]   liaison avec la carte ; --wheels : sens des roues (roues en l'air)\n  robot_tool teleop --sim|--real [--no-camera] [--record] [--demo]  pilotage au clavier : Z/S avancer/reculer, Q/D tourner, ESPACE stop\n  robot_tool calib --sim|--real [--short]  identification du modèle moteur -> config/robot.yaml\n";
  return 1;
}
