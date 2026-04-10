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

