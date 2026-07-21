// bench : simulation sans affichage, aussi vite que possible, sur une suite de scénarios
// (config/scenarios/<suite>.yaml) et plusieurs graines. Affiche un tableau Markdown.
//
//   bench --suite basic --seeds 5                  pose vraie, Pure Pursuit
//   bench --suite basic --mode full --seeds 5      caméra simulée + EKF
//   bench --suite narrow --controller mppi
//
// Options : --config <dossier>, --mode gt|full, --controller pp|mppi, --seeds N, --out <fichier.md>,
//   --log (log.csv et events.csv de chaque essai dans runs/), --latency <s>, --no-latency-comp,
//   --gain-scale <k> (vrais gains moteurs x k, 0.5 = pile faible), --speed-scale <k> (v_max x k)

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "app.hpp"
#include "core.hpp"

namespace {

struct Options {
  std::string config_dir = "config";
  std::string suite = "basic";
  std::string mode;  // vide = déduit de la suite
  std::string controller = "pp";
  std::string out;
  int seeds = 5;
  bool keep_logs = false;
  double latency = -1.0;     // latence caméra forcée (s)
  bool latency_comp = true;  // false = ablation de la compensation de latence
  double gain_scale = 1.0;   // facteur sur les vrais gains moteurs (0.5 = pile à plat)
  double speed_scale = 1.0;  // facteur sur v_max (vitesse réduite)
};

Options parseArgs(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("valeur manquante après " + a);
      return argv[++i];
    };
    if (a == "--config") o.config_dir = next();
    else if (a == "--suite") o.suite = next();
    else if (a == "--mode") o.mode = next();
    else if (a == "--controller") o.controller = next();
    else if (a == "--seeds") o.seeds = std::stoi(next());
    else if (a == "--out") o.out = next();
    else if (a == "--log") o.keep_logs = true;
    else if (a == "--latency") o.latency = std::stod(next());
    else if (a == "--no-latency-comp") o.latency_comp = false;
    else if (a == "--gain-scale") o.gain_scale = std::stod(next());
    else if (a == "--speed-scale") o.speed_scale = std::stod(next());
    else throw std::runtime_error("option inconnue : " + a);
  }
  if (o.mode.empty()) o.mode = "gt";
  return o;
}

double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t i = static_cast<std::size_t>(std::ceil(p * v.size())) - 1;
  return v[std::min(i, v.size() - 1)];
}

double mean(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) s += x;
  return v.empty() ? 0.0 : s / v.size();
}

std::string fmt(double x, int decimals) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(decimals) << x;
  return os.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parseArgs(argc, argv);
    SimSetup setup = loadSimSetup(opt.config_dir);
    setup.camera_latency = opt.latency;
    setup.latency_compensation = opt.latency_comp;
    setup.world.robot.left.gain *= opt.gain_scale;
    setup.world.robot.right.gain *= opt.gain_scale;
    setup.nav.limits.v_max *= opt.speed_scale;
    const auto scenarios = loadScenarios(
        opt.config_dir + "/scenarios/" + opt.suite + ".yaml", setup.arena.obstacle_height);

    EpisodeOptions eo;
    eo.mode = opt.mode == "full" ? PoseMode::Full : PoseMode::GroundTruth;
    eo.controller = opt.controller == "mppi" ? ControllerKind::Mppi
                                             : ControllerKind::PurePursuit;
    const std::string run_dir = opt.keep_logs ? makeRunDirectory("runs") : "";

    std::ostringstream table;
    table << "| Scénario | Succès | Temps (s) | Long. parcourue / planifiée | Écart lat. RMS (mm) "
             "| Dist. min obstacles (mm) | Err. pos. (mm) | Err. cap (°) | Collisions |\n"
          << "|---|---|---|---|---|---|---|---|---|\n";

    int total = 0, successes = 0, collisions = 0, low_battery = 0, rejections = 0;
    std::vector<double> all_compute, all_lateral, all_pos, all_head, all_clear, all_time, all_ratio;
    std::map<std::string, int> failures;

    for (const Scenario& sc : scenarios) {
      std::vector<double> times, ratios, lateral, clear, pos, head;
      int ok = 0, coll = 0;
      for (int seed = 0; seed < opt.seeds; ++seed) {
        eo.seed = static_cast<unsigned>(seed);
        eo.log_dir = run_dir.empty() ? "" : run_dir + "/" + sc.name + "_seed" + std::to_string(seed);
        const EpisodeMetrics m = runEpisode(setup, sc, eo);
        ++total;
        if (m.success) {
          ++ok;
        } else {
          ++failures[m.failure];
          std::cerr << "  échec " << sc.name << " graine " << seed << " : " << m.failure
                    << " (pos " << fmt(m.final_position_error * 1000, 1) << " mm, cap "
                    << fmt(radToDeg(m.final_heading_error), 1) << "°)\n";
        }
        coll += m.collisions;
        low_battery += m.low_battery_alert ? 1 : 0;
        rejections += m.pose_rejections;
        times.push_back(m.time_to_goal);
        if (m.planned_length > 0) ratios.push_back(m.traveled_length / m.planned_length);
        lateral.push_back(m.lateral_rms);
        if (m.min_clearance < 1e8) clear.push_back(m.min_clearance);
        pos.push_back(m.final_position_error);
        head.push_back(m.final_heading_error);
        all_compute.insert(all_compute.end(), m.compute_ms.begin(), m.compute_ms.end());
      }
      successes += ok;
      collisions += coll;
      all_time.insert(all_time.end(), times.begin(), times.end());
      all_ratio.insert(all_ratio.end(), ratios.begin(), ratios.end());
      all_lateral.insert(all_lateral.end(), lateral.begin(), lateral.end());
      all_clear.insert(all_clear.end(), clear.begin(), clear.end());
      all_pos.insert(all_pos.end(), pos.begin(), pos.end());
      all_head.insert(all_head.end(), head.begin(), head.end());
      table << "| " << sc.name << " | " << ok << "/" << opt.seeds << " | " << fmt(mean(times), 1)
            << " | " << fmt(mean(ratios), 2) << " | " << fmt(mean(lateral) * 1000, 1) << " | "
            << (clear.empty() ? "-" : fmt(*std::min_element(clear.begin(), clear.end()) * 1000, 0))
            << " | " << fmt(mean(pos) * 1000, 1) << " | " << fmt(radToDeg(mean(head)), 1)
            << " | " << coll << " |\n";
    }

    std::ostringstream report;
    report << "## bench --suite " << opt.suite << " --mode " << opt.mode << " --controller "
           << opt.controller << " --seeds " << opt.seeds;
    if (opt.latency >= 0) report << " --latency " << opt.latency;
    if (!opt.latency_comp) report << " --no-latency-comp";
    if (opt.gain_scale != 1.0) report << " --gain-scale " << opt.gain_scale;
    if (opt.speed_scale != 1.0) report << " --speed-scale " << opt.speed_scale;
    report << "\n\n"
           << "- Taux de succès : **" << successes << "/" << total << " ("
           << fmt(100.0 * successes / std::max(1, total), 1) << " %)**\n"
           << "- Collisions : " << collisions << "\n"
           << "- Alertes pile faible : " << low_battery << "/" << total << "\n"
           << "- Mesures rejetées par l'EKF (moyenne par essai) : "
           << fmt(static_cast<double>(rejections) / std::max(1, total), 1) << "\n"
           << "- Temps jusqu'au but (moyen) : " << fmt(mean(all_time), 2) << " s\n"
           << "- Longueur parcourue / planifiée (moyenne) : " << fmt(mean(all_ratio), 3) << "\n"
           << "- Écart latéral RMS (moyen) : " << fmt(mean(all_lateral) * 1000, 1) << " mm\n"
           << "- Distance minimale aux obstacles : "
           << (all_clear.empty() ? "-" : fmt(*std::min_element(all_clear.begin(), all_clear.end()) * 1000, 0) + " mm") << "\n"
           << "- Erreur finale de position : médiane " << fmt(percentile(all_pos, 0.5) * 1000, 1)
           << " mm, max " << fmt(percentile(all_pos, 1.0) * 1000, 1) << " mm\n"
           << "- Erreur finale de cap : médiane " << fmt(radToDeg(percentile(all_head, 0.5)), 2)
           << "°, max " << fmt(radToDeg(percentile(all_head, 1.0)), 2) << "°\n"
           << "- Temps de calcul par cycle : p50 " << fmt(percentile(all_compute, 0.5), 3)
           << " ms, p99 " << fmt(percentile(all_compute, 0.99), 3) << " ms\n";
    for (const auto& f : failures) {
      report << "- Échecs « " << f.first << " » : " << f.second << "\n";
    }
    report << "\n" << table.str();

    std::cout << report.str();
    if (!opt.out.empty()) {
      std::ofstream(opt.out) << report.str();
      std::cout << "\nTableau écrit dans " << opt.out << "\n";
    }
    if (!run_dir.empty()) std::cout << "Logs : " << run_dir << "\n";
  } catch (const std::exception& e) {
    std::cerr << "Erreur : " << e.what() << "\n";
    return 1;
  }
  return 0;
}
