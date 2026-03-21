// Outils caméra.
//   camera_tool markers [--draw]                 tags et plateau ChArUco à imprimer (tools/print/)
//   camera_tool calib [--screen --square-mm X]   intrinsèques de l'iPhone -> config/camera.yaml
//   camera_tool latency [--write]                retard de la caméra (filmer l'écran)
//   camera_tool arena --sim|--real [--once]      pose de la caméra par rapport au sol -> config/arena_extrinsics.yaml
//   camera_tool arena --real --view              regarder la caméra (sans recalculer)

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core.hpp"
#include "tools.hpp"
#include "vision.hpp"

namespace {

constexpr double kDpi = 300.0;
constexpr double kPxPerMm = kDpi / 25.4;
const cv::Size kA4(static_cast<int>(210 * kPxPerMm), static_cast<int>(297 * kPxPerMm));

int mm(double v) { return static_cast<int>(std::lround(v * kPxPerMm)); }

cv::Mat blankPage() { return cv::Mat(kA4, CV_8UC1, cv::Scalar(255)); }

void text(cv::Mat& page, const std::string& s, double x_mm, double y_mm, double size_mm = 4.0) {
  const double scale = size_mm * kPxPerMm / 22.0;
  cv::putText(page, s, {mm(x_mm), mm(y_mm)}, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0),
              std::max(1, static_cast<int>(scale * 2)), cv::LINE_AA);
}

void ruler(cv::Mat& page, double x_mm, double y_mm) {
  cv::line(page, {mm(x_mm), mm(y_mm)}, {mm(x_mm + 100), mm(y_mm)}, cv::Scalar(0), 3);
  for (int i = 0; i <= 10; ++i) {
    const int len = (i % 5 == 0) ? mm(5) : mm(3);
    cv::line(page, {mm(x_mm + 10 * i), mm(y_mm)}, {mm(x_mm + 10 * i), mm(y_mm) - len}, cv::Scalar(0), 3);
  }
  text(page, "regle : 10 cm exactement (sinon : imprimer en taille reelle / 100 %)", x_mm, y_mm + 7, 3.0);
}

void placeTag(cv::Mat& page, int id, double tag_mm, double margin_mm, double x_mm, double y_mm) {
  cv::Mat tag;
  cv::aruco::generateImageMarker(markerDictionary(), id, mm(tag_mm), tag);
  tag.copyTo(page(cv::Rect(mm(x_mm + margin_mm), mm(y_mm + margin_mm), tag.cols, tag.rows)));
  const int outer = mm(tag_mm + 2 * margin_mm);
  for (int k = 0; k < outer; k += mm(4)) {  // pointillés de découpe
    const int e = std::min(outer, k + mm(2));
    const int x0 = mm(x_mm), y0 = mm(y_mm);
    cv::line(page, {x0 + k, y0}, {x0 + e, y0}, cv::Scalar(120), 2);
    cv::line(page, {x0 + k, y0 + outer}, {x0 + e, y0 + outer}, cv::Scalar(120), 2);
    cv::line(page, {x0, y0 + k}, {x0, y0 + e}, cv::Scalar(120), 2);
    cv::line(page, {x0 + outer, y0 + k}, {x0 + outer, y0 + e}, cv::Scalar(120), 2);
  }
}

void arrowUp(cv::Mat& page, double x_mm, double y_bottom_mm, double length_mm) {
  cv::arrowedLine(page, {mm(x_mm), mm(y_bottom_mm)}, {mm(x_mm), mm(y_bottom_mm - length_mm)},
                  cv::Scalar(0), mm(1.5), cv::LINE_AA, 0, 0.3);
}

std::vector<cv::Mat> floorTagPages(const ArenaConfig& arena) {
  // Un tag de 12 cm + 2 cm de marge = 16 cm : un seul par page A4.
  std::vector<cv::Mat> pages;
  const double tag = arena.floor_tag_size * 1000.0;  // 120 mm
  const double margin = tag / 6.0;                   // une case de marge blanche
  const double outer = tag + 2 * margin;
  for (const FloorTag& t : arena.floor_tags) {
    cv::Mat page = blankPage();
    text(page, "Tag SOL ID " + std::to_string(t.id) + " (" + std::to_string(static_cast<int>(tag)) +
                   " mm). Decouper sur les pointilles.", 15, 15);
    text(page, "Le poser au sol avec la FLECHE vers +y (direction du tag 0 vers le tag 3).", 15, 22);
    const double x = (210 - outer) / 2, y = 75;
    arrowUp(page, 105, y - 5, 35);
    text(page, "HAUT (+y)", 112, y - 25, 5);
    placeTag(page, t.id, tag, margin, x, y);
    ruler(page, 30, 280);
    pages.push_back(page);
  }
  return pages;
}

cv::Mat robotTagPage(const RobotConfig& robot, int id) {
  cv::Mat page = blankPage();
  const double tag = robot.marker_size * 1000.0;  // 90 mm
  const double margin = 10.0;                     // toit de 110 mm
  text(page, "Tag ROBOT (ID " + std::to_string(id) + ", " + std::to_string(static_cast<int>(tag)) +
                 " mm) a coller sur le toit en carton de 11 x 11 cm.", 15, 15);
  text(page, "La fleche AVANT doit pointer vers l'avant de la voiture.", 15, 22);
  const double x = (210 - tag - 2 * margin) / 2;
  placeTag(page, id, tag, margin, x, 80);
  arrowUp(page, 105, 75, 30);
  text(page, "AVANT", 112, 60, 6);
  ruler(page, 30, 280);
  return page;
}

cv::Mat charucoPage(const ConfigNode& charuco) {
  const int sx = charuco.getInt("squares_x", 3, 20);
  const int sy = charuco.getInt("squares_y", 3, 20);
  const double square = charuco.getDouble("square_mm", 5.0, 100.0);
  const double marker = charuco.getDouble("marker_mm", 3.0, square);
  const cv::aruco::CharucoBoard board(cv::Size(sx, sy), static_cast<float>(square / 1000.0),
                                      static_cast<float>(marker / 1000.0),
                                      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100));
  cv::Mat img;
  board.generateImage(cv::Size(mm(sx * square), mm(sy * square)), img, 0, 1);
  cv::Mat page = blankPage();
  img.copyTo(page(cv::Rect(mm((210 - sx * square) / 2), mm(20), img.cols, img.rows)));
  text(page, "Plateau ChArUco " + std::to_string(sx) + "x" + std::to_string(sy) + " : carre " +
                 std::to_string(static_cast<int>(square)) + " mm. Coller bien a plat (carton).", 15, 12);
  ruler(page, 30, 282);
  return page;
}

std::string drawingInstructions(int id, double cell_mm, const std::string& png_path) {
  cv::Mat bits;
  cv::aruco::generateImageMarker(markerDictionary(), id, 6, bits, 1);  // 1 px = 1 case
  std::ostringstream out;
  out << "### Tag " << id << " — grille 6 × 6, cases de " << cell_mm << " mm (côté "
      << 6 * cell_mm << " mm)\n\nCases à noircir (colonne lettre A à F, de gauche à droite ; "
      << "ligne 1 à 6, de haut en bas) :\n\n";
  const int cell = 80;
  cv::Mat img(8 * cell, 8 * cell, CV_8UC1, cv::Scalar(255));
  for (int r = 0; r < 6; ++r) {
    out << "- ligne " << r + 1 << " :";
    for (int c = 0; c < 6; ++c) {
      const bool black = bits.at<uchar>(r, c) < 128;
      if (black) out << " " << static_cast<char>('A' + c) << r + 1;
      cv::rectangle(img, cv::Rect((c + 1) * cell, (r + 1) * cell, cell, cell),
                    black ? cv::Scalar(0) : cv::Scalar(255), cv::FILLED);
      cv::rectangle(img, cv::Rect((c + 1) * cell, (r + 1) * cell, cell, cell), cv::Scalar(150), 1);
    }
    out << "\n";
    cv::putText(img, std::to_string(r + 1), {cell / 3, (r + 1) * cell + cell * 2 / 3},
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0), 2);
  }
  for (int c = 0; c < 6; ++c) {
    cv::putText(img, std::string(1, static_cast<char>('A' + c)), {(c + 1) * cell + cell / 3, cell * 2 / 3},
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0), 2);
  }
  cv::putText(img, "HAUT", {3 * cell, 7 * cell + cell * 2 / 3}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0), 2);
  cv::imwrite(png_path, img);
  out << "\nImage d'aide : `" << png_path << "` (le haut de la grille est le HAUT du tag).\n\n";
  return out.str();
}

void writePdf(const std::string& path, const std::vector<cv::Mat>& pages) {
  std::ofstream f(path, std::ios::binary);
  std::vector<std::size_t> offsets;
  auto object = [&](const std::string& body) {
    offsets.push_back(static_cast<std::size_t>(f.tellp()));
    f << offsets.size() << " 0 obj\n" << body << "\nendobj\n";
  };
  f << "%PDF-1.4\n";
  const int n = static_cast<int>(pages.size());
  std::ostringstream kids;
  for (int i = 0; i < n; ++i) kids << 3 + 3 * i << " 0 R ";
  object("<< /Type /Catalog /Pages 2 0 R >>");
  object("<< /Type /Pages /Kids [" + kids.str() + "] /Count " + std::to_string(n) + " >>");
  const std::string a4 = "595.276 841.89";
  for (int i = 0; i < n; ++i) {
    const int page_id = 3 + 3 * i;
    object("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 " + a4 + "] /Resources << /XObject << /Im0 " +
           std::to_string(page_id + 2) + " 0 R >> >> /Contents " + std::to_string(page_id + 1) + " 0 R >>");
    const std::string content = "q " + std::string("595.276 0 0 841.89 0 0 cm /Im0 Do Q");
    object("<< /Length " + std::to_string(content.size()) + " >>\nstream\n" + content + "\nendstream");
    std::vector<uchar> jpeg;
    cv::imencode(".jpg", pages[i], jpeg, {cv::IMWRITE_JPEG_QUALITY, 100});
    offsets.push_back(static_cast<std::size_t>(f.tellp()));
    f << offsets.size() << " 0 obj\n<< /Type /XObject /Subtype /Image /Width " << pages[i].cols
      << " /Height " << pages[i].rows << " /ColorSpace /DeviceGray /BitsPerComponent 8"
      << " /Filter /DCTDecode /Length " << jpeg.size() << " >>\nstream\n";
    f.write(reinterpret_cast<const char*>(jpeg.data()), static_cast<std::streamsize>(jpeg.size()));
    f << "\nendstream\nendobj\n";
  }
  const std::size_t xref = static_cast<std::size_t>(f.tellp());
  f << "xref\n0 " << offsets.size() + 1 << "\n0000000000 65535 f \n";
  for (std::size_t o : offsets) {
    char line[24];
    std::snprintf(line, sizeof(line), "%010zu 00000 n \n", o);
    f << line;
  }
  f << "trailer\n<< /Size " << offsets.size() + 1 << " /Root 1 0 R >>\nstartxref\n" << xref << "\n%%EOF\n";
}

}  // namespace

// tags à imprimer
int cmdMarkers(int argc, char** argv) {
  try {
    std::string config_dir = "config", out_dir = "tools/print";
    bool draw = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--draw") draw = true;
      else if (a == "--config" && i + 1 < argc) config_dir = argv[++i];
      else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
      else throw std::runtime_error("option inconnue : " + a);
    }
    const auto arena = loadArenaConfig(ConfigNode::loadFile(config_dir + "/robot.yaml").child("arena"));
    const auto robot = loadRobotConfig(ConfigNode::loadFile(config_dir + "/robot.yaml").child("robot"));
    const auto charuco = ConfigNode::loadFile(config_dir + "/params.yaml").child("perception").child("charuco");
    std::filesystem::create_directories(out_dir);

    std::vector<cv::Mat> pages = floorTagPages(arena);
    pages.push_back(robotTagPage(robot, arena.robot_tag_id));
    pages.push_back(charucoPage(charuco));
    std::vector<std::string> names;
    for (const auto& t : arena.floor_tags) names.push_back("tag_sol_" + std::to_string(t.id));
    names.push_back("tag_robot");
    names.push_back("charuco_a4");
    for (std::size_t i = 0; i < pages.size(); ++i) cv::imwrite(out_dir + "/" + names[i] + ".png", pages[i]);
    writePdf(out_dir + "/a_imprimer.pdf", pages);
    std::cout << "Écrit : " << out_dir << "/a_imprimer.pdf (" << pages.size()
              << " pages A4) et les PNG.\nImprimer en « taille réelle / 100 % », puis vérifier la règle de 10 cm.\n";

    if (draw) {
      std::ofstream md(out_dir + "/tags_a_dessiner.md");
      md << "# Tags à dessiner à la main\n\n"
         << "1. Sur un carton blanc et plat, tracer au crayon une grille 6 × 6 de la bonne taille.\n"
         << "2. Écrire les lettres A à F au-dessus des colonnes et les chiffres 1 à 6 à gauche des lignes.\n"
         << "3. Noircir au feutre noir les cases listées (bien remplir, bords nets).\n"
         << "4. Laisser une marge blanche d'une case tout autour (ne pas la colorier).\n"
         << "5. Le HAUT de la grille est le haut du tag (vers +y pour les tags sol, vers l'avant pour le robot).\n\n";
      for (const auto& tag : arena.floor_tags) {
        md << drawingInstructions(tag.id, arena.floor_tag_size * 1000 / 6,
                                  out_dir + "/dessin_tag_" + std::to_string(tag.id) + ".png");
      }
      md << drawingInstructions(arena.robot_tag_id, robot.marker_size * 1000 / 6,
                                out_dir + "/dessin_tag_" + std::to_string(arena.robot_tag_id) + ".png");
      std::cout << "Mode dessin : " << out_dir << "/tags_a_dessiner.md\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Erreur : " << e.what() << "\n";
    return 1;
  }
  return 0;
}


// calibration des intrinsèques
int cmdCalib(int argc, char** argv) {
  try {
    std::string config_dir = "config", source;
    bool screen = false;
    double square_mm = 0.0;
    int wanted_views = 20;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("valeur manquante après " + a);
        return argv[++i];
      };
      if (a == "--screen") screen = true;
      else if (a == "--square-mm") square_mm = std::stod(next());
      else if (a == "--camera") source = next();
      else if (a == "--config") config_dir = next();
      else if (a == "--views") wanted_views = std::stoi(next());
      else throw std::runtime_error("option inconnue : " + a);
    }
    if (screen && square_mm <= 0.0) {
      throw std::runtime_error("mode écran : donnez le côté d'une case mesuré à l'écran, ex. --square-mm 31.5");
    }
    const ConfigNode cam = ConfigNode::loadFile(config_dir + "/params.yaml").child("hardware");
    if (source.empty()) source = cam.getString("camera_source");
    const CharucoSpec spec = loadCharucoSpec(
        ConfigNode::loadFile(config_dir + "/params.yaml").child("perception").child("charuco"), screen ? square_mm : 0.0);

    if (screen) {
      // Plateau en plein écran, sur fond blanc (marge = zone blanche autour des cases).
      cv::Mat board_img;
      makeBoard(spec).generateImage(cv::Size(1000, 1400), board_img, 60, 1);
      cv::namedWindow("plateau", cv::WINDOW_NORMAL);
      cv::setWindowProperty("plateau", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
      cv::imshow("plateau", board_img);
      std::cout << "Plateau affiché. Vérifiez à la règle qu'une case mesure bien " << square_mm
                << " mm, puis filmez l'écran (luminosité au maximum).\n";
    }

    SteadyClock clock;
    OpenCvCamera camera(source, cam.getInt("camera_width", 16, 10000), cam.getInt("camera_height", 16, 10000),
                                    0.0, clock);
    CameraCalibrator calibrator(spec);
    std::cout << "ESPACE = garder la vue, C = calibrer, ÉCHAP = quitter. Objectif : " << wanted_views
              << " vues variées (inclinées, proches, loin, dans les coins de l'image).\n";
    std::string message;
    while (true) {
      const auto frame = camera.grab();
      if (frame) {
        cv::Mat view = frame->image.clone();
        const auto corners = calibrator.detectCorners(view);
        for (const auto& c : corners) cv::circle(view, c, 4, cv::Scalar(0, 255, 0), cv::FILLED);
        drawStatusLine(view, "vues : " + std::to_string(calibrator.views()) + " / " +
                                      std::to_string(wanted_views) + "   coins vus : " + std::to_string(corners.size()), 0);
        drawStatusLine(view, "ESPACE garder la vue   C calibrer   ECHAP quitter", 1, cv::Scalar(0, 255, 255));
        if (!message.empty()) drawStatusLine(view, message, 2, cv::Scalar(0, 200, 255));
        cv::resize(view, view, cv::Size(), 0.75, 0.75);
        cv::imshow("camera_tool calib", view);
      }
      const int key = cv::waitKey(10);
      if (key == 27) break;
      if (key == ' ' && frame) {
        message = calibrator.addView(frame->image) ? "vue gardee" : "pas assez de coins : rapprochez le plateau";
      }
      if (key == 'c' || key == 'C') {
        CameraIntrinsics intr;
        const double rms = calibrator.calibrate(intr);
        if (rms < 0) {
          message = "il faut au moins 4 vues (20 conseillees)";
          continue;
        }
        const std::string path = config_dir + "/camera.yaml";
        saveIntrinsics(path, intr, rms);
        std::cout << "Erreur de reprojection RMS : " << rms << " px (objectif < 0,5 px)\n"
                  << "fx " << intr.K(0, 0) << "  fy " << intr.K(1, 1) << "  cx " << intr.K(0, 2)
                  << "  cy " << intr.K(1, 2) << "\nÉcrit dans " << path << "\n";
        message = "rms " + std::to_string(rms) + " px, enregistre dans " + path;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Erreur : " << e.what() << "\n";
    return 1;
  }
  return 0;
}


// mesure de la latence
int cmdLatency(int argc, char** argv) {
  try {
    std::string config_dir = "config", source;
    int samples_wanted = 150;
    bool write = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--camera" && i + 1 < argc) source = argv[++i];
      else if (a == "--samples" && i + 1 < argc) samples_wanted = std::stoi(argv[++i]);
      else if (a == "--config" && i + 1 < argc) config_dir = argv[++i];
      else if (a == "--write") write = true;
      else throw std::runtime_error("option inconnue : " + a);
    }
    const std::string hw_path = config_dir + "/params.yaml";
    const ConfigNode cam = ConfigNode::loadFile(hw_path).child("hardware");
    if (source.empty()) source = cam.getString("camera_source");

    SteadyClock clock;
    // Latence 0 : on veut l'heure de RÉCEPTION brute des images.
    OpenCvCamera camera(source, cam.getInt("camera_width", 16, 10000), cam.getInt("camera_height", 16, 10000),
                                    0.0, clock);
    const MarkerDetector detector;
    std::array<Time, 50> shown_at{};
    shown_at.fill(-1.0);
    std::vector<double> latencies;
    cv::namedWindow("latence", cv::WINDOW_NORMAL);
    int counter = 0;
    std::cout << "Filmez la fenêtre « latence » avec l'iPhone. ÉCHAP pour arrêter.\n";
    while (static_cast<int>(latencies.size()) < samples_wanted) {
      const int id = counter++ % 50;
      cv::Mat tag, screen(700, 700, CV_8UC1, cv::Scalar(255));
      cv::aruco::generateImageMarker(markerDictionary(), id, 500, tag);
      tag.copyTo(screen(cv::Rect(100, 100, 500, 500)));
      cv::imshow("latence", screen);
      if (cv::waitKey(1) == 27) break;
      shown_at[id] = clock.now();  // l'image est à l'écran (à un rafraîchissement près)

      if (const auto frame = camera.grab()) {
        for (const auto& d : detector.detect(frame->image)) {
          if (d.id < 0 || d.id >= 50 || shown_at[d.id] < 0) continue;
          const double latency = frame->stamp - shown_at[d.id];
          if (latency > 0.0 && latency < 1.0) latencies.push_back(latency);
        }
      }
    }
    if (latencies.empty()) throw std::runtime_error("aucun tag vu : la caméra filme-t-elle l'écran ?");
    std::sort(latencies.begin(), latencies.end());
    const double median = latencies[latencies.size() / 2];
    std::cout << "Latence caméra : médiane " << median * 1000 << " ms (min " << latencies.front() * 1000
              << ", max " << latencies.back() * 1000 << ", " << latencies.size() << " mesures)\n";
    if (write) {
      std::ifstream in(hw_path);
      std::stringstream text;
      text << in.rdbuf();
      std::ostringstream value;
      value << "  camera_latency: " << median;
      const std::string updated =
          std::regex_replace(text.str(), std::regex("  camera_latency: [0-9.eE+-]+"), value.str(),
                             std::regex_constants::format_first_only);
      std::ofstream(hw_path) << updated;
      std::cout << hw_path << " mis à jour (hardware.camera_latency).\n";
    } else {
      std::cout << "Relancez avec --write pour l'enregistrer dans " << hw_path << ".\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Erreur : " << e.what() << "\n";
    return 1;
  }
  return 0;
}


namespace {
bool arenaFullyVisible(const CameraModel& camera, const ArenaConfig& a) {
  const std::vector<cv::Point3f> corners{{static_cast<float>(a.x_min), static_cast<float>(a.y_min), 0},
                                         {static_cast<float>(a.x_max), static_cast<float>(a.y_min), 0},
                                         {static_cast<float>(a.x_max), static_cast<float>(a.y_max), 0},
                                         {static_cast<float>(a.x_min), static_cast<float>(a.y_max), 0}};
  for (const cv::Point2f& p : camera.project(corners)) {
    if (p.x < 0 || p.y < 0 || p.x >= camera.intrinsics().width || p.y >= camera.intrinsics().height) return false;
  }
  return true;
}
}  // namespace

// extrinsèques et contrôle de l'arène
int cmdArena(int argc, char** argv) {
  try {
    std::vector<std::string> rest;
    const RigOptions options = parseRigOptions(argc, argv, rest);
    const bool once = std::find(rest.begin(), rest.end(), "--once") != rest.end();
    // --view : juste regarder la caméra, avec la pose du sol déjà enregistrée (rien n'est réécrit).
    const bool view_only = std::find(rest.begin(), rest.end(), "--view") != rest.end();
    Rig rig(options);
    const CameraModel camera = rig.cameraModel(!view_only);
    const auto& arena = rig.setup().arena;
    const bool visible = arenaFullyVisible(camera, arena);
    std::cout << "Toute l'arène est dans l'image : " << (visible ? "oui" : "NON (monter ou recentrer le téléphone)") << "\n";
    if (!rig.isSim() && !view_only) std::cout << "Extrinsèques enregistrés dans " << rig.extrinsicsPath() << "\n";
    if (once) return 0;

    ArenaCalibrator checker(rig.intrinsics(), arena);
    const MarkerDetector detector;
    bool placed = false;
    while (true) {
      // --view : n'envoie rien au robot (rig.tick renverrait des ordres "stop" à chaque tour).
      if (view_only) std::this_thread::sleep_for(std::chrono::milliseconds(15));
      else rig.tick(1.0 / 30.0);
      const auto frame = rig.camera().grab();
      if (!frame) continue;
      cv::Mat view = frame->image.clone();
      const auto detections = detector.detect(view);
      drawDetections(view, detections);
      drawWorldGrid(view, camera, arena);
      // Contrôle continu : la caméra a-t-elle bougé ?
      const double err = checker.reprojectionError(frame->image, camera.rvec(), camera.tvec());
      const bool ok = err >= 0 && err < 3.0;  // tags posés à la main : 2 à 3 px est normal
      drawStatusLine(view, "reprojection des tags sol : " + (err < 0 ? std::string("tags non vus") : std::to_string(err) + " px"),
                          0, ok ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255));
      if (err > 5.0) drawStatusLine(view, "ATTENTION : la camera a bouge ? relancer camera_tool arena", 1, cv::Scalar(0, 0, 255));
      drawStatusLine(view, std::string("arene entiere visible : ") + (visible ? "oui" : "NON"), 2);
      // --view : petite fenêtre gardée au-dessus des autres (pour la laisser ouverte pendant les essais).
      const char* window = view_only ? "camera" : "camera_tool arena";
      cv::resize(view, view, cv::Size(), view_only ? 0.33 : 0.75, view_only ? 0.33 : 0.75);
      cv::imshow(window, view);
      if (view_only && !placed) {
        cv::setWindowProperty(window, cv::WND_PROP_TOPMOST, 1);
        cv::moveWindow(window, 0, 0);  // coin en haut à gauche, pour ne pas cacher robot_nav
        placed = true;
      }
      if (cv::waitKey(1) == 27) break;
    }
  } catch (const std::exception& e) {
    std::cerr << "Erreur : " << e.what() << "\n";
    return 1;
  }
  return 0;
}

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "markers") return cmdMarkers(argc - 1, argv + 1);
  if (cmd == "calib") return cmdCalib(argc - 1, argv + 1);
  if (cmd == "latency") return cmdLatency(argc - 1, argv + 1);
  if (cmd == "arena") return cmdArena(argc - 1, argv + 1);
  std::cerr << "Outils caméra.\n  camera_tool markers [--draw]                 tags et plateau ChArUco à imprimer (tools/print/)\n  camera_tool calib [--screen --square-mm X]   intrinsèques de l'iPhone -> config/camera.yaml\n  camera_tool latency [--write]                retard de la caméra (filmer l'écran)\n  camera_tool arena --sim|--real [--once]      pose de la caméra par rapport au sol -> config/arena_extrinsics.yaml\n  camera_tool arena --real --view              regarder la caméra (sans recalculer)\n";
  return 1;
}
