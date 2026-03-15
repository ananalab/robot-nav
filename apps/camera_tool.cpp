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
