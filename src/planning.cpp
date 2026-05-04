#include "planning.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <utility>

#include "core.hpp"

OccupancyGrid::OccupancyGrid(double x_min, double y_min, int cols, int rows, double resolution)
    : x_min_(x_min), y_min_(y_min), cols_(cols), rows_(rows), resolution_(resolution),
      cells_(static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows), CellState::Free) {}

void OccupancyGrid::fill(CellState s) { std::fill(cells_.begin(), cells_.end(), s); }

GridIndex OccupancyGrid::worldToCell(double x, double y) const {
  return {static_cast<int>(std::floor((x - x_min_) / resolution_)),
          static_cast<int>(std::floor((y - y_min_) / resolution_))};
}

Eigen::Vector2d OccupancyGrid::cellToWorld(const GridIndex& c) const {
  return {x_min_ + (c.col + 0.5) * resolution_, y_min_ + (c.row + 0.5) * resolution_};
}

bool pointInPolygon(const Eigen::Vector2d& p, const std::vector<Eigen::Vector2d>& polygon) {
  // On lance une demi-droite horizontale vers la droite et on compte les côtés
  // traversés : un nombre impair veut dire "dedans".
  bool inside = false;
  const std::size_t n = polygon.size();
  for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
    const Eigen::Vector2d& a = polygon[i];
    const Eigen::Vector2d& b = polygon[j];
    const bool crosses = (a.y() > p.y()) != (b.y() > p.y());
    if (crosses && p.x() < (b.x() - a.x()) * (p.y() - a.y()) / (b.y() - a.y()) + a.x()) {
      inside = !inside;
    }
  }
  return inside;
}

void OccupancyGrid::fillPolygon(const std::vector<Eigen::Vector2d>& polygon, CellState s) {
  if (polygon.size() < 3) {
    return;
  }
  // On ne teste que les cellules du rectangle englobant.
  double min_x = polygon[0].x(), max_x = min_x, min_y = polygon[0].y(), max_y = min_y;
  for (const auto& p : polygon) {
    min_x = std::min(min_x, p.x());
    max_x = std::max(max_x, p.x());
    min_y = std::min(min_y, p.y());
    max_y = std::max(max_y, p.y());
  }
  const GridIndex lo = worldToCell(min_x, min_y);
  const GridIndex hi = worldToCell(max_x, max_y);
  for (int row = std::max(0, lo.row); row <= std::min(rows_ - 1, hi.row); ++row) {
    for (int col = std::max(0, lo.col); col <= std::min(cols_ - 1, hi.col); ++col) {
      if (pointInPolygon(cellToWorld({col, row}), polygon)) {
        set({col, row}, s);
      }
    }
  }
}

OccupancyGrid makeArenaGrid(const ArenaConfig& arena, double resolution) {
  // Bordure de 3 cellules autour de l'arène, marquée occupée : le bord de
  // l'arène se comporte comme un mur pour l'inflation et la planification.
  const int border = 3;
  const double x0 = arena.x_min - border * resolution;
  const double y0 = arena.y_min - border * resolution;
  const int cols = static_cast<int>(std::ceil((arena.x_max - arena.x_min) / resolution)) + 2 * border;
  const int rows = static_cast<int>(std::ceil((arena.y_max - arena.y_min) / resolution)) + 2 * border;
  OccupancyGrid grid(x0, y0, cols, rows, resolution);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const Eigen::Vector2d c = grid.cellToWorld({col, row});
      const bool in_arena = c.x() >= arena.x_min && c.x() <= arena.x_max &&
                            c.y() >= arena.y_min && c.y() <= arena.y_max;
      grid.set({col, row}, in_arena ? CellState::Free : CellState::Occupied);
    }
  }
  return grid;
}

std::vector<Eigen::Vector2d> rectangleCorners(double cx, double cy, double length, double width,
                                              double yaw) {
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  const double hl = length / 2.0;
  const double hw = width / 2.0;
  std::vector<Eigen::Vector2d> corners;
  const double lx[4] = {hl, -hl, -hl, hl};
  const double ly[4] = {hw, hw, -hw, -hw};
  for (int k = 0; k < 4; ++k) {
    corners.emplace_back(cx + c * lx[k] - s * ly[k], cy + s * lx[k] + c * ly[k]);
  }
  return corners;
}

OccupancyGrid randomBoxMap(int cols, int rows, double resolution, int num_boxes,
                           std::mt19937& rng) {
  OccupancyGrid grid(0.0, 0.0, cols, rows, resolution);
  const double width = cols * resolution;
  const double height = rows * resolution;
  std::uniform_real_distribution<double> ux(0.0, width), uy(0.0, height);
  std::uniform_real_distribution<double> usize(0.05, 0.25), uyaw(0.0, 3.14159);
  for (int i = 0; i < num_boxes; ++i) {
    grid.fillPolygon(rectangleCorners(ux(rng), uy(rng), usize(rng), usize(rng), uyaw(rng)));
  }
  return grid;
}

namespace {
// "Infini" fini : plus grand que toute distance au carré possible dans une grille,
// mais sans risque de calcul inf - inf = NaN.
constexpr double kFar = 1e12;
}  // namespace

InflationConfig loadInflationConfig(const ConfigNode& section, double robot_radius) {
  InflationConfig cfg;
  cfg.robot_radius = robot_radius;
  cfg.margin = section.getDouble("margin", 0.0, 0.2);
  cfg.alpha = section.getDouble("alpha", 0.1, 500.0);
  cfg.d_max = section.getDouble("d_max", robot_radius + cfg.margin, 2.0);
  return cfg;
}

void squaredDistance1D(const std::vector<double>& f, std::vector<double>& d) {
  // Enveloppe inférieure des paraboles y = (q - p)^2 + f[p] (Felzenszwalb 2012).
  // v[k] : sommet de la k-ième parabole de l'enveloppe ;
  // z[k], z[k+1] : intervalle où cette parabole est la plus basse.
  const int n = static_cast<int>(f.size());
  d.assign(n, 0.0);
  if (n == 0) {
    return;
  }
  std::vector<int> v(n);
  std::vector<double> z(n + 1);
  int k = 0;
  v[0] = 0;
  z[0] = -kFar;
  z[1] = kFar;
  for (int q = 1; q < n; ++q) {
    // Abscisse où la parabole de q coupe celle de v[k].
    double s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k]);
    while (s <= z[k]) {
      --k;
      s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k]);
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = kFar;
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) {
      ++k;
    }
    const double dq = q - v[k];
    d[q] = dq * dq + f[v[k]];
  }
}

std::vector<double> euclideanDistanceTransform(const OccupancyGrid& grid) {
  // La distance euclidienne au carré est séparable : on fait la transformée 1D
  // sur chaque colonne, puis sur chaque ligne du résultat.
  const int cols = grid.cols();
  const int rows = grid.rows();
  std::vector<double> sq(static_cast<std::size_t>(cols) * rows);
  for (int i = 0; i < cols * rows; ++i) {
    sq[i] = grid.at(grid.fromLinear(i)) == CellState::Free ? kFar : 0.0;
  }
  std::vector<double> f, d;
  f.resize(rows);
  for (int col = 0; col < cols; ++col) {
    for (int row = 0; row < rows; ++row) f[row] = sq[row * cols + col];
