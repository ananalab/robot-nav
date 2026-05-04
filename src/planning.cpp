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
    squaredDistance1D(f, d);
    for (int row = 0; row < rows; ++row) sq[row * cols + col] = d[row];
  }
  f.resize(cols);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) f[col] = sq[row * cols + col];
    squaredDistance1D(f, d);
    for (int col = 0; col < cols; ++col) sq[row * cols + col] = std::sqrt(d[col]);
  }
  return sq;
}

CostMap::CostMap(const OccupancyGrid& grid, const InflationConfig& config)
    : grid_(grid), config_(config) {
  distances_ = euclideanDistanceTransform(grid);
  costs_.resize(distances_.size());
  const double r = config.robot_radius + config.margin;
  for (std::size_t i = 0; i < distances_.size(); ++i) {
    const double dist = distances_[i] * grid.resolution();
    distances_[i] = dist;
    if (dist <= r) {
      costs_[i] = kLethalCost;
    } else if (dist >= config.d_max) {
      costs_[i] = 0;
    } else {
      costs_[i] = static_cast<std::uint8_t>(kMaxNonLethalCost * std::exp(-config.alpha * (dist - r)));
    }
  }
}

double CostMap::distanceAt(double x, double y) const {
  const GridIndex c = grid_.worldToCell(x, y);
  return grid_.inside(c) ? distance(c) : 0.0;
}

Eigen::Vector2d CostMap::distanceGradientAt(double x, double y) const {
  const double h = grid_.resolution();
  const double gx = (distanceAt(x + h, y) - distanceAt(x - h, y)) / (2.0 * h);
  const double gy = (distanceAt(x, y + h) - distanceAt(x, y - h)) / (2.0 * h);
  return {gx, gy};
}

namespace {
constexpr double kSqrt2 = 1.41421356237309504880;

// Les 8 voisins : 4 orthogonaux puis 4 diagonaux.
constexpr int kDcol[8] = {1, -1, 0, 0, 1, 1, -1, -1};
constexpr int kDrow[8] = {0, 0, 1, -1, 1, -1, 1, -1};

// Un pas diagonal ne doit pas passer entre deux cellules dont l'une est létale.
bool cutsCorner(const CostMap& map, const GridIndex& from, int dcol, int drow) {
  if (dcol == 0 || drow == 0) {
    return false;
  }
  return map.lethal({from.col + dcol, from.row}) || map.lethal({from.col, from.row + drow});
}

std::vector<GridIndex> rebuildPath(const OccupancyGrid& grid, const std::vector<int>& parent,
                                   int goal) {
  std::vector<GridIndex> cells;
  for (int i = goal; i != -1; i = parent[i]) {
    cells.push_back(grid.fromLinear(i));
  }
  std::reverse(cells.begin(), cells.end());
  return cells;
}
}  // namespace

double octileDistance(const GridIndex& a, const GridIndex& b) {
  const int dx = std::abs(a.col - b.col);
  const int dy = std::abs(a.row - b.row);
  return std::max(dx, dy) + (kSqrt2 - 1.0) * std::min(dx, dy);
}

GridSearchResult searchGrid(const CostMap& map, const GridIndex& start, const GridIndex& goal,
                            double w_cost, bool use_heuristic) {
  GridSearchResult result;
  if (map.lethal(start) || map.lethal(goal)) {
    return result;
  }
  const OccupancyGrid& grid = map.grid();
  const double res = grid.resolution();
  const int n = grid.cols() * grid.rows();
  std::vector<double> g(n, std::numeric_limits<double>::infinity());  // meilleur coût connu
  std::vector<int> parent(n, -1);
  std::vector<bool> closed(n, false);

  // File de priorité (f = g + h, indice) ; les entrées périmées sont ignorées
  // à la sortie grâce au tableau "closed" (plus simple qu'une mise à jour de clé).
  using Entry = std::pair<double, int>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
  const int start_i = grid.linear(start);
  const int goal_i = grid.linear(goal);
  g[start_i] = 0.0;
  open.push({use_heuristic ? res * octileDistance(start, goal) : 0.0, start_i});

  while (!open.empty()) {
    const int current = open.top().second;
    open.pop();
    if (closed[current]) {
      continue;
    }
    closed[current] = true;
    ++result.expanded;
    if (current == goal_i) {
      result.found = true;
      result.cost = g[goal_i];
      result.cells = rebuildPath(grid, parent, goal_i);
      return result;
    }
    const GridIndex c = grid.fromLinear(current);
    for (int k = 0; k < 8; ++k) {
      const GridIndex next{c.col + kDcol[k], c.row + kDrow[k]};
      if (map.lethal(next) || cutsCorner(map, c, kDcol[k], kDrow[k])) {
        continue;
      }
      const int ni = grid.linear(next);
      if (closed[ni]) {
        continue;
      }
      const double length = (k < 4 ? 1.0 : kSqrt2) * res;
      const double step = length * (1.0 + w_cost * map.cost(next) / kMaxNonLethalCost);
      if (g[current] + step < g[ni]) {
        g[ni] = g[current] + step;
        parent[ni] = current;
        const double h = use_heuristic ? res * octileDistance(next, goal) : 0.0;
        open.push({g[ni] + h, ni});
      }
    }
  }
  return result;  // pas de chemin
}

bool nearestFreeCell(const CostMap& map, const GridIndex& start, GridIndex& free_cell) {
  const OccupancyGrid& grid = map.grid();
  if (!grid.inside(start)) {
    return false;
  }
  std::vector<bool> seen(grid.cols() * grid.rows(), false);
  std::queue<GridIndex> queue;
  queue.push(start);
  seen[grid.linear(start)] = true;
  while (!queue.empty()) {
    const GridIndex c = queue.front();
    queue.pop();
    if (!map.lethal(c)) {
      free_cell = c;
      return true;
    }
    for (int k = 0; k < 8; ++k) {
      const GridIndex next{c.col + kDcol[k], c.row + kDrow[k]};
      if (grid.inside(next) && !seen[grid.linear(next)]) {
        seen[grid.linear(next)] = true;
        queue.push(next);
      }
    }
  }
  return false;
}

SmootherConfig loadSmootherConfig(const ConfigNode& section) {
  SmootherConfig cfg;
  cfg.w_data = section.getDouble("w_data", 0.0, 1.0);
  cfg.w_smooth = section.getDouble("w_smooth", 0.0, 0.5);
  cfg.w_obstacle = section.getDouble("w_obstacle", 0.0, 1.0);
  cfg.obstacle_distance = section.getDouble("obstacle_distance", 0.0, 1.0);
  cfg.iterations = section.getInt("iterations", 0, 10000);
  cfg.resample_step = section.getDouble("resample_step", 0.001, 0.1);
  return cfg;
}

bool lineOfSight(const CostMap& map, const GridIndex& a, const GridIndex& b) {
  // Bresenham généralisé à toutes les directions : on avance d'une cellule à la
  // fois sur l'axe principal, et l'erreur accumulée décide quand bouger sur l'autre.
  int col = a.col, row = a.row;
  const int dcol = std::abs(b.col - a.col), drow = -std::abs(b.row - a.row);
  const int step_col = a.col < b.col ? 1 : -1;
  const int step_row = a.row < b.row ? 1 : -1;
