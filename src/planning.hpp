#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "core.hpp"

// Grille d'occupation 2D : une cellule = un carré de "resolution" mètres.
// La cellule (col, row) couvre [x_min + col*res, x_min + (col+1)*res[ en x, idem en y.

enum class CellState : std::uint8_t { Free, Occupied, Unknown };

struct GridIndex {
  int col = 0;
  int row = 0;
};

inline bool operator==(const GridIndex& a, const GridIndex& b) {
  return a.col == b.col && a.row == b.row;
}

class OccupancyGrid {
 public:
  OccupancyGrid() = default;
  OccupancyGrid(double x_min, double y_min, int cols, int rows, double resolution);

  int cols() const { return cols_; }
  int rows() const { return rows_; }
  double resolution() const { return resolution_; }
  double xMin() const { return x_min_; }
  double yMin() const { return y_min_; }

  bool inside(const GridIndex& c) const {
    return c.col >= 0 && c.row >= 0 && c.col < cols_ && c.row < rows_;
  }
  // Indice linéaire (ligne par ligne), utile pour les tableaux parallèles.
  int linear(const GridIndex& c) const { return c.row * cols_ + c.col; }
  GridIndex fromLinear(int i) const { return {i % cols_, i / cols_}; }

  CellState at(const GridIndex& c) const { return cells_[linear(c)]; }
  void set(const GridIndex& c, CellState s) { cells_[linear(c)] = s; }
  void fill(CellState s);

  // Cellule contenant le point (x, y) ; peut être hors de la grille (vérifier inside()).
  GridIndex worldToCell(double x, double y) const;
  // Centre de la cellule, en mètres.
  Eigen::Vector2d cellToWorld(const GridIndex& c) const;

  // Marque comme occupées les cellules dont le centre est dans le polygone (repère W).
  void fillPolygon(const std::vector<Eigen::Vector2d>& polygon, CellState s = CellState::Occupied);

 private:
  double x_min_ = 0.0, y_min_ = 0.0;
  int cols_ = 0, rows_ = 0;
  double resolution_ = 0.01;
  std::vector<CellState> cells_;
};

// Point dans un polygone (règle pair-impair).
bool pointInPolygon(const Eigen::Vector2d& p, const std::vector<Eigen::Vector2d>& polygon);

// Grille couvrant l'arène plus une bordure ; tout ce qui est hors de l'arène est occupé.
OccupancyGrid makeArenaGrid(const ArenaConfig& arena, double resolution);

// Coins d'un rectangle orienté (centre, dimensions, lacet), dans l'ordre.
std::vector<Eigen::Vector2d> rectangleCorners(double cx, double cy, double length, double width,
                                              double yaw);

// Carte aléatoire (tests) : n boîtes posées au hasard dans une grille libre.
OccupancyGrid randomBoxMap(int cols, int rows, double resolution, int num_boxes,
                           std::mt19937& rng);

// Inflation des obstacles.
//
// 1) Transformée de distance euclidienne exacte (algorithme de Felzenszwalb &
//    Huttenlocher), écrite à la main : distance de chaque cellule à l'obstacle le
//    plus proche.
// 2) Coût de chaque cellule à partir de cette distance d :
//      d <= r            -> létal (le robot touche l'obstacle), r = r_robot + marge
//      r < d < d_max     -> c = 253 * exp(-alpha * (d - r))
//      d >= d_max        -> 0

constexpr std::uint8_t kLethalCost = 254;
constexpr double kMaxNonLethalCost = 253.0;

struct InflationConfig {
  double robot_radius = 0.10;
  double margin = 0.02;
  double alpha = 20.0;  // décroissance du coût (1/m)
  double d_max = 0.25;  // au-delà, coût nul (m)
};

InflationConfig loadInflationConfig(const ConfigNode& section, double robot_radius);

// Transformée de distance 1D sur des distances AU CARRÉ (f = 0 sur un obstacle,
// très grand ailleurs). Résultat : d[q] = min_p ((q - p)^2 + f[p]).
void squaredDistance1D(const std::vector<double>& f, std::vector<double>& d);

// Distance euclidienne (en cellules) de chaque cellule à la cellule occupée la
// plus proche (les cellules Unknown comptent comme occupées). Ordre linéaire de la grille.
std::vector<double> euclideanDistanceTransform(const OccupancyGrid& grid);

class CostMap {
 public:
  CostMap() = default;
  CostMap(const OccupancyGrid& grid, const InflationConfig& config);

  const OccupancyGrid& grid() const { return grid_; }
  const InflationConfig& config() const { return config_; }
