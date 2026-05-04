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

  std::uint8_t cost(const GridIndex& c) const { return costs_[grid_.linear(c)]; }
  // Hors grille = létal.
  bool lethal(const GridIndex& c) const { return !grid_.inside(c) || cost(c) >= kLethalCost; }
  bool lethalAt(double x, double y) const { return lethal(grid_.worldToCell(x, y)); }
  // Distance au plus proche obstacle (m).
  double distance(const GridIndex& c) const { return distances_[grid_.linear(c)]; }
  // Distance au plus proche obstacle au point (x, y) ; 0 hors de la grille.
  double distanceAt(double x, double y) const;
  // Gradient de la distance (différences centrées), pointe à l'opposé des obstacles.
  Eigen::Vector2d distanceGradientAt(double x, double y) const;

 private:
  OccupancyGrid grid_;
  InflationConfig config_;
  std::vector<double> distances_;     // m
  std::vector<std::uint8_t> costs_;   // 0..254
};

// Recherche de chemin sur la grille : A* à 8 voisins.
//
//   coût d'un pas = longueur du pas (m) * (1 + w_cost * c / 253)
//   heuristique   = distance octile (m), admissible car le facteur de coût est >= 1
//
// Un pas en diagonale est interdit s'il "coupe le coin" d'une cellule létale.

struct GridSearchResult {
  bool found = false;
  std::vector<GridIndex> cells;  // du départ au but inclus
  double cost = 0.0;             // somme des coûts de pas
  int expanded = 0;              // nombre de cellules développées (pour comparer A* et Dijkstra)
};

// A* ; avec use_heuristic = false, c'est l'algorithme de Dijkstra (même code).
GridSearchResult searchGrid(const CostMap& map, const GridIndex& start, const GridIndex& goal,
                            double w_cost, bool use_heuristic = true);

// Distance octile entre deux cellules, en cellules : max + (sqrt(2) - 1) * min.
double octileDistance(const GridIndex& a, const GridIndex& b);

// Recherche en largeur de la cellule non létale la plus proche (départ en zone
// létale). Renvoie false si aucune n'existe.
bool nearestFreeCell(const CostMap& map, const GridIndex& start, GridIndex& free_cell);

// Post-traitement du chemin A* :
//  1) raccourcis par ligne de vue (Bresenham sur les cellules létales) ;
//  2) densification puis lissage par descente de gradient ;
//  3) rééchantillonnage tous les "step" mètres, calcul du cap et de la courbure.
// Garantie : aucun point du résultat n'est en zone létale (sinon repli sur le chemin brut).

struct SmootherConfig {
  double w_data = 0.1;          // attache aux points d'origine
  double w_smooth = 0.3;        // lissage (rapproche chaque point du milieu de ses voisins)
  double w_obstacle = 0.3;      // éloignement des obstacles
  double obstacle_distance = 0.15;  // les obstacles plus loin que ça n'agissent pas (m)
  int iterations = 100;
  double resample_step = 0.01;  // (m)
};

SmootherConfig loadSmootherConfig(const ConfigNode& section);

// Vrai si le segment a-b ne traverse aucune cellule létale (tracé de Bresenham).
bool lineOfSight(const CostMap& map, const GridIndex& a, const GridIndex& b);

// Garde seulement les cellules nécessaires : depuis chaque point, on saute au
// point le plus lointain encore visible.
std::vector<GridIndex> shortcutPath(const CostMap& map, const std::vector<GridIndex>& cells);

// Rééchantillonne une polyligne tous les "step" mètres (extrémités conservées).
std::vector<Eigen::Vector2d> resamplePolyline(const std::vector<Eigen::Vector2d>& points,
                                              double step);

// Calcule s, cap et courbure de chaque point.
Path makePath(const std::vector<Eigen::Vector2d>& points);

// Toute la chaîne : cellules A* -> chemin lisse et sûr.
Path smoothPath(const CostMap& map, const std::vector<GridIndex>& cells,
                      const SmootherConfig& config);

// Planificateur complet : A* + lissage, gestion du départ en zone létale,
// et tests de replanification.

struct PlannerConfig {
  double resolution = 0.01;
  InflationConfig inflation;
  double w_cost = 2.0;
  SmootherConfig smoother;
  double replan_deviation = 0.10;  // écart au chemin qui déclenche une replanification (m)
};

// Lit params.yaml (planning) ; le rayon du robot vient de robot.yaml.
PlannerConfig loadPlannerConfig(const ConfigNode& root, double robot_radius);

enum class PlanStatus { Ok, GoalOutsideMap, GoalInObstacle, StartTrapped, NoPath };

const char* toString(PlanStatus status);

struct PlanResult {
  PlanStatus status = PlanStatus::NoPath;
  Path path;
  bool start_escaped = false;          // le départ était en zone létale (journaliser)
  std::vector<GridIndex> raw_cells;    // chemin A* brut (affichage)
  bool ok() const { return status == PlanStatus::Ok; }
};

PlanResult planPath(const CostMap& map, const Pose2D& start, const Eigen::Vector2d& goal,
                    std::optional<double> goal_heading, const PlannerConfig& config);

// Vrai si un point du chemin, à partir de from_index, est devenu létal.
bool pathBlocked(const CostMap& map, const Path& path, std::size_t from_index);

// Distance du point au chemin (au segment le plus proche), en m.
double distanceToPath(const Path& path, double x, double y);
