# Robot Nav

Camera-guided navigation for a self-built two-wheeled robot, written from scratch in C++17. An overhead camera localises the robot through fiducial markers and detects obstacles; an extended Kalman filter, an A* planner and two controllers (Pure Pursuit and MPPI) drive it to goals clicked on screen. The same code runs in a simulator and on the real robot.

![Real robot, MPPI](docs/media/real_mppi.gif)

## Pipeline

| Stage | Method |
|---|---|
| Camera | iPhone, 1920 × 1080 at 30 fps; ChArUco intrinsics; floor pose from 4 ArUco markers |
| Localisation | rays through the roof-marker corners intersected with the roof plane |
| Obstacles | HSV segmentation, floor projection, 20-frame vote |
| Estimation | EKF on (x, y, θ, v, ω); camera delay compensated by replaying the last 2 s of commands; Mahalanobis gating |
| Planning | A* on a 1 cm occupancy grid inflated with a linear-time distance transform; shortcutting and smoothing |
| Control | Pure Pursuit; MPPI (512 samples, 1.5 s horizon) |
| Robot | Arduino firmware applies wheel PWM received over UDP (ESP8266) or USB; 300 ms watchdog |

## Results

Real robot, 3 flat red obstacles, 5 chained goals (videos: [Pure Pursuit](docs/media/real_pp.mp4), [MPPI](docs/media/real_mppi.mp4)):

| Controller | Goals reached | Final error (median / max) | Time per goal (median) | Obstacle contacts |
|---|---|---|---|---|
| Pure Pursuit | 5 / 5 | 0.9 / 1.7 cm | 7.4 s | 2 |
| MPPI | 5 / 5 | 1.0 / 1.1 cm | 7.6 s | 0 |

Simulation, 5 seeds per scenario, camera rendering and EKF in the loop:

| Scenarios | Controller | Goal reached (2 cm, no collision) | Collisions |
|---|---|---|---|
| 20 random, 0–2 obstacles | Pure Pursuit | 90 / 100 | 8 |
| | MPPI | 100 / 100 | 0 |
| 9 narrow passages | Pure Pursuit | 26 / 45 | 14 |
| | MPPI | 43 / 45 | 0 |

Details: [docs/results.md](docs/results.md). Hardware: [docs/hardware.md](docs/hardware.md).

## Build

```bash
scripts/install_deps.sh        # macOS (Homebrew); Linux without root: --local
cmake -B build -G Ninja -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build
ctest --test-dir build         # 85 tests
```

Dependencies: OpenCV 4/5, Eigen, yaml-cpp, GoogleTest (fetched), arduino-cli for the firmware.

## Usage

| Command | |
|---|---|
| `build/robot_nav --mode sim --realtime` | simulator; click a goal, drag for a heading, `c` switches controller |
| `build/robot_nav --mode real` | real robot |
| `build/bench --suite basic --mode full --controller mppi --seeds 5` | simulated benchmark |
| `scripts/real_trials pp mppi` | real-robot trial series, recorded |
| `build/camera_tool calib` / `arena --real` | camera intrinsics / floor pose |
| `build/robot_tool calib --real` | motor identification |
| `build/robot_tool check --wheels` / `teleop --real` | link test / keyboard driving |
| `scripts/flash_firmware` | firmware upload |

## Layout

| Path | |
|---|---|
| `src/` | core, vision, estimation, planning, control, simulation, hardware link, application |
| `apps/` | `robot_nav`, `bench`, `robot_tool`, `camera_tool` |
| `tests/` | GoogleTest unit and closed-loop tests |
| `config/` | robot and arena (`robot.yaml`), parameters (`params.yaml`), benchmark scenarios |
| `firmware/` | Arduino firmware and hardware test sketches |
| `docs/` | hardware, results, media |
