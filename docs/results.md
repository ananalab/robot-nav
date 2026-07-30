# Results

Configuration: `config/` as committed (arena 1.47 × 0.99 m, robot radius 0.12 m + 1 cm margin, flat 10 × 10 cm obstacles).

## Real robot

Three flat red obstacles, five chained goals per controller, robot pose measured by the camera.

| Goal (m) | Pure Pursuit error | MPPI error | Pure Pursuit time | MPPI time |
|---|---|---|---|---|
| (0.11, 0.11) | 0.8 cm | 1.0 cm | 7.4 s | 5.5 s |
| (1.12, 0.64) | 1.3 cm | 1.1 cm | 10.3 s | 10.1 s |
| (0.24, 0.62) | 0.9 cm | 1.0 cm | 6.0 s | 7.6 s |
| (1.21, 0.11) | 0.6 cm | 0.9 cm | 9.9 s | 9.7 s |
| (0.69, 0.33) | 1.7 cm | 0.6 cm | 3.9 s | 5.4 s |

| | Pure Pursuit | MPPI |
|---|---|---|
| Goals reached | 5 / 5 | 5 / 5 |
| Obstacle contacts (robot disc over a sheet) | 2, up to 2.7 cm | 0, smallest clearance 0.3 cm |

Videos: [Pure Pursuit](media/real_pp.mp4), [MPPI](media/real_mppi.mp4).

## Simulation

`build/bench --suite <basic|narrow> --mode <gt|full> --controller <pp|mppi> --seeds 5`

The simulated motors differ from the model used by the controller (left/right gains 0.61 / 0.43 m/s against 0.58 / 0.45, smaller dead zones, 0.10 s delay, 3 % wheel slip). `gt`: true pose given to the controller. `full`: rendered camera image (150 ms delay, 3 % dropped frames, blur, noise), marker detection and EKF in the loop.

| Scenarios | Pose | Controller | Goal reached (2 cm, no collision) | Collisions | Final error (median) |
|---|---|---|---|---|---|
| 20 random, 0–2 obstacles | True pose | Pure Pursuit | 97 / 100 | 3 | 6.5 mm |
|  | True pose | MPPI | 100 / 100 | 0 | 4.6 mm |
|  | Camera + EKF | Pure Pursuit | 90 / 100 | 8 | 7.7 mm |
|  | Camera + EKF | MPPI | 100 / 100 | 0 | 7.2 mm |
| 9 narrow passages and slaloms | True pose | Pure Pursuit | 40 / 45 | 5 | 8.8 mm |
|  | True pose | MPPI | 45 / 45 | 0 | 5.0 mm |
|  | Camera + EKF | Pure Pursuit | 26 / 45 | 14 | 8.1 mm |
|  | Camera + EKF | MPPI | 43 / 45 | 0 | 7.1 mm |

Half of the basic scenarios and 3 narrow ones also request a final heading within 5°. It is met with the true pose in most runs but not with the camera in the loop: the minimum wheel command that overcomes static friction on the real robot makes the final in-place rotation too coarse for 5°.

## Perception (`build/test_vision`)

| Measurement | Result |
|---|---|
| Camera pose from the 4 floor markers (30 images) | 5.1 mm, 0.12°, reprojection 0.30 px |
| Robot position, 500 images, ray–plane method | median 2.9 mm, p95 3.5 mm, heading p95 0.32° |
| Robot position, 500 images, `solvePnP` IPPE_SQUARE | median 3.4 mm, p95 5.3 mm, heading p95 0.41° |
| Flat obstacles, 20-image vote, grid IoU | 0.98 |
| ChArUco calibration on synthetic views | fx within 0.3 %, RMS 0.18 px |
