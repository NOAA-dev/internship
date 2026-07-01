# 1/14th-Scale Connected Autonomous Vehicle (CAV) — Full-Stack Navigation

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)
![Python](https://img.shields.io/badge/Python-3-blue)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin%20Nano-76B900)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)

A ROS 2 full-stack autonomous navigation system for a 1/14th-scale Ackermann-steered
Connected Autonomous Vehicle (CAV): occupancy-grid map generation, a kinematically-feasible
Hybrid A* global planner, a Linear Time-Varying MPC (with a Stanley + PID baseline),
multi-sensor EKF localization (VESC odometry, BNO055 IMU, AprilTag IPS), a VESC hardware
interface, and a camera-based traffic-sign response pipeline.

> Developed at the **AMS Laboratory, IIT Bombay**, during a two-month summer internship.

---

## Overview

This repository is the real-time, on-hardware evolution of an inherited, simulation-only
planning/control codebase. It now runs closed-loop on a physical Ackermann-steered chassis
with a VESC-driven BLDC drivetrain, fusing wheel odometry, an IMU, and AprilTag fiducial
detections into a single pose estimate that drives either a reactive Stanley + PID controller
or a predictive LTV-MPC controller along a Hybrid A*-generated path.

The project's goal was to take planning/control algorithms validated only in offline
simulation and turn them into a working, closed-loop robot: wrapping the algorithms as ROS 2
nodes, building the hardware bridge to the VESC, calibrating sensors and actuators, and
replacing the reactive baseline controller with a constrained predictive controller once its
limitations (corner-cutting, no wall awareness) were exposed on the physical track.

---

## Features

- **Hybrid A\* global planner** using an RK4-integrated kinematic bicycle model, an 8-point
  oriented-footprint collision check, and a dual heuristic (2D Dijkstra distance + heading
  penalty).
- **LTV-MPC trajectory tracker** — condensed QP formulation with per-step (time-varying)
  linearization, solved in real time with OSQP, including actuator/rate constraints and an
  EDT-based (Euclidean Distance Transform) obstacle/wall-clearance fallback.
- **Stanley + PID baseline controller** (C++) used to validate the planner and hardware
  pipeline before MPC development.
- **Multi-sensor EKF localization** (`robot_localization`) fusing VESC-derived odometry, an
  IMU yaw measurement, and AprilTag absolute position corrections.
- **VESC hardware interface** translating velocity/steering commands into ERPM and servo PWM,
  with slew-rate limiting and an asymmetric steering-linkage calibration table.
- **Offline map pre-processor** converting a track-layout image into a dilated occupancy grid
  (for planning) and an un-dilated distance-transform grid (for MPC clearance costs).
- **Camera-based traffic-sign inference node** (YOLO-style ONNX model via OpenVINO/PyTorch)
  publishing sign classifications consumed by the MPC's velocity scheduler.
- **Shadow simulator** (`sim_vehicle_tf_generator`) that mirrors commanded control inputs
  through an ideal bicycle model, for isolating controller bugs from localization/hardware
  noise.
- **CSV data recorder** for time-synchronized logging of odometry, AprilTag detections, and
  visualized paths.

---

## Hardware Platform

| Component | Detail |
|---|---|
| Compute | NVIDIA Jetson Orin Nano (ARM, Ubuntu) † |
| Motor controller | VESC Mini V6 ("Skyflip"), driving a BLDC motor † |
| IMU | Bosch BNO055 (I²C), auto-fused orientation, relative-gyro mode `0x08` |
| Positioning | AprilTag fiducial-based Indoor Positioning System (camera + `apriltag` detector) |
| Camera | OAK-D (traffic-sign detector node: `bno_test_pkg` — YOLOv8-style ONNX model) † |
| LiDAR | YDLIDAR (driver package `ydlidar_ros2_driver` present in repo; not referenced in the launch file — see *Information Missing*) |
| Chassis | Ackermann-steered 1/14th-scale CAV, wheelbase 0.21 m † |
| Steering range | Asymmetric: approx. −10° (left lock) to +18.75°/+22° (right lock) † |
| Battery | 4S LiPo † |
| Encoder (evaluated, not deployed) | AS5600 Hall-effect magnetic rotary encoder (`as5966.py`) |

† Sourced from the internship report; not independently verifiable from repository source alone.

---

## Software Stack

| Layer | Technology |
|---|---|
| Middleware | ROS 2 (Humble-style `package.xml` / `setup.py` layout) |
| Global planning | Custom Hybrid A* (Python), NumPy, `nav2_map_server` for map serving |
| Local control | Custom LTV-MPC (Python) solved with **OSQP**; Stanley controller in **C++** |
| Localization | `robot_localization` EKF node (`ekf_node`) |
| Hardware I/O | `vesc_driver`, `vesc_ackermann`, `vesc_msgs` (VESC ROS 2 stack, vendored from [f1tenth/vesc](https://github.com/f1tenth/vesc)) |
| Perception | OpenCV, PyTorch/TorchVision, ONNX Runtime / OpenVINO (traffic-sign inference) |
| LiDAR driver | `ydlidar_ros2_driver` |
| Visualization | RViz2, custom `visualization_msgs/Marker` publishers |
| SLAM (optional mode) | `slam_toolbox` (`slam.launch.xml`, `slam_toolbox.yaml`) |

---

## Repository Structure

```
.
├── src/
│   ├── global_planer/                # Hybrid A*, map pre-processing, hardware relay, sim TF, data recorder
│   │   └── global_planer/
│   │       ├── hyprid_A_star.py              # HybridAStar global planner node
│   │       ├── map_pre_prcoessor.py          # Track image -> occupancy grid (.pgm/.yaml)
│   │       ├── real_robot_command_and_odom_relay.py   # VESC hardware bridge + odometry fusion fallback
│   │       ├── sim_vehicle_tf_generator.py   # Shadow bicycle-model simulator / path visualizer
│   │       └── data_recorder.py              # CSV logger (odometry, AprilTag, path)
│   │
│   ├── local_planner_mpc/            # LTV-MPC trajectory tracking controller (Python, OSQP)
│   │   └── local_planner_mpc/mpc_node.py
│   │
│   ├── local_planner_stanley/        # Stanley + PID baseline controller (C++)
│   │   └── src/
│   │       ├── stanley_for_real_bot.cpp
│   │       └── stanley_for_sim.cpp
│   │
│   ├── bno_test_pkg/                 # IMU driver, encoder evaluation, traffic-sign inference
│   │   └── bno_test_pkg/
│   │       ├── bno05.py                      # BNO055 IMU ROS 2 node (imu_node)
│   │       ├── as5966.py                     # AS5600 Hall-effect encoder evaluation (not deployed)
│   │       ├── inference.py                  # OAK-D / ONNX traffic-sign detector (inference_node)
│   │       └── result/                       # ONNX / OpenVINO / blob model weights
│   │
│   ├── custom_interfaces/            # Custom ROS 2 msg/srv definitions
│   │   ├── msg/Path.msg                      # float64[] x, y, theta
│   │   └── srv/HybridAStar.srv               # goal request -> feedback string
│   │
│   ├── program_bringup/              # Top-level launch files, RViz config, EKF config
│   │   ├── launch/iitb_bot.xml               # Main system launch file
│   │   ├── launch/slam.launch.xml            # Optional SLAM-mode launch
│   │   ├── config/efk_iitb.yaml              # EKF (robot_localization) parameters
│   │   └── config/iitb_view.rviz             # RViz2 layout
│   │
│   ├── robot_maps/                   # Pre-built track maps (.pgm/.yaml/.npy) and reference images
│   │   └── maps/iitb_maps/
│   │
│   ├── vesc/                         # Vendored VESC ROS 2 driver stack (vesc_driver, vesc_ackermann, vesc_msgs)
│   │
│   └── ydlidar_ros2_driver/          # Vendored YDLIDAR ROS 2 driver
│
└── docs/images/                      # Diagrams extracted for this README
```

---

## System Architecture

The stack is organized into five cooperating subsystems: **map processing/perception**,
**global path planning**, **local motion control**, **state estimation/sensor fusion**, and
**hardware interfacing**. All coordination happens over standard ROS 2 topics/services.

![Workflow Diagram](docs/images/workflow_diagram.png)
*System workflow: an offline map pre-processor feeds a dilated occupancy grid to Hybrid A*
and the map server/RViz; Hybrid A* publishes `/Path` to the active controller (LTV-MPC or
Stanley); the controller's `/cmd_vell`-style output is converted to VESC motor commands by the
hardware relay node, which also fuses AprilTag and IMU inputs into the EKF (`EKF_NODE`) at
50 Hz, closing the loop back to the controller via `/Odometry/filtered`. (Source: internship
report, Fig. 1 — topic names shown are as illustrated in the report; see the ROS 2 Graph
section below for the exact topic names used in this repository's source code.)*

**Subsystem responsibilities:**

- **Map processing** (`map_pre_prcoessor.py`) — converts a track-layout image into a dilated
  occupancy grid for planning and a separate un-dilated distance-transform grid for MPC
  clearance costs.
- **Global planning** (`hyprid_A_star.py`) — kinematically feasible search over the occupancy
  grid.
- **Local control** (`mpc_node.py`, `stanley_for_real_bot.cpp`) — tracks the planned path in
  real time.
- **State estimation** (`ekf_node`, `real_robot_command_and_odom_relay.py`, `bno05.py`) —
  fuses odometry, IMU, and AprilTag inputs.
- **Hardware interface** (`real_robot_command_and_odom_relay.py`) — converts control commands
  to VESC ERPM/servo signals and back-converts telemetry to odometry.

---

## Workflow

Runtime data flow, goal to actuation:

```
Goal (service call to /goal)
        │
        ▼
  Hybrid A* planner  ──publishes──▶  /path
        │
        ▼
  Active controller (MPC or Stanley)
        │  reads /odometry/filtered, /path
        │  publishes /velocity_steer (MPC) or /test_twist (Stanley)
        ▼
  Hardware interface node (real_robot_command_and_odom_relay)
        │  publishes /commands/motor/speed, /commands/servo/position
        ▼
       VESC  ──▶  Vehicle motion
        │
        ▼
   Sensors: VESC /sensors/core, BNO055 /imu, AprilTag /apriltag/tag2/pose
        │
        ▼
  real_robot_command_and_odom_relay  ──publishes──▶  /odom/imu
        │
        ▼
   ekf_filter_node (robot_localization)  ──publishes──▶  /odometry/filtered
        │
        └──────────────► feeds back into Hybrid A* and the active controller (loop)
```

---

## ROS2 Graph

### Nodes

| Node (ROS 2 name) | Executable | Package | Role |
|---|---|---|---|
| `hybrid_astar_planner` | `hybrid_a_star_` | `global_planer` | Global path planning |
| `mpc_controller` | `mpc_` | `local_planner_mpc` | Primary LTV-MPC trajectory tracker |
| `stanley_controller` / `stanley_controlle_sim` | `Stanley_real` / `Stanley` | `local_planner_stanley` | Baseline reactive tracker (real / simulated) |
| `spwan_real_vehicle` | `spwan_real_vehicle` | `global_planer` | Hardware interface bridge + odometry fusion fallback |
| `spawn_sim_vehicle` | `spawn_sim_vehicle` | `global_planer` | Shadow bicycle-model simulator |
| `ekf_filter_node` | `ekf_node` | `robot_localization` | Multi-sensor EKF fusion (50 Hz) |
| `map_gen_` | `map_gen_` | `global_planer` | Track image → occupancy grid (offline) |
| `data_recorder` | `data_collect` | `global_planer` | Time-synchronized CSV data logging |
| `imu_node` (unnamed at launch) | `imu_node` | `bno_test_pkg` | BNO055 IMU driver |
| *(not in main launch file)* | `inference_node` | `bno_test_pkg` | Traffic-sign detector (OAK-D + ONNX) |
| `map_server` | — | `nav2_map_server` | Serves the static occupancy grid |
| `lifecycle_manager_localization` | — | `nav2_lifecycle_manager` | Lifecycle-manages `map_server` |

> **Information Missing:** the traffic-sign `inference_node` (which publishes to
> `/sign_action_node/command`, consumed by the MPC) is defined in `bno_test_pkg/setup.py` but
> is **not** instantiated in `program_bringup/launch/iitb_bot.xml`. It must be launched
> separately, or the launch file is incomplete relative to what the MPC subscribes to.

### Key Topics

| Topic | Type | Publisher(s) | Subscriber(s) |
|---|---|---|---|
| `/path` | `custom_interfaces/Path` | `hybrid_astar_planner` | `mpc_controller`, `spawn_sim_vehicle` |
| `/odometry/filtered` | `nav_msgs/Odometry` | `ekf_filter_node` | `hybrid_astar_planner`, `mpc_controller`, hardware relay, data recorder |
| `/odom/imu` | `nav_msgs/Odometry` | `spwan_real_vehicle` | `ekf_filter_node` |
| `/imu` | `sensor_msgs/Imu` | `imu_node` (`bno05.py`) | `ekf_filter_node` |
| `/apriltag/tag2/pose` | `geometry_msgs/PoseStamped` | AprilTag detector (external) | hardware relay, data recorder |
| `/velocity_steer` | `std_msgs/Float64MultiArray` | `mpc_controller` | `spawn_sim_vehicle` |
| `/test_twist` | `geometry_msgs/Twist` | `mpc_controller`, Stanley controller | `spwan_real_vehicle` |
| `/commands/motor/speed`, `/commands/servo/position`, `/commands/motor/duty_cycle` | `std_msgs/Float64` | `spwan_real_vehicle` | VESC driver |
| `/sensors/core` | `vesc_msgs/VescStateStamped` | VESC driver | `spwan_real_vehicle` |
| `/sign_action_node/command` | `std_msgs/String` | `inference_node` (not launched by default — see note above) | `mpc_controller` |
| `/robot_actual_path`, `/mpc_predicted_path`, `/mpc_reference_path`, `/mpc_nominal_path` | `nav_msgs/Path` | `mpc_controller` | RViz2 |
| `/mpc_nearby_obstacles`, `/car_bounds_marker`, `/car_sim_bounds_marker` | `visualization_msgs/Marker` | `mpc_controller`, hardware relay, sim TF | RViz2 |
| `/path_vis` | `nav_msgs/Path` | `spawn_sim_vehicle` | `data_recorder` |
| `bno/yaw`, `bno/gyro`, `bno/accel` | `std_msgs/Float64` / `geometry_msgs/Vector3` | `imu_node` | debugging/RViz |

### Services

| Service | Type | Server | Client |
|---|---|---|---|
| `/goal` | `custom_interfaces/HybridAStar` | `hybrid_astar_planner` | `mpc_controller` (also used for replanning) |

### TF Frames

| Frame | Published by |
|---|---|
| `map` → `odom` | static transform (`map_to_odom_broadcaster_3`), disabled in SLAM mode |
| `base_footprint_imu` → `imu_link` | static transform (`map_to_odom_broadcaster_2`) |
| `odom` → `base_link` (or equivalent) | `ekf_filter_node` (`publish_tf: true`) |

### Parameters (selected, from `efk_iitb.yaml`)

| Parameter | Value |
|---|---|
| `frequency` | 50.0 Hz |
| `two_d_mode` | true |
| `map_frame` / `odom_frame` / `base_link_frame` / `world_frame` | `map` / `odom` / `base_footprint_imu` / `odom` |
| `imu0` | `imu` (yaw only — `imu0_config` enables only the yaw/orientation field) |
| `odom0` | `/odom/imu` (x, y position and forward velocity) |

---

## Path Planning — Hybrid A*

`hyprid_A_star.py` implements a Hybrid A* planner over the occupancy grid built by the map
pre-processor. Motion primitives are propagated forward with a continuous-time kinematic
bicycle model (rear-axle reference), integrated using 4th-order Runge-Kutta for accuracy over
long horizons:

```
ẋ = v·cos(θ)      ẏ = v·sin(θ)      θ̇ = (v / L)·tan(δ)
```

- **Heuristic:** a holonomic 2D Dijkstra distance-to-goal map (obstacle-aware) combined with a
  heading-penalty term near the goal. The analytical Reed-Shepp expansion used in some classic
  Hybrid A* formulations was intentionally omitted to keep planning latency predictable on
  embedded hardware.
- **Collision checking:** an 8-point oriented bounding-box footprint check (replacing an
  earlier single-point/ray check that caused corner-clipping).
- **Steering limits:** asymmetric, matching the physical linkage (~−10° to +18.75°/+22°).
- **QoS:** the `/path` publisher uses **Transient Local** durability so late-joining
  subscribers (e.g. RViz2) immediately receive the last published plan.

![Track layout](docs/images/track_layout.jpeg)
*Reference track layout image used as the source for occupancy-grid generation.*

---

## Motion Control

### Stanley + PID (baseline)

Implemented in C++ (`stanley_for_real_bot.cpp` / `stanley_for_sim.cpp`). Control law:

```
δ = e_ψ + arctan( (Kp·e_y + Ki·∫e_y + Kd·de_y/dt) / (v + ε) )
```

where `e_ψ` is heading error, `e_y` is cross-track error at a front-axle projection point, and
error is evaluated at a "carrot" waypoint offset ahead of the nearest path point. Final tuned
gains (per the internship report): `Kp = 0.37, Ki = 0.00, Kd = 0.10`. This controller has no
predictive horizon, no native obstacle/wall awareness, and is vulnerable to path-index
"loop-trapping" on closed tracks — it served primarily to validate the planner and the
hardware pipeline.

### LTV-MPC (primary controller)

Implemented in `mpc_node.py`. A linear time-varying model predictive controller with:

- **Per-step Jacobian linearization** of the kinematic bicycle model around a rolled-out
  nominal trajectory (rather than a single fixed linearization point), assembled into a
  condensed (state-free) QP: `X = Φ·x₀ + Γ·U + C`.
- **Solver:** OSQP, warm-started each cycle from the previous shifted control sequence.
- **Constraints:** actuator bounds, steering/acceleration slew-rate limits, and velocity
  bounds — all expressed as linear inequality constraints on the stacked control sequence.
- **Obstacle handling:** not a native QP constraint; a post-QP Euclidean Distance Transform
  (EDT) check classifies the predicted trajectory as SAFE / SLOW / STOP and can override the
  QP output with an open-loop escape maneuver.
- **Replanning:** `check_position_jump()` detects large discontinuities in the EKF pose
  (typically from an AprilTag re-lock after dropout) and triggers a new Hybrid A* request via
  the `/goal` service.
- **Traffic-sign response:** subscribes to `/sign_action_node/command` and applies velocity
  overrides for `STOP`, `SLOW_DOWN`, and `PARK` classifications.

### Comparison

| Category | Stanley + PID | LTV-MPC |
|---|---|---|
| Kinematic feasibility | Moderate — clipping applied post-hoc | High — RK4 rollout + per-step linearization |
| Obstacle/boundary handling | None (external logic only) | EDT-based post-QP fallback |
| Wall awareness | Absent at controller level | Explicit SAFE/SLOW/STOP zones |
| Corner cutting | Frequent on tight curves | Reduced by predictive horizon |
| Computational cost | Very low (closed-form) | Moderate (condensed QP via OSQP) |
| Constraint handling | Post-hoc clamping | Hard constraints in the optimizer |
| Sign/perception commands | External override only | Native subscription in the controller |
| Replanning on pose jump | Not handled | Triggers a new Hybrid A* request |

---

## Localization

State estimation is provided by `ekf_filter_node` (`robot_localization`) at 50 Hz, publishing
`/odometry/filtered`.

![EKF localization data flow](docs/images/ekf_dataflow_diagram.png)
*EKF data flow: BNO055 IMU yaw and VESC-derived odometry (with AprilTag position override
applied inside the hardware relay node) are fused into `/odometry/filtered`, consumed by both
the controller and the global planner.*

- **IMU (`bno05.py`):** BNO055 in relative-gyro fusion mode (`0x08`), publishing an
  already-fused absolute yaw (not a raw angular rate) to `/imu`. The EKF config
  (`efk_iitb.yaml`) enables only the yaw/orientation field from this input.
- **Odometry:** wheel-speed-derived translational odometry, computed in
  `real_robot_command_and_odom_relay.py` from VESC ERPM feedback, published on `/odom/imu`.
- **AprilTag IPS:** when a fresh detection is available on `/apriltag/tag2/pose`, the hardware
  relay node overrides the dead-reckoned x/y position before publishing `/odom/imu`, rather
  than adding AprilTag as a second EKF measurement stream.
- **Sensor fusion architecture note:** feeding IMU angular velocity *and* odometry position
  simultaneously was tried and abandoned (double-integration of heading); the deployed
  configuration cleanly separates translational state (odometry) from rotational state
  (absolute IMU yaw) to keep the fusion deterministic.
- **AS5600 encoder** (`as5966.py`) was evaluated as an alternative velocity source but was not
  mechanically stable enough for deployment; VESC ERPM telemetry remains the sole velocity
  source.

---

## Hardware Interface

`real_robot_command_and_odom_relay.py` (node name `twist_test`, executable `spwan_real_vehicle`)
bridges high-level `geometry_msgs/Twist` commands to the VESC:

- **Command path:** subscribes to `/test_twist` → publishes `/commands/motor/speed`,
  `/commands/servo/position`, `/commands/motor/duty_cycle` (`std_msgs/Float64`).
- **Steering calibration:** a piecewise interpolation table maps normalized servo commands to
  effective wheel angle, correcting for the asymmetric Ackermann linkage (approx. −10° left to
  +18.5–22° right).
- **Slew-rate limiting:** steering and speed commands are rate-limited before being sent to
  the actuators to emulate/respect physical actuator transit time.
- **Odometry:** subscribes to `/sensors/core` (VESC state) and `/odometry/filtered` (EKF) and
  to `/apriltag/tag2/pose`; publishes fused `/odom/imu`, plus debug topics
  (`/debug/wheel_speed_ms`, `/debug/wheel_angle_rad`, `/debug/cmd_speed_ms`,
  `/debug/theta1_deg`, `/debug/theta2_deg`, `/debug/tag_age_sec`).

![Actuator calibration / IPS fallback logic](docs/images/actuator_calibration_flowchart.png)
*Localization fallback logic inside the hardware relay: fresh AprilTag detections trigger a
hard position overwrite; stale detections fall back to EKF-blended dead-reckoning.*

---

## Installation

> **Information Missing:** no top-level `README`, `Dockerfile`, `rosdep` keys file, or
> `requirements.txt` exists in the repository, so the exact list of system/Python
> dependencies below is inferred from `package.xml`/`import` statements and should be verified
> against your ROS 2 distribution before use.

```bash
# System prerequisites (ROS 2 — Humble or compatible distro assumed from package.xml syntax)
sudo apt update
sudo apt install -y \
  ros-$ROS_DISTRO-nav2-map-server \
  ros-$ROS_DISTRO-nav2-lifecycle-manager \
  ros-$ROS_DISTRO-robot-localization \
  ros-$ROS_DISTRO-slam-toolbox \
  ros-$ROS_DISTRO-tf-transformations \
  ros-$ROS_DISTRO-joy \
  ros-$ROS_DISTRO-joy-teleop \
  ros-$ROS_DISTRO-xacro \
  ros-$ROS_DISTRO-rviz2 \
  python3-serial

# Python dependencies used by the MPC / planning / perception nodes
pip install --break-system-packages numpy scipy osqp opencv-python torch torchvision onnxruntime
```

Clone into a ROS 2 workspace:

```bash
mkdir -p ~/cav_ws/src
cd ~/cav_ws/src
git clone https://github.com/NOAA-dev/internship.git
```

---

## Build

```bash
cd ~/cav_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

---

## Running

```bash
# Default: track-map mode with the LTV-MPC controller
ros2 launch program_bringup iitb_bot.xml

# Use the Stanley baseline controller instead
ros2 launch program_bringup iitb_bot.xml controller:=stanley

# Run in SLAM mode instead of a pre-built static map
ros2 launch program_bringup iitb_bot.xml mode:=slam

# Trigger a goal (custom_interfaces/srv/HybridAStar)
ros2 service call /goal custom_interfaces/srv/HybridAStar "{goal: [x, y, theta]}"
```

RViz2 launches automatically using `program_bringup/config/iitb_view.rviz`, showing the map,
planned path, MPC predicted/reference/nominal paths, and vehicle footprint markers.

The traffic-sign inference node is **not** included in the default launch file and must be run
separately if sign-response behavior is required:

```bash
ros2 run bno_test_pkg inference_node
```

**Expected behavior:** the vehicle should plan a Hybrid A*-feasible path to the requested
goal, track it with the active controller, and slow/stop for detected obstacles (EDT fallback,
MPC only) or classified traffic signs (if `inference_node` is running).

---

## Experimental Results

Four physical validation runs (from the internship report) compared EKF localization
configurations while holding the MPC controller and planner fixed.

| Metric | IPS + IMU (production) | IPS Only | IMU Only | Sign-Detection Run |
|---|---|---|---|---|
| Run duration (s) | 20.3 | 39.3 | 25.7 | 47.6 |
| Cross-track error, mean (cm) | 4.46 | 8.65 | 3.84 | 6.84 |
| Cross-track error, p95 (cm) | 7.59 | 11.05 | 11.55 | 17.67 |
| Localization error vs. AprilTag ground truth, mean (cm) | 0.74 | 0.71 | 22.66 | 0.43 |
| Mean moving velocity (m/s) | 0.252 | 0.207 | 0.313 | 0.274 |

<table>
<tr>
<td><img src="docs/images/run1_ips_imu_trajectory.png" width="420"/><br/><sub>Run 1 — IPS + IMU (production)</sub></td>
<td><img src="docs/images/run2_ips_only_trajectory.png" width="420"/><br/><sub>Run 2 — IPS only</sub></td>
</tr>
<tr>
<td><img src="docs/images/run3_imu_only_trajectory.png" width="420"/><br/><sub>Run 3 — IMU only (dead-reckoning)</sub></td>
<td><img src="docs/images/run4_sign_detection_trajectory.png" width="420"/><br/><sub>Run 4 — Sign-detection run</sub></td>
</tr>
</table>

**Key findings** (report, Section V-E):

- The **IPS + IMU** configuration gave the best combined result: sub-2 cm absolute
  localization error (p95) and the lowest cross-track error among the two-sensor
  configurations.
- **IMU-only** (dead-reckoning, no AprilTag correction) produced the *lowest* cross-track
  error numerically, but this is a self-consistency artifact — the controller tracked its own
  drifting reference frame accurately while true position error grew to ~22–28 cm.
- **IPS-only** (no IMU yaw) showed degraded heading accuracy during brief AprilTag dropouts,
  roughly doubling cross-track error relative to the production configuration.
- The sign-detection run confirmed STOP/SLOW_DOWN/PARK velocity-override behavior, including
  two 5-second STOP holds matching the MPC's designed dwell duration to within 20 ms.

> Cross-track error is a controller-tracking metric measured against the EKF's own belief of
> its position, not an absolute-accuracy metric; localization error against AprilTag ground
> truth is the absolute-accuracy metric.

---

## Performance

- **Planning:** unconstrained open-set search could take 10–20 s under tight turning
  conditions before optimization (state-space pruning, coarser discretization, distance-field
  clearance costs).
- **Tracking:** switching from single-point to per-step (true LTV) linearization eliminated a
  systematic parallel-offset tracking failure in the MPC; the controller then converged from
  15–20 cm initial lateral offsets within 2–3 seconds.
- **Controller improvements:** LTV-MPC materially reduced corner-cutting relative to the
  Stanley baseline by using a finite prediction horizon and boundary-penalty constraints
  instead of post-hoc clamping.
- **EKF performance:** the final IMU-yaw + AprilTag-position configuration was the first to
  produce run-to-run reproducible trajectories, after two earlier configurations
  (raw-angular-velocity fusion, weighted yaw merging) failed due to double-integration and
  non-deterministic message-timing sensitivity, respectively.
- **MPC solve time:** OSQP warm-starting reduced ADMM iterations from ~100 (cold start) to
  ~10–20 per cycle, keeping solve time within the 20 ms budget of the 50 Hz control loop.

---

## Future Improvements

From the internship report's prioritized future-work list:

- Resolve intermittent QP infeasibility triggered by aggressive velocity-weight tuning.
- Replace the AutoCAD-derived static occupancy grid with live SLAM-based map generation.
- Integrate obstacle avoidance natively into the MPC cost function (linearized distance-field
  penalty) instead of the current post-QP EDT fallback.
- Characterize and compensate the IMU's lever-arm mounting offset.
- Expand the traffic-sign vocabulary and add per-sign velocity profiles.
- Migrate the traffic-sign CV pipeline from CPU to GPU execution.

---

## Acknowledgements

Developed during a summer internship at the **AMS Laboratory, Indian Institute of Technology
Bombay**, under the guidance of **Angshuman Baruah**. Includes vendored third-party ROS 2
packages: [`vesc`](https://github.com/f1tenth/vesc) (BSD license) and `ydlidar_ros2_driver`
(MIT license).

---

## License

> **Information Missing:** no top-level `LICENSE` file was found in this repository. The
> vendored `src/vesc` package is BSD-licensed and `src/ydlidar_ros2_driver` is MIT-licensed
> (per their own `package.xml`/`LICENSE` files); all other packages declare
> `TODO: License declaration` in their `package.xml` and have no license specified.

---

## Information Missing

The following items could not be reliably inferred from the repository source or could not be
fully reconciled between the repository and the internship report, and are flagged rather than
guessed:

- **No top-level README, LICENSE, `.gitignore`, CI config, or `rosdep` dependency list** exists
  in the repository; installation steps above are inferred from `package.xml` and `import`
  statements.
- The **traffic-sign `inference_node`** (`bno_test_pkg`) is defined and installable but is
  **not launched** by `program_bringup/launch/iitb_bot.xml`, even though `mpc_node.py`
  subscribes to its output topic (`/sign_action_node/command`). Whether this is an intentional
  manual-launch step or an incomplete launch file could not be determined from source.
- The AprilTag detector node itself (publisher of `/apriltag/tag2/pose`) is **not present in
  this repository** — it is an external dependency assumed to be running separately.
- Exact **ROS 2 distribution** (e.g. Humble vs. Iron) is not pinned anywhere in the repository;
  it is inferred from `package_format3.xsd` usage and general API style.
- Precise Jetson model, VESC firmware version, camera model (OAK-D), and battery specifications
  are taken from the internship report only and have no corresponding configuration/constant in
  the source confirming them.
- The `robot_maps/maps/iitb_maps/shrisha.jpeg` and `track_2.jpeg`/`track_3.png` files exist in
  the repository but their specific purpose (alternate track layouts vs. reference photos)
  could not be determined from surrounding code or comments.
- No `Dockerfile` or containerized build process is present.
