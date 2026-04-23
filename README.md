# AI-Powered Smart Traffic Control System

This research project implements an intelligent adaptive traffic signal control system for Indian roads using computer vision and reinforcement learning.

## Project Goals

- Replace fixed-cycle traffic signals with an adaptive system.
- Use live CCTV footage to detect and classify vehicles in real time with YOLOv8.
- Estimate traffic density and feed state information into a reinforcement learning (PPO/DQN) agent.
- Prioritize emergency vehicles automatically.
- Coordinate across multiple connected intersections.
- Account for mixed traffic and low lane discipline typical of Indian road conditions.
- Train policies in a calibrated SUMO simulation and deploy to edge hardware like Jetson Nano.
- Target performance improvements:
  - 30–40% reduction in average wait time
  - 60% faster emergency vehicle clearance

## Architecture

- `src/perception/`: Camera input and YOLO-based vehicle detection.
- `src/rl/`: Reinforcement learning environment, agent, and training logic.
- `src/simulation/`: SUMO integration for simulation-based training.
- `src/deployment/`: Edge runtime and controller integration.
- `src/utils/`: Shared helpers, configuration, and stream handling.

## Getting Started

1. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
   If you see an import error for `cv2`, use:
   ```bash
   python -m pip install opencv-python
   ```
   If you see an import error for `ultralytics`, use:
   ```bash
   python -m pip install ultralytics
   ```
2. Configure your cameras and simulation paths in `src/utils/config.py`.
3. Run the perception pipeline to verify detection:
   ```bash
   python -m src.perception.yolo_detector
   ```
4. Train the RL agent using the SUMO environment.

## Notes

- This repository is intended as a research prototype and proof-of-concept.
- Deployment on Jetson Nano requires hardware-specific optimizations and driver setup.
- Real-world deployment should be tested carefully with municipal partners.
“The system uses configurable lane regions, making it adaptable to any intersection layout.”