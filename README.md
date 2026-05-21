# Smart Urban Traffic Control (SUTC)

AI-driven adaptive traffic signal control system using computer vision and reinforcement learning.

## Overview

Smart Urban Traffic Control (SUTC) is a traffic signal optimization system designed to dynamically manage lane prioritization using real-time vehicle density.

The system combines:

- **YOLOv8n** for vehicle detection and lane-wise traffic estimation
- **Proximal Policy Optimization (PPO)** for signal decision-making
- **Fairness override logic** to prevent lane starvation
- **Streamlit dashboard** for real-time monitoring and telemetry

Unlike conventional fixed-timer traffic systems, SUTC makes decisions based on observed traffic density from video input.

---

## System Architecture

The system follows a perception → decision → control pipeline.

```text
Video Feed
     │
     ▼
YOLOv8 Vehicle Detection
     │
     ▼
Lane-wise Vehicle Counting
     │
     ▼
PPO Reinforcement Learning Agent
     │
     ├── Fairness Override Check
     │
     ▼
Traffic Signal Decision
     │
     ▼
Real-Time Dashboard
```

---

## Core Workflow

The system operates through an observation–action–reward loop.

### 1. Observation

Traffic frames are processed using YOLOv8n to detect vehicles.

Detected vehicle classes include:

- Car
- Bus
- Truck
- Motorcycle
- Bicycle

Each detected vehicle is assigned to a lane using region-of-interest (ROI) segmentation.

### 2. Decision

The PPO agent selects one of two actions:

| Action | Description |
|--------|-------------|
| 0 | Give green signal to Lane 1 |
| 1 | Give green signal to Lane 2 |

The RL policy is trained to minimize vehicle congestion.

### 3. Reward

The reward function penalizes waiting vehicles:

\[
Reward = -(Lane1\_Count + Lane2\_Count)
\]

The agent learns policies that reduce congestion by minimizing queue lengths.

---

## Perception Layer

### YOLOv8n Vehicle Detection

The perception module uses **YOLOv8n** for real-time inference.

**Model:** YOLOv8n  
**Framework:** Ultralytics  
**Input:** Video stream / pre-recorded traffic footage  
**Output:** Bounding boxes, class labels, confidence scores

Detected vehicles are assigned to lanes using bounding box center coordinates.

Example logic:

```python
cx = (xmin + xmax) // 2

if lane1_roi[0] <= cx <= lane1_roi[2]:
    lane1_count += 1
else:
    lane2_count += 1
```

---

## Reinforcement Learning Module

### PPO Agent

The system uses **Proximal Policy Optimization (PPO)** implemented through **Stable-Baselines3**.

The RL environment is built using **Gymnasium**.

### Observation Space

```python
[lane1_vehicle_count, lane2_vehicle_count]
```

### Action Space

```python
0 -> Lane 1 Green
1 -> Lane 2 Green
```

### Training

The PPO model is trained using a custom `TrafficEnv` environment.

Training script:

```bash
python train_model.py
```

The trained model is stored at:

```text
models/ppo_traffic_model.zip
```

---

## Fairness Override Mechanism

A purely RL-based controller may repeatedly prioritize high-density lanes and ignore low-density lanes.

To prevent indefinite waiting, the system includes a deterministic fairness rule.

### Wait Limit Rule

If a lane exceeds **120 seconds of waiting time** and contains vehicles, the PPO decision is overridden and the lane receives a green signal.

Example logic:

```python
WAIT_LIMIT = 120

if lane1_wait >= WAIT_LIMIT:
    return 0

if lane2_wait >= WAIT_LIMIT:
    return 1
```

This ensures fairness while preserving RL-based optimization.

---

## Dashboard

The system includes a real-time monitoring dashboard built using **Streamlit**.

### Available Telemetry

- Lane-wise vehicle count
- Signal state
- Wait time tracking
- PPO confidence scores
- RL vs fairness override indication

The UI uses `st.empty()` placeholders for smooth real-time updates without flickering.

---

## Project Structure

```text
SMARTTRAFFIC/
│
├── models/
│   └── ppo_traffic_model.zip
│
├── src/
│   ├── perception/
│   │   └── yolo_detector.py
│   │
│   ├── rl/
│   │   ├── env.py
│   │   └── agent.py
│   │
│   ├── pipeline/
│   │   └── main_pipeline.py
│   │
│   └── utils/
│       └── config.py
│
├── app.py
├── train_model.py
└── README.md
```

---

## Technologies Used

| Component | Technology |
|------------|-------------|
| Language | Python |
| Object Detection | YOLOv8n |
| Reinforcement Learning | PPO |
| RL Framework | Stable-Baselines3 |
| Environment API | Gymnasium |
| Computer Vision | OpenCV |
| Deep Learning | PyTorch |
| Dashboard | Streamlit |

---

## Installation

Install dependencies:

```bash
pip install ultralytics
pip install opencv-python
pip install stable-baselines3
pip install gymnasium
pip install streamlit
pip install torch
```

Or using requirements:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train the PPO Model

```bash
python train_model.py
```

### Start Dashboard

```bash
streamlit run app.py
```

The application will launch locally at:

```text
http://localhost:8501
```

---

## Current Implementation Status

| Module | Status |
|--------|--------|
| YOLOv8 Vehicle Detection | Implemented |
| Lane-wise Vehicle Counting | Implemented |
| PPO Decision Engine | Implemented |
| Fairness Override | Implemented |
| Streamlit Dashboard | Implemented |
| Real-time Confidence Monitoring | Implemented |
| Model Training Pipeline | Implemented |

---

## Author

**Gunturu Akash Kumar**

AI/ML | Computer Vision | Reinforcement Learning
