"""Shared configuration values for the smart traffic control system."""

PERCEPTION_CONFIG = {
    'camera_url': 'src/data/video.mp4',
    'yolo_model_path': 'yolov8n.pt',
}

SIMULATION_CONFIG = {
    'sumo_binary': 'sumo',
    'sumo_config': 'simulation/network.sumocfg',
    'step_length': 1.0,
}

DEPLOYMENT_CONFIG = {
    'controller_uri': 'http://127.0.0.1:5000/api/signal',
}

RL_CONFIG = {
    'algorithm': 'PPO',
    'timesteps': 200_000,
}
