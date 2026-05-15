"""SUMO integration layer for traffic simulation and training."""

import os
import subprocess
import traci
from src.utils.config import SIMULATION_CONFIG


class SUMOInterface:
    def __init__(self, sumo_binary: str = SIMULATION_CONFIG['sumo_binary'], config_file: str = SIMULATION_CONFIG['sumo_config']):
        self.sumo_binary = sumo_binary
        self.config_file = config_file

    def launch(self):
        cmd = [self.sumo_binary, '-c', self.config_file, '--step-length', str(SIMULATION_CONFIG['step_length'])]
        return subprocess.Popen(cmd)

    def run_training_episode(self):
        process = self.launch()
        process.wait()
        return process.returncode

    def get_state(self):
        # Placeholder for interaction via TraCI or SUMO network files.
        return {}

    def sync_simulation(self, lane_counts):
        """
        Injects vehicles into SUMO based on YOLO detections
        to keep the 'Digital Twin' accurate.
        """
        for lane_id, count in enumerate(lane_counts):
            for _ in range(count):
                # Add a vehicle to the simulation at the specific lane
                veh_id = f"veh_{lane_id}_{traci.simulation.getTime()}"
                traci.vehicle.add(veh_id, routeID=f"route_{lane_id}")
