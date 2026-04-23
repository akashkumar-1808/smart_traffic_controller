"""Edge deployment runtime for Jetson Nano and signal controller integration."""

import time
from src.utils.config import DEPLOYMENT_CONFIG


class EdgeRunner:
    def __init__(self, controller_uri: str = DEPLOYMENT_CONFIG['controller_uri']):
        self.controller_uri = controller_uri

    def apply_signal_plan(self, plan):
        # Replace with actual protocol for signal controller integration.
        print(f'Applying signal plan to controller at {self.controller_uri}: {plan}')

    def run(self, policy, camera_stream):
        for frame in camera_stream:
            state = self._extract_state(frame)
            action = policy.predict(state)
            self.apply_signal_plan(action)
            time.sleep(1)

    def _extract_state(self, frame):
        return frame
