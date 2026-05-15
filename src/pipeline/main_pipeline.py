import time
import numpy as np
from src.perception.yolo_detector import TrafficProcessor, load_model
from src.rl.agent import TrafficAgent
from src.utils.config import RL_CONFIG

class IntegratedController:
    def __init__(self):
        yolo_model = load_model()
        self.processor = TrafficProcessor(yolo_model)

        # FIX: Pass the path to the model you just created
        self.agent = TrafficAgent(
            algorithm=RL_CONFIG['algorithm'], 
            model_path="models/ppo_traffic_model.zip" 
        )

        self.last_green_time = {0: time.time(), 1: time.time()}
        self.max_wait_seconds = 120

    def get_decision(self, frame):
        """
        Processes a frame and decides which lane gets the green light.
        """
        # 1. PERCEPTION: This defines 'counts' and 'annotated_frame'
        result = self.processor.process_frame(frame)
        counts = [result['lane1_count'], result['lane2_count']]
        annotated_frame = result['frame']
        current_time = time.time()

        # 2. FAIRNESS RULE: Override AI if wait time > 120s
        for lane_id in [0, 1]:
            if counts[lane_id] > 0:
                wait_time = current_time - self.last_green_time[lane_id]
                if wait_time > self.max_wait_seconds:
                    self.last_green_time[lane_id] = current_time 
                    # Return 1.0 (100%) confidence for manual overrides
                    return lane_id, annotated_frame, True, 1.0 

        # 3. RL DECISION: Let the Brain think
        obs = np.array(counts, dtype=np.float32)
        
        # This calls the updated predict from agent.py
        action, confidence = self.agent.predict(obs) 
        
        # Ensure 'action' is an int for the dictionary
        action = int(action)

        # Update the timer for the lane that gets the green light
        self.last_green_time[action] = current_time

        return action, annotated_frame, False, confidence
if __name__ == "__main__":
    print("Pipeline Initialized. Waiting for data...")