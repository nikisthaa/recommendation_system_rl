import os

ROOT_DIR = os.getcwd()
DATA_DIR = ROOT_DIR+"/data"
ALLOWED_ALGO = {
    "DQN": "Deep Q Network",
    "DDQN": "Double Deep Q Network"
}
SEED_VALUE=50
STATE_SIZE = 10 # Window size: representing number of items seen in current window
MODEL_PATH = {
    "DQN": {
    "Q_NET": ROOT_DIR +"/rl_model/dqn/q_network_100_fixed.h5",
    "TAR_NET": ROOT_DIR +"/rl_model/dqn/target_network_100_fixed.h5"
    },
    "DDQN": {
    "Q_NET": ROOT_DIR +"/rl_model/ddqn/q_network_100_fixed.h5",
    "TAR_NET": ROOT_DIR +"/rl_model/ddqn/target_network_100_fixed.h5"
    } 
}
