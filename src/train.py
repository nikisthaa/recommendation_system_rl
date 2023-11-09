import os
import sys
import random
import numpy as np

import tensorflow as tf

from utils import get_data
from environment import RecEnv
from agents import DqnAgent, DdqnAgent
from constants import DATA_DIR, ALLOWED_ALGO, STATE_SIZE, SEED_VALUE


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(SEED_VALUE)
    tf.random.set_seed(SEED_VALUE)

    _,_, movies_list, ratings_df,_,movies_df = get_data()
    
    users_dict = np.load(os.path.join(DATA_DIR,'user_dict_final.npy'), allow_pickle=True)
    users_history_lens = np.load(os.path.join(DATA_DIR,'users_history_lens_dict_final.npy'), allow_pickle=True)

    users_num = max(ratings_df["user_id"])+1
    items_num = max(ratings_df["movie_id"])+1

    # Training setting
    train_users_num = int(users_num * 0.8)
    train_items_num = items_num
    train_users_dict = {k:users_dict.item().get(k) for k in range(1, train_users_num+1)}
    train_users_history_lens = {k:users_history_lens.item().get(k) for k in range(1, train_users_num+1)}

    MAX_EPISODE_NUM = train_users_num
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}

    algorithm_choice = sys.argv[1]
    if algorithm_choice in ALLOWED_ALGO:
        print(f"Training: {ALLOWED_ALGO[algorithm_choice]}")
        env = RecEnv(train_users_dict, train_users_history_lens, movies_id_to_movies, STATE_SIZE)
        ACTION_SIZE = len(movies_df)
        if algorithm_choice == "DQN":
            recommender = DqnAgent(env, users_num, items_num, STATE_SIZE, ACTION_SIZE)
        else:
            recommender = DdqnAgent(env, users_num, items_num, STATE_SIZE, ACTION_SIZE)
        recommender.build_network()
        recommender.train(env, MAX_EPISODE_NUM, load_model=False)
    else:
        raise ValueError("Improper algorithm. Pass \"DQN\" or \"DDQN\"")
