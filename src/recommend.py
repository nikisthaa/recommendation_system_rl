import os
import sys
import numpy as np

from utils import get_data
from environment import RecEnv
from agents import DdqnAgent, DqnAgent
from constants import DATA_DIR, ALLOWED_ALGO, STATE_SIZE, MODEL_PATH


if __name__ == "__main__":
    
    _,_, movies_list, ratings_df,_,movies_df = get_data() 
    ACTION_SIZE = len(movies_df)

    users_dict = np.load(os.path.join(DATA_DIR,'user_dict_final.npy'), allow_pickle=True)
    users_history_lens = np.load(os.path.join(DATA_DIR,'users_history_lens_dict_final.npy'), allow_pickle=True)

    users_num = max(ratings_df["user_id"])+1
    items_num = max(ratings_df["movie_id"])+1

    # Evaluation setting
    eval_users_num = int(users_num * 0.2)
    eval_items_num = items_num
    eval_users_dict = {k:users_dict.item().get(k) for k in range(users_num-eval_users_num, users_num)}
    eval_users_history_lens = {k:users_history_lens.item().get(k) for k in range(users_num-eval_users_num, users_num)}

    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}
    # Parse inputs
    algorithm_choice = sys.argv[1]
    user_id = int(sys.argv[2])
    if algorithm_choice in ALLOWED_ALGO:
        env = RecEnv(eval_users_dict, eval_users_history_lens, movies_id_to_movies, STATE_SIZE, user_id)    
        user_id, items, _ = env.reset()
        if algorithm_choice == "DQN":
            recommender = DqnAgent(env, users_num, items_num, STATE_SIZE, ACTION_SIZE)
        else:
            recommender = DdqnAgent(env, users_num, items_num, STATE_SIZE, ACTION_SIZE)
        recommender.build_network()
        recommender.load_model(MODEL_PATH[algorithm_choice]['Q_NET'], MODEL_PATH[algorithm_choice]['TAR_NET'])
        recommended_item_id = recommender.recommend_item(user_id, items)
        print(f"\nRecommending for user: {user_id} using : {ALLOWED_ALGO[algorithm_choice]}")
        print("------------------------------------------------------------\n")

        print(f"User current watch history:")
        print("-------------------------------------------------------------")
        for item in items:
            print(movies_id_to_movies[str(item)])
        print("Recommended movie:")
        print("-------------------------------------------------------------")
        print(movies_id_to_movies[str(recommended_item_id)])
    else:
        raise ValueError("Improper algorithm. Pass \"DQN\" or \"DDQN\"")
