import random
import numpy as np


class RecEnv():
    def __init__(self, users_dict, users_history_lens, movies_id_to_movies, state_size, fix_user_id=None):
        self.users_dict = users_dict
        self.users_history_lens = users_history_lens
        self.items_id_to_name = movies_id_to_movies
        self.state_size = state_size
        self.available_users = self._generate_available_users()

        self.fix_user_id = fix_user_id

        self.user = fix_user_id if fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        self.done_count = 450
        
    def _generate_available_users(self):
        available_users = []
        for user_id, history_lens in zip(self.users_dict.keys(), self.users_history_lens):
            if history_lens > self.state_size:
                available_users.append(user_id)
        return available_users

    def reset(self):
        self.user = self.fix_user_id if self.fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        return self.user, self.items, self.done
    
    def step(self, action):                
        recommended_item = action + 1

        # Calculate diversity bonus
        diversity_bonus = 0
        if recommended_item not in self.recommended_items:
            diversity_bonus = 0.1

        # Calculate user preference bonus
        user_preference_bonus = 0
        if recommended_item in self.user_items.keys():
            user_preference_bonus = self.user_items[recommended_item]

        # Calculate redundancy penalty
        redundancy_penalty = 0
        if recommended_item in self.recommended_items:
            redundancy_penalty = -0.5

        # Calculate final reward
        reward = user_preference_bonus + redundancy_penalty + diversity_bonus

        # Update the state
        if recommended_item in self.user_items.keys() and recommended_item not in self.recommended_items:
            self.items = self.items[1:] + [recommended_item]
        else:
            random.shuffle(self.items)

        self.recommended_items.add(recommended_item)

        # Check if the episode is done
        if len(self.recommended_items) > self.done_count or len(self.recommended_items) >= len(self.users_history_lens[self.user]):
            self.done = True
        
        return self.items, reward, self.done, self.recommended_items

    def get_items_names(self, items_ids):
        items_names = []
        for id in items_ids:
            try:
                items_names.append(self.items_id_to_name[str(id)])
            except:
                items_names.append(list(['Not in list']))
        return items_names
