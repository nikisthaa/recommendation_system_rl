import os
import numpy as np
import pandas as pd

from constants import DATA_DIR


class SumTree:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.tree = np.zeros((buffer_size * 2 - 1))
        self.index = buffer_size - 1

    def update_tree(self, index):
        while True:
            index = (index - 1) // 2
            left = (index * 2) + 1
            right = (index * 2) + 2
            self.tree[index] = self.tree[left] + self.tree[right]
            if index == 0:
                break

    def add_data(self, priority):
        if self.index == self.buffer_size * 2 - 1:
            self.index = self.buffer_size - 1

        self.tree[self.index] = priority
        self.update_tree(self.index)
        self.index += 1

    def search(self, num):
        current = 0
        while True:
            left = (current * 2) + 1
            right = (current * 2) + 2

            if num <= self.tree[left]:
                current = left
            else:
                num -= self.tree[left]
                current = right
            
            if current >= self.buffer_size - 1:
                break

        return self.tree[current], current, current - self.buffer_size + 1

    def update_prioirty(self, priority, index):
        self.tree[index] = priority
        self.update_tree(index)

    def sum_all_prioirty(self):
        return float(self.tree[0])


class MinTree:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.tree = np.ones((buffer_size * 2 - 1))
        self.index = buffer_size - 1

    def update_tree(self, index):
        while True:
            index = (index - 1) // 2
            left = (index * 2) + 1
            right = (index * 2) + 2
            if self.tree[left] > self.tree[right]:
                self.tree[index] = self.tree[right]
            else:
                self.tree[index] = self.tree[left]
            if index == 0:
                break

    def add_data(self, priority):
        if self.index == self.buffer_size * 2 - 1:
            self.index = self.buffer_size - 1

        self.tree[self.index] = priority
        self.update_tree(self.index)
        self.index += 1

    def update_prioirty(self, priority, index):
        self.tree[index] = priority
        self.update_tree(index)

    def min_prioirty(self):
        return float(self.tree[0])


def get_data():

    # Loading datasets

    ratings_cols = ["user_id", "movie_id", "ratings", "timestamp"]
    movies_cols = ["movie_id", "title", "genres"]
    user_cols = ["user_id", "age", "gender", "occupation", "zip-codes"]

    ratings_list = [i.strip().split(",") for i in open(os.path.join(DATA_DIR,'ratings.csv'), 'r').readlines()]
    users_list = [i.strip().split("|") for i in open(os.path.join(DATA_DIR,'u.user'), 'r').readlines()]
    movies_list = [i.strip().split(",") for i in open(os.path.join(DATA_DIR,'movies.csv'),encoding='latin-1').readlines()]

    ratings_df = pd.read_csv(os.path.join(DATA_DIR,'ratings.csv'))
    ratings_df.columns = ratings_cols
    users_df = pd.read_csv(os.path.join(DATA_DIR,'u.user'), sep="|", names = user_cols)
    movies_df = pd.read_csv(os.path.join(DATA_DIR,'movies.csv'), encoding='latin-1')

    # Coverting string to int/numeric
    movies_df['movie_id'] = movies_df['movie_id'].apply(pd.to_numeric)
    ratings_df = ratings_df.applymap(int)

    return ratings_list, users_list, movies_list, ratings_df, users_df, movies_df
