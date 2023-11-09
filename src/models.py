import tensorflow as tf


# DNN for our DQN/DDQN agent

class DNN(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(DNN, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(name='input_layer', input_shape=(3*embedding_dim,))
        self.flatten_layer = tf.keras.layers.Flatten()
        self.fc_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(output_dim, activation='linear')
        ])
        
    def call(self, x):
        x = self.input_layer(x)
        x = self.flatten_layer(x)
        return self.fc_layers(x)
    

# Embedding layers for representing user_movie iteraction
class UserMovieEmbedding(tf.keras.Model):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(UserMovieEmbedding, self).__init__()
        self.m_u_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        # embedding
        self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users, output_dim=embedding_dim)
        self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_dim)
       
        # dot product
        self.m_u_merge = tf.keras.layers.Dot(name='movie_user_dot', normalize=False, axes=1)
        # output
        self.m_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, x):
        x = self.m_u_input(x)
        uemb = self.u_embedding(x[0])
        memb = self.m_embedding(x[1])
        m_u = self.m_u_merge([memb, uemb])
        return self.m_u_fc(m_u)
    


class StateRepresentation(tf.keras.Model):
    """
    This class is used to represent our state space. i.e user, user*items, items interactions
    """
    def __init__(self, embedding_dim):
        super(StateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.wav = tf.keras.layers.Conv1D(1, 1, 1)
        self.concat = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()
        
    def call(self, x):
        items_eb = tf.transpose(x[1], perm=(0,2,1))/self.embedding_dim
        wav = self.wav(items_eb)
        wav = tf.transpose(wav, perm=(0,2,1))
        wav = tf.squeeze(wav, axis=1)
        user_wav = tf.keras.layers.multiply([x[0], wav])
        concat = self.concat([x[0], user_wav, wav])
        return self.flatten(concat)
