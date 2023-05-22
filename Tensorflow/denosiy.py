import tensorflow as tf
import numpy as np
import sklearn

import gym

# reinforcement learning to replace the PID controller

env = gym.make('CartPole-v0', render_mode='rgb_array')

# input is the state of the cartpole
# which the shape is (4,)
# output is the action of the cartpole
# which the shape is (2,)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# train the model with reinforcement learning
# the input is the state of the cartpole
# the output is the action of the cartpole
# the reward is the score of the cartpole
# the loss is the score of the cartpole
# the accuracy is the score of the cartpole
EPISODES = 1000
for episode in range(EPISODES):
    state = env.reset()
    done = False
    while not done:
        action = model.predict(state.reshape(1, -1))
        state, reward, done, info, _ = env.step(np.argmax(action))
        model.fit(state.reshape(1, -1), np.argmax(action), verbose=1)
        env.render()
    print(f'Episode {episode} finished after {info["time"]} timesteps')
env.close()
