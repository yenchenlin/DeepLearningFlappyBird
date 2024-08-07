import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Lambda
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird'  # the name of the game being played for log files
ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100000  # timesteps to observe before training
EXPLORE = 2000000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1

def build_model(input_shape, action_space):
    inputs = layers.Input(shape=input_shape)
    layer = layers.Conv2D(32, (8, 8), strides=(4, 4), padding='same', activation='relu')(inputs)
    layer = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(layer)
    layer = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(layer)
    layer = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(layer)
    layer = layers.Flatten()(layer)
    layer = layers.Dense(512, activation='relu')(layer)
    
    value_fc = layers.Dense(1)(layer)
    advantage_fc = layers.Dense(action_space)(layer)
    
    def dueling_dqn(value, advantage):
        mean_advantage = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage)
        return value + (advantage - mean_advantage)
    
    policy = dueling_dqn(value_fc, advantage_fc)
    
    model = models.Model(inputs=inputs, outputs=policy)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-6), loss='mse')
    return model

def preprocess(image):
    image = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return image

def trainNetwork(model, target_model):
    game_state = game.GameState()

    D = deque()

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = preprocess(x_t)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    s_t = np.expand_dims(s_t, axis=0)

    checkpoint_path = "saved_networks/" + GAME + "-dqn"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print("Successfully loaded:", checkpoint_manager.latest_checkpoint)
    else:
        print("Could not find old network weights")

    epsilon = INITIAL_EPSILON
    t = 0
    while True:
        readout_t = model.predict(s_t)[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # do nothing

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = preprocess(x_t1_colored)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(s_t[:, :, :, 1:], np.expand_dims(x_t1, axis=0), axis=3)

        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE:
            minibatch = random.sample(D, BATCH)

            s_j_batch = np.array([d[0] for d in minibatch])
            a_batch = np.array([d[1] for d in minibatch])
            r_batch = np.array([d[2] for d in minibatch])
            s_j1_batch = np.array([d[3] for d in minibatch])

            y_batch = []
            readout_j1_batch = model.predict(s_j1_batch)
            readout_j1_target_batch = target_model.predict(s_j1_batch)
            for i in range(len(minibatch)):
                terminal = minibatch[i][4]
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    max_action = np.argmax(readout_j1_batch[i])
                    y_batch.append(r_batch[i] + GAMMA * readout_j1_target_batch[i][max_action])

            y_batch = np.array(y_batch)
            a_batch = np.array(a_batch)
            target_f = model.predict(s_j_batch)
            for i in range(len(minibatch)):
                target_f[i][np.argmax(a_batch[i])] = y_batch[i]

            model.fit(s_j_batch, target_f, epochs=1, verbose=0)

        s_t = s_t1
        t += 1

        if t % 10000 == 0:
            checkpoint_manager.save()
            target_model.set_weights(model.get_weights())

        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print(f"TIMESTEP {t} / STATE {state} / EPSILON {epsilon} / ACTION {action_index} / REWARD {r_t} / Q_MAX {np.max(readout_t)}")

def playGame():
    input_shape = (80, 80, 4)
    action_space = ACTIONS
    model = build_model(input_shape, action_space)
    target_model = build_model(input_shape, action_space)
    target_model.set_weights(model.get_weights())
    trainNetwork(model, target_model)

if __name__ == "__main__":
    playGame()
