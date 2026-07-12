# Using Deep Q-Network to Learn How To Play Flappy Bird

<img src="./images/flappy_bird_demp.gif" width="250">

7 mins version: [DQN for flappy bird](https://www.youtube.com/watch?v=THhUXIhjkCM)

## Overview
This project follows the description of the Deep Q Learning algorithm described in Playing Atari with Deep Reinforcement Learning [2] and shows that this learning algorithm can be further generalized to the notorious Flappy Bird.

## Installation Dependencies:
* Python 2.7 or 3
* TensorFlow 0.7
* pygame
* OpenCV-Python

## How to Run?
```
git clone https://github.com/yenchenlin1994/DeepLearningFlappyBird.git
cd DeepLearningFlappyBird
python deep_q_network.py
```

## What is Deep Q-Network?
It is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards.

For those who are interested in deep reinforcement learning, I highly recommend to read the following post:

[Demystifying Deep Reinforcement Learning](http://www.nervanasys.com/demystifying-deep-reinforcement-learning/)

## Deep Q-Network Algorithm

The pseudo-code for the Deep Q Learning algorithm, as given in [1], can be found below:

```
Initialize replay memory D to size N
Initialize action-value function Q with random weights
for episode = 1, M do
    Initialize state s_1
    for t = 1, T do
        With probability ϵ select random action a_t
        otherwise select a_t=max_a  Q(s_t,a; θ_i)
        Execute action a_t in emulator and observe r_t and s_(t+1)
        Store transition (s_t,a_t,r_t,s_(t+1)) in D
        Sample a minibatch of transitions (s_j,a_j,r_j,s_(j+1)) from D
        Set y_j:=
            r_j for terminal s_(j+1)
            r_j+γ*max_(a^' )  Q(s_(j+1),a'; θ_i) for non-terminal s_(j+1)
        Perform a gradient step on (y_j-Q(s_j,a_j; θ_i))^2 with respect to θ
    end for
end for
```

## Experiments

#### Environment
Since deep Q-network is trained on the raw pixel values observed from the game screen at each time step, [3] finds that remove the background appeared in the original game can make it converge faster. This process can be visualized as the following figure:

<img src="./images/preprocess.png" width="450">

#### Network Architecture
According to [1], I first preprocessed the game screens with following steps:

1. Convert image to grayscale
2. Resize image to 80x80
3. Stack last 4 frames to produce an 80x80x4 input array for network

The architecture of the network is shown in the figure below. The first layer convolves the input image with an 8x8x4x32 kernel at a stride size of 4. The output is then put through a 2x2 max pooling layer. The second layer convolves with a 4x4x32x64 kernel at a stride of 2. We then max pool again. The third layer convolves with a 3x3x64x64 kernel at a stride of 1. We then max pool one more time. The last hidden layer consists of 256 fully connected ReLU nodes.

<img src="./images/network.png">

The final output layer has the same dimensionality as the number of valid actions which can be performed in the game, where the 0th index always corresponds to doing nothing. The values at this output layer represent the Q function given the input state for each valid action. At each time step, the network performs whichever action corresponds to the highest Q value using a ϵ greedy policy.


#### Training
At first, I initialize all weight matrices randomly using a normal distribution with a standard deviation of 0.01, then set the replay memory with a max size of 500,00 experiences.

I start training by choosing actions uniformly at random for the first 10,000 time steps, without updating the network weights. This allows the system to populate the replay memory before training begins.

Note that unlike [1], which initialize ϵ = 1, I linearly anneal ϵ from 0.1 to 0.0001 over the course of the next 3000,000 frames. The reason why I set it this way is that agent can choose an action every 0.03s (FPS=30) in our game, high ϵ will make it **flap** too much and thus keeps itself at the top of the game screen and finally bump the pipe in a clumsy way. This condition will make Q function converge relatively slow since it only start to look other conditions when ϵ is low.
However, in other games, initialize ϵ to 1 is more reasonable.

During training time, at each time step, the network samples minibatches of size 32 from the replay memory to train on, and performs a gradient step on the loss function described above using the Adam optimization algorithm with a learning rate of 0.000001. After annealing finishes, the network continues to train indefinitely, with ϵ fixed at 0.001.

## FAQ

#### Checkpoint not found
Change [first line of `saved_networks/checkpoint`](https://github.com/yenchenlin1994/DeepLearningFlappyBird/blob/master/saved_networks/checkpoint#L1) to

`model_checkpoint_path: "saved_networks/bird-dqn-2920000"`

#### How to reproduce?
1. Comment out [these lines](https://github.com/yenchenlin1994/DeepLearningFlappyBird/blob/master/deep_q_network.py#L108-L112)

2. Modify `deep_q_network.py`'s parameter as follow:
```python
OBSERVE = 10000
EXPLORE = 3000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
```

## References

[1] Mnih Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. **Human-level Control through Deep Reinforcement Learning**. Nature, 529-33, 2015.

[2] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. **Playing Atari with Deep Reinforcement Learning**. NIPS, Deep Learning workshop

[3] Kevin Chen. **Deep Reinforcement Learning for Flappy Bird** [Report](http://cs229.stanford.edu/proj2015/362_report.pdf) | [Youtube result](https://youtu.be/9WKBzTUsPKc)

## Disclaimer
This work is highly based on the following repos:

1. [sourabhv/FlapPyBird] (https://github.com/sourabhv/FlapPyBird)
2. [asrivat1/DeepLearningVideoGames](https://github.com/asrivat1/DeepLearningVideoGames)

