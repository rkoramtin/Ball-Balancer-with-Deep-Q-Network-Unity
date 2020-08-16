# Ball-Balancer-with-Deep-Q-Network-Unity
In this project we train a Ball with Q-learning To stay on the platform.

# Reinforcement learning:
![Alt Text](http://uupload.ir/files/ra8_1.png)   
In Reinforcement learning agents learn to perform actions in an environment in order to to maximize a reward.
The key difference between reinforcement learning from supervised or unsupervised learning is presence of two things: <br>
   An environment <br>
   An agent <br>
# Q-Learning:
Q-learning is a reinforcement learning algorithm that seeks to find the best action to take given the current state.
Q-Learning is based on a Q-function. <br>
![Alt Text](http://uupload.ir/files/erw1_2.png) <br>
Which means that the maximum return from state "s" and action "a" is the sum of the immediate reward r and the maximum reward from the next state " s' " .
# Deep Q-Learning:
Deep Q-learning makes use of neural networks and The Deep Q-Network algorithm was developed by DeepMind in 2015. It actually enhance Q-Learning which is a classic Reinforcement learning algorithm, with deep neural networks and a technique called experience replay.
# Experience Replay:
At each time step of data collection, the transitions are added to a circular buffer called the replay buffer. Then during training, instead of using just the latest transition to compute the loss and its gradient, we compute them using a mini-batch of transitions sampled from the replay buffer.
This is called Experience Replay which makes the network updates more stable and has the following benefits: <br>
    A better data efficiency by by make use of each transition in many updates. <br>
    A better stability using uncorrelated transitions in a batch. <br>
# Our Network:
![Alt Text](http://uupload.ir/files/ed9m_3.png) <br>
For input we use Platform X Rotation, Ball Z Position, and Ball's X Velocity. <br>
The outputs are Quality Values of how quality to the left and the right of the platform is. <br>
# Training Process: <br>
![Alt Text](http://uupload.ir/files/wg5x_ezgif.com-video-to-gif.gif) <br>

