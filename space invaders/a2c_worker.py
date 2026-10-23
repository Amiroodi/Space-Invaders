#a2c_worker
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random

from collections import deque

import ale_py
gym.register_envs(ale_py)

# Set the device
cpu_device = torch.device("cpu")

class A2CNetwork(nn.Module):
    def __init__(self, input_channels, action_space):
        super(A2CNetwork, self).__init__()
        
        # Define the four convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)  # New conv layer
        
        # Calculate the input size to the fully connected layer after conv4
        # Assuming the input image size is 84x84 as in many RL environments
        self.fc1 = nn.Linear(128 * 5 * 5, 512)  # Update the input size here
        
        # Actor and Critic heads
        self.actor = nn.Linear(512, action_space)
        self.critic = nn.Linear(512, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  # Pass through the new convolutional layer
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

    def select_action(self, x):
        logits, _ = self.forward(x)
        action_probs = F.softmax(logits, dim=1)
        action_probs = torch.clamp(action_probs, 1e-6, 1.0)  # Prevent sampling errors
        action = torch.multinomial(action_probs, 1).item()
        return action

class A2CWorker:
    def __init__(self, global_network_state, input_dim, action_space, worker_id, exploration_steps, epsilon):
        self.local_network = A2CNetwork(input_dim[0], action_space).to(cpu_device)
        self.local_network.load_state_dict(global_network_state)
        self.worker_id = worker_id
        self.episode_experiences = []
        self.exploration_steps = exploration_steps  
        self.current_global_step = 0  
        self.env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")
        self.epsilon = epsilon
        self.reward_track = []
        ##self.env.seed(worker_id)

    def preprocess_frame(self, frame, frame_size=(84, 84)):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
        return frame / 255.0

    def stack_frames(self, stacked_frames, state, is_new_episode):
        frame = self.preprocess_frame(state)
        if is_new_episode:
            stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for i in range(4)], maxlen=4)
            for _ in range(4):
                stacked_frames.append(frame)
        else:
            stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=0)
        return stacked_state, stacked_frames

    def select_action(self, state, epsilon=0.1):
        if self.current_global_step < self.exploration_steps or random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(cpu_device)
            action = self.local_network.select_action(state_tensor)
            return action

    def run_step(self, update_frequency):
        total_steps = 0
        state, _ = self.env.reset()
        stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for i in range(4)], maxlen=4)  # Use float32 for consistency
        stacked_state, _ = self.stack_frames(stacked_frames, state, True)

        while total_steps < update_frequency:
            done = False
            episode_reward = 0
            while not done:
                action = self.select_action(stacked_state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_stacked_state, stacked_frames = self.stack_frames(stacked_frames, next_state, False)

                # Ensure final experience is stored even if done is True
                self.episode_experiences.append((stacked_state, action, reward, next_stacked_state, done))

                stacked_state = next_stacked_state
                total_steps += 1
                episode_reward += reward
                self.current_global_step += 1

                if done:
                    # Ensure the final experience is stored before resetting the environment
                    #print(f"Worker {self.worker_id}, Reward: {episode_reward}")
                    self.reward_track.append(episode_reward)
                    break

            # Reset environment after the episode finishes
            state, _ = self.env.reset()
            stacked_state, stacked_frames = self.stack_frames(stacked_frames, state, True)


    def get_experience(self):
        if len(self.episode_experiences) > 0:
            states, actions, rewards, next_states, dones = zip(*self.episode_experiences)
            return states, actions, rewards, next_states, dones, self.reward_track
        return None