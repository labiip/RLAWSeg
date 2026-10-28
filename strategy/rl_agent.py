"""DQN智能体实现"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state))
        next_state = torch.FloatTensor(np.array(next_state))
        action = torch.LongTensor(np.array(action)).unsqueeze(1)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1)
        done = torch.FloatTensor(np.array(done)).unsqueeze(1)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """深度Q网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.net(state)


class DQNAgent:
    """DQN智能体"""

    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-4,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.1,
                 epsilon_decay=500, buffer_size=10000, batch_size=64,
                 target_update=50, device='cuda'):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device
        self.steps = 0

        # 网络
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # 经验回放
        self.memory = ReplayBuffer(buffer_size)

        # 初始化目标网络
        self.update_target()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state, valid_actions=None):
        """ε-greedy策略选择单个动作"""
        self.steps += 1
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.steps / self.epsilon_decay) *
            (self.epsilon_start - self.epsilon_end)
        )

        if valid_actions is None:
            valid_actions = self.action_dim

        if random.random() < self.epsilon:
            return random.randint(0, valid_actions - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                if valid_actions < self.action_dim:
                    q_values[0, valid_actions:] = -float('inf')
                return q_values.argmax(dim=1).item()

    def select_topk_actions(self, state, k, valid_actions=None):
        """选择Top-K个动作（用于批量采样）"""
        if valid_actions is None:
            valid_actions = self.action_dim

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze()
            if valid_actions < self.action_dim:
                q_values[valid_actions:] = -float('inf')
            topk_values, topk_indices = torch.topk(q_values, min(k, valid_actions))

        return topk_indices.cpu().numpy().tolist(), topk_values.mean().item()

    def learn(self):
        """从经验回放中学习"""
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 当前Q值
        current_q = self.q_network(states).gather(1, actions)

        # 目标Q值
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # 损失
        loss = nn.MSELoss()(current_q, target_q)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # 更新目标网络
        if self.steps % self.target_update == 0:
            self.update_target()

        return loss.item()

    def save(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']
