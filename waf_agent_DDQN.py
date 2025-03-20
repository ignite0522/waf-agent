import tensorflow as tf
import gym
from envs.env import Env
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers.legacy import Adam


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=0.9, epsilon_decay=0.995,
                 epsilon_min=0.01, update_target_freq=10):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.update_target_freq = update_target_freq  # 目标网络更新频率
        self.model = self.build_model()  # 在线网络（Q 网络）
        self.target_model = self.build_model()  # 目标网络
        self.target_model.set_weights(self.model.get_weights())  # 初始化目标网络权重
        self.train_step = 0

    def build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # 选择 Q 值最大的动作

    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)  # 随机选取 batch_size 个样本
        for idx in minibatch:
            state, action, reward, next_state, done = self.memory[idx]

            # Double DQN：用在线网络选择动作
            next_action = np.argmax(self.model.predict(next_state, verbose=0)[0])

            # 用目标网络计算 Q 值
            target_q_value = self.target_model.predict(next_state, verbose=0)[0][next_action]

            # 计算目标 Q 值
            target = reward if done else reward + self.gamma * target_q_value

            # 计算新 Q 值
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            # 训练模型
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # **减少探索率**
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # **定期更新目标网络**
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
            print(f"更新目标网络（Step: {self.train_step}）")


def train_dqn(env, agent, episodes=100, batch_size=32):
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        total_reward = 0
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)  # next_state 是 observation
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        print(f"Episode: {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)


if __name__ == '__main__':
    env = Env()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    train_dqn(env, agent, episodes=100, batch_size=32)
    agent.model.save("dqn_model_DDQN.h5")
