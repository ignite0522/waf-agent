import gym
import numpy as np
from gym import spaces
import random
from sklearn.model_selection import train_test_split
from envs.xss_manipulator import Xss_Manipulator
from envs.features import Features
from envs.waf import Waf_Check
# from envs.WAF_PLUS import Waf_Check


samples_file = "/Users/guyuwei/security_ai/大佬项目/ItBaizhan/代码/waf_agent/envs/xss-samples-all.txt"
samples = []
with open(samples_file) as f:
    for line in f:
        line = line.strip('\n')
        samples.append(line)

samples_train, samples_test = train_test_split(samples, test_size=0.4)

ACTION_LOOKUP = {i:act for i,act in enumerate(Xss_Manipulator.ACTION_TABLE.keys())}
'''
i为原动作字典的下标0123，act为原动作字典的key即免杀操作名

ACTION_LOOKUP字典的作用是将动作的下标映射到动作名，方便后续使用
    ACTION_LOOKUP = {
    0: 'charTo16',
    1: 'charTo10',
    2: 'addComment',
    3: 'addTab',
    4: 'addZero',
    5: 'addEnter'
    }
'''

class Env(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))
        self.current_sample = ""
        self.features = Features()
        self.waf_check = Waf_Check()
        self.xss_manipulator = Xss_Manipulator()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(257,), dtype=np.float32) # np.inf是无穷的意思，构造观测空间

    def reset(self, seed=None, options=None):
        self.current_sample = random.choice(samples_train)
        observation = self.features.extract(self.current_sample)
        # print(f'observation初始样本为：{observation.shape}')
        return observation

    def step(self, action):
        r = 0
        done = False  # 默认本轮学习未结束
        truncated = False  # 表示回合是否被截断

        _action = ACTION_LOOKUP[action]
        #调用xss_manipulator的modify方法，对当前样本进行免杀操作
        modified_sample = self.xss_manipulator.modify(self.current_sample, _action)

        #调用waf_check的check方法，检测当前样本是否存在waf漏洞
        if not self.waf_check.check_xss(modified_sample):
            r = 10  # 免杀成功，奖励10分
            print(repr(f'免杀成功！,原样本:{self.current_sample},免杀样本:{modified_sample}'))


        # observation = self.features.extract(self.current_sample)
        observation = self.features.extract(modified_sample)
        return observation, r, done, truncated, {}

    def render(self, mode='human', close=False):
        """
        渲染环境（可选实现）。
        """
        return

if __name__ == '__main__':
    env = Env()
    t = env.action_space
    s = env.observation_space
    print(s.shape[0])
    print(t.n)
    print(env.reset())