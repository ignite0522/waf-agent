from gym.envs.registration import register
import gym

# 将自定义环境添加到gym的注册表中，以便可以使用make创建
register(
    id='Waf-v0',    # 环境名字(要这样命名，否则“Currently all IDs must be of the form ^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$.“)
    entry_point='waf_agent.envs.env2:Env',    # 环境进入点，即环境对应的类
)


