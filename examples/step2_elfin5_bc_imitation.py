# -*- coding: utf-8 -*-
"""
本示例演示如何用 imitation 库对 elfin5 机械臂进行行为克隆（BC）模仿学习：
1. 加载已训练好的 RL 专家策略。
2. 用专家采集高质量演示数据。
3. 用 imitation 库的 BC 算法做模仿学习。
4. 分别用 Mujoco 仿真窗口展示 RL 专家和 BC 学生的效果。
"""
import os
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from imitation.algorithms import bc
from imitation.data import rollout
import time

# elfin5 机械臂 MJCF 模型文件路径
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../xacro-to-urdf-to-mjcf-converter/mjcf_models/elfin5/elfin5.xml')

class Elfin5ReachEnv(gym.Env):
    """
    elfin5机械臂 Mujoco gym 环境，实现"末端到达目标点"任务。
    
    任务描述：
    - 观测空间：6个关节角度 + 目标点3D坐标（9维向量）
    - 动作空间：6个关节的角度增量（6维向量）
    - 奖励函数：分层奖励，距离越近奖励越高
    - 终止条件：末端距离目标点小于0.05米，或步数超限
    - 目标点可视化：用红色小球实时显示目标位置
    """
    def __init__(self, show_render=True):
        super().__init__()
        # 加载 MJCF 机械臂模型和物理数据
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        
        # 机械臂关节配置
        self.joint_names = [f"elfin_joint{i+1}" for i in range(6)]  # 6个关节名称
        self.joint_idxs = [self.model.joint(name).qposadr[0] for name in self.joint_names]  # 关节在qpos中的索引
        self.n_joints = 6  # 关节数量
        
        # 每个关节的角度范围限制（弧度）
        self.joint_range = np.array([
            [-3.14, 3.14], [-2.35, 2.35], [-2.61, 2.61],  # 关节1-3的范围
            [-3.14, 3.14], [-2.56, 2.56], [-3.14, 3.14]   # 关节4-6的范围
        ])
        
        # 末端执行器的body名称（用于获取末端位置）
        self.ee_body = "elfin_link6"
        
        # 定义观测空间：6个关节角度 + 3个目标点坐标
        self.observation_space = spaces.Box(
            low=np.concatenate([self.joint_range[:,0], [-1, -1, -1]]),  # 最小值：关节下限 + 目标点下限
            high=np.concatenate([self.joint_range[:,1], [1, 1, 1]]),    # 最大值：关节上限 + 目标点上限
            dtype=np.float32
        )
        
        # 定义动作空间：6个关节的角度增量
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(6,), dtype=np.float32  # 每步最大角度变化±0.05弧度
        )
        
        # 仿真参数
        self.max_steps = 200  # 每个episode的最大步数
        self.show_render = show_render  # 是否显示Mujoco可视化窗口
        self._viewer = None  # Mujoco窗口查看器

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态
        
        Returns:
            obs (np.ndarray): 初始观测（9维：6关节角度 + 3目标点坐标）
            info (dict): 额外信息（暂时为空）
        """
        # 重置所有关节角度为0（机械臂回到初始姿态）
        qpos = np.zeros(self.n_joints)
        mujoco.mj_resetData(self.model, self.data)  # 重置Mujoco物理数据
        for i, idx in enumerate(self.joint_idxs):
            self.data.qpos[idx] = qpos[i]
        mujoco.mj_forward(self.model, self.data)  # 前向动力学计算
        
        # 随机生成目标点（在机械臂工作空间内）
        self.target_pos = np.array([
            np.random.uniform(0.4, 0.5),  # x坐标范围
            np.random.uniform(0.4, 0.5),  # y坐标范围  
            np.random.uniform(0.4, 0.6)   # z坐标范围
        ])
        
        # 重置步数计数器
        self.step_count = 0
        
        # 同步目标小球位置到目标点（可视化）
        self._set_target_ball(self.target_pos)
        
        # 如果开启渲染，显示仿真窗口
        if self.show_render:
            self.render()
            
        return self._get_obs().astype(np.float32), {}

    def step(self, action):
        """
        执行一步动作
        
        Args:
            action (np.ndarray): 6个关节的角度增量
            
        Returns:
            obs (np.ndarray): 新的观测
            reward (float): 奖励值
            done (bool): 是否终止
            truncated (bool): 是否截断（此处恒为False）
            info (dict): 包含末端位置、目标位置、距离等信息
        """
        # 限制动作在合法范围内
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 执行动作：更新每个关节的角度
        for i, idx in enumerate(self.joint_idxs):
            self.data.qpos[idx] += action[i]  # 角度增量
            # 确保关节角度不超出物理限制
            self.data.qpos[idx] = np.clip(self.data.qpos[idx], self.joint_range[i,0], self.joint_range[i,1])
        
        # 执行物理仿真
        mujoco.mj_forward(self.model, self.data)  # 前向动力学
        mujoco.mj_step(self.model, self.data)     # 仿真一步
        
        # 更新步数
        self.step_count += 1
        
        # 获取新的观测
        obs = self._get_obs().astype(np.float32)
        
        # 计算末端执行器与目标点的距离
        ee_pos = self.data.body(self.ee_body).xpos.copy()  # 末端位置
        dist = np.linalg.norm(ee_pos - self.target_pos)    # 欧几里得距离
        
        # 分层奖励函数：距离越近奖励越高
        if dist < 0.01:
            reward = 100  # 极高精度：距离<1cm，给极高奖励
        elif dist < 0.05:
            reward = 10   # 合格精度：距离<5cm，给高奖励
        else:
            reward = -dist  # 连续引导：奖励 = 负距离，引导靠近目标
        
        # 终止条件：到达目标或超时
        done = dist < 0.05 or self.step_count >= self.max_steps
        
        # 收集信息用于调试和记录
        info = {
            'ee_pos': ee_pos,           # 末端当前位置
            'target_pos': self.target_pos,  # 目标位置
            'dist': dist                # 当前距离
        }
        
        # 同步目标小球位置（可视化更新）
        self._set_target_ball(self.target_pos)
        
        # 如果开启渲染，刷新仿真窗口
        if self.show_render:
            self.render()
            
        return obs, reward, done, False, info

    def _get_obs(self):
        """
        获取当前观测
        
        Returns:
            np.ndarray: 9维观测向量（6关节角度 + 3目标点坐标）
        """
        # 获取所有关节的当前角度
        qpos = np.array([self.data.qpos[idx] for idx in self.joint_idxs])
        
        # 拼接关节角度和目标点坐标
        return np.concatenate([qpos, self.target_pos]).astype(np.float32)

    def render(self, mode='human'):
        """
        渲染仿真环境（显示Mujoco可视化窗口）
        
        Args:
            mode (str): 渲染模式，默认'human'
        """
        if self.show_render:
            if self._viewer is None:
                # 首次调用：启动Mujoco被动查看器
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                # 后续调用：同步数据到窗口（刷新显示）
                self._viewer.sync()

    def _set_target_ball(self, pos):
        """
        将目标小球的位置设置为目标点坐标（用于可视化）
        
        Args:
            pos (np.ndarray): 目标点的三维坐标 (x, y, z)
        """
        # 获取目标小球的自由关节在qpos中的地址
        ball_qpos_addr = self.model.joint('target_ball_freejoint').qposadr
        
        # 设置小球位置（前3个自由度为xyz坐标）
        self.data.qpos[ball_qpos_addr[0]:ball_qpos_addr[0]+3] = pos
        
        # 设置小球旋转为单位四元数（无旋转，后4个自由度为四元数）
        self.data.qpos[ball_qpos_addr[0]+3:ball_qpos_addr[0]+7] = np.array([1, 0, 0, 0])

# 兼容 imitation rollout 的简单包装器
# imitation库期望gym环境的reset/step返回格式与新版gymnasium略有不同
class SimpleObsEnv(gym.Wrapper):
    """
    Gymnasium环境包装器，用于兼容imitation库
    - reset只返回obs（而非obs, info）
    - step只返回obs, reward, done, info（而非obs, reward, terminated, truncated, info）
    """
    def reset(self, **kwargs):
        """重置环境，只返回观测"""
        obs, _ = self.env.reset(**kwargs)
        return obs
        
    def step(self, action):
        """执行动作，兼容旧版gym格式"""
        result = self.env.step(action)
        if len(result) == 5:  # 新版gymnasium格式
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated  # 合并终止条件
            return obs, reward, done, info
        return result[:4]  # 旧版gym格式，直接返回前4个

def expert_rollout_one(policy, env, rng, max_steps=200):
    """
    用专家策略在环境中运行一个完整的episode，采集演示轨迹
    
    Args:
        policy: 专家策略（如训练好的PPO模型）
        env: 包装后的环境
        rng: 随机数生成器
        max_steps: 最大步数
        
    Returns:
        TrajectoryWithRew: 包含obs, acts, rewards, infos的轨迹数据
    """
    # 存储轨迹数据的列表
    obs_list, act_list, info_list, rew_list = [], [], [], []
    
    # 重置环境并开始episode
    obs = env.reset()
    obs_list.append(obs)
    done = False
    steps = 0
    
    # 运行episode直到终止或超时
    while not done and steps < max_steps:
        # 专家策略预测动作（确定性策略，不加噪声）
        action, _ = policy.predict(obs, deterministic=True)
        
        # 执行动作并获取反馈
        next_obs, reward, done, info = env.step(action)
        
        # 记录这一步的数据
        act_list.append(action)      # 动作
        info_list.append(info)       # 环境信息
        rew_list.append(reward)      # 奖励
        obs_list.append(next_obs)    # 下一状态
        
        # 更新状态和步数
        obs = next_obs
        steps += 1
    
    # 将轨迹数据转换为imitation库需要的格式
    from imitation.data.types import TrajectoryWithRew
    return TrajectoryWithRew(
        obs=np.array(obs_list),      # 观测序列 (T+1, obs_dim)
        acts=np.array(act_list),     # 动作序列 (T, act_dim)
        infos=info_list,             # 信息列表
        rews=np.array(rew_list),     # 奖励序列 (T,)
        terminal=done,               # 是否正常终止
    )

if __name__ == '__main__':
    # ==== 1. 加载已训练好的 RL 专家策略 ====
    print('加载 RL 专家策略...')
    env = Elfin5ReachEnv(show_render=True)  # 创建环境（开启可视化）
    expert = PPO.load('elfin5_ppo_expert', env=env)  # 加载专家模型

    # ==== 2. 用专家采集高质量演示数据 ====
    print('用RL专家采集演示数据...')
    # 创建用于数据采集的环境（兼容imitation库）
    rollout_env = SimpleObsEnv(Elfin5ReachEnv(show_render=True))
    
    # 采集50条专家演示轨迹
    expert_trajectories = [
        expert_rollout_one(expert, rollout_env, rng=np.random.default_rng(i), max_steps=200)
        for i in range(50)  # 采集50个episode
    ]
    
    # 将轨迹展平为transition格式（用于BC训练）
    transitions = rollout.flatten_trajectories(expert_trajectories)
    print(f'采集完成，共获得 {len(transitions)} 个transition')

    # ==== 3. 用 imitation 库 BC 算法做模仿学习 ====
    print('用BC算法做模仿学习...')
    rng = np.random.default_rng(0)  # 固定随机种子确保可重复性
    
    # 创建BC训练器
    bc_trainer = bc.BC(
        observation_space=env.observation_space,  # 观测空间
        action_space=env.action_space,            # 动作空间
        demonstrations=transitions,               # 专家演示数据
        rng=rng,                                 # 随机数生成器
        batch_size=16,                           # 批大小
    )
    
    # 开始BC训练（模仿学习）
    bc_trainer.train(n_epochs=5)  # 训练5个epoch
    print('BC训练完成')
    
    # 保存BC训练好的模型
    from imitation.util import util
    util.save_policy(bc_trainer.policy, "elfin5_bc_policy.pt")
    print('BC策略已保存到 elfin5_bc_policy.pt')

    # ==== 4. 展示 RL 专家和 BC 学生的仿真效果 ====
    
    # 4.1 展示RL专家策略效果
    print('展示RL专家策略效果...')
    env.show_render = True  # 确保开启可视化
    
    for ep in range(2):  # 展示2个episode
        obs, _ = env.reset()  # 重置环境
        done = False
        
        # 运行一个完整episode
        while not done:
            action, _ = expert.predict(obs, deterministic=True)  # 专家决策
            obs, reward, done, _, info = env.step(action)       # 执行动作
            time.sleep(0.05)  # 每步暂停50ms，便于观察
            
        # 打印这个episode的结果
        print(f'RL专家第{ep+1}次演示累计reward: {info["dist"]:.3f}')
        time.sleep(1)  # episode间暂停5秒

    # 4.2 展示BC学生策略效果
    print('展示BC学生策略效果...')
    
    for ep in range(2):  # 展示2个episode
        obs, _ = env.reset()  # 重置环境
        done = False
        
        # 运行一个完整episode
        while not done:
            # BC策略决策（注意：返回格式略有不同）
            action = bc_trainer.policy.predict(obs, deterministic=True)[0]
            obs, reward, done, _, info = env.step(action)  # 执行动作
            time.sleep(0.05)  # 每步暂停50ms，便于观察
            
        # 打印这个episode的结果
        print(f'BC学生第{ep+1}次演示累计reward: {info["dist"]:.3f}')
        time.sleep(3)  # episode间暂停10秒