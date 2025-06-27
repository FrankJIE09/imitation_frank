# -*- coding: utf-8 -*-
"""
本示例演示如何用 Mujoco + stable-baselines3 训练 elfin5 机械臂的 RL 专家（末端到达目标点任务）。
"""
import os
import time

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import mujoco.viewer
import sys

# elfin5 机械臂 MJCF 模型文件路径
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../xacro-to-urdf-to-mjcf-converter/mjcf_models/elfin5/elfin5_with_sphere.xml')

class Elfin5ReachEnv(gym.Env):
    """
    elfin5机械臂 Mujoco gym 环境，实现"末端到达目标点"任务。
    观测空间：6个关节角度 + 目标点3D坐标
    动作空间：6个关节的增量
    奖励：末端与目标点距离的负数
    终止条件：末端距离目标点小于0.05米，或步数超限
    """
    def __init__(self, show_render=True):
        super().__init__()
        # 加载 MJCF 模型和数据
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        # 关节名称和索引
        self.joint_names = [f"elfin_joint{i+1}" for i in range(6)]
        self.joint_idxs = [self.model.joint(name).qposadr[0] for name in self.joint_names]
        self.n_joints = 6
        # 关节角度范围
        self.joint_range = np.array([
            [-3.14, 3.14], [-2.35, 2.35], [-2.61, 2.61],
            [-3.14, 3.14], [-2.56, 2.56], [-3.14, 3.14]
        ])
        # 末端 body 名称
        self.ee_body = "elfin_link6"
        # 观测空间：6个关节角度 + 目标点3D坐标
        self.observation_space = spaces.Box(
            low=np.concatenate([self.joint_range[:,0], [-1, -1, -1]]),
            high=np.concatenate([self.joint_range[:,1], [1, 1, 1]]),
            dtype=np.float32
        )
        # 动作空间：6个关节的增量
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(6,), dtype=np.float32
        )
        self.max_steps = 200  # 每个 episode 最大步数
        self.show_render = show_render  # 是否显示 mujoco 可视化窗口
        self._viewer = None

    def reset(self, seed=None, options=None):
        # 重置关节角度为0
        qpos = np.zeros(self.n_joints)
        mujoco.mj_resetData(self.model, self.data)
        for i, idx in enumerate(self.joint_idxs):
            self.data.qpos[idx] = qpos[i]
        mujoco.mj_forward(self.model, self.data)
        # 随机生成目标点（在工作空间内）
        self.target_pos = np.array([
            np.random.uniform(0.4, 0.5),
            np.random.uniform(0.4, 0.5),
            np.random.uniform(0.4, 0.6)
        ])
        self.step_count = 0
        self._set_target_ball(self.target_pos)  # 同步小球到目标点
        if self.show_render:
            self.render()
        return self._get_obs().astype(np.float32), {}

    def step(self, action):
        # 执行动作（关节增量），并裁剪到合法范围
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for i, idx in enumerate(self.joint_idxs):
            self.data.qpos[idx] += action[i]
            self.data.qpos[idx] = np.clip(self.data.qpos[idx], self.joint_range[i,0], self.joint_range[i,1])
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        obs = self._get_obs().astype(np.float32)
        # 计算末端与目标点距离
        ee_pos = self.data.body(self.ee_body).xpos.copy()
        dist = np.linalg.norm(ee_pos - self.target_pos)
        if dist < 0.01:
            reward = 100  # 极高奖励
        elif dist < 0.05:
            reward = 10   # 高奖励  
        else:
            reward = -dist  # 连续奖励
        done = dist < 0.05 or self.step_count >= self.max_steps
        info = {'ee_pos': ee_pos, 'target_pos': self.target_pos, 'dist': dist}
        self._set_target_ball(self.target_pos)  # 同步小球到目标点
        if self.show_render:
            self.render()
        return obs, reward, done, False, info

    def _get_obs(self):
        # 返回观测：6个关节角度 + 目标点坐标
        qpos = np.array([self.data.qpos[idx] for idx in self.joint_idxs])
        return np.concatenate([qpos, self.target_pos]).astype(np.float32)

    def render(self, mode='human'):
        # 显示 mujoco 可视化窗口，并在每次调用时刷新内容
        if self.show_render:
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self._viewer.sync()  # 每次调用都刷新窗口

    def _set_target_ball(self, pos):
        """
        将目标小球的位置设置为目标点坐标
        :param pos: 目标点的三维坐标 (x, y, z)
        """
        # 获取小球自由关节的qpos索引（前3个为位置，后4个为四元数旋转）
        ball_qpos_addr = self.model.joint('target_ball_freejoint').qposadr
        self.data.qpos[ball_qpos_addr[0]:ball_qpos_addr[0]+3] = pos  # 设置位置
        self.data.qpos[ball_qpos_addr[0]+3:ball_qpos_addr[0]+7] = np.array([1, 0, 0, 0])  # 单位四元数，无旋转

if __name__ == '__main__':
    # 判断是否只做展示
    if len(sys.argv) > 1 and sys.argv[1] == 'show':
        # 只做展示，不训练
        print('加载已训练模型，Mujoco窗口展示强化学习效果...')
        env = Elfin5ReachEnv(show_render=True)
        model = PPO.load('elfin5_ppo_expert', env=env)
        for i in range(3):  # 连续演示3次
            obs, _ = env.reset()
            total_reward = 0
            for _ in range(100):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                if done:
                    break
            print(f'第{i+1}次演示累计reward: {total_reward:.3f}，末端距离目标点: {info["dist"]:.3f}')
            time.sleep(10)
    else:
        # 训练+周期性展示
        print('用PPO强化学习训练专家...')
        env = Elfin5ReachEnv(show_render=False)  # 关闭渲染加速训练
        check_env(env)  # 检查环境接口合法性
        if os.path.exists('elfin5_ppo_expert.zip'):
            print('检测到已有模型，加载并继续训练...')
            ppo_model = PPO.load('elfin5_ppo_expert', env=env)
        else:
            print('无现有模型，重新训练...')
            ppo_model = PPO('MlpPolicy', env, verbose=1)
        # 训练时每1000步展示一次当前策略效果
        total_steps = 100000
        eval_interval = 1000
        demo_env = Elfin5ReachEnv(show_render=True)  # 只创建一次可视化环境
        for step in range(0, total_steps, eval_interval):
            # 训练eval_interval步
            ppo_model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
            # 展示当前策略效果
            print(f'第{step+eval_interval}步，展示当前策略效果...')
            obs, _ = demo_env.reset()
            total_reward = 0
            for _ in range(100):
                action, _ = ppo_model.predict(obs, deterministic=True)
                obs, reward, done, _, info = demo_env.step(action)
                total_reward += reward
                if done:
                    break
            print(f'当前策略累计reward: {total_reward:.3f}，末端距离目标点: {info["dist"]:.3f}')
        # 训练完成后保存模型
        ppo_model.save('elfin5_ppo_expert')  # 保存专家策略
        print('RL专家训练完成并已保存为 elfin5_ppo_expert') 