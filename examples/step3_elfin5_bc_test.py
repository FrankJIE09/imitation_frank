# -*- coding: utf-8 -*-
"""
本脚本用于加载已保存的BC模仿学习模型并测试其效果。
可以单独运行，无需重新训练。
"""
import os
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from imitation.algorithms import bc
import time

# elfin5 机械臂 MJCF 模型文件路径
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../xacro-to-urdf-to-mjcf-converter/mjcf_models/elfin5/elfin5_with_sphere.xml')

class Elfin5ReachEnv(gym.Env):
    """
    elfin5机械臂 Mujoco gym 环境（与训练时完全相同）
    """
    def __init__(self, show_render=True):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        
        self.joint_names = [f"elfin_joint{i+1}" for i in range(6)]
        self.joint_idxs = [self.model.joint(name).qposadr[0] for name in self.joint_names]
        self.n_joints = 6
        
        self.joint_range = np.array([
            [-3.14, 3.14], [-2.35, 2.35], [-2.61, 2.61],
            [-3.14, 3.14], [-2.56, 2.56], [-3.14, 3.14]
        ])
        
        self.ee_body = "elfin_link6"
        
        self.observation_space = spaces.Box(
            low=np.concatenate([self.joint_range[:,0], [-1, -1, -1]]),
            high=np.concatenate([self.joint_range[:,1], [1, 1, 1]]),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(6,), dtype=np.float32
        )
        
        self.max_steps = 200
        self.show_render = show_render
        self._viewer = None

    def reset(self, seed=None, options=None):
        qpos = np.zeros(self.n_joints)
        mujoco.mj_resetData(self.model, self.data)
        for i, idx in enumerate(self.joint_idxs):
            self.data.qpos[idx] = qpos[i]
        mujoco.mj_forward(self.model, self.data)
        
        self.target_pos = np.array([
            np.random.uniform(0.4, 0.5),
            np.random.uniform(0.4, 0.5),
            np.random.uniform(0.4, 0.6)
        ])
        
        self.step_count = 0
        self._set_target_ball(self.target_pos)
        
        if self.show_render:
            self.render()
            
        return self._get_obs().astype(np.float32), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        for i, idx in enumerate(self.joint_idxs):
            self.data.qpos[idx] += action[i]
            self.data.qpos[idx] = np.clip(self.data.qpos[idx], self.joint_range[i,0], self.joint_range[i,1])
        
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1
        obs = self._get_obs().astype(np.float32)
        
        ee_pos = self.data.body(self.ee_body).xpos.copy()
        dist = np.linalg.norm(ee_pos - self.target_pos)
        
        if dist < 0.01:
            reward = 100
        elif dist < 0.05:
            reward = 10
        else:
            reward = -dist
        
        done = dist < 0.05 or self.step_count >= self.max_steps
        
        info = {
            'ee_pos': ee_pos,
            'target_pos': self.target_pos,
            'dist': dist
        }
        
        self._set_target_ball(self.target_pos)
        
        if self.show_render:
            self.render()
            
        return obs, reward, done, False, info

    def _get_obs(self):
        qpos = np.array([self.data.qpos[idx] for idx in self.joint_idxs])
        return np.concatenate([qpos, self.target_pos]).astype(np.float32)

    def render(self, mode='human'):
        if self.show_render:
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self._viewer.sync()

    def _set_target_ball(self, pos):
        ball_qpos_addr = self.model.joint('target_ball_freejoint').qposadr
        self.data.qpos[ball_qpos_addr[0]:ball_qpos_addr[0]+3] = pos
        self.data.qpos[ball_qpos_addr[0]+3:ball_qpos_addr[0]+7] = np.array([1, 0, 0, 0])


if __name__ == '__main__':
    # 检查保存的BC模型是否存在
    bc_policy_path = "elfin5_bc_policy.pt"
    if not os.path.exists(bc_policy_path):
        print(f'错误：未找到保存的BC模型 {bc_policy_path}')
        print('请先运行 elfin5_bc_imitation.py 训练并保存模型')
        exit(1)
    
    # 创建环境
    print('创建测试环境...')
    env = Elfin5ReachEnv(show_render=True)
    
    # 加载保存的BC模型
    print(f'加载BC模型: {bc_policy_path}')
    bc_policy = bc.reconstruct_policy(bc_policy_path)
    
    # 测试BC模型效果
    print('开始测试BC模型效果...')
    
    total_episodes = 5
    success_count = 0
    total_reward = 0
    
    for ep in range(total_episodes):
        print(f'\n=== 第 {ep+1}/{total_episodes} 次测试 ===')
        
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # BC策略预测动作
            action = bc_policy.predict(obs, deterministic=True)[0]
            obs, reward, done, _, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            time.sleep(0.05)  # 便于观察
        
        # 判断是否成功到达目标
        final_dist = info['dist']
        success = final_dist < 0.05
        if success:
            success_count += 1
        
        total_reward += episode_reward
        
        print(f'步数: {steps}, 最终距离: {final_dist:.4f}m, 成功: {"是" if success else "否"}')
        print(f'Episode奖励: {episode_reward:.2f}')
        
        time.sleep(2)  # episode间暂停
    
    # 统计结果
    print(f'\n=== 测试结果统计 ===')
    print(f'总episode数: {total_episodes}')
    print(f'成功次数: {success_count}')
    print(f'成功率: {success_count/total_episodes*100:.1f}%')
    print(f'平均奖励: {total_reward/total_episodes:.2f}')
    
    print('\n测试完成！') 