# imitation/examples 目录说明

本目录包含基于 Mujoco、imitation、stable-baselines3 等库的模仿学习与强化学习示例脚本，适用于机械臂等机器人任务。

## 主要示例脚本

### 1. quickstart.py
- imitation 官方最简快速上手示例。
- 演示如何用 CartPole 环境做行为克隆（BC）模仿学习。
- 适合初学者快速体验 imitation 库。

### 2. elfin5_rl_expert.py
- 用 PPO 强化学习算法训练 elfin5 机械臂完成"末端到达目标点"任务。
- 训练完成后会保存专家策略模型（elfin5_ppo_expert）。
- 依赖：Mujoco、stable-baselines3。

### 3. elfin5_bc_imitation.py
- 加载已训练好的 RL 专家（elfin5_ppo_expert），采集专家演示数据。
- 用 imitation 库的 BC 算法进行模仿学习。
- 仿真窗口中可直观展示 RL 专家和 BC 学生的效果，并实体化目标点。
- 依赖：Mujoco、imitation、stable-baselines3。

### 4. 其它脚本
- elfin5_reach_demo.py：elfin5 机械臂"到达目标点"任务的最小 demo，适合自定义环境开发参考。

## 运行环境依赖
- Python 3.8/3.9
- mujoco >= 2.2.0
- gymnasium 或 gym
- stable-baselines3
- imitation
- numpy

建议先激活 conda 虚拟环境，并确保已正确安装上述依赖。

## 运行方法示例
```bash
cd imitation_examples
# 训练 RL 专家
python elfin5_rl_expert.py

# 用 RL 专家采集演示并做 BC 模仿学习
python elfin5_bc_imitation.py

# 运行 imitation 官方 quickstart
python quickstart.py
```

## 备注
- elfin5 相关脚本需确保 MJCF 模型路径正确，且有对应 STL/mesh 文件。
- 如需自定义任务、reward、专家策略等，可参考各脚本详细中文注释进行扩展。 