import os
import time
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import safe_mean
import gymnasium as gym
from gymnasium import spaces
from rlplay_reward_calculator import RlplayRewardCalculator

class MLPlayArgsSaver:
    def __init__(self):
        self.name = None
        self.init_kwargs = None
        self.observations = None
        self.keyboard = None

mlplayArgs = MLPlayArgsSaver()

rlplayRewardCalculator = RlplayRewardCalculator()

class ObservationProcessor:
    def __init__(self, observation_structure):
        self.observation_structure = observation_structure
        self.observation_size = self._calculate_observation_size(observation_structure)
        print(f"Observation size calculated: {self.observation_size}")

    def get_size(self):
        return self.observation_size

    def _calculate_observation_size(self, observation_structure):
        total_size = 0

        for item in observation_structure:
            item_type = item.get("type", "")
            item_key = item.get("key", "")

            if item_key == "flattened":
                vector_size = item.get("vector_size", 0)
                return vector_size

            if item_type == "Vector3":
                total_size += 3
            elif item_type == "Vector2":
                total_size += 2
            elif item_type == "float" or item_type == "int" or item_type == "bool":
                total_size += 1
            elif item_type == "Grid":
                grid_size = item.get("grid_size", 0)
                sub_items = item.get("items", [])
                sub_item_size = self._calculate_observation_size(sub_items)
                total_size += sub_item_size * grid_size * grid_size
            elif item_type == "List":
                sub_items = item.get("items", [])
                sub_item_size = self._calculate_observation_size(sub_items)
                sub_item_count = item.get("item_count", 0)

                if sub_item_count > 0:
                    total_size += sub_item_size * sub_item_count
                else:
                    total_size += sub_item_size

        return total_size

class ActionProcessor:
    def __init__(self, action_space_info):
        self.action_space_info = action_space_info

        if action_space_info.is_continuous():
            self.action_type = "continuous"
            self.action_size = action_space_info.continuous_size
        elif action_space_info.is_discrete():
            self.action_type = "discrete"
            self.action_size = sum(action_space_info.discrete_branches)
            self.discrete_branches = action_space_info.discrete_branches
        else:
            self.action_type = "hybrid"
            self.continuous_size = action_space_info.continuous_size
            self.discrete_branches = action_space_info.discrete_branches
            self.discrete_size = sum(action_space_info.discrete_branches)
            self.action_size = self.continuous_size + self.discrete_size

        print(f"Action space detected: {self.action_type}")
        if self.action_type == "continuous":
            print(f"  Continuous size: {self.action_size}")
        elif self.action_type == "discrete":
            print(f"  Discrete branches: {self.discrete_branches}")
        else:
            print(f"  Continuous size: {self.continuous_size}")
            print(f"  Discrete branches: {self.discrete_branches}")
            print(f"  Unified Box space size: {self.action_size}")

    def create_action(self, network_output):
        if self.action_type == "continuous":
            return network_output
        elif self.action_type == "discrete":
            return self._process_discrete_action(network_output)
        else:
            return self._process_hybrid_action(network_output)

    def action_to_network_output(self, action):
        if self.action_type == "continuous":
            return action
        elif self.action_type == "discrete":
            return self._process_discrete_to_network_output(action)
        else:
            return self._process_hybrid_to_network_output(action)

    def get_size(self):
        return self.action_size

    def get_gym_action_space(self):
        if self.action_type == "continuous":
            return spaces.Box(low=-1.0, high=1.0, shape=(self.action_size,), dtype=np.float32)
        elif self.action_type == "discrete":
            if len(self.discrete_branches) == 1:
                return spaces.Discrete(self.discrete_branches[0])
            else:
                return spaces.MultiDiscrete(self.discrete_branches)
        else:
            return spaces.Box(low=-1.0, high=1.0, shape=(self.action_size,), dtype=np.float32)

    def _process_discrete_action(self, network_output):
        if isinstance(network_output, np.ndarray):
            if len(self.discrete_branches) == 1:
                return np.array([network_output], dtype=np.int32)
            else:
                return network_output.astype(np.int32)
        else:
            return np.array([network_output], dtype=np.int32)

    def _process_hybrid_action(self, network_output):
        continuous_part = network_output[:self.continuous_size]
        discrete_part = network_output[self.continuous_size:]

        continuous_action = continuous_part
        discrete_action = self._continuous_to_discrete(discrete_part)

        return (continuous_action, discrete_action)

    def _continuous_to_discrete(self, continuous_values):
        discrete_actions = []
        value_idx = 0

        for branch_size in self.discrete_branches:
                discrete_action = 0
                max_continuous_val = float("-inf")
                for i in range(branch_size):
                        if value_idx + i < len(continuous_values):
                                continuous_val = continuous_values[value_idx + i]
                                if continuous_val > max_continuous_val:
                                        discrete_action = i
                                        max_continuous_val = continuous_val
                discrete_actions.append(discrete_action)
                value_idx += branch_size

        return np.array(discrete_actions, dtype=np.int32)

    def _process_discrete_to_network_output(self, action):
        if isinstance(action, np.ndarray) and len(self.discrete_branches) == 1 and len(action) == 1:
            return action[0]
        return action

    def _process_hybrid_to_network_output(self, action):
        continuous_action, discrete_action = action
        discrete_continuous = self._discrete_to_continuous(discrete_action)
        return np.concatenate([continuous_action, discrete_continuous])

    def _discrete_to_continuous(self, discrete_values):
        continuous_actions = []
        value_idx = 0

        for branch_size in self.discrete_branches:
            if value_idx < len(discrete_values):
                discrete_val = discrete_values[value_idx]
                for i in range(branch_size):
                    if i == discrete_val:
                        continuous_actions.append(1.0)
                    else:
                        continuous_actions.append(-1.0)
            value_idx += 1

        return np.array(continuous_actions, dtype=np.float32)

class EnvWrapper(gym.Env):
    def __init__(self, observation_structure, action_space_info):
        super().__init__()
        self.observation_processor = ObservationProcessor(observation_structure)
        self.action_processor = ActionProcessor(action_space_info)

        obs_size = self.observation_processor.get_size()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = self.action_processor.get_gym_action_space()
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return dummy_obs, {}

    def step(self, action):
        self.step_count += 1
        dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return dummy_obs, reward, terminated, truncated, info

class MLPlay:
    def __init__(self, observation_structure, action_space_info, name, *args, **kwargs):
        mlplayArgs.name = name
        mlplayArgs.init_kwargs = kwargs

        rlplayRewardCalculator.reset()
        self.RLPlay = RLPlay()

        self.env_wrapper = EnvWrapper(observation_structure, action_space_info)
        self.config = {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "clip_range": 0.2,
            "gamma": 0.99,
            "ent_coef": 0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "tensorboard_log": os.path.join(os.path.dirname(__file__), "tensorboard"),
            "policy_kwargs": {
                "net_arch": [64, 64],
                "activation_fn": torch.nn.Tanh
            }
        }
        self.prev_observation = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None
        self.episode_rewards = []
        self.total_steps = 0
        self.episode_count = 1
        self.update_count = 0
        self.start_time = time.strftime("%Y%m%d_%H%M%S")
        self.model_save_dir = os.path.join(os.path.dirname(__file__), "models", self.start_time)
        self.model_path = os.path.join(os.path.dirname(__file__), 'model' + ".zip")

        os.makedirs(self.model_save_dir, exist_ok=True)

        self._initialize_model()
        print(f"PPO initialized in training mode")

    def reset(self):
        if self.episode_rewards:
                total_reward = sum(self.episode_rewards)
                print(f"Episode {self.episode_count}: Total Reward = {total_reward:.2f}, Steps = {len(self.episode_rewards)}")
                self.episode_rewards = []

        self._update_policy()

        self.prev_observation = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None
        self.episode_count += 1

        rlplayRewardCalculator.reset()
        self.RLPlay.reset()

    def update(self, observations, done, info, keyboard=set(), *args, **kwargs):
        mlplayArgs.observations = observations
        mlplayArgs.keyboard = keyboard
        rlplayRewardCalculator.update(observations)
        observation = observations["flattened"]

        reward, not_used_for_training = self.RLPlay.update()
        action, log_prob, value = self._predict_with_info(observation)

        if self.prev_observation is not None:
            self.episode_rewards.append(reward)

            if not not_used_for_training and not self.model.rollout_buffer.full:
                self._add_to_rollout_buffer(
                    obs=self.prev_observation,
                    action=self.prev_action,
                    reward=reward,
                    done=done,
                    value=self.prev_value,
                    log_prob=self.prev_log_prob
                )
                if self.model.rollout_buffer.full:
                    done_tensor = np.array([done])
                    value_tensor = torch.as_tensor(value).unsqueeze(0) if value.ndim == 0 else torch.as_tensor(value)
                    self.model.rollout_buffer.compute_returns_and_advantage(last_values=value_tensor, dones=done_tensor)

        self.prev_observation = observation
        self.prev_action = action
        self.prev_log_prob = log_prob
        self.prev_value = value
        self.total_steps += 1

        return self.env_wrapper.action_processor.create_action(action)

    def _initialize_model(self):
        print(f"Initializing PPO model...")
        if os.path.exists(self.model_path):
            try:
                self.model = PPO.load(self.model_path, env=self.env_wrapper, **self.config, verbose=1)
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading model from {self.model_path}: {e}")
                print("Creating new model...")
                self.model = PPO("MlpPolicy", env=self.env_wrapper, **self.config, verbose=1)
        else:
            print(f"No pre-trained model found at {self.model_path}. Creating new model...")
            self.model = PPO("MlpPolicy", env=self.env_wrapper, **self.config, verbose=1)
        self.model.learn(total_timesteps=0, tb_log_name=f"PPO_{self.start_time}")

    def _save_model(self):
        if self.model is not None:
            self.model.save(self.model_path)
            print(f"Model saved to {self.model_path}")

            update_path = f"{self.model_save_dir}/ppo_model_{self.update_count}.zip"
            self.model.save(update_path)
            print(f"Model saved to {update_path}")

    def _predict_with_info(self, obs):
        obs_tensor = torch.as_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, value, log_prob = self.model.policy(obs_tensor)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten(), value.cpu().numpy().flatten()

    def _add_to_rollout_buffer(self, obs, action, reward, done, value, log_prob):
        if not self.model.rollout_buffer.full:
            self.model.rollout_buffer.add(
                obs=torch.as_tensor(obs).unsqueeze(0),
                action=torch.as_tensor(action).unsqueeze(0),
                reward=torch.as_tensor([reward]),
                episode_start=torch.as_tensor([done]),
                value=torch.as_tensor(value).unsqueeze(0) if value.ndim == 0 else torch.as_tensor(value),
                log_prob=torch.as_tensor(log_prob).unsqueeze(0) if log_prob.ndim == 0 else torch.as_tensor(log_prob)
            )

    def _update_policy(self):
        if self.model.rollout_buffer.size() == 0 or not self.model.rollout_buffer.full:
            return

        print(f"Updating PPO policy with {self.model.rollout_buffer.size()} experiences...")

        self.model.num_timesteps += self.model.rollout_buffer.size()
        self.model.train()
        self.update_count += 1

        self.model.logger.record("train/mean_reward", safe_mean(self.model.rollout_buffer.rewards))
        self.model.logger.record("param/n_steps", self.model.n_steps)
        self.model.logger.record("param/batch_size", self.model.batch_size)
        self.model.logger.record("param/n_epochs", self.model.n_epochs)
        self.model.logger.record("param/gamma", self.model.gamma)
        self.model.logger.record("param/gae_lambda", self.model.gae_lambda)
        self.model.logger.record("param/ent_coef", self.model.ent_coef)
        self.model.logger.record("param/vf_coef", self.model.vf_coef)
        self.model.logger.record("param/max_grad_norm", self.model.max_grad_norm)
        self.model._dump_logs(self.update_count)

        self.model.rollout_buffer.reset()
        print("PPO policy updated successfully")

        self._save_model()


class RLPlay:
    def __init__(self):
        """
        [初始化設定]
        目標：設定這局遊戲需要的計數器或狀態變數。

        提示與引導：
        1. 你可能需要變數來記錄：
           - self.stagnation_timer: 用來偵測是否卡牆太久 (防摸魚機制)。
           - self.last_checkpoint: 紀錄上一次的檢查點，用來判斷有沒有在前進。
        """
        self.stagnation_timer = 0
        self.last_checkpoint = 0
        pass

    def update(self):
        """
        [主更新迴圈] (每一幀都會執行)
        目標：計算這一瞬間的總得分，並告訴 AI 這一局是否結束 (Done)。

        提示與引導：
        1. **呼叫獎勵計算機**：
           使用 rlplayRewardCalculator.calculate_...() 取得各個項目的分數。

        2. **加總獎勵 (Total Reward)**：
           total_reward = 檢查點分數 + 距離分數 + 活著的分數 + ...
           (這是 AI 這一瞬間真正拿到的「糖果」或「鞭子」)

        3. **判斷是否結束 (Done Flag)**：
           除了遊戲原本的結束 (贏了/輸了)，我們通常還會加上「強制結束」條件：
           - 是否卡在原地太久？ (Stagnation) -> 如果是，回傳 True (結束這局)。
           - 是否反覆撞牆？
           (提早結束沒希望的局，可以節省大量的訓練時間！)

        回傳值：
        - total_reward (float): 這一幀的總分。
        - is_done (bool): 是否結束這回合。
        """
        obs = rlplayRewardCalculator.observation

        # 1. 計算各項分數
        checkpoint_reward = rlplayRewardCalculator.calculate_checkpoint_reward(weight=2.0)
        distance_reward = rlplayRewardCalculator.calculate_distance_reward(close_weight=0.05, leave_weight=-0.05)
        health_reward = rlplayRewardCalculator.calculate_health_reward(
            death_weight=-2.0,
            increase_weight=0.5,
            decrease_weight=-0.5
        )
        water_reward = rlplayRewardCalculator.calculate_water_reward(
            threshold=1.5,
            close_weight=-1.0
        )
        hazard_reward = rlplayRewardCalculator.calculate_mud_reward(
            threshold=1.5,
            leave_weight=0.1,
            close_weight=-0.2
        )

        # 2. 加總
        total_reward = checkpoint_reward + distance_reward + health_reward + water_reward + hazard_reward

        # 3. 進度/卡住偵測 (Anti-AFK)
        not_used_for_training = False
        if obs is None:
            return 0.0, True

        if obs.get("is_respawning", False):
            not_used_for_training = True

        current_checkpoint = obs.get("last_checkpoint_index", None)
        if current_checkpoint is not None and current_checkpoint > self.last_checkpoint:
            self.last_checkpoint = current_checkpoint
            self.stagnation_timer = 0
        else:
            self.stagnation_timer += 1

        reached_final = obs.get("reached_final_checkpoint", False)

        if self.stagnation_timer > 600:
            return total_reward, True

        if reached_final:
            return total_reward, True

        return total_reward, not_used_for_training

    def reset(self):
        """
        [重置回合]
        目標：當一局結束，準備開始下一局時，把所有計數器歸零。

        提示與引導：
        1. 就像玩遊戲接關一樣，計時器、狀態變數都要重新開始。
        2. 確保把 self.stagnation_timer 清空。
        3. **重要**：也要記得呼叫 rlplayRewardCalculator.reset()。
        """
        self.stagnation_timer = 0
        rlplayRewardCalculator.reset()
        pass
