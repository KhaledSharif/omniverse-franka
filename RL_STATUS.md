# RL Infrastructure Status

## Current Status Summary

### ✅ **Implemented - Data Collection**

The repository has **complete** infrastructure for collecting RL training data:

#### 1. **Observation Space (23D)**
Location: `src/franka_keyboard_control.py:1041` - `ObservationBuilder` class

```
[0:7]   - Joint positions (7)
[7:10]  - End-effector position (3)
[10:14] - End-effector orientation quaternion (4)
[14:15] - Gripper width (1)
[15:18] - Cube position (3)
[18:21] - Goal position (3)
[21:22] - Cube grasped flag (1)
[22:23] - Distance to cube (1)
```

**Status:** ✅ Fully implemented and tested (191 tests passing)

#### 2. **Action Space (7D)**
Location: `src/franka_keyboard_control.py:987` - `ActionMapper` class

```
[0] - delta_x (forward/backward)
[1] - delta_y (left/right)
[2] - delta_z (up/down)
[3] - delta_roll
[4] - delta_pitch
[5] - delta_yaw
[6] - gripper command
```

**Status:** ✅ Fully implemented and tested

#### 3. **Reward Computation**
Location: `src/franka_keyboard_control.py:1094` - `RewardComputer` class

**Supports two modes:**

- **Sparse rewards:** +10.0 only on task completion
- **Dense rewards:** Shaped rewards with:
  - Distance-based shaping (reaching cube, placing at goal)
  - Grasp bonus: +5.0
  - Drop penalty: -5.0
  - Task completion: +10.0

**Status:** ✅ Fully implemented with both sparse/dense modes

#### 4. **Demo Recording**
Location: `src/franka_keyboard_control.py:773` - `DemoRecorder` class

**Features:**
- Records (observation, action, reward, done) tuples
- Episode segmentation with success/failure labels
- Auto-save every 5 seconds (checkpoint system)
- Saves to `.npz` format compatible with RL libraries

**Recorded data format:**
```python
{
    'observations': (N, 23),
    'actions': (N, 7),
    'rewards': (N,),
    'dones': (N,),
    'episode_starts': episode boundaries,
    'episode_lengths': frames per episode,
    'episode_returns': cumulative reward per episode,
    'episode_success': boolean success flags,
    'metadata': additional info
}
```

**Usage:**
```bash
./run.sh --enable-recording          # Start recording
# Press ` to toggle recording
# Press [ to mark success
# Press ] to mark failure
```

**Status:** ✅ Fully implemented and working (existing demos in `demos/`)

#### 5. **Scene Management**
Location: `src/franka_keyboard_control.py:608` - `SceneManager` class

**Features:**
- Spawns cube and goal marker in simulation
- Randomizes positions within workspace bounds
- Grasp detection (`check_grasp()`)
- Task completion detection (`check_task_complete()`)

**Status:** ✅ Fully implemented

---

## ❌ **Not Implemented - RL Training & Inference**

### Missing Components:

#### 1. **Gymnasium Environment Wrapper**
**What's needed:**
- A `gym.Env` subclass that wraps the Isaac Sim environment
- Implements standard methods: `reset()`, `step()`, `render()`, `close()`
- Handles episode termination and truncation
- Properly manages Isaac Sim simulation lifecycle

**Why it's needed:**
- Required for using Stable-Baselines3, RLlib, CleanRL, etc.
- Provides standard RL interface for training algorithms

**Suggested location:** `src/franka_rl_env.py`

#### 2. **RL Training Script**
**What's needed:**
- Script to train RL agents (PPO, SAC, TD3, etc.)
- Uses Stable-Baselines3 or similar library
- Loads environment wrapper
- Configures hyperparameters
- Saves checkpoints during training
- Logs metrics (rewards, success rate, etc.)

**Suggested location:** `src/train_rl.py`

**Example usage:**
```bash
./run.sh train_rl.py --algo ppo --timesteps 1000000
```

#### 3. **Policy Inference/Evaluation**
**What's needed:**
- Script to load trained RL policies
- Run policy in Isaac Sim for visual evaluation
- Compute success rate over N episodes
- Compare with human demonstrations

**Suggested location:** `src/eval_policy.py`

**Example usage:**
```bash
./run.sh eval_policy.py models/ppo_policy.zip --episodes 100
```

#### 4. **Pre-training from Demonstrations (Optional)**
**What's needed:**
- Use recorded demos to pre-train RL policy
- Either through behavioral cloning warm-start
- Or using expert demonstrations in replay buffer

**Note:** The repo already has behavioral cloning (`train_bc.py`), but it's not integrated with RL training.

---

## 📊 **Current Capabilities vs Gaps**

| Capability | Status | Location |
|------------|--------|----------|
| **Data Collection** | ✅ Complete | `DemoRecorder` |
| Observation space | ✅ Complete | `ObservationBuilder` |
| Action space | ✅ Complete | `ActionMapper` |
| Reward function | ✅ Complete | `RewardComputer` |
| Scene setup | ✅ Complete | `SceneManager` |
| Demo recording | ✅ Complete | Recording mode |
| Demo replay | ✅ Complete | `replay.py` |
| **Imitation Learning** | ✅ Complete | `train_bc.py` |
| Behavioral cloning | ✅ Complete | PyTorch/imitation |
| **RL Training** | ❌ Missing | N/A |
| Gym environment | ❌ Missing | N/A |
| RL algorithms | ❌ Missing | N/A |
| Training script | ❌ Missing | N/A |
| **RL Evaluation** | ❌ Missing | N/A |
| Policy loading | ❌ Missing | N/A |
| Inference script | ❌ Missing | N/A |
| Success metrics | ❌ Missing | N/A |

---

## 🚀 **Recommended Implementation Plan**

### Phase 1: Gymnasium Environment Wrapper

Create `src/franka_rl_env.py`:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FrankaPickPlaceEnv(gym.Env):
    """Gymnasium environment for Franka pick-and-place task."""

    def __init__(self, headless=False, reward_mode='dense'):
        super().__init__()

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(23,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(7,), dtype=np.float32
        )

        # Initialize Isaac Sim components
        self.sim_app = SimulationApp({"headless": headless})
        self.world = World(...)
        self.franka = ...
        self.scene_manager = SceneManager(self.world)
        self.obs_builder = ObservationBuilder()
        self.reward_computer = RewardComputer(mode=reward_mode)

    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        super().reset(seed=seed)

        # Reset robot to home position
        # Randomize cube and goal positions
        # Return initial observation

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        """Execute action and return transition."""
        # Apply action to robot
        # Step simulation
        # Compute observation, reward, termination

        obs = self._get_obs()
        reward = self._compute_reward(...)
        terminated = self._check_termination()
        truncated = self._check_truncation()
        info = {}

        return obs, reward, terminated, truncated, info

    def close(self):
        """Cleanup Isaac Sim."""
        self.sim_app.close()
```

### Phase 2: RL Training Script

Create `src/train_rl.py`:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from franka_rl_env import FrankaPickPlaceEnv

def train(algo='ppo', timesteps=1000000, seed=0):
    # Create environment
    env = DummyVecEnv([lambda: FrankaPickPlaceEnv(headless=True)])

    # Create agent
    if algo == 'ppo':
        model = PPO('MlpPolicy', env, verbose=1)

    # Train
    model.learn(total_timesteps=timesteps)

    # Save
    model.save(f"models/{algo}_policy")

if __name__ == '__main__':
    train()
```

### Phase 3: Policy Evaluation

Create `src/eval_policy.py`:

```python
from stable_baselines3 import PPO
from franka_rl_env import FrankaPickPlaceEnv

def evaluate(policy_path, num_episodes=100):
    env = FrankaPickPlaceEnv(headless=False)
    model = PPO.load(policy_path)

    successes = 0
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if info.get('task_complete'):
                successes += 1

    print(f"Success rate: {successes}/{num_episodes} = {100*successes/num_episodes:.1f}%")
    env.close()
```

---

## 📝 **Summary**

### What Works Today:
- ✅ **Data collection:** Record expert demonstrations with rewards
- ✅ **Imitation learning:** Train behavioral cloning policies
- ✅ **Visualization:** Replay recorded demos

### What's Missing for RL:
- ❌ **Gym environment wrapper** (most critical)
- ❌ **RL training script** (PPO, SAC, etc.)
- ❌ **Policy evaluation script**

### Bottom Line:
The repository has **excellent infrastructure** for collecting RL data (observations, actions, rewards), but **does not yet support** training or evaluating RL policies. All the hard work (reward functions, observation spaces, scene management) is done - you just need to wrap it in a Gymnasium environment and add training scripts.

### Effort Estimate:
- Gymnasium wrapper: ~200-300 lines
- Training script: ~100-150 lines
- Evaluation script: ~100 lines
- **Total: ~400-550 lines of code**

Given the quality of existing infrastructure, adding full RL support would be straightforward.
