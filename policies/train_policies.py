from stable_baselines3 import SAC, TD3, PPO
import gymnasium as gym


def train_policy(algo: str = "SAC"):
    
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    
    if algo == "SAC":
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="logs/sac_halfcheetah_logs/")
    elif algo == "TD3":
        model = TD3("MlpPolicy", env, verbose=1, tensorboard_log="logs/td3_halfcheetah_logs/")
    elif algo == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/ppo_halfcheetah_logs/")
    
    model.learn(total_timesteps=500_000)
    

    model.save(f"policies/HalfCheetah-v5_{algo}")
    

    if hasattr(model, "replay_buffer"):
        model.save_replay_buffer(f"buffers/{algo}_replay.pkl")
    
    return model


train_policy("SAC")
train_policy("TD3")
train_policy("PPO")