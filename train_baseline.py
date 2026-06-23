import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from surgical_tray_env import SurgicalTrayEnv
from gymnasium.wrappers import TimeLimit

def main():
    print("Initializing SurgicalTrayEnv for Training...")
    
    # Initialize the base environment
    env = SurgicalTrayEnv(render_mode=None)
    
    # 1. Wrap with TimeLimit (e.g., 500 steps max per episode)
    env = TimeLimit(env, max_episode_steps=500)
    
    # 2. Wrap with Monitor to track episodic rewards and lengths automatically
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    
    # check_env(env, warn=True)
    
    print("Environment verified. Setting up SAC Agent...")
    
    # Initialize Soft Actor-Critic (SAC) - Excellent for continuous robotic control
    # Using MultiInputPolicy to natively support the Dict observation space!
    model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")
    
    # Setup Evaluation Callback to test the model during training
    eval_callback = EvalCallback(
        env, 
        best_model_save_path='./models/best_model/',
        log_path='./logs/', 
        eval_freq=10000, 
        deterministic=True, 
        render=False
    )
    
    print("Starting Final Training Run (1,000,000 timesteps)...")
    # This is the massive training run for the paper
    model.learn(total_timesteps=1000000, callback=eval_callback, progress_bar=True)
    
    # Save the final model
    model.save("./models/final_sac_surgical_model")
    print("Training complete! Model saved to ./models/")

if __name__ == "__main__":
    main()
