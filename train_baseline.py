import os
import argparse
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from surgical_tray_env import SurgicalTrayEnv
from gymnasium.wrappers import TimeLimit

def main():
    parser = argparse.ArgumentParser(description="Train SAC on SurgicalTrayEnv")
    parser.add_argument("--reward_type", type=str, default="dense", choices=["dense", "binary"], help="Type of reward to use.")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps.")
    args = parser.parse_args()

    print(f"Initializing SurgicalTrayEnv for Training with {args.reward_type} rewards...")
    
    # Initialize the base environment
    env = SurgicalTrayEnv(render_mode=None, reward_type=args.reward_type)
    
    # 1. Wrap with TimeLimit (e.g., 500 steps max per episode)
    env = TimeLimit(env, max_episode_steps=500)
    
    # 2. Wrap with Monitor to track episodic rewards and lengths automatically
    log_dir = f"./logs/{args.reward_type}/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    
    # check_env(env, warn=True)
    
    print("Environment verified. Setting up SAC Agent...")
    
    # Initialize Soft Actor-Critic (SAC) - Excellent for continuous robotic control
    # Using MultiInputPolicy to natively support the Dict observation space!
    model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log=f"./tensorboard_logs/{args.reward_type}/")
    
    # Setup Evaluation Callback to test the model during training
    eval_callback = EvalCallback(
        env, 
        best_model_save_path=f'./models/best_{args.reward_type}_model/',
        log_path=f'./logs/{args.reward_type}/', 
        eval_freq=10000, 
        deterministic=True, 
        render=False
    )
    
    print(f"Starting Training Run for {args.reward_type} reward ({args.timesteps} timesteps)...")
    # This is the massive training run for the paper
    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=True)
    
    # Save the final model
    model.save(f"./models/final_sac_{args.reward_type}_model")
    print(f"Training complete! Model saved to ./models/final_sac_{args.reward_type}_model")

if __name__ == "__main__":
    main()
