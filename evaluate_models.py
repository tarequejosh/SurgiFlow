import argparse
import numpy as np
from stable_baselines3 import SAC
from surgical_tray_env import SurgicalTrayEnv
from gymnasium.wrappers import TimeLimit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model .zip file")
    parser.add_argument("--reward_type", type=str, default="dense", choices=["dense", "binary"], help="Reward type env was trained with")
    parser.add_argument("--trials", type=int, default=1000, help="Number of evaluation trials")
    args = parser.parse_args()

    print(f"Loading environment with {args.reward_type} reward setting...")
    env = SurgicalTrayEnv(render_mode=None, reward_type=args.reward_type)
    env = TimeLimit(env, max_episode_steps=500)
    
    print(f"Loading model from {args.model_path}...")
    model = SAC.load(args.model_path)
    
    success_count = 0
    collision_count = 0
    drop_count = 0
    completion_times = []
    
    print(f"Running {args.trials} evaluation trials...")
    
    for i in range(args.trials):
        obs, info = env.reset()
        done = False
        
        episode_success = False
        episode_collision = False
        episode_dropped = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if info.get("collision", False):
                episode_collision = True
            if info.get("dropped", False):
                episode_dropped = True
                
            done = terminated or truncated
            
            if info.get("is_success"):
                episode_success = True
                if "completion_time_seconds" in info:
                    completion_times.append(info["completion_time_seconds"])
        
        if episode_success:
            success_count += 1
        if episode_collision:
            collision_count += 1
        if episode_dropped:
            drop_count += 1
            
        if (i+1) % 100 == 0:
            print(f"Completed {i+1}/{args.trials} trials. Current Success Rate: {success_count/(i+1)*100:.1f}%")
            
    sr = success_count / args.trials * 100
    cr = collision_count / args.trials * 100
    dr = drop_count / args.trials * 100
    avg_t = np.mean(completion_times) if completion_times else 0.0
    
    print("\n" + "="*40)
    print(f"EVALUATION RESULTS: {args.model_path}")
    print("="*40)
    print(f"Total Trials: {args.trials}")
    print(f"Success Rate: {sr:.2f}%")
    print(f"Collision Rate: {cr:.2f}%")
    print(f"Drop Rate: {dr:.2f}%")
    print(f"Average Completion Time: {avg_t:.2f}s")
    print("="*40)

if __name__ == "__main__":
    main()
