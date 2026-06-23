import os
import imageio
from surgical_tray_env import SurgicalTrayEnv
from gymnasium.wrappers import FlattenObservation
try:
    from stable_baselines3 import SAC
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

def render_env():
    print("Initializing SurgicalTrayEnv for rendering...")
    # Initialize the environment with render mode
    env = SurgicalTrayEnv(render_mode="rgb_array")
    
    # Check if we have a trained model
    model_path = "./models/best_model/best_model.zip"
    model = None
    if HAS_SB3 and os.path.exists(model_path):
        print(f"Found trained model at {model_path}. Loading SAC policy...")
        model = SAC.load(model_path)
    else:
        print("No trained model found. Using random actions...")
    
    obs, info = env.reset()
    frames = []
    
    steps = 150 if model is not None else 50
    print(f"Taking {steps} steps for video generation...")
    for _ in range(steps):
        if model is not None:
            action, _states = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
            
        if terminated or truncated:
            obs, info = env.reset()

    # Save to a GIF
    if len(frames) > 0:
        print("Saving video to visual_representation.gif...")
        imageio.mimsave('visual_representation.gif', frames, fps=30)
        
        # Save a single frame
        import matplotlib.pyplot as plt
        plt.imsave('initial_state.png', frames[0])
        print("Saving initial_state.png...")
    
    print("Done!")

if __name__ == "__main__":
    render_env()
