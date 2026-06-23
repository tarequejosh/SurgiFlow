import sys
from surgical_tray_env import SurgicalTrayEnv

def test_env():
    try:
        print("Initializing SurgicalTrayEnv...")
        env = SurgicalTrayEnv()
        print("Environment initialized successfully.")
        
        print("Observation Space:", env.observation_space)
        
        print("Resetting environment...")
        obs, info = env.reset()
        print("Reset successful. Initial object states:")
        print(env.object_states)
        
        print(f"Target sequence: {env.target_sequence}")
        print("Taking 5 random steps...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: Reward: {reward}, Success: {info.get('is_success')}")
        
        print("Test complete and successful.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_env()
