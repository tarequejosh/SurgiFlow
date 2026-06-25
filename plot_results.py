import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_comparative_curves(log_dirs, labels, output_file="comparative_learning_curve.png"):
    plt.figure(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, log_dir in enumerate(log_dirs):
        csv_file = os.path.join(log_dir, "monitor.csv")
        if not os.path.exists(csv_file):
            print(f"Warning: Could not find {csv_file}")
            continue
            
        df = pd.read_csv(csv_file, skiprows=1)
        df['cumulative_timesteps'] = df['l'].cumsum()
        
        window = min(len(df) // 10, 50)
        if window < 1:
            window = 1
        df['smoothed_reward'] = df['r'].rolling(window=window, min_periods=1).mean()
        
        color = colors[i % len(colors)]
        plt.plot(df['cumulative_timesteps'], df['r'], alpha=0.15, color=color)
        plt.plot(df['cumulative_timesteps'], df['smoothed_reward'], color=color, linewidth=2, label=f'{labels[i]} (MA window={window})')
        
    plt.title('Reward Structure Comparison: Dense vs Binary', fontsize=16)
    plt.xlabel('Timesteps', fontsize=14)
    plt.ylabel('Episodic Reward', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300)
    print(f"Saved comparative learning curve to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense_log", type=str, default="./logs/dense/", help="Path to dense model log directory")
    parser.add_argument("--binary_log", type=str, default="./logs/binary/", help="Path to binary model log directory")
    args = parser.parse_args()
    
    plot_comparative_curves([args.dense_log, args.binary_log], ["Dense Reward", "Binary Reward"])
