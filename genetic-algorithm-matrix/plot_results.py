import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from datetime import datetime
import inspect


def export_ga_run(population_size, mutation_rate, crossover_rate, 
                  tournament_size, fitness_vs_time, matrix,
                  base_dir="ga_run_results", matrix_dir="ga_matrices_results"):

    frame = inspect.currentframe().f_back
    caller_filename = os.path.basename(frame.f_code.co_filename)
    caller_name = os.path.splitext(caller_filename)[0]  # Remove .py extension

    timestamp = datetime.now().strftime("%y%m%d%H%M")
    run_name = f"{timestamp}_pop{population_size}_mut{mutation_rate}_cross{crossover_rate}_tour{tournament_size}"

    run_subdir = os.path.join(base_dir, caller_name)
    matrix_subdir = os.path.join(matrix_dir, caller_name)
    
    os.makedirs(run_subdir, exist_ok=True)
    os.makedirs(matrix_subdir, exist_ok=True)

    # Save fitness results CSV
    csv_file = os.path.join(run_subdir, f"{run_name}.csv")
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['iteration', 'fitness', 'population_size', 
                        'mutation_rate', 'crossover_rate', 'tournament_size'])

        for iteration, fitness in fitness_vs_time.items():
            writer.writerow([iteration, fitness, population_size, 
                           mutation_rate, crossover_rate, tournament_size])
    
    # Save matrix 
    matrix_file = os.path.join(matrix_subdir, f"{run_name}.txt")
    np.savetxt(matrix_file, matrix, fmt='%.6f', delimiter='\t')
    
    print(f"Results exported to: {csv_file}")
    print(f"Matrix exported to: {matrix_file}")


def load_matrix(run_name, python_file_name, matrix_dir="ga_matrices_results"):
    """Load a saved matrix by run name and python file name"""
    matrix_subdir = os.path.join(matrix_dir, python_file_name)
    matrix_file = os.path.join(matrix_subdir, f"{run_name}.txt")
    if os.path.exists(matrix_file):
        return np.loadtxt(matrix_file, delimiter='\t')
    else:
        print(f"Matrix file not found: {matrix_file}")
        return None


def plot_all_runs(base_dir="ga_run_results"):
    """Plot all GA runs from all subdirectories"""
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist")
        return
    
    all_runs = {}
    
    # Iterate through all subdirectories (python file names)
    for subdir_name in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir_name)
        
        if os.path.isdir(subdir_path):
            print(f"Processing runs from: {subdir_name}")
            
            # Iterate through all CSV files in this subdirectory
            for filename in os.listdir(subdir_path):
                if filename.endswith('.csv'):
                    run_name = filename[:-4]  # Remove .csv extension
                    full_run_name = f"{subdir_name}_{run_name}"  # Prefix with subdir name
                    csv_file = os.path.join(subdir_path, filename)
                    
                    all_runs[full_run_name] = {'iterations': [], 'fitness': []}
                    
                    with open(csv_file, 'r') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            all_runs[full_run_name]['iterations'].append(int(row['iteration']))
                            all_runs[full_run_name]['fitness'].append(float(row['fitness']))
    
    if not all_runs:
        print("No run data found")
        return
    
    # Plot all runs
    plt.figure(figsize=(14, 8))
    
    # Use different colors for different subdirectories
    subdirs = list(set(run_name.split('_')[0] for run_name in all_runs.keys()))
    colors = plt.cm.tab10(range(len(subdirs)))
    color_map = {subdir: colors[i] for i, subdir in enumerate(subdirs)}
    
    for run_name, data in all_runs.items():
        subdir = run_name.split('_')[0]
        plt.plot(data['iterations'], data['fitness'], 
                label=run_name, marker='o', markersize=2, 
                color=color_map[subdir], alpha=0.8)
    
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title('GA Fitness Evolution - All Runs (All Files)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_runs_by_file(python_file_name, base_dir="ga_run_results"):
    """Plot runs from a specific python file"""
    
    subdir_path = os.path.join(base_dir, python_file_name)
    
    if not os.path.exists(subdir_path):
        print(f"Directory {subdir_path} does not exist")
        return
    
    runs = {}
    
    # Iterate through all CSV files in the specific subdirectory
    for filename in os.listdir(subdir_path):
        if filename.endswith('.csv'):
            run_name = filename[:-4]  # Remove .csv extension
            csv_file = os.path.join(subdir_path, filename)

            runs[run_name] = {'iterations': [], 'fitness': []}

            with open(csv_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    runs[run_name]['iterations'].append(int(row['iteration']))
                    runs[run_name]['fitness'].append(float(row['fitness']))

    if not runs:
        print(f"No run data found for {python_file_name}")
        return

    plt.figure(figsize=(12, 8))

    for run_name, data in runs.items():
        plt.plot(data['iterations'], data['fitness'], label=run_name, 
                marker='o', markersize=2)

    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title(f'GA Fitness Evolution - {python_file_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def test():
    plot_runs_by_file("evolved_paulix")
    plot_runs_by_file("evolved_paulix_new")


if __name__ == "__main__":
    test()
