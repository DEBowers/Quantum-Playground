import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from datetime import datetime


def export_ga_run(population_size, mutation_rate, crossover_rate, 
                  tournament_size, fitness_vs_time, matrix,
                  base_dir="ga_run_results", matrix_dir="ga_matrices_results"):

    timestamp = datetime.now().strftime("%y%m%d%H%M")

    run_name = f"{timestamp}_pop{population_size}_mut{mutation_rate}_cross{crossover_rate}_tour{tournament_size}"

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(matrix_dir, exist_ok=True)

    csv_file = os.path.join(base_dir, f"{run_name}.csv")

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['iteration', 'fitness', 'population_size', 
                        'mutation_rate', 'crossover_rate', 'tournament_size'])

        for iteration, fitness in fitness_vs_time.items():
            writer.writerow([iteration, fitness, population_size, 
                           mutation_rate, crossover_rate, tournament_size])

    matrix_file = os.path.join(matrix_dir, f"{run_name}.txt")
    np.savetxt(matrix_file, matrix, fmt='%.6f', delimiter='\t')    
    print(f"Results exported to: {csv_file}")
    print(f"Matrix exported to: {matrix_file}")
    print(f"Results exported to: {csv_file}")

def plot_all_runs(base_dir="ga_run_results"):
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist")
        return

    runs = {}

    for filename in os.listdir(base_dir):
        if filename.endswith('.csv'):
            run_name = filename[:-4]
            csv_file = os.path.join(base_dir, filename)

            runs[run_name] = {'iterations': [], 'fitness': []}

            with open(csv_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    runs[run_name]['iterations'].append(int(row['iteration']))
                    runs[run_name]['fitness'].append(float(row['fitness']))
    if not runs:
        print("No run data found")
        return
    plt.figure(figsize=(12, 8))

    for run_name, data in runs.items():
        plt.plot(data['iterations'], data['fitness'], label=run_name, 
                marker='o', markersize=2)

    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title('GA Fitness Evolution - All Runs')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def test():
    plot_all_runs()


if __name__ == "__main__":
    test()

