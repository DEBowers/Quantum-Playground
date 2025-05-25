import matplotlib.pyplot as plt
import csv
import os

def export_ga_run(population_size, mutation_rate, crossover_rate, 
                  tournament_size, fitness_vs_time, filename="ga_results.csv"):
    run_name = f"pop{population_size}_mut{mutation_rate}_cross{crossover_rate}_tour{tournament_size}"
    file_exists = os.path.exists(filename)

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['run_name', 'iteration', 'fitness', 'population_size', 
                           'mutation_rate', 'crossover_rate', 'tournament_size'])

        for iteration, fitness in fitness_vs_time.items():
            writer.writerow([run_name, iteration, fitness, population_size, 
                           mutation_rate, crossover_rate, tournament_size])

def plot_all_runs(filename="ga_results.csv"):
    runs = {}

    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            run_name = row['run_name']
            if run_name not in runs:
                runs[run_name] = {'iterations': [], 'fitness': [], 'params': None}

            runs[run_name]['iterations'].append(int(row['iteration']))
            runs[run_name]['fitness'].append(float(row['fitness']))

            # Store params (same for all rows of a run)
            if runs[run_name]['params'] is None:
                runs[run_name]['params'] = (row['population_size'], row['mutation_rate'], 
                                          row['crossover_rate'], row['tournament_size'])

    plt.figure(figsize=(10, 6))

    for run_name, data in runs.items():
        plt.plot(data['iterations'], data['fitness'], label=run_name, marker='o', markersize=2)

    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title('GA Fitness Evolution - All Runs')
    plt.legend()
    plt.grid(True)
    plt.show()

def test():
    plot_all_runs()


if __name__ == "__main__":
    test()
