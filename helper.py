import matplotlib.pyplot as plt
import os
from MWSATInstance import MWSATInstance
from MWSATSolution import MWSATSolution
from simulated_annealing import simulated_annealing
import numpy as np

def evaluate_algorithm_performance(instance_paths, solutions_dict, params, n_runs=4, title="Algorithm Evaluation"):
    """
    Runs the Simulated Annealing algorithm multiple times on a set of instances,
    plots the convergence history, and prints a summary statistics table.

    Args:
        instance_paths (list): List of file paths to .mwcnf instances.
        solutions_dict (dict): Dictionary mapping instance names (no ext) to optimal weights.
        params (dict): Dictionary of hyperparameters for the algorithm.
        n_runs (int): Number of times to run the algorithm per instance.
        title (str): Title for the entire figure.
    """
    
    n_instances = len(instance_paths)
    
    # --- Setup Canvas ---
    # Rows = Instances, Columns = Individual Runs
    # Adjust height dynamically: 4 inches per row
    fig, axes = plt.subplots(nrows=n_instances, ncols=n_runs, 
                             figsize=(5 * n_runs, 4 * n_instances), 
                             sharey='row', sharex=True)
    
    # Handle case where n_instances=1 (axes is 1D array)
    if n_instances == 1:
        axes = np.array([axes])
    # Handle case where n_runs=1 (axes needs to be 2D)
    if n_runs == 1 and n_instances > 1:
         axes = axes.reshape(-1, 1)
    if n_instances == 1 and n_runs == 1:
        axes = np.array([[axes]])

    summary_data = []

    print(f"--- Starting {title} ({n_instances * n_runs} total runs) ---")
    print(f"Params: {params}")

    # --- Main Loop ---
    for i, path in enumerate(instance_paths):
        filename = os.path.basename(path)
        key = filename.split(".")[0]
        optimal_weight = solutions_dict.get(key, 0) # Default to 0 if unknown
        
        print(f"Processing Instance {i+1}/{n_instances}: {filename}")
        
        # Aggregators for this instance
        inst_scores = []
        inst_steps = []
        inst_solved_count = 0
        inst_optimal_count = 0

        for j in range(n_runs):
            ax = axes[i, j]
            
            # 1. Run Algorithm
            instance = MWSATInstance(path)
            best_state, history = simulated_annealing(instance, **params)
            
            # 2. Collect Metrics
            is_solved = (best_state.clauses_satisfied == instance.num_clauses)
            # If optimal_weight is 0 (unknown), we can't strictly check optimality, 
            # so we assume optimal if solved and weight > 0 (basic check)
            is_optimal = is_solved and (best_state.current_score >= optimal_weight)
            
            if is_solved: inst_scores.append(best_state.current_score)
            inst_steps.append(len(history))
            if is_solved: inst_solved_count += 1
            if is_optimal: inst_optimal_count += 1

            # 3. Plotting
            ax.plot(history, color='tab:blue', linewidth=1, alpha=0.9)
            
            if optimal_weight > 0:
                ax.axhline(y=optimal_weight, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimum')

            # 4. Status Box
            status_color = "green" if is_solved else "red"
            status_text = "SOLVED" if is_solved else "STUCK"
            info_text = f"{status_text}\nScore: {best_state.current_score}"
            
            # Position text dynamically
            text_y = 0.05 if (optimal_weight > 0 and best_state.current_score > optimal_weight * 0.9) else 0.85
            
            ax.text(0.05, text_y, info_text, transform=ax.transAxes, 
                    bbox=dict(facecolor='white', edgecolor=status_color, linewidth=2, alpha=0.9),
                    fontsize=9, fontweight='bold', color=status_color)

            # 5. Labels & Grid
            ax.grid(True, linestyle=':', alpha=0.5)
            
            if j == 0:
                ylabel = f"{filename}"
                if optimal_weight > 0: ylabel += f"\nOpt: {optimal_weight}"
                ax.set_ylabel(ylabel, fontsize=10, fontweight='bold', labelpad=10)
            
            if i == 0:
                ax.set_title(f"Run {j+1}", fontsize=11, fontweight='bold')
            
            if i == n_instances - 1:
                ax.set_xlabel("Steps", fontsize=10)

        # Instance Summary
        if len(inst_scores) == 0 :
            inst_scores.append(0)
        avg_score = np.mean(inst_scores)
        avg_steps = np.mean(inst_steps)
        avg_pct_optimal = (avg_score / optimal_weight * 100) if optimal_weight > 0 else 0.0
        
        summary_data.append({
            "Instance": filename,
            "Runs": n_runs,
            "Solved": inst_solved_count,
            "Optimal": inst_optimal_count,
            "Avg_Score": avg_score,
            "Avg_Pct_Opt": avg_pct_optimal,
            "Avg_Steps": avg_steps
        })

    # --- Finalize Layout ---
    fig.suptitle(f"{title}\nParams: {params}", fontsize=15, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Leave room for suptitle
    plt.show()

    # --- Print Table ---
    print("\n" + "="*95)
    print(f"{'Instance':<25} | {'Runs':<4} | {'Solved':<6} | {'Optimal':<7} | {'Avg Score':<10} | {'% of Opt':<8} | {'Avg Steps':<9}")
    print("-" * 95)
    for row in summary_data:
        print(f"{row['Instance']:<25} | {row['Runs']:<4} | {row['Solved']:<6} | {row['Optimal']:<7} | {row['Avg_Score']:<10.1f} | {row['Avg_Pct_Opt']:<6.2f}%  | {row['Avg_Steps']:<9.1f}")
    print("="*95)
    
    return summary_data



def plot_algorithm_run(instance, solutions_dict, params, ax=None, title=None):
    """
    Runs the algorithm on the given instance and plots the history on a Matplotlib axes.
    Matches the visual style of the grid evaluation (Status Box, Optima Line, etc.).

    Args:
        instance (MWSATInstance): The problem instance object to solve.
        solutions_dict (dict): Dictionary mapping instance names to optimal weights.
        params (dict): Dictionary of hyperparameters for the simulated_annealing function.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. Creates new fig if None.
        title (str, optional): Custom title. If None, uses the instance filename.
    """
    
    # 1. Run the Algorithm
    # We run it here inside the function as requested
    best_state, history = simulated_annealing(instance, **params)

    # 2. Extract Data for Plotting
    filename = os.path.basename(instance.filepath)
    instance_key = filename.split(".")[0]
    optima = solutions_dict.get(instance_key, 0) # Default to 0 if unknown
    
    # Determine Status
    is_solved = (best_state.clauses_satisfied == instance.num_clauses)
    final_score = best_state.current_score

    # 3. Setup Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # 4. Plot Trace
    ax.plot(history, color='tab:blue', linewidth=1, alpha=0.9, label='Run History')
    
    # 5. Plot Optimum Line
    if optima > 0:
        ax.axhline(y=optima, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimum')
    
    # 6. Status Box Logic
    if is_solved:
        status_color = "green"
        status_text = "SOLVED"
    else:
        status_color = "red"
        status_text = "STUCK"

    # Box Text
    info_text = f"{status_text}\nFinal: {final_score}"
    if optima > 0:
        info_text += f"\nOpt: {optima}"
    
    # Dynamic Positioning: Move box to bottom if the curve is high up
    text_y = 0.05 if (optima > 0 and final_score > optima * 0.85) else 0.85
    
    ax.text(0.05, text_y, info_text, transform=ax.transAxes, 
            bbox=dict(facecolor='white', edgecolor=status_color, linewidth=2, alpha=0.9),
            fontsize=10, fontweight='bold', color=status_color)
    
    # 7. Styling
    plot_title = title if title else f"{filename}"
    ax.set_title(plot_title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Steps', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    if optima > 0:
        ax.legend(loc='upper right')
    
    return ax

def get_solution_dict(filepath):
    result_dict = {}
    with open(filepath) as file:
        lines = [line.rstrip() for line in file]

        for line in lines:
            tokens = line.split(" ")
            result_dict["w"+tokens[0]] = int(tokens[1])
    return result_dict


def run_tuning_experiment(base_params, param_name, param_values, instance_path, optimal_weight, n_runs=4):
    """
    Runs a White Box experiment to tune a specific parameter.
    
    Args:
        base_params (dict): Dictionary of fixed parameters (e.g., {'cooling_coefficient': 0.95, ...})
        param_name (str): The name of the parameter to tune (e.g., "P0")
        param_values (list): List of values to test for that parameter
        instance_path (str): Path to the .mwcnf instance file
        optimal_weight (int): The known optimal weight for this instance
        n_runs (int): Number of runs per parameter value (default 4)
    """
    
    # --- Setup Canvas ---
    n_rows = len(param_values)
    n_cols = n_runs
    
    # Dynamic Figure Size: 4 inches height per row
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 4 * n_rows), sharey='row', sharex=True)
    
    # If there is only 1 row, axes is a 1D array, make it 2D for consistency
    if n_rows == 1: axes = np.array([axes])
    
    filename = os.path.basename(instance_path)
    summary_data = []
    
    print(f"--- Tuning '{param_name}' on {filename} ({n_rows * n_cols} total runs) ---")
    
    # --- Main Loops ---
    for i, val in enumerate(param_values):
        print(f"Testing {param_name} = {val}...")
        
        # Stats tracking for this parameter value
        val_scores = []
        val_steps = []
        val_solved = 0
        val_optimal = 0
        
        for j in range(n_cols):
            ax = axes[i, j]
            
            # 1. Merge Base Params with Current Tuning Value
            current_params = base_params.copy()
            current_params[param_name] = val
            
            # 2. Run Algorithm
            instance = MWSATInstance(instance_path)
            best_state, history = simulated_annealing(instance, **current_params)
            
            # 3. Collect Data
            is_solved = (best_state.clauses_satisfied == instance.num_clauses)
            is_optimal = is_solved and (best_state.current_score >= optimal_weight)
            
            if is_solved: val_scores.append(best_state.current_score)
            else: val_scores.append(0)
            val_steps.append(len(history))
            if is_solved: val_solved += 1
            if is_optimal: val_optimal += 1
            
            # 4. Plot
            ax.plot(history, color='tab:blue', linewidth=1, alpha=0.9)
            if optimal_weight > 0:
                ax.axhline(y=optimal_weight, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimum')
            
            # 5. Text & Styling
            status_color = "green" if is_solved else "red"
            status_text = "SOLVED" if is_solved else "STUCK"
            
            ax.text(0.05, 0.05, f"{status_text}\nScore: {best_state.current_score}", 
                    transform=ax.transAxes, 
                    bbox=dict(facecolor='white', edgecolor=status_color, linewidth=2, alpha=0.9),
                    fontsize=9, fontweight='bold', color=status_color)
            
            ax.grid(True, linestyle=':', alpha=0.5)
            
            if j == 0:
                ax.set_ylabel(f"{param_name} = {val}", fontsize=14, fontweight='bold', labelpad=15)
            if i == 0:
                ax.set_title(f"Run {j+1}", fontsize=12, fontweight='bold')
            if i == n_rows - 1:
                ax.set_xlabel("Steps", fontsize=10)

        # Calculate Averages for Table
        avg_score = np.mean(val_scores)
        avg_pct = (avg_score / optimal_weight * 100) if optimal_weight > 0 else 0
        
        summary_data.append({
            "Val": val,
            "Runs": n_cols,
            "Solved": val_solved,
            "Optimal": val_optimal,
            "Avg_Score": avg_score,
            "Avg_Pct": avg_pct,
            "Avg_Steps": np.mean(val_steps)
        })

    # Final Layout
    fig.suptitle(f"Tuning Parameter: {param_name} (Instance: {filename})\nOptimal: {optimal_weight}", fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Print Table ---
    print("\n" + "="*95)
    print(f"{param_name:<15} | {'Runs':<4} | {'Solved':<6} | {'Optimal':<7} | {'Avg Score':<10} | {'% of Opt':<8} | {'Avg Steps':<9}")
    print("-" * 95)
    for row in summary_data:
        print(f"{str(row['Val']):<15} | {row['Runs']:<4} | {row['Solved']:<6} | {row['Optimal']:<7} | {row['Avg_Score']:<10.1f} | {row['Avg_Pct']:<6.2f}%  | {row['Avg_Steps']:<9.1f}")
    print("="*95)

    
def evaluate_param_tuning_no_plot(instance_paths, solutions_dict, base_params, param_name, param_values, n_runs_per_instance=4):
    """
    Evaluates a parameter's performance across multiple instances without plotting.
    Prints a summary table aggregating the results.

    Args:
        instance_paths (list): List of paths to .mwcnf instance files.
        solutions_dict (dict): Dictionary mapping instance names (no ext) to optimal weights.
        base_params (dict): The fixed baseline parameters for the algorithm.
        param_name (str): The name of the parameter to tune (e.g., "P0").
        param_values (list): List of values to test for the parameter.
        n_runs_per_instance (int): Number of runs to perform per instance per parameter value.
    """
    
    print(f"\n--- Tuning Parameter: '{param_name}' (Across {len(instance_paths)} instances) ---")
    
    summary_rows = []

    # 1. Iterate through each parameter value we want to test
    for val in param_values:
        # Aggregators for this specific parameter value across ALL instances
        total_runs = 0
        total_solved = 0
        total_optimal = 0
        all_pct_opts = []
        all_steps = []
        
        # 2. Run on every instance in the list
        for path in instance_paths:
            filename = os.path.basename(path)
            key = filename.split(".")[0]
            optimal_weight = solutions_dict.get(key, 0)
            
            # Load instance
            instance = MWSATInstance(path) 
            
            # 3. Repeat N times per instance to smooth out randomness
            for _ in range(n_runs_per_instance):
                # Setup Params: Merge base with current test value
                current_params = base_params.copy()
                current_params[param_name] = val
                
                # Run Algorithm
                best_state, history = simulated_annealing(instance, **current_params)
                
                # Calculate Metrics
                is_solved = (best_state.clauses_satisfied == instance.num_clauses)
                is_optimal = is_solved and (optimal_weight > 0) and (best_state.current_score >= optimal_weight)
                
                pct_opt = 0.0
                if optimal_weight > 0:
                    pct_opt = (best_state.current_score / optimal_weight) * 100
                
                # Update Aggregates
                total_runs += 1
                if is_solved: total_solved += 1
                if is_optimal: total_optimal += 1
                if is_solved: all_pct_opts.append(pct_opt)
                all_steps.append(len(history))
        
        # 4. Summarize stats for this parameter value
        avg_pct = np.mean(all_pct_opts) if all_pct_opts else 0
        avg_steps = np.mean(all_steps) if all_steps else 0
        
        summary_rows.append({
            "Val": val,
            "Runs": total_runs,
            "Solved": total_solved,
            "Optimal": total_optimal,
            "Avg_Pct_Opt": avg_pct,
            "Avg_Steps": avg_steps
        })

    # 5. Print Summary Table
    print("\n" + "="*90)
    print(f" {param_name:<15} | {'Runs':<6} | {'Solved':<8} | {'Optimal':<8} | {'Avg % Opt':<12} | {'Avg Steps':<10}")
    print("-" * 90)
    for row in summary_rows:
        # Helper for clean formatting of the value
        val_str = str(row['Val'])
        print(f" {val_str:<15} | {row['Runs']:<6} | {row['Solved']:<8} | {row['Optimal']:<8} | {row['Avg_Pct_Opt']:<12.2f} | {row['Avg_Steps']:<10.1f}")
    print("="*90)
    
    return summary_rows