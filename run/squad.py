import os
import sys
import torch
import cvxpy as cp
import json
from tqdm import tqdm
import time
import numpy as np
import datetime 
from scipy.special import comb
import concurrent.futures
import random
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from structures import squad
from data import squad_utils
from models import hf_inference
from utils import coverage_plots, pred_set_size_plots

# ======= User-Defined Parameters =======
# This section allows users to set custom parameters for the experiment.
SEEDS = [2, 42, 123, 516, 2024]
CALIBRATION_RATIO = 0.5
MAX_WORKERS = 500

# Year range
MIN_YEAR = 1970
MAX_YEAR = 2020

# Coverage guarantee
GUARANTEE = "marginal"   # "marginal" or "pac"

# Hyperparameters
EPSILON_VALUES = [0.05, 0.1, 0.2, 0.4]
PAC_DELTA_VALUES = [0.1, 0.01, 0.001]
M_VALUES = [1, 2, 4, 8]

# Threshold candidates
THRESHOLDS = [1, 0.999, 0.995, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
# =======================================



def calculate_leaf_probs(start_year, end_year, filtered_example_dicts, year_dag):
    generation_filename = f'../collected/squad/years_{start_year}_to_{end_year}_n{len(filtered_example_dicts)}.json'
    example_probs_dicts = []
    if not os.path.exists(generation_filename):
        model = hf_inference.HFInferenceModel(model_name='meta-llama/Meta-Llama-3.1-70B-Instruct')
        print(f"Computing probs for years {start_year} to {end_year} ...")
        for i, example_dict in enumerate(tqdm(filtered_example_dicts, desc="Computing probs")):
            example_probs_dict = _calculate_leaf_probs_helper(model, year_dag, example_dict)
            example_probs_dicts.append(example_probs_dict)
        with open(generation_filename, 'w') as f:
            json.dump(example_probs_dicts, f, indent=4)
        print(f"Saved {len(example_probs_dicts)} examples w/ probs to {generation_filename}.")
    else:
        with open(generation_filename, 'r') as f:
            example_probs_dicts = json.load(f)
    print(f"Loaded {len(example_probs_dicts)} examples w/ probs.")
    return example_probs_dicts



def _calculate_leaf_probs_helper(model, year_dag, example_dict):
    leaves = year_dag.get_leaves()
    leaf_log_prob_map = {}
    prompt = example_dict['prompt']

    # Process leaves in batches of 25
    batch_size = 25
    for i in range(0, len(leaves), batch_size):
        batch_leaves = leaves[i:i+batch_size]
        batch_prompts = [prompt] * len(batch_leaves)
        batch_years = [str(leaf.label[0]) for leaf in batch_leaves]
        
        # Pad the batch if it's not full
        while len(batch_years) < batch_size:
            batch_prompts.append(prompt)
            batch_years.append(batch_years[-1])  # Duplicate the last year for padding
        
        log_probs = model.get_log_prob(batch_prompts, batch_years)
        
        # Assign log probabilities to leaves
        for j, leaf in enumerate(batch_leaves):
            leaf_log_prob_map[leaf.label[0]] = log_probs[j]

    example_dict['log_probs'] = leaf_log_prob_map
    return example_dict



def add_leaves_prob(example_dict, year_dag):
    leaves = year_dag.get_leaves()
    log_probs_map = example_dict['log_probs']

    log_probs_pairs = list(log_probs_map.items())
    years, log_probs = zip(*log_probs_pairs)
    log_probs_tensor = torch.tensor(log_probs)
    softmax_probs = torch.nn.functional.softmax(log_probs_tensor, dim=0)
    softmax_probs_map = {years[i]: softmax_probs[i].item() for i in range(len(years))}

    for leaf in leaves:
        leaf.prob.value = softmax_probs_map[str(leaf.label[0])]



def generate_pred_sets(example_probs_dicts, tau, tauParam, year_dag, problem, warm_start_values):
    solve_times = [] # time for solving each integer programming problem
    tauParam.value = tau
    pred_sets = []
    labels = []
    for j, example_dict in enumerate(example_probs_dicts):
        add_leaves_prob(example_dict, year_dag)
        if warm_start_values is not None:
            for node in year_dag.get_nodes():
                if node in warm_start_values:
                    node.alpha.value = int(warm_start_values[node].get('alpha', None))
                    node.beta.value = int(warm_start_values[node].get('beta', None))
        
        start_time = time.time()
        problem.solve(solver=cp.GUROBI, warm_start=True)
        end_time = time.time()
        solve_time = end_time - start_time  
        solve_times.append(solve_time)

        pred_set = [node.label for node in year_dag.get_nodes() if node.alpha.value >= 0.5]
        pred_sets.append(pred_set)
        labels.append(example_dict['answer'])
    
        warm_start_values = {
            node: {'alpha': node.alpha.value, 'beta': node.beta.value} for node in year_dag.get_nodes()
        }
    return pred_sets, labels, sum(solve_times) / len(solve_times)



def compute_coverage_and_avg_size(pred_sets, labels):
    coverage = []
    sizes = []
    for pred_set_list, label in zip(pred_sets, labels):
        label_set = set()
        label_set.update(label)

        pred_set = set()
        for pred in pred_set_list:
            for p in range(pred[0], pred[1] + 1):
                pred_set.add(p)
        
        if any(l in pred_set for l in label_set):
            coverage.append(1)
        else:
            coverage.append(0)
        sizes.append(len(pred_set))
    return coverage, sizes



def evaluate_model(example_dicts):
    coverage = []
    correct_years = []
    for example_dict in example_dicts:
        labels = set()
        labels.update(example_dict['answer'])

        log_probs_pairs = list(example_dict['log_probs'].items())
        years, log_probs = zip(*log_probs_pairs)
        log_probs_tensor = torch.tensor(log_probs)
        softmax_probs = torch.nn.functional.softmax(log_probs_tensor, dim=0)
        softmax_probs_map = {years[i]: softmax_probs[i].item() for i in range(len(years))}

        most_probable_year = max(softmax_probs_map, key=softmax_probs_map.get)
        coverage.append(int(most_probable_year) in labels)

        if int(most_probable_year) in labels:
            correct_years.append(int(most_probable_year))

    accuracy = sum(coverage) / len(coverage)
    print(f"SQuAD year accuracy: {accuracy}")
    return accuracy



def read_worker(results_path, qualitative_samples, start_year, end_year, test_probs_dicts, epsilon, delta, m):
    df = pd.read_csv(results_path)

    filtered_df = df[(df['seed'] == 2) & 
                     (df['epsilon'] == epsilon) & 
                     (df['delta'] == delta) & 
                     (df['m'] == m)]

    if filtered_df.shape[0] != 1:
        raise ValueError("Filtered DataFrame does not contain exactly one row. Check input parameters or CSV data.")
    
    threshold = filtered_df['threshold'].values[0]

    # Construct year tree
    year_dag = squad.construct_year_dag(start_year, end_year)
    warm_start_values = None

    # Build constraint problem
    mParam = cp.Parameter(integer=True)
    tauParam = cp.Parameter(nonneg=True)
    problem = year_dag.build_constraint_problem(mParam, tauParam)
    mParam.value = m

    test_pred_sets, _, _ = generate_pred_sets(test_probs_dicts, threshold, tauParam, year_dag, problem, warm_start_values)
    for i in range(len(test_pred_sets)):
        if GUARANTEE == "marginal":
            key = f"e={epsilon},m={m}"
        elif GUARANTEE == "pac":
            key = f"e={epsilon},m={m},d={delta}"
        qualitative_samples[i][key] = test_pred_sets[i]



def compute_worker(start_year, end_year, cal_probs_dicts, test_probs_dicts, epsilon, delta, m, seed):
    print(f"({seed}) Starting experiment for m={m}, epsilon={epsilon}, delta={delta} ...")
    # Construct DAG
    year_dag = squad.construct_year_dag(start_year, end_year)
    warm_start_values = None

    # Build constraint problem
    mParam = cp.Parameter(integer=True)
    tauParam = cp.Parameter(nonneg=True)
    problem = year_dag.build_constraint_problem(mParam, tauParam)
    mParam.value = m

    # Compute threshold given input m, epsilon, (and delta) on calibration data
    n_cal = len(cal_probs_dicts)
    threshold_idx = -1
    cal_coverage = None
    cal_avg_size = None
    phi = np.zeros(len(THRESHOLDS))

    start_time = time.time()
    for i, tau in enumerate(THRESHOLDS):
        cal_pred_sets, cal_labels, _ = generate_pred_sets(cal_probs_dicts, tau, tauParam, year_dag, problem, warm_start_values)
        coverage, size = compute_coverage_and_avg_size(cal_pred_sets, cal_labels)

        def compute_h(n, epsilon, delta):
            cumulative_sum = 0  
            h_hat = 0
            for h in range(n + 1):  
                binomial_term = comb(n, h) * (epsilon ** h) * ((1 - epsilon) ** (n - h))
                cumulative_sum += binomial_term
                if cumulative_sum >= delta:
                    break  
                h_hat = h
            return h_hat

        if GUARANTEE == 'marginal':
            phi[i] = (n_cal - sum(coverage)) <= (epsilon * (n_cal + 1))
        elif GUARANTEE == 'pac':
            phi[i] = (n_cal - sum(coverage)) <= compute_h(n_cal, epsilon, delta)

        # Validate tau
        if np.sum(phi[:i+1] == 0) <= 0:
            threshold_idx = i
            cal_coverage = coverage
            cal_avg_size = size
        else:
            end_time = time.time()
            threshold_avg_estimate_time = end_time - start_time  
            if i == 0:
                raise Exception("No valid threshold candidate found. Try using larger threshold values.")
            threshold = THRESHOLDS[threshold_idx]
            print(f"({seed}) Found threshold={threshold} given m={m}, epsilon={epsilon}, delta={delta} ...")

            # Compute coverage rate and average size for testing data
            test_pred_sets, test_labels, avg_problem_solve_time = generate_pred_sets(test_probs_dicts, threshold, tauParam, year_dag, problem, warm_start_values)
            test_coverage, test_avg_size = compute_coverage_and_avg_size(test_pred_sets, test_labels)
            print(f"with test coverage: {np.mean(test_coverage)}, test avg_size: {np.mean(test_avg_size)}.")

            result = [seed, epsilon, delta, m, threshold, np.mean(cal_coverage), np.mean(cal_avg_size), np.mean(test_coverage), np.mean(test_avg_size), threshold_avg_estimate_time, avg_problem_solve_time]
            detailed_result = []
            for test_c, test_s in zip(test_coverage, test_avg_size):
                detailed_result.append([seed, epsilon, delta, m, threshold, test_c, test_s, np.mean(test_coverage), np.mean(test_avg_size)])
            return result, detailed_result



def guarantee_tests(start_year, end_year, cal_probs_dicts, test_probs_dicts, seed):
    if GUARANTEE == "pac":
        delta_values = PAC_DELTA_VALUES
    else:
        delta_values = [-1]

    total_tasks = len(EPSILON_VALUES) * len(delta_values) * len(M_VALUES)
    completed_tasks = 0

    results = []
    detailed_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_params = {
            executor.submit(compute_worker, start_year, end_year, cal_probs_dicts, test_probs_dicts, epsilon, delta, m, seed): (epsilon, delta, m)
            for epsilon in EPSILON_VALUES
            for delta in delta_values
            for m in M_VALUES
        }
        for future in concurrent.futures.as_completed(future_to_params):
            epsilon, delta, m = future_to_params[future]
            result, detailed_result = future.result()
            results.append(result)
            detailed_results.extend(detailed_result)
            completed_tasks += 1
            print(f"({seed}) Completed task {completed_tasks}/{total_tasks}: m={m}, epsilon={epsilon}, delta={delta}.")
    return results, detailed_results



def save_qualitative_examples(qual_results_path, results_path, test_probs_dicts):
    if GUARANTEE == "pac":
        delta_values = PAC_DELTA_VALUES
    else:
        delta_values = [-1]

    total_tasks = len(EPSILON_VALUES) * len(delta_values) * len(M_VALUES)
    completed_tasks = 0

    qualitative_samples = []
    for test_dicts_prob in test_probs_dicts:
        qualitative_sample = {}
        qualitative_sample["question"] = test_dicts_prob["question"]
        qualitative_sample["answer"] = test_dicts_prob["answer"]
        qualitative_samples.append(qualitative_sample)

    for epsilon in EPSILON_VALUES:
        for m in M_VALUES:
            for delta in delta_values:
                read_worker(results_path, qualitative_samples, start_year, end_year, test_probs_dicts, epsilon, delta, m)
                completed_tasks += 1
                print(f"(Qual) Completed {completed_tasks}/{total_tasks} tasks.")
        
    with open(qual_results_path, 'w') as f:
        json.dump(qualitative_samples, f, indent=4)
    print(f"Saved qualitative samples to {qual_results_path} successfully.")

if __name__ == '__main__':
    # Load data
    example_dicts = squad_utils.load_data()
    print(f"Loaded {len(example_dicts)} SQuAD year problems.")

    # Filter problem according to year range
    filtered_example_dicts = []
    start_year = 10000
    end_year = -1 
    for example_dict in example_dicts:
        valid_example = False
        for answer in example_dict['answer']:
            if MIN_YEAR <= answer <= MAX_YEAR:
                start_year = min(start_year, answer)
                end_year = max(end_year, answer)
                valid_example = True
        if valid_example:
            filtered_example_dicts.append(example_dict)
    print(f"Filtered {len(filtered_example_dicts)} problems for years {MIN_YEAR} to {MAX_YEAR}.")

    # Construct DAG
    year_dag = squad.construct_year_dag(start_year, end_year)

    # Calculate and save leaves probability
    example_probs_dicts = calculate_leaf_probs(start_year, end_year, filtered_example_dicts, year_dag)
    model_accuracy = evaluate_model(example_probs_dicts)

    # Conduct guarantee tests
    seed_results = []
    seed_detailed_results = []
    for seed in SEEDS:
        print(f"Set seed={seed}.")
        random.seed(seed)
        indices = list(range(len(example_probs_dicts)))
        random.shuffle(indices)
        n_cal = int(CALIBRATION_RATIO * len(indices))
        cal_probs_dicts = [example_probs_dicts[i] for i in indices[:n_cal]]
        test_probs_dicts = [example_probs_dicts[i] for i in indices[n_cal:]]

        results, detailed_results = guarantee_tests(start_year, end_year, cal_probs_dicts, test_probs_dicts, seed)
        seed_results.append(results)
        seed_detailed_results.append(detailed_results)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save all results
    results_dir = "../experiments/squad"
    results_filename = f"{GUARANTEE}_years_{start_year}_to_{end_year}_calib{CALIBRATION_RATIO}_{timestamp}.csv"
    results_path = os.path.join(results_dir, results_filename)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(results_path, 'w') as f:
        f.write("seed,epsilon,delta,m,threshold,cal_coverage,cal_avg_size,test_coverage,test_avg_size,time_solve_threshold,avg_time_per_test\n")
        for seed_result in seed_results:
            for result in seed_result:
                f.write(','.join(map(str, result)) + '\n')
    print(f"Results saved to {results_filename} successfully.")

    # Save all detailed results
    detailed_results_dir = "../experiments/squad/details"
    detailed_results_filename = f"{GUARANTEE}_years_{start_year}_to_{end_year}_calib{CALIBRATION_RATIO}_details_{timestamp}.csv"
    detailed_results_path = os.path.join(detailed_results_dir, detailed_results_filename)
    if not os.path.exists(detailed_results_dir):
        os.makedirs(detailed_results_dir)
    with open(detailed_results_path, 'w') as f:
        f.write("seed,epsilon,delta,m,threshold,test_covered,test_size,test_coverage,test_avg_size\n")
        for seed_result in seed_detailed_results:
            for result in seed_result:
                f.write(','.join(map(str, result)) + '\n')
    print(f"Detailed results saved to {detailed_results_filename} successfully.")

    # Save qualitative examples
    random.seed(2)
    indices = list(range(len(example_probs_dicts)))
    random.shuffle(indices)
    n_cal = int(CALIBRATION_RATIO * len(indices))
    test_probs_dicts = [example_probs_dicts[i] for i in indices[n_cal:]]
    qual_results_dir = "../qual"
    qual_results_filename = f"years_{start_year}_to_{end_year}_{GUARANTEE}_seed2_qual.json"
    qual_results_path = os.path.join(qual_results_dir, qual_results_filename)
    if not os.path.exists(qual_results_dir):
        os.makedirs(qual_results_dir)
    save_qualitative_examples(qual_results_path, results_path, test_probs_dicts)

    # Plot results
    plots_dir = "../plots/squad"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    coverage_plots.coverage_holds_all_m("squad", results_path, model_accuracy=model_accuracy)
    pred_set_size_plots.size_vs_m("squad", results_path)
    if GUARANTEE == "pac":
        coverage_plots.coverage_holds_all_d("squad", results_path, model_accuracy=model_accuracy)
        pred_set_size_plots.size_vs_d("squad", results_path)
