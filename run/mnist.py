import os
import sys
import numpy as np
import cvxpy as cp
import datetime
from scipy.special import comb
import concurrent.futures
import random
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from structures import mnist
from utils import coverage_plots, pred_set_size_plots

# ======= User-Defined Parameters =======
# This section allows users to set custom parameters for the experiment.
SEEDS = [2, 42, 123, 516, 2024]
CALIBRATION_RATIO = 0.2
MAX_WORKERS = 500

# MNIST k-digit
NUM_DIGITS = 2   # 2 or 3

# Number of samples
N = 1000

# Coverage guarantee
GUARANTEE = "marginal"   # "marginal" or "pac"

# Hyperparameters
EPSILON_VALUES = [0.05, 0.1, 0.2, 0.4]
PAC_DELTA_VALUES = [0.1, 0.01, 0.001]
M_VALUES = [1, 2, 4, 8]

# Threshold candidates
THRESHOLDS = [0.999, 0.995, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]
# =======================================



def get_prediction_set(digit_list, tau, m):
    label_tree = mnist.construct_mnist_tree(digit_list)
    problem = label_tree.build_constraint_problem(m, tau)
    start_time = time.time()
    problem.solve(solver=cp.GUROBI)
    end_time = time.time()
    solve_time = end_time - start_time
    return [node.label for node in label_tree.get_nodes() if node.alpha.value >= 0.5], solve_time

def eval_conf(conf, xs, tau):
    coverage = 0
    avg_size = 0
    for x in xs:
        cover, size = conf(x, tau)
        coverage += cover
        avg_size += size
    return coverage / len(xs), avg_size / len(xs)

def fit_conf(conf, xs, taus, eps):
    for tau in taus:
        print(tau)
        coverage, avg_size = eval_conf(conf, xs, tau)
        if coverage >= 1 - eps:
            return tau, avg_size
    return 100

def check_coverage(prediction_set, label):
    for prediction in prediction_set:
        if len(prediction) != len(label):
            continue
        if_match = True
        for p_char, l_char in zip(prediction, label):
            if p_char != 'X' and p_char != l_char:
                if_match = False
                break
        if if_match:
            return 1
    return 0

def compute_prediction_set_size(prediction_set):
    total_size = 0
    for prediction in prediction_set:
        prediction_size = 1
        for digit in prediction:
            if digit == 'X':
                prediction_size *= 10
        total_size += prediction_size
    return total_size

def evaluate_model(digit_lists):
    correct = 0
    total = 0

    for digit_list in digit_lists:
        label = ''.join(str(digit.label) for digit in digit_list)
        max_prob = 0
        predicted_label = None

        if NUM_DIGITS == 3:
            for d1 in range(10):
                for d2 in range(10):
                    for d3 in range(10):
                        prob_product = digit_list[0].probs[d1] * digit_list[1].probs[d2] * digit_list[2].probs[d3]
                        if prob_product > max_prob:
                            max_prob = prob_product
                            predicted_label = f"{d1}{d2}{d3}"
        elif NUM_DIGITS == 2:
            for d1 in range(10):
                for d2 in range(10):
                    prob_product = digit_list[0].probs[d1] * digit_list[1].probs[d2]
                    if prob_product > max_prob:
                        max_prob = prob_product
                        predicted_label = f"{d1}{d2}"

        if predicted_label == label:
            correct += 1
        total += 1
    accuracy = correct / total
    print(f"MNIST {NUM_DIGITS}-digit problem accuracy: ", accuracy)
    return accuracy

def compute_worker(cal_digit_lists, test_digit_lists, epsilon, delta, m, seed):
    print(f"({seed}) Starting experiment for  m={m}, epsilon={epsilon}, delta={delta} ...")
    # Compute threshold given input m, epsilon, (and delta) on calibration data
    n_cal = len(cal_digit_lists)
    threshold_idx = -1
    cal_coverage = None
    cal_avg_size = None
    phi = np.zeros(len(THRESHOLDS))

    start_time = time.time()
    for i, tau in enumerate(THRESHOLDS):
        coverage = []
        size = []
        for j, digit_list in enumerate(cal_digit_lists):
            prediction_set, _ = get_prediction_set(digit_list, tau, m)
            label = ''.join(str(digit.label) for digit in digit_list)
            coverage.append(check_coverage(prediction_set, label))
            size.append(compute_prediction_set_size(prediction_set))

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
                raise Exception("No valid tau candidate.")
            threshold = THRESHOLDS[threshold_idx]
            print(f"({seed}) Found threshold={threshold} given m={m}, epsilon={epsilon}, delta={delta} ...")

            # Compute coverage rate and average size for testing data
            test_coverage = []
            test_avg_size = []
            solve_times = [] # each element is the time of solving one integer programming problem
            for j, digit_list in enumerate(test_digit_lists):
                prediction_set, solve_time = get_prediction_set(digit_list, threshold, m)
                solve_times.append(solve_time)
                label = ''.join(str(digit.label) for digit in digit_list)
                test_coverage.append(check_coverage(prediction_set, label))
                test_avg_size.append(compute_prediction_set_size(prediction_set))
            print(f"with test coverage: {np.mean(test_coverage)}, test avg_size: {np.mean(test_avg_size)}.")

            avg_problem_solve_time = sum(solve_times) / len(solve_times)

            result = [seed, epsilon, delta, m, threshold, np.mean(cal_coverage), np.mean(cal_avg_size), np.mean(test_coverage), np.mean(test_avg_size), threshold_avg_estimate_time, avg_problem_solve_time]
            detailed_result = []
            for test_c, test_s in zip(test_coverage, test_avg_size):
                detailed_result.append([seed, epsilon, delta, m, threshold, test_c, test_s, np.mean(test_coverage), np.mean(test_avg_size)])
            return result, detailed_result
    print(f"*** Did not find threshold for epsilon={epsilon}, delta={delta}, m={m}")

def guarantee_tests(cal_digit_lists, test_digit_lists, seed):
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
            executor.submit(compute_worker, cal_digit_lists, test_digit_lists, epsilon, delta, m, seed): (epsilon, delta, m)
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
            print(f"({seed}) Completed task {completed_tasks}/{total_tasks}:  m={m}, epsilon={epsilon}, delta={delta}.")
    return results, detailed_results

if __name__ == '__main__':
    digit_lists = mnist.construct_mnist_lists(NUM_DIGITS)
    model_accuracy = evaluate_model(digit_lists)

    # Conduct guarantee tests
    seed_results = []
    seed_detailed_results = []
    for seed in SEEDS:
        print(f"Set seed={seed}.")
        random.seed(seed)
        indices = list(range(len(digit_lists)))
        random.shuffle(indices)
        n_cal = int(CALIBRATION_RATIO * N)
        cal_digit_lists = [digit_lists[i] for i in indices[:n_cal]]
        test_digit_lists = [digit_lists[i] for i in indices[n_cal:N]]

        results, detailed_results = guarantee_tests(cal_digit_lists, test_digit_lists, seed)
        seed_results.append(results)
        seed_detailed_results.append(detailed_results)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all results
    results_dir = "../experiments/mnist"
    results_filename = f"{GUARANTEE}_{NUM_DIGITS}digits_n{N}_calib{CALIBRATION_RATIO}_{timestamp}.csv"
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
    detailed_results_dir = "../experiments/mnist/details"
    detailed_results_filename = f"{GUARANTEE}_{NUM_DIGITS}digits_n{N}_calib{CALIBRATION_RATIO}_details_{timestamp}.csv"
    detailed_results_path = os.path.join(detailed_results_dir, detailed_results_filename)
    if not os.path.exists(detailed_results_dir):
        os.makedirs(detailed_results_dir)
    with open(detailed_results_path, 'w') as f:
        f.write("seed,epsilon,delta,m,threshold,test_covered,test_size,test_coverage,test_avg_size\n")
        for seed_result in seed_detailed_results:
            for result in seed_result:
                f.write(','.join(map(str, result)) + '\n')
    print(f"Detailed results saved to {detailed_results_filename} successfully.")

    # Plot results
    plots_dir = f"../plots/mnist/{NUM_DIGITS}digits"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    coverage_plots.coverage_holds_all_m("mnist", results_path, model_accuracy=model_accuracy, num_digits=NUM_DIGITS)
    pred_set_size_plots.size_vs_m("mnist", results_path, num_digits=NUM_DIGITS)
    if GUARANTEE == "pac":
        coverage_plots.coverage_holds_all_d("mnist", results_path, model_accuracy=model_accuracy, num_digits=NUM_DIGITS)
        pred_set_size_plots.size_vs_d("mnist", results_path, num_digits=NUM_DIGITS)
