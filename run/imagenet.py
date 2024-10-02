import os
import sys
import numpy as np
import cvxpy as cp
import json
from torchvision.models import resnet50, ResNet50_Weights
import torchvision
import torch
import datetime
import time
from scipy.special import comb
import concurrent.futures
import random
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import imagenet_utils
from structures import imagenet
from utils import coverage_plots, pred_set_size_plots

# ======= User-Defined Parameters =======
# This section allows users to set custom parameters for the experiment.
SEEDS = [2, 42, 123, 516, 2024]
CALIBRATION_RATIO = 0.2
MAX_WORKERS = 500

# Number of samples
N = 1000

# Coverage guarantee
GUARANTEE = "marginal"   # "marginal" or "pac"

# Hyperparameters
EPSILON_VALUES = [0.05, 0.1, 0.2, 0.4]
PAC_DELTA_VALUES = [0.1, 0.01, 0.001]
M_VALUES = [1, 2, 4, 8]

# Threshold candidates
THRESHOLDS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
# =======================================



def add_leaves_prob(prob, hierarchical_tree, category_names):
    leaves = hierarchical_tree.get_leaves()
    for leaf in leaves:
        leaf.prob.value = prob[category_names.index(leaf.label)]



def generate_pred_sets(probs, tau, tauParam, hierarchical_tree, category_names, problem, warm_start_values):
    solve_times = [] # time for solving each integer programming problem
    tauParam.value = tau
    pred_sets = []
    for j, prob in enumerate(probs):
        add_leaves_prob(prob, hierarchical_tree, category_names)
        if warm_start_values is not None:
            for node in hierarchical_tree.get_nodes():
                if node in warm_start_values:
                    node.alpha.value = int(warm_start_values[node].get('alpha', None))
                    node.beta.value = int(warm_start_values[node].get('beta', None))

        start_time = time.time()
        problem.solve(solver=cp.GUROBI, warm_start=True)
        end_time = time.time()
        solve_time = end_time - start_time  
        solve_times.append(solve_time)

        pred_set = [node.label for node in hierarchical_tree.get_nodes() if node.alpha.value >= 0.5]
        pred_sets.append(pred_set)

        warm_start_values = {
            node: {'alpha': node.alpha.value, 'beta': node.beta.value} for node in hierarchical_tree.get_nodes()
        }

    return pred_sets, sum(solve_times) / len(solve_times)



def compute_coverage_and_avg_size(pred_sets, labels, tree_nodes_dict, category_names):
    coverage = []
    sizes = []
    for i, pred_set in enumerate(pred_sets):
        label_name = category_names[labels[i]]
        leaves_name = []
        for pred in pred_set:
            for leaf in tree_nodes_dict[pred].get_leaves():
                leaves_name.append(leaf.label)
        
        sizes.append(len(leaves_name))
        coverage.append(1 if label_name in leaves_name else 0)
    
    return coverage, sizes



def evaluate_model(probs, labels):
    predictions = np.argmax(probs, axis=1)
    accuracy = np.mean(predictions == labels)
    print(f"ImageNet accuracy: {accuracy}%")
    return accuracy



def read_worker(results_path, qualitative_samples, hierarchy, category_names, test_probs, test_labels, epsilon, delta, m):
    df = pd.read_csv(results_path)

    filtered_df = df[(df['seed'] == 2) & 
                     (df['epsilon'] == epsilon) & 
                     (df['delta'] == delta) & 
                     (df['m'] == m)]
    
    if filtered_df.shape[0] != 1:
        raise ValueError("Filtered DataFrame does not contain exactly one row. Check input parameters or CSV data.")
    
    threshold = filtered_df['threshold'].values[0]

    # Construct hierarchical tree
    hierarchical_tree = imagenet.construct_imagenet_tree(hierarchy)
    tree_nodes_dict = {}
    for n in hierarchical_tree.get_nodes():
        tree_nodes_dict[n.label] = n
    warm_start_values = None 

    # Build constraint problem
    mParam = cp.Parameter(integer=True)
    tauParam = cp.Parameter(nonneg=True)
    problem = hierarchical_tree.build_constraint_problem(mParam, tauParam)
    mParam.value = m

    test_pred_sets, _ = generate_pred_sets(test_probs, threshold, tauParam, hierarchical_tree, category_names, problem, warm_start_values)
    for i in range(len(qualitative_samples)):
        if GUARANTEE == "marginal":
            key = f"e={epsilon},m={m}"
        elif GUARANTEE == "pac":
            key = f"e={epsilon},m={m},d={delta}"
        if qualitative_samples[i]['label'] == category_names[test_labels[i]]:
            qualitative_samples[i][key] = test_pred_sets[i]
        else:
            raise Exception("Encountered mismatched label name.")
        


def compute_worker(hierarchy, category_names, cal_probs, cal_labels, test_probs, test_labels, epsilon, delta, m, seed):
    print(f"({seed}) Starting experiment for  m={m}, epsilon={epsilon}, delta={delta} ...")
    # Construct hierarchical tree
    hierarchical_tree = imagenet.construct_imagenet_tree(hierarchy)
    tree_nodes_dict = {}
    for n in hierarchical_tree.get_nodes():
        tree_nodes_dict[n.label] = n
    warm_start_values = None 

    # Build constraint problem
    mParam = cp.Parameter(integer=True)
    tauParam = cp.Parameter(nonneg=True)
    problem = hierarchical_tree.build_constraint_problem(mParam, tauParam)
    mParam.value = m

    # Compute threshold given input m, epsilon, (and delta) on calibration data
    n_cal = len(cal_probs)
    threshold_idx = -1
    cal_coverage = None
    cal_avg_size = None
    phi = np.zeros(len(THRESHOLDS))

    start_time = time.time()
    for i, tau in enumerate(THRESHOLDS):
        cal_pred_sets, _ = generate_pred_sets(cal_probs, tau, tauParam, hierarchical_tree, category_names, problem, warm_start_values)
        coverage, size = compute_coverage_and_avg_size(cal_pred_sets, cal_labels, tree_nodes_dict, category_names)
        
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
            test_pred_sets, avg_problem_solve_time = generate_pred_sets(test_probs, threshold, tauParam, hierarchical_tree, category_names, problem, warm_start_values)
            test_coverage, test_avg_size = compute_coverage_and_avg_size(test_pred_sets, test_labels, tree_nodes_dict, category_names)
            print(f"with test coverage: {np.mean(test_coverage)}, test avg_size: {np.mean(test_avg_size)}.")

            result = [seed, epsilon, delta, m, threshold, np.mean(cal_coverage), np.mean(cal_avg_size), np.mean(test_coverage), np.mean(test_avg_size), threshold_avg_estimate_time, avg_problem_solve_time]
            detailed_result = []
            for test_c, test_s in zip(test_coverage, test_avg_size):
                detailed_result.append([seed, epsilon, delta, m, threshold, test_c, test_s, np.mean(test_coverage), np.mean(test_avg_size)])
            return result, detailed_result



def guarantee_tests(hierarchy, category_names, cal_probs, cal_labels, test_probs, test_labels, seed):
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
            executor.submit(compute_worker, hierarchy, category_names, cal_probs, cal_labels, test_probs, test_labels, epsilon, delta, m, seed): (epsilon, delta, m)
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



def save_qualitative_examples(qual_results_path, results_path, hierarchy, category_names, test_probs, test_labels, image_indices):
    if GUARANTEE == "pac":
        delta_values = PAC_DELTA_VALUES
    else:
        delta_values = [-1]

    total_tasks = len(EPSILON_VALUES) * len(delta_values) * len(M_VALUES)
    completed_tasks = 0
    
    qualitative_samples = []
    for i, test_label in enumerate(test_labels):
        qualitative_sample = {}
        qualitative_sample["label"] = category_names[test_label]
        qualitative_sample["image_idx"] = image_indices[i]
        qualitative_samples.append(qualitative_sample)

    for epsilon in EPSILON_VALUES:
        for m in M_VALUES:
            for delta in delta_values:
                read_worker(results_path, qualitative_samples, hierarchy, category_names, test_probs, test_labels, epsilon, delta, m)
                completed_tasks += 1
                print(f"Completed {completed_tasks}/{total_tasks} tasks.")

    with open(qual_results_path, 'w') as f:
        json.dump(qualitative_samples, f, indent=4)
    print(f"Saved qualitative samples to {qual_results_path} successfully.")

if __name__ == '__main__':
    weights = ResNet50_Weights.DEFAULT
    category_names = weights.meta["categories"]

    if os.path.exists('../collected/imagenet/predictions.npy') and os.path.exists('../collected/imagenet/labels.npy'):
        with open('../collected/imagenet/predictions.npy', 'rb') as f:
            probs = np.load(f) # 25000 x 1000
        with open('../collected/imagenet/labels.npy', 'rb') as f:
            labels = np.load(f) # 25000
    else:
        # Step 1: Initialize model with the best available weights
        model = resnet50(weights=weights).to("cuda")
        model.eval()

        # Step 2: Initialize the inference transforms
        preprocess = weights.transforms()

        imagenet_data = torchvision.datasets.ImageNet('../data/imagenet', split='val', transform=preprocess)
        data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                  batch_size=64,
                                                  shuffle=True)
        probs_list = []
        labels_list = []
        with torch.no_grad():
            for i, (images, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
                # input images to the model
                pred = model(images.cuda())
                # compute softmax probabilities
                pred = pred.softmax(1)
                probs_list.append(pred.cpu().detach().numpy())
                labels_list.append(labels.numpy())

        probs = np.concatenate(probs_list)
        labels = np.concatenate(labels_list)

        with open('../collected/imagenet/predictions.npy', 'wb') as f:
            np.save(f, probs)
        with open('../collected/imagenet/labels.npy', 'wb') as f:
            np.save(f, labels)
        print(f"Saved ImageNet data.")

    print(f"Loaded ImageNet data.")
    model_accuracy = evaluate_model(probs, labels)

    # Get all parent labels
    hierarchy = imagenet_utils.read_hierarchy(category_names)
    parents = imagenet_utils.find_all_parents(hierarchy)

    # Conduct guarantee tests
    seed_results = []
    seed_detailed_results = []
    for seed in SEEDS:
        print(f"Set seed={seed}.")
        random.seed(seed)
        indices = list(range(len(probs)))
        random.shuffle(indices)
        n_cal = int(CALIBRATION_RATIO * N)
        cal_probs = [probs[i] for i in indices[:n_cal]]
        cal_labels = [labels[i] for i in indices[:n_cal]]
        test_probs = [probs[i] for i in indices[n_cal:N]]
        test_labels = [labels[i] for i in indices[n_cal:N]]

        results, detailed_results = guarantee_tests(hierarchy, category_names, cal_probs, cal_labels, test_probs, test_labels, seed)
        seed_results.append(results)
        seed_detailed_results.append(detailed_results)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save all results
    results_dir = "../experiments/imagenet"
    results_filename = f"{GUARANTEE}_n{N}_calib{CALIBRATION_RATIO}_{timestamp}.csv"
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
    detailed_results_dir = "../experiments/imagenet/details"
    detailed_results_filename = f"{GUARANTEE}_n{N}_calib{CALIBRATION_RATIO}_details_{timestamp}.csv"
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
    with open('../collected/imagenet/probs_qual.npy', 'rb') as f:
        probs_qual = np.load(f)
    with open('../collected/imagenet/labels_qual.npy', 'rb') as f:
        labels_qual = np.load(f)
    random.seed(2)
    indices = list(range(len(probs_qual)))
    random.shuffle(indices)
    image_indices = indices[:1000]
    probs_qual = [probs_qual[i] for i in image_indices]
    labels_qual = [labels_qual[i] for i in image_indices]
    qual_results_dir = "../qual"
    qual_results_filename = f"imagenet_{GUARANTEE}_n{len(image_indices)}_seed2_qual.json"
    qual_results_path = os.path.join(qual_results_dir, qual_results_filename)
    if not os.path.exists(qual_results_dir):
        os.makedirs(qual_results_dir)
    save_qualitative_examples(qual_results_path, results_path, hierarchy, category_names, probs_qual, labels_qual, image_indices)

    # Plot results
    plots_dir = "../plots/imagenet"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    coverage_plots.coverage_holds_all_m("imagenet", results_path, model_accuracy=model_accuracy)
    pred_set_size_plots.size_vs_m("imagenet", results_path)
    if GUARANTEE == "pac":
        coverage_plots.coverage_holds_all_d("imagenet", results_path, model_accuracy=model_accuracy)
        pred_set_size_plots.size_vs_d("imagenet", results_path)