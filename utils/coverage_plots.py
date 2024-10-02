import os
import pandas as pd
import matplotlib.pyplot as plt

FONTSIZE = 50
ALPHA = 0.8
CAP_ALPHA = 0.6
CAPSIZE = 20
LINEWIDTH = 8

def coverage_holds_all_m(task_name, results_path, model_accuracy=0, num_digits=2):
    results_filename = os.path.basename(os.path.splitext(results_path)[0])
    guarantee = results_filename.split("_")[0]
    if guarantee == "pac":
        delta = 0.01
    else:
        delta = ""

    if task_name == 'mnist':
        plots_path = f"../plots/{task_name}/{num_digits}digits/" + results_filename + f"_d{delta}_coverage_holds_all_m.jpg"
    else:
        plots_path = f"../plots/{task_name}/" + results_filename + f"_d{delta}_coverage_holds_all_m.jpg"

    df = pd.read_csv(results_path)

    plt.figure(figsize=(16, 12))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plot for each m
    unique_m_values = sorted(df['m'].unique())
    for idx, m in enumerate(unique_m_values):
        if guarantee == "marginal":
            subset = df[df['m'] == m]
        elif guarantee == "pac":
            subset = df[(df['m'] == m) & (df['delta'] == delta)]

        grouped = subset.groupby(1 - subset['epsilon']).agg({
            'test_coverage': ['mean', 'std']
        }).reset_index()

        x_values = grouped.iloc[:, 0]  # 1 - epsilon values
        y_mean_values = grouped[('test_coverage', 'mean')]
        y_std_values = grouped[('test_coverage', 'std')]

        color = color_cycle[idx % len(color_cycle)]

        plt.plot(x_values, y_mean_values, '-', color=color, alpha=ALPHA, label=rf'$m={m}$', linewidth=LINEWIDTH)
        plt.errorbar(x_values, y_mean_values, yerr=y_std_values, fmt='o', color=color, alpha=CAP_ALPHA, capsize=CAPSIZE, elinewidth=LINEWIDTH, capthick=LINEWIDTH)

    if model_accuracy == 0:
        if task_name == 'squad':
            model_accuracy = 0.3320610687022901
        elif task_name == 'mnist' and num_digits == 2:
            model_accuracy = 0.8068806880688069
        elif task_name == 'mnist' and num_digits == 3:
            model_accuracy = 0.7280456091218244
        elif task_name == 'imagenet':
            model_accuracy = 0.80936
    plt.axhline(y=model_accuracy, linestyle='--', color='magenta', label=f'Model Acc.', linewidth=LINEWIDTH)

    all_epsilon = 1 - df['epsilon']
    plt.plot([all_epsilon.min(), all_epsilon.max()], [all_epsilon.min(), all_epsilon.max()], '--', color='gray', label='Reference', linewidth=LINEWIDTH)
    
    plt.legend(title=None, fontsize=FONTSIZE)

    plt.xlabel(r'Desired Coverage Rate ($1-\epsilon$)', fontsize=FONTSIZE)
    plt.ylabel('Coverage Rate', fontsize=FONTSIZE)

    all_x_values = sorted(df['epsilon'].apply(lambda e: 1 - e).unique())
    plt.xticks(all_x_values, fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.grid(True)

    plt.savefig(plots_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plots_path}.")



def coverage_holds_all_d(task_name, results_path, model_accuracy=0, num_digits=2):
    m = 2
    results_filename = os.path.basename(os.path.splitext(results_path)[0])
    guarantee = results_filename.split("_")[0]

    if task_name == "mnist":
        plots_path = f"../plots/{task_name}/{num_digits}digits/" + results_filename + f"_m{m}_coverage_holds_all_d.jpg"
    else:
        plots_path = f"../plots/{task_name}/" + results_filename + f"_m{m}_coverage_holds_all_d.jpg"

    df = pd.read_csv(results_path)

    plt.figure(figsize=(16, 12))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plot for each d
    unique_d_values = sorted(df['delta'].unique())
    for idx, d in enumerate(unique_d_values):
        subset = df[(df['m'] == m) & (df['delta'] == d)]

        grouped = subset.groupby(1 - subset['epsilon']).agg({
            'test_coverage': ['mean', 'std']
        }).reset_index()

        x_values = grouped.iloc[:, 0]  # 1 - epsilon values
        y_mean_values = grouped[('test_coverage', 'mean')]
        y_std_values = grouped[('test_coverage', 'std')]

        color = color_cycle[idx % len(color_cycle)]

        plt.plot(x_values, y_mean_values, '-', color=color, alpha=ALPHA, label=rf'$\delta={d}$', linewidth=LINEWIDTH)
        plt.errorbar(x_values, y_mean_values, yerr=y_std_values, fmt='o', color=color, alpha=CAP_ALPHA, capsize=CAPSIZE, elinewidth=LINEWIDTH, capthick=LINEWIDTH)

    if model_accuracy == 0:
        if task_name == 'squad':
            model_accuracy = 0.3320610687022901
        elif task_name == 'mnist' and num_digits == 2:
            model_accuracy = 0.8068806880688069
        elif task_name == 'mnist' and num_digits == 3:
            model_accuracy = 0.7280456091218244
        elif task_name == 'imagenet':
            model_accuracy = 0.80936
    plt.axhline(y=model_accuracy, linestyle='--', color='magenta', label=f'Model Acc.', linewidth=LINEWIDTH)

    all_epsilon = 1 - df['epsilon']
    plt.plot([all_epsilon.min(), all_epsilon.max()], [all_epsilon.min(), all_epsilon.max()], '--', color='gray', label='Reference', linewidth=LINEWIDTH) 

    plt.xlabel(r'Desired Coverage Rate ($1-\epsilon$)', fontsize=FONTSIZE)
    plt.ylabel('Coverage Rate', fontsize=FONTSIZE)

    plt.legend(title=None, fontsize=FONTSIZE, loc='lower right')

    all_x_values = sorted(df['epsilon'].apply(lambda e: 1 - e).unique())
    plt.xticks(all_x_values, fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.grid(True)

    plt.savefig(plots_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plots_path}")



def coverage_comparison(task_name, marginal_path, pac_path, model_accuracy=0, num_digits=2):
    marginal_filename = os.path.basename(os.path.splitext(marginal_path)[0])
    pac_filename = os.path.basename(os.path.splitext(pac_path)[0])

    if task_name == "mnist":
        plots_path = f"../plots/{task_name}/{num_digits}digits/" + "comb_" + marginal_filename + pac_filename + ".jpg"
    else:
        plots_path = f"../plots/{task_name}/" + "comb_" + marginal_filename + pac_filename + ".jpg"
    m = 2
    delta = 0.01

    df_mar = pd.read_csv(marginal_path)
    df_pac = pd.read_csv(pac_path)

    plt.figure(figsize=(16, 12))

    subset_mar = df_mar[df_mar['m'] == m]
    grouped_mar = subset_mar.groupby(1 - subset_mar['epsilon']).agg({
        'test_coverage': ['mean', 'std']
    }).reset_index()
    x_values_mar = grouped_mar.iloc[:, 0]  # 1 - epsilon values
    y_mean_values_mar = grouped_mar[('test_coverage', 'mean')]
    y_std_values_mar = grouped_mar[('test_coverage', 'std')]
    plt.plot(x_values_mar, y_mean_values_mar, '-', color='blue', alpha=ALPHA, label=f'marginal', linewidth=LINEWIDTH)
    plt.errorbar(x_values_mar, y_mean_values_mar, yerr=y_std_values_mar, fmt='o', color='blue', alpha=CAP_ALPHA, capsize=CAPSIZE, elinewidth=LINEWIDTH, capthick=LINEWIDTH)

    subset_pac = df_pac[(df_pac['m'] == m) & (df_pac['delta'] == delta)]
    grouped_pac = subset_pac.groupby(1 - subset_pac['epsilon']).agg({
        'test_coverage': ['mean', 'std']
    }).reset_index()
    x_values_pac = grouped_pac.iloc[:, 0]  # 1 - epsilon values
    y_mean_values_pac = grouped_pac[('test_coverage', 'mean')]
    y_std_values_pac = grouped_pac[('test_coverage', 'std')]
    plt.plot(x_values_pac, y_mean_values_pac, '-', color='green', alpha=ALPHA, label=f'PAC', linewidth=LINEWIDTH)
    plt.errorbar(x_values_pac, y_mean_values_pac, yerr=y_std_values_pac, fmt='o', color='green', alpha=CAP_ALPHA, capsize=CAPSIZE, elinewidth=LINEWIDTH, capthick=LINEWIDTH)

    all_epsilon = 1 - df_mar['epsilon']
    plt.plot([all_epsilon.min(), all_epsilon.max()], [all_epsilon.min(), all_epsilon.max()], '--', color='gray', label='Reference', linewidth=LINEWIDTH) 

    if model_accuracy == 0:
        if task_name == 'squad':
            model_accuracy = 0.3320610687022901
        elif task_name == 'mnist' and num_digits == 2:
            model_accuracy = 0.8068806880688069
        elif task_name == 'mnist' and num_digits == 3:
            model_accuracy = 0.7280456091218244
        elif task_name == 'imagenet':
            model_accuracy = 0.80936
    plt.axhline(y=model_accuracy, linestyle='--', color='magenta', label=f'Model Acc.', linewidth=LINEWIDTH)

    plt.xlabel(r'Desired Coverage Rate ($1-\epsilon$)', fontsize=FONTSIZE)
    plt.ylabel('Coverage Rate', fontsize=FONTSIZE)
    plt.legend(title=None, fontsize=FONTSIZE, loc='lower right')

    all_x_values = sorted(df_mar['epsilon'].apply(lambda e: 1 - e).unique())
    plt.xticks(all_x_values, fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.grid(True)

    plt.savefig(plots_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plots_path}")



def get_thresholds(task_name, results_path, num_digits=2):
    results_filename = os.path.basename(os.path.splitext(results_path)[0])
    guarantee = results_filename.split("_")[0]

    if task_name == 'mnist':
        plots_path = f"../plots/{task_name}/{num_digits}digits/" + results_filename + f"_thresholds.jpg"
    else:
        plots_path = f"../plots/{task_name}/" + results_filename + f"_thresholds.jpg"

    if guarantee == "pac":
        delta = 0.01
    else:
        delta = -1

    df = pd.read_csv(results_path)

    plt.figure(figsize=(16, 12))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plot for each m
    unique_m_values = sorted(df['m'].unique())
    for idx, m in enumerate(unique_m_values):
        subset = df[(df['m'] == m) & (df['delta'] == delta)]

        grouped = subset.groupby(1 - subset['epsilon']).agg({
            'threshold': ['mean', 'std']
        }).reset_index()

        x_values = grouped.iloc[:, 0]  # 1 - epsilon values
        y_mean_values = grouped[('threshold', 'mean')]
        y_std_values = grouped[('threshold', 'std')]

        color = color_cycle[idx % len(color_cycle)]

        plt.plot(x_values, y_mean_values, '-', color=color, alpha=ALPHA, label=f'$m={m}$', linewidth=LINEWIDTH)
        plt.errorbar(x_values, y_mean_values, yerr=y_std_values, fmt='o', color=color, alpha=CAP_ALPHA, capsize=CAPSIZE, elinewidth=LINEWIDTH, capthick=LINEWIDTH)

    plt.legend(title=None, fontsize=FONTSIZE)

    plt.xlabel(r'Desired Coverage Rate ($1-\epsilon$)', fontsize=FONTSIZE)
    plt.ylabel('Threshold', fontsize=FONTSIZE)
    all_x_values = sorted(df['epsilon'].apply(lambda e: 1 - e).unique())
    plt.xticks(all_x_values, fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.grid(True)

    plt.savefig(plots_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plots_path}")

if __name__ == '__main__':
    plots_dir = "../plots/squad"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plots_dir = "../plots/mnist/2digits"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plots_dir = "../plots/mnist/3digits"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plots_dir = "../plots/imagenet"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # SQuAD
    results_dir = "../results/experiments/squad"
    mar_filename = "marginal_years_1970_to_2020_calib0.5_20240929_130920.csv"
    pac_filename = "pac_years_1970_to_2020_calib0.5_20240929_221436.csv"
    mar_results_path = os.path.join(results_dir, mar_filename)
    pac_results_path = os.path.join(results_dir, pac_filename)
    coverage_holds_all_m("squad", mar_results_path)
    coverage_holds_all_m("squad", pac_results_path)
    coverage_holds_all_d("squad", pac_results_path)

    # MNIST 3-digit
    results_dir = "../results/experiments/mnist"
    mar_filename = "marginal_3digits_n1000_calib0.2_20240930_215901.csv"
    pac_filename = "pac_3digits_n1000_calib0.2_20240930_212731.csv"
    mar_results_path = os.path.join(results_dir, mar_filename)
    pac_results_path = os.path.join(results_dir, pac_filename)
    coverage_holds_all_m("mnist", mar_results_path, num_digits=3)
    coverage_holds_all_m("mnist", pac_results_path, num_digits=3)
    coverage_holds_all_d("mnist", pac_results_path, num_digits=3)

    # MNIST 2-digit
    results_dir = "../results/experiments/mnist"
    mar_filename = "marginal_2digits_n1000_calib0.2_20240929_022026.csv"
    pac_filename = "pac_2digits_n1000_calib0.2_20240929_013259.csv"
    mar_results_path = os.path.join(results_dir, mar_filename)
    pac_results_path = os.path.join(results_dir, pac_filename)
    coverage_holds_all_m("mnist", mar_results_path, num_digits=2)
    coverage_holds_all_m("mnist", pac_results_path, num_digits=2)
    coverage_holds_all_d("mnist", pac_results_path, num_digits=2)

    # ImageNet
    results_dir = "../results/experiments/imagenet"
    mar_filename = "marginal_n1000_calib0.2_20240928_033830.csv"
    pac_filename = "pac_n1000_calib0.2_20240929_002032.csv"
    mar_results_path = os.path.join(results_dir, mar_filename)
    pac_results_path = os.path.join(results_dir, pac_filename)
    coverage_holds_all_m("imagenet", mar_results_path)
    coverage_holds_all_m("imagenet", pac_results_path)
    coverage_holds_all_d("imagenet", pac_results_path)
