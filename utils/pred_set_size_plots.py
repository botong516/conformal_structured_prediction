import os
import pandas as pd
import matplotlib.pyplot as plt

FONTSIZE = 50
ALPHA = 0.8
CAP_ALPHA = 0.6
CAPSIZE = 20
LINEWIDTH = 8

def size_vs_m(task_name, results_path, num_digits=2):
    results_filename = os.path.basename(os.path.splitext(results_path)[0])
    guarantee = results_filename.split("_")[0]
    if guarantee == "pac":
        delta = 0.1
    else:
        delta = ""

    if task_name == 'mnist':
        plots_path = f"../plots/{task_name}/{num_digits}digits/" + results_filename + f"_d{delta}_size_vs_m.jpg"
    else:
        plots_path = f"../plots/{task_name}/" + results_filename + f"_d{delta}_size_vs_m.jpg"

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
            'test_avg_size': ['mean', 'std']
        }).reset_index()

        x_values = grouped.iloc[:, 0]  # 1 - epsilon values
        y_mean_values = grouped[('test_avg_size', 'mean')]
        y_std_values = grouped[('test_avg_size', 'std')]

        color = color_cycle[idx % len(color_cycle)]

        plt.plot(x_values, y_mean_values, '-', color=color, alpha=ALPHA, label=rf'$m={m}$', linewidth=LINEWIDTH)
        plt.errorbar(x_values, y_mean_values, yerr=y_std_values, fmt='o', color=color, alpha=CAP_ALPHA, capsize=CAPSIZE, elinewidth=LINEWIDTH, capthick=LINEWIDTH)

    plt.legend(title=None, fontsize=FONTSIZE)

    plt.xlabel(r'Desired Coverage Rate ($1-\epsilon$)', fontsize=FONTSIZE)
    plt.ylabel('Average Size', fontsize=FONTSIZE)

    all_x_values = sorted(df['epsilon'].apply(lambda e: 1 - e).unique())
    plt.xticks(all_x_values, fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.grid(True)

    plt.savefig(plots_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plots_path}.")



def size_vs_d(task_name, results_path, num_digits=2):
    results_filename = os.path.basename(os.path.splitext(results_path)[0])

    if task_name == "squad":
        m = 4
    else:
        m = 2

    if task_name == "mnist":
        plots_path = f"../plots/{task_name}/{num_digits}digits/" + results_filename + f"_m{m}_size_vs_d.jpg"
    else:
        plots_path = f"../plots/{task_name}/" + results_filename + f"_m{m}_size_vs_d.jpg"

    df = pd.read_csv(results_path)

    plt.figure(figsize=(16, 12))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plot for each delta
    unique_d_values = sorted(df['delta'].unique())
    for idx, d in enumerate(unique_d_values):
        subset = df[(df['m'] == m) & (df['delta'] == d)]

        grouped = subset.groupby(1 - subset['epsilon']).agg({
            'test_avg_size': ['mean', 'std']
        }).reset_index()

        x_values = grouped.iloc[:, 0]  # 1 - epsilon values
        y_mean_values = grouped[('test_avg_size', 'mean')]
        y_std_values = grouped[('test_avg_size', 'std')]

        color = color_cycle[idx % len(color_cycle)]

        plt.plot(x_values, y_mean_values, '-', color=color, alpha=ALPHA, label=rf'$\delta={d}$', linewidth=LINEWIDTH)
        plt.errorbar(x_values, y_mean_values, yerr=y_std_values, fmt='o', color=color, alpha=CAP_ALPHA, capsize=CAPSIZE, elinewidth=LINEWIDTH, capthick=LINEWIDTH)

    plt.xlabel(r'Desired Coverage Rate ($1-\epsilon$)', fontsize=FONTSIZE)
    plt.ylabel('Average Size', fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)

    all_x_values = sorted(df['epsilon'].apply(lambda e: 1 - e).unique())
    plt.xticks(all_x_values, fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.grid(True)

    plt.savefig(plots_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plots_path}.")

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
    size_vs_m("squad", mar_results_path)
    size_vs_m("squad", pac_results_path)
    size_vs_d("squad", pac_results_path)

    # # MNIST 3-digit
    results_dir = "../results/experiments/mnist"
    mar_filename = "marginal_3digits_n1000_calib0.2_20240930_215901.csv"
    pac_filename = "pac_3digits_n1000_calib0.2_20240930_212731.csv"
    mar_results_path = os.path.join(results_dir, mar_filename)
    pac_results_path = os.path.join(results_dir, pac_filename)
    size_vs_m("mnist", mar_results_path, num_digits=3)
    size_vs_m("mnist", pac_results_path, num_digits=3)
    size_vs_d("mnist", pac_results_path, num_digits=3)

    # MNIST 2-digit
    results_dir = "../results/experiments/mnist"
    mar_filename = "marginal_2digits_n1000_calib0.2_20240929_022026.csv"
    pac_filename = "pac_2digits_n1000_calib0.2_20240929_013259.csv"
    mar_results_path = os.path.join(results_dir, mar_filename)
    pac_results_path = os.path.join(results_dir, pac_filename)
    size_vs_m("mnist", mar_results_path, num_digits=2)
    size_vs_m("mnist", pac_results_path, num_digits=2)
    size_vs_d("mnist", pac_results_path, num_digits=2)

    # ImageNet
    results_dir = "../results/experiments/imagenet"
    mar_filename = "marginal_n1000_calib0.2_20240928_033830.csv"
    pac_filename = "pac_n1000_calib0.2_20240929_002032.csv"
    mar_results_path = os.path.join(results_dir, mar_filename)
    pac_results_path = os.path.join(results_dir, pac_filename)
    size_vs_m("imagenet", mar_results_path)
    size_vs_m("imagenet", pac_results_path)
    size_vs_d("imagenet", pac_results_path)