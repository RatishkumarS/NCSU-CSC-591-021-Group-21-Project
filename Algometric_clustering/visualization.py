import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data for each algorithm: mean and standard deviation
algorithms = {
    'K-Means': ([15882.07492002, 400.83412978], [2915.54910377, 354.27446179]),
    'SMO-GMM': ([17683.95, 199.0415], [621.23373017, 7.55288241]),
    'SVM': ([19040.17924697, 190.5923851], [297.48590422, 10.96539187]),
    'Algometric-Clustering': ([14874.22815326, 516.68867292], [1439.94804304, 191.63764115]),
    'SMO': ([17866.34217781, 210.01384061], [1011.0172236, 24.8845294])
}

# Extracting means and standard deviations for plotting
means = [values[0] for values in algorithms.values()]
std_devs = [values[1] for values in algorithms.values()]

# Number of algorithms
num_algorithms = len(algorithms)

# Set seaborn style and color palette
sns.set_style("whitegrid")
palette = sns.color_palette("husl", num_algorithms)

# Plotting
fig, ax = plt.subplots()

for i, (algo, (mean, std_dev)) in enumerate(algorithms.items()):
    x = np.arange(len(mean)) + i * 0.2  # Shift bars horizontally for each algorithm
    ax.bar(x, mean, yerr=std_dev, width=0.2, color=palette[i], label=algo)

ax.set_xticks(np.arange(len(mean)) + 0.2 * (num_algorithms - 1) / 2)
ax.set_xticklabels(['Throughput', 'Latency'], fontsize=18)  # Increase font size for x-labels
ax.set_ylabel('Value', fontsize=14)  # Increase font size for y-labels
ax.set_title('Mean and Standard deviation for different algorithms', fontsize=16)  # Increase font size for title
ax.legend(fontsize=12)  # Increase font size for legend

plt.show()
