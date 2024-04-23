import sys
from config import CONFIG
from data import DATA
# from hw.w4.src.test import *
import random
from test import *
import matplotlib.pyplot as plt
import numpy as np

temp = CONFIG()


def cli():
    args = sys.argv[1:]
    if args[0] == '--help' or args[0] == "-h":
        print(temp.gethelp())
    elif args[0] == '--file' or args[0] == "-f":
        file = str(args[1])
        gate20(file)
        dataobj = DATA(file)
        if args[2] == "--test" or args[2]=="-t":
            if args[3] == "sym":
                print(test_sym())
            elif args[3] == "stats":
                fname = args[1].split("/")[-1]
                fstats = dataobj.stats()
                print(fstats)
            elif args[3] == "config":
                print(test_seed_cohen())
        
    else:
        attr = args[0][2:]
        val = args[1]
        temp.setthe(attr,val)

def plot_statistics(means, std_devs):
    # Plot error bars
    plt.figure(figsize=(10, 6))
    x = np.arange(len(means))
    plt.errorbar(x, means, yerr=std_devs, fmt='o', capsize=5)
    plt.xticks(x, ['Feature 1', 'Feature 2'])  # Replace with actual feature names
    plt.xlabel('Features')
    plt.ylabel('Means')
    plt.title('Means with Error Bars (Aggregated)')
    plt.grid(True)
    plt.show()

def gate20(file):
        all_means = []
        all_std_devs = []

        random_seeds = random.sample(range(100), 20)
        for random_seed in random_seeds:
            print("========================================================================================================================")
            print("Current random seed: ", random_seed)
            data_new = DATA(file)
            # print(data_new.cols.y.values())
            _, _, means, std_devs = data_new.gate(random_seed)
            all_means.append(means)
            all_std_devs.append(std_devs)
        print("========================================================================================================================")

        all_means = np.array(all_means)
        all_std_devs = np.array(all_std_devs)
        final_means = np.mean(all_means, axis=0)
        final_std_devs = np.mean(all_std_devs, axis=0)

        print(final_means)
        print(final_std_devs)

        # Plot aggregated statistics
        plot_statistics(final_means, final_std_devs)

cli()
