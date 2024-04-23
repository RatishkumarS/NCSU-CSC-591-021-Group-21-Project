import warnings
import numpy as np
from row import ROW
from cols import COLS
from sklearn.mixture import GaussianMixture
from helpers import *
import csv
import time

# Filter out the specific warning
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")


class DATA:
    def __init__(self, src, fun=None):
        self.row = []
        self.cols = None
        if isinstance(src, str):
            with open(src, 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    self.add(row, fun)
        else:
            for x in src:
                self.add(x,fun)

    def add(self, t, fun=None, row=None):
        if isinstance(t, ROW):
            row = t
        else:
            row = ROW(t)

        if self.cols:
            if fun:
                fun(self, row)
            self.row.append(self.cols.add(row))
        else:
            self.cols = COLS(t)

    def mid(self, cols=None):
        if cols is None: 
            cols=self.cols.all
        u = []
        for col in cols:
            u.append(col.mid())
        return ROW(u)
    
    def div(self,cols,u):
        u = []
        for col in (self.cols or self.cols.all):
            u.append(col.div())
        return ROW(u)
    
    def small(self,u):
        u = []
        for col in self.cols.all:
            u.append(col.small())
        return ROW(u)
    
    def stats(self, cols=None, fun=None, ndivs=None, u=None,lists=None):
        u = {".N":len(self.row)}
        for i,j in zip(self.cols.names,self.cols.all):
            if i in ['Lbs-','Acc+','Mpg+']:
                u[i] = round(j.mid(),2)
        return u

    def gate(self, random_seed, budget0=4, budget=10, some=0.5):
        start=time.time()
        random.seed(random_seed)
        list_1, list_2, list_3, list_4, list_5, list_6 = [], [], [], [], [], []
        print("['Cost-','Score-','Idle-']")

        # shuffling the rows
        rows = random.sample(self.row, len(self.row))
        # list_1.append(1)
        list_1.append(f"1. top6:{[r.cells[len(r.cells) - 3:] for r in rows[:6]]}")
        list_2.append(f"2. top50:{[[r.cells[len(r.cells) - 3:] for r in rows[:50]]]}")

        # sorting rows based on d2h
        rows.sort(key=lambda r: r.d2h(self))
        list_3.append(f"3. most: {rows[0].cells[len(rows[0].cells) - 3:]}")

        # shuffling rows again
        rows = random.sample(self.row, len(self.row))

        # train and test
        lite = rows[:budget0]  # train-data
        dark = rows[budget0:]  # test-data

        stats, bests = [], []
        first_rows = []
        mean_values = []
        d2h_values = []
        all_statistics = []

        for i in range(20):
            best, rest = self.best_rest(lite, len(lite) ** some)
            todo, selected = self.split(best, rest, lite, dark)

            selected_rows_rand = random.sample(dark, budget0 + i)
            y_values_rand = []
            for row in selected_rows_rand:
                y_val = list(map(coerce, row.cells[-3:]))
                y_values_rand.append(y_val)

            first_row = selected.row[0] if selected.row else None
            first_rows.append(first_row)

            mean_mid = selected.mid().cells[len(selected.mid().cells) - 2:]
            all_statistics.append(mean_mid)

            d2h_value = first_row.d2h(self)
            d2h_values.append(d2h_value)

            list_4.append(f"4: rand:{np.mean(np.array(y_values_rand), axis=0)}")
            list_5.append(f"5: mid: {selected.mid().cells[len(selected.mid().cells) - 3:]}")
            list_6.append(f"6: top: {best.row[0].cells[len(best.row[0].cells) - 3:]}")
            stats.append(selected.mid())
            bests.append(best.row[0])
            lite.append(dark.pop(todo))

        # print('\n'.join(map(str, list_1)))
        # print('\n'.join(map(str, list_2)))
        # print('\n'.join(map(str, list_3)))
        # print('\n'.join(map(str, list_4)))
        # print('\n'.join(map(str, list_5)))
        # print('\n'.join(map(str, list_6)))
        
        all_statistics = np.array(all_statistics)
        means = np.mean(all_statistics, axis=0)
        std_devs = np.std(all_statistics, axis=0)

        # Calculate error bars (e.g., standard error)
        error_bars = std_devs / np.sqrt(len(all_statistics))

        # Print or return the aggregated statistics with error bars
        print("Means with Error Bars:")
        for idx, (mean, error) in enumerate(zip(means, error_bars)):
            print(f"Feature {idx+1}: Mean={mean}, Error={error}")

        # print(d2h_values)
        # end=time.time()
        # temp=end-start
        # print(f"Time Taken to run SMO-GMM on the dataset = {temp}")

        return stats, bests, means, std_devs

    def best_rest(self, rows, want):
        rows.sort(key=lambda r: r.d2h(self))
        best, rest = [self.cols.names], [self.cols.names]
        for i, row in enumerate(rows):
            if i < want:
                best.append(row)
            else:
                rest.append(row)
        return DATA(best), DATA(rest)


    def extract_features(self, row):
        return [float(cell) for cell in row.cells]

    def split(self, best, rest, lite, dark):
        selected = DATA([self.cols.names])
        max_likelihood = -float('inf')
        best_index = -1

        # Extract features from ROW objects in 'dark'
        dark_data = [self.extract_features(row) for row in dark]

        # Convert 'dark_data' to a numpy array
        dark_data_np = np.array(dark_data)

        # Fit a Gaussian Mixture Model
        gmm = GaussianMixture(n_components=2)  # Assuming 2 components for simplicity
        gmm.fit(dark_data_np)

        # Compute the likelihood of each data point under the GMM
        likelihoods = gmm.score_samples(dark_data_np)

        # Select the data point with the highest likelihood
        for i, likelihood in enumerate(likelihoods):
            if likelihood > max_likelihood:
                best_index = i
                max_likelihood = likelihood

        # Add the selected data point to the 'selected' set
        if best_index != -1:
            selected.add(dark[best_index])

        return best_index, selected
