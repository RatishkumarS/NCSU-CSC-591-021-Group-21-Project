
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree
from sklearn.svm import SVC
from row import ROW
from cols import COLS
import random
from helpers import *
import csv
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from config import CONFIG
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM

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
                fun(self,row)
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

    def gate(self, random_seed,file, budget0=4, budget=10, some=0.5, algo='svm'):
        start=time.time()
        random.seed(random_seed)
        list_1, list_2, list_3, list_4, list_5, list_6 = [], [], [], [], [], []
        if(file[8]=='p'):
            print(self.cols.names[-3:])
        else:
            print(self.cols.names[-4:])

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

        for i in range(budget):
            if algo == 'split':
                best, rest = self.best_rest(lite, len(lite) ** some)  
                todo, selected = self.split(best, rest, lite, dark)
            elif algo == 'svm':
                todo, selected, top = self.svm(lite, dark)
            
            selected_rows_rand = random.sample(dark, budget0 + i)
            y_values_rand = []
            for row in selected_rows_rand:
                y_val = list(map(coerce, row.cells[-3:]))
                y_values_rand.append(y_val)

            if(file[8].startswith('x')):
                y_values_flat = [value for sublist in y_values_rand for value in sublist]
                y_values_float = [float(value) for value in y_values_flat]
            
            list_4.append(f"4: rand:{np.mean(np.array(y_values_rand), axis=0)}")
            list_5.append(f"5: mid: {selected.mid().cells[len(selected.mid().cells) - 3:]}")
            stats.append(selected.mid())
            lite.append(dark.pop(todo))
            
            if algo == 'split':
                list_6.append(f"6: top: {best.row[0].cells[len(best.row[0].cells) - 3:]}")
                bests.append(best.row[0])
            elif algo == 'kMeans':
                list_6.append(f"6: top: {top.cells[len(top.cells) - 3:]}")
                bests.append(top)

        print('\n'.join(map(str, list_1)))
        print('\n'.join(map(str, list_2)))
        print('\n'.join(map(str, list_3)))
        print('\n'.join(map(str, list_4)))
        print('\n'.join(map(str, list_5)))
        print('\n'.join(map(str, list_6)))
        end=time.time()
        temp=end-start
        print(f"Time Taken to run SVM on the dataset = {temp}")

        return stats, bests

    def best_rest(self, rows, want):
        rows.sort(key=lambda r: r.d2h(self))
        best, rest = [self.cols.names], [self.cols.names]
        for i, row in enumerate(rows):
            if i < want:
                best.append(row)
            else:
                rest.append(row)
        
        return DATA(best), DATA(rest)

    def split(self, best, rest, lite, dark):
        selected = DATA([self.cols.names])
        max_val = 1E30
        out = 1

        for i, row in enumerate(dark):
            b = row.like(best, len(lite), 2)
            r = row.like(rest, len(lite), 2)

            if b > r:
                selected.add(row)

            tmp = abs(b + r) / abs(b - r + 1E-300)

            if tmp > max_val:
                out, max_val = i, tmp
        return out, selected



    def svm(self, lite, dark):
        selected = DATA([self.cols.names])
        temp = CONFIG()
        rows_list = np.array([r.cells for r in lite])

        # Step 1: Preprocess your data
        scaler = StandardScaler()
        scaled_rows = scaler.fit_transform(rows_list)

        # Step 2: Apply PCA
        pca = PCA(n_components=2)  # Specify the number of components you want to keep
        pca_result = pca.fit_transform(scaled_rows)

        # Step 3: Apply KD-trees
        kdtree = KDTree(scaled_rows)

        # Step 4: Apply One-Class SVM
        svm = OneClassSVM(kernel='linear')
        svm.fit(scaled_rows)

        out = 1
        top = None

        for row, scaled_row in zip(dark, scaler.transform([row.cells for row in dark])):
            pred = svm.predict([scaled_row])
            if pred == 1:  # In OCSVM, 1 represents inliers, -1 represents outliers
                selected.add(row)

        # Now you can use 'pca_result' for visualization or further analysis
        # You can also use 'kdtree' for efficient nearest neighbor search

        return out, selected, top

