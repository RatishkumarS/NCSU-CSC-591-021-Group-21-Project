
import numpy as np
import pandas as pd
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

    def gate(self, random_seed,file, budget0=4, budget=10, some=0.5, algo='algometric_clustering'):
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
            elif algo == 'algometric_clustering':
                todo, selected, top = self.agglomerative_clustering(lite, dark)
            
            selected_rows_rand = random.sample(dark, budget0 + i)
            y_values_rand = []
            for row in selected_rows_rand:
                y_val = list(map(coerce, row.cells[-3:]))
                y_values_rand.append(y_val)
            if(file[8].startswith('x')):
                y_values_flat = [value for sublist in y_values_rand for value in sublist]
                y_values_float = [float(value) for value in y_values_flat]
            
            list_4.append(f"4: rand:{np.mean(np.array(y_values_float), axis=0)}")
            list_5.append(f"5: mid: {selected.mid().cells[len(selected.mid().cells) - 3:]}")
            stats.append(selected.mid())
            lite.append(dark.pop(todo))
            
            if algo == 'split':
                list_6.append(f"6: top: {best.row[0].cells[len(best.row[0].cells) - 3:]}")
                bests.append(best.row[0])
            elif algo == 'kmeans':
                list_6.append(f"6: top: {top.cells[len(top.cells) - 3:]}")
                bests.append(top)

        print('\n'.join(map(str, list_1)))
        print('\n'.join(map(str, list_2)))
        print('\n'.join(map(str, list_3)))
        print('\n'.join(map(str, list_4)))
        print('\n'.join(map(str, list_5)))
        print('\n'.join(map(str, list_6)))

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
    
    def convert_to_numeric(self, data):
        numeric_data = []
        for row in data:
            numeric_row = [float(value) for value in row]
            numeric_data.append(numeric_row)
        return numeric_data

    def agglomerative_clustering(self, lite, dark):
        selected = DATA([self.cols.names])
        temp = CONFIG()
        # Convert lite data to numeric values
        lite_numeric = self.convert_to_numeric([r.cells for r in lite])
        rows_list = np.array(lite_numeric)
        clustering = AgglomerativeClustering(n_clusters=2)
        clustering.fit(rows_list)
        out = 1
        top = None  # Assigning a default value to top
        # To get top from Agglomerative Clustering clusters
        for ind, value in enumerate(clustering.labels_):
            if value == 0:
                top = lite[ind]
                break
        
        for i, row in enumerate(dark):
            # Convert dark row to numeric values
            dark_numeric = self.convert_to_numeric([row.cells])
            dark_numeric_flat = np.array(dark_numeric).flatten()  # Flatten to 1D array
            pred = clustering.fit_predict(np.vstack((rows_list, dark_numeric_flat)))
            if pred[-1] == 0:
                selected.add(row)
        
        return out, selected, top