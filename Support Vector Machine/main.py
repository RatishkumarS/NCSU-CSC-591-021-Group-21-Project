import sys
from config import CONFIG
from data import DATA
# from hw.w4.src.test import *
import random
from test import *

temp = CONFIG()


def cli():
    args = sys.argv[1:]
    if args[0] == '--help' or args[0] == "-h":
        print(temp.gethelp())
    elif args[0] == '--file' or args[0]=="-f":
        file = str(args[1])
        gate20(file)
        dataobj = DATA(file)
        
    else:
        attr = args[0][2:]
        val = args[1]
        temp.setthe(attr,val)


def gate20(file):
    random_seeds = random.sample(range(100), 20)
    count = 1
    for random_seed in random_seeds:
        print("========================================================================================================================")
        print("Current random seed: ", random_seed)
        data_new = DATA(file)
        # print(data_new.cols.y.values())
        data_new.gate(random_seed,file)
        print("========================================================================================================================")

cli()
