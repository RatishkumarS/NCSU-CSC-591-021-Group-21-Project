from sym import SYM
from num import NUM
from config import CONFIG

def test_sym():
    obj = SYM()
    for x in [1,1,1,2,2,2,2,3,4]:
        obj.add(x)

    return obj.mid()==2 and round(obj.div(),2)==1.75

def test_stat(name,vals):
    if name=="auto93.csv":
        print(vals['.N']==398 and vals['Lbs-']==8.38 and vals['Acc+']==5.94 and vals['Mpg+']==1.77)

def test_seed_cohen():
    obj = CONFIG()
    obj.setthe("seed",12345)
    obj.setthe("cohen",0.67)
    return obj.the['seed']==12345 and obj.the['cohen']==0.67
