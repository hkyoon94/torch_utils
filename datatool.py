######################################################################################
#------------------------ DATA STRUCTURE MANAGER FOR PYTORCH ------------------------#
######################################################################################

import os, sys, dill; cwd = os.getcwd()+"/"
from math import *
from collections import Counter
import torch as t
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def sizeof(obj):
    m = sys.getsizeof(obj)
    m_kb = m/1024; m_mb = m_kb/1024; m_gb = m_mb/1024
    print(f"Size in memory: {m_kb:.4f}KB ({m_mb:.4f}MB, {m_gb:.4f}GB)")

def mkdircwd(cwdpath):
    if not os.path.exists(f"{cwd}{cwdpath}"):
        os.makedirs(f"{cwd}{cwdpath}")
        
def saveobjcwd(obj,cwdpath):
    with open(f"{cwd}{cwdpath}",'wb') as f:
        dill.dump(obj,f)
        
def loadobjcwd(cwdpath):
    with open(f"{cwd}{cwdpath}",'rb') as f:
        return dill.load(f)

def readrows(filePath, row_indices=[]):
    with open(filePath) as f:
        for i, line in enumerate(f):
            if i in row_indices:
                yield line
                
                
#--------------------------------------------------------------------------------------------------#
def splitter(num_nodes,p,name1='set1',name2='set2',seed=3):
        t.manual_seed(seed)
        randvec = t.rand(num_nodes)        
        mask1 = randvec <= p; mask2 = randvec > p
        print(f"Masked {name1}: {sum(mask1)}, {name2}: {sum(mask2)} out of {num_nodes} nodes")
        return mask1, mask2

def k_fold_splitter(num_nodes,k):
    pass

class DataStruct(Dataset):
    def __init__(self,x_train,y_train,G_train=None):
        super().__init__()
        self.tasktype = 'simple_classification' #! SHOULD BE OVERRIDDEN FOR DIFFERENT TASKS
        self.x_train = x_train; self.y_train = y_train; self.G_train = G_train
        self.num_features = self.x_train[0].shape[-1]
        self.num_targets = len(set(self.y_train.tolist()))
        self.num_train = len(self.y_train)
        self.x_val = None; self.y_val = None; self.x_test = None; self.y_test = None

    def __getitem__(self, idx): #! SHOULD BE OVERRIDDEN FOR DIFFERENT TASKS
        if self.G_train is None:
            return self.x_train[idx], self.y_train[idx]
        else:
            return self.x_train[idx], self.y_train[idx], self.G_train[idx]
    
    def __len__(self): 
        return len(self.x_train)
    
    def set_val(self, x_val,y_val,G_val=None):
        self.x_val = x_val; self.y_val = y_val; self.G_val = G_val
        self.num_val = len(self.y_val)
        
    def set_test(self, x_test,y_test,G_test=None):
        self.x_test = x_test; self.y_test = y_test; self.G_test = G_test
        self.num_test = len(self.y_test)

    def sort_by_target(self):
        self.y_train, sorted_indices = t.sort(self.train_y)
        self.x_train = self.x_train[sorted_indices,:]
        if self.G_train is not None:
            self.G_train = self.G_train[sorted_indices,:]




#--------------------------------------------------------------------------------------------------#
class GNN_node_classification(DataStruct):
    def __init__(self,X,Y,G,trm,tm,vm):
        super().__init__(X,Y)
        self.tasktype = 'GNN_node_classification'
        self.x = X; self.y = Y; self.G = G
        self.trm = trm; self.tm = tm; self.vm = vm
    



#--------------------------------------------------------------------------------------------------#
class GNN_graph_classification(DataStruct):
    def __init__(self,*args,G):
        super().__init__(*args)
        self.tasktype = 'GNN_graph_classification'
        self.G = G
        
    def __getitem__(self, idx):
        return self.x_train[idx], self.G[idx], self.y_train[idx]





#----------------------------------------------------------------------------------------------------#
def my_data_generator(remainder_twist=True, fig_show=False):

    t.manual_seed(3)
    data_dim = 4
    data_num = 500
    data_tensor = t.rand(data_num,data_dim)

    coeffs = t.randn(2,4)    # AUXILLIARY FUNCTION FOR ASSIGNING DATA CLASS
    def f(x,y,z,w):                
        val1 = t.dot(coeffs[0,:], t.tensor([sin(x),cos(y),asin(z),acos(w)]))
        val2 = t.dot(coeffs[1,:], t.tensor([exp(x),exp(y),cosh(z),sinh(w)]))
        return val1, val2

    y1 = t.zeros(data_num); y2 = t.zeros(data_num) 

    for i in range(data_num):
        y1[i], y2[i] = f(*data_tensor[i,:])
        
    fig,axes = plt.subplots(1,2)
    x = t.arange(1,data_num+1)
    axes[0].scatter(x,y1); axes[1].scatter(x,y2)
    axes[0].set_title("$f_1(\mathbf{x}_i)$", fontsize=15)
    axes[1].set_title("$f_2(\mathbf{x}_i)$", fontsize=15)
    axes[0].set_xlabel("Data index $i$"); axes[1].set_xlabel("Data index $i$")
    fig.set_size_inches(12,5)
    if fig_show is True:
        fig.show()
    plt.close()

    fact1 = y1 < -0.5; fact2 = y2 < 0.6            
    data_class = t.zeros(data_num,dtype=t.long)    
    #!!! CLASS TAGS MUST BE AN INTEGER. If not, torch's loss function returns error.
    for i in range(0,data_num):
        if (fact1[i] == True) and (fact2[i] == True):
            data_class[i] = 0
        elif (fact1[i] == True) and (fact2[i] == False):
            data_class[i] = 1
        elif (fact1[i] == False) and (fact2[i] == True):
            data_class[i] = 2
        else:
            data_class[i] = 3
            
    print(f"Class 0: {sum(data_class == 0)} nodes")
    print(f"Class 1: {sum(data_class == 1)} nodes")
    print(f"Class 2: {sum(data_class == 2)} nodes")
    print(f"Class 3: {sum(data_class == 3)} nodes")    
    
    if remainder_twist is True:
        data_tensor = t.remainder(data_tensor,0.5)
    
    return data_tensor, data_class