import os, sys, shutil, dill, glob
from turtle import forward; cwd = os.getcwd()+"/"
from itertools import compress
import torch_utils as utils
from torch_utils import pipeline, nnframe, task, monitor, datatool, auxs
import torch as t
from torch.utils.data import Dataset
import torch.nn.functional as Func
from torch.nn.parameter import Parameter as Param
import torch_geometric as tg
import numpy as np
import matplotlib.pyplot as plt
from math import nan

X,Y = datatool.my_data_generator(remainder_twist=False)
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
Cora = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())[0]
X = Cora.x; Y = Cora.y; G = Cora.edge_index
trm = Cora.train_mask; vm = Cora.val_mask; tm = Cora.test_mask


#--------------------------------------------------------------------------------------------------#
# Using pin_memory method

data = datatool.DataStruct(X,Y)

model = nnframe.FFNN(layers = 2*[t.nn.Linear],
                     args = [{'in_features':data.num_features, 'out_features':16},
                            {'in_features':16, 'out_features':data.num_targets}],
                     activations = 2*[t.sigmoid],
                     dropouts = 2*[0.5])

work = pipeline.Builder(data,model)
work.learn( criterion = t.nn.CrossEntropyLoss(),
            optimizer = t.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5),
            tasklist = [(work.loss,5),
                        (work.acc_maskwise,5),
                        #(work.P_max_mean,100),
                        (work.loss_log,20),
                        (work.acc_log,100),
                        (work.LA_plot,100),
                        #(work.P_show,nan),
                        #(work.P_max_mean_log,100),
                        #(work.P_max_mean_window,100),
                        #(work.out_2d,nan),
                        ],
            total_epoch=1000, batch_size=500, 
            forward_val=False, forward_test=False, streaming=True, gpu=True,)

work.infer(data.x_test,data.y_test)
work.infer(data.x_val,data.y_val)

#--------------------------------------------------------------------------------------------------#


# datapath = "my_data"
# datatool.mkdir(datapath)
# for i in range(len(Y)):
#   np.savetxt(f"{datapath}/data{i}.txt",X[i])
# np.savetxt(f"{datapath}/target.txt",Y)


data = datatool.GNN_node_classification(X,Y,G,trm,vm,tm)
model = nnframe.GNN(data = data,
                    layers = [tg.nn.GCNConv, tg.nn.GCNConv],
                    activations = 2*[t.sigmoid],
                    dropouts = [0,0],
                    hidden_channels = [16])

work = pipeline.Builder(data,model)
work.learn( criterion = t.nn.CrossEntropyLoss(),
            optimizer = t.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-5),
            tasklist = [(work.loss,5),
                        (work.acc_maskwise,5),
                        (work.loss_log,20),
                        (work.acc_log,100),
                        (work.LA_plot,100),
                        ],
            total_epoch=1000, streaming=True, gpu=True,)


