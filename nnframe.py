######################################################################################
#--------------------- NEURAL NETWORK MODEL FRAMES FOR PYTORCH ----------------------#
######################################################################################

import torch as t
import torch.nn.functional as Func
from inspect import getfullargspec


#---------------------- BASE CLASS OF GENERAL FEED-FORWARD NEURAL NETWORKS -----------------------#
class FFNN(t.nn.Module):
    def __init__(self, layers, args, activations, 
                 dropouts=False, batch_normalize=False, seed=False):
        super().__init__()
        self.activations = activations; self.dropouts = dropouts; 
        self.batch_normalize = batch_normalize
        self.num_layers = len(layers)
        if seed is not False: ##### building layers
            t.manual_seed(seed)
        self.layers = t.nn.ModuleList([])
        for i in range(self.num_layers):
            if type(args[i]) is dict:
                self.layers.append(layers[i](**args[i]))
            else:
                self.layers.append(layers[i](*args[i]))
        self.params = list(self.parameters())
        self.n_params = len(self.params)
            
        print(f"\n----------- MODEL -----------: \n{self}")
    
    def __override__(self,F,asname='forward'):
        setattr(self, asname, F.__get__(self, self.__class__))

#! OVERRIDE FOR DIFFERENT TASKS
    def forward(self, x): # building forward message passer
        for i in range(0,self.num_layers):
            if self.batch_normalize is not False:
                pass
            x = self.activations[i](self.layers[i](x))
            if self.dropouts is not False:
                x = Func.dropout(x, p=self.dropouts[i], training=self.training)
        return x
    

#-------------------------------------------------------------------------------------------------#
class GNN(FFNN):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
    def forward(self, x,G):
        for i in range(0,self.num_layers):
            if self.batch_normalize is not False:
                pass
            try:
                x = self.activations[i](self.layers[i](x))
            except:
                x = self.activations[i](self.layers[i](x,G))
            if self.dropouts is not False:
                x = Func.dropout(x, p=self.dropouts[i], training=self.training)
        return x