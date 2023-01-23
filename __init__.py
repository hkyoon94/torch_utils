######################################################################################
#-------------------------- LEARNING UTILITIES FOR PYTORCH --------------------------#
######################################################################################

import torch_utils.auxs as auxs
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from math import nan


#----------------------------------------------#
def get_acc(out,y):
    pred = out.argmax(dim=1)
    correct = pred == y
    acc = 100*correct.sum()/len(y)
    return acc.item()


#----------------------------------------------#
def get_P_max_mean(params):
    if type(params) is list:
        n = len(params)
        Mag = np.zeros(n); mag = np.zeros(n)
        Mav = np.zeros(n); mav = np.zeros(n)
        i=0
        for param in params:
            try:
                Mag[i] = param.grad.abs().max().item()
                mag[i] = param.grad.abs().mean().item()
            except: # if p's gradient is None
                Mag[i] = nan; mag[i] = nan
            Mav[i] = param.abs().max().item()
            mav[i] = param.abs().mean().item()
            i+=1
    else: # for single input
        try:
            Mag = params.grad.abs().max().item()
            mag = params.grad.abs().mean().item()
        except: # if p's gradient is None
            Mag = nan; mag = nan
        Mav = params.abs().max().item()
        mav = params.abs().mean().item()

    return Mag, mag, Mav, mav


#----------------------------------------------#
def P_max_mean(params, Mag=None, mag=None, Mav=None, mav=None):
    print("---------------------------------------------------------------------")
    print("\tsize\tmax|grad|\tmean|grad|\tmax|val|\tmean|val|".expandtabs(12))
    print("---------------------------------------------------------------------")
    try:
        i=0
        for p in params:
            try: 
                tag = p.tag
            except: # if p has no tag
                tag = [" ",list(p.shape)]
            print(f"Param {tag[0]}\t{tag[1]}\t{Mag[i]:.6f}\t{mag[i]:.6f}\t{Mav[i]:.6f}\t{mav[i]:.6f}".expandtabs(12))
            i+=1
        print("---------------------------------------------------------------------")
            
    except: # if statistics input is None
        Mag, mag, Mav, mav = get_P_max_mean(params)
        try:
            tag = params.tag
        except: # if p has no tag
            tag = [" ",list(params.shape)]
        print(f"Param {tag[0]}\t{tag[1]}\t{Mag:.6f}\t{mag:.6f}\t{Mav:.6f}\t{mav:.6f}".expandtabs(12))
        print("---------------------------------------------------------------------")


#----------------------------------------------#
def P_draw(params,size=(9,4)):
    if type(params) is list or type(params) is tuple:
        
        n = len(params)
        
        fig,ax = plt.subplots(n,2)
        fig.set_size_inches(size[0],size[1]*n)
            
        i=0
        for param in params:
            ax[i,0].set_title(f"Param {i} grads, size:{list(param.shape)}")
            ax[i,1].set_title(f"Param {i} values, size:{list(param.shape)}")
            
            param_sq = t.squeeze(param)
            if param.grad is not None: # drawing grad
                if len(param_sq.shape)==1: # for 1d tensor, drawing bar-graph
                    ax[i,0].set_ylim(-1e-4,1e-4)
                    ax[i,0].bar( list(range(0,len(param_sq))),t.squeeze(param.grad.detach()).cpu() )
                    
                elif len(param_sq.shape)==2: # for 2d tensor, drawing heatmap
                    ax[i,0].imshow(param.grad.detach().cpu(),
                        cmap='jet',aspect='auto',vmin=-1e-3,vmax=1e-3)
                    
                else: # for 3d tensor, resizing into pseudo-square shape, and drawing heatmap
                    ax[i,0].imshow(param.grad.view(-1,auxs.near_sqrt_divisor(param.numel())).detach().cpu(),
                        cmap='jet',aspect='auto',vmin=-1e-3,vmax=1e-3)
                
                # drawing values
            if len(param_sq.shape)==1: # for 1d tensor, drawing bar-graph
                ax[i,1].bar( list(range(0,len(param_sq))),t.squeeze(param.detach()).cpu() )
                ax[i,1].set_title(f"Param {i} values, size:{list(param.shape)}")
                
            elif len(param_sq.shape)==2: # for 2d tensor, drawing heatmap
                ax[i,1].imshow(param.detach().cpu(),
                    cmap='jet',aspect='auto')
                
            else: # for 3d tensor, resizing into pseudo-square shape, and drawing heatmap
                ax[i,1].imshow(param.view(-1,auxs.near_sqrt_divisor(param.numel())).detach().cpu(),
                    cmap='jet',aspect='auto')
            i+=1
            
        return fig
        
    else: # if params is not a list (i.e., a single parameter)
        param = params
        
        fig,ax = plt.subplots(1,2)
        fig.set_size_inches(size[0],size[1])
        
        ax[0].set_title(f"Param grads, size:{list(param.shape)}")
        ax[1].set_title(f"Param values, size:{list(param.shape)}")
        
        param_sq = t.squeeze(param)
        if param.grad is not None: # drawing grad
            
            if len(param_sq.shape)==1:
                ax[0].set_ylim(-1e-4,1e-4)
                ax[0].bar( list(range(0,len(param_sq))),t.squeeze(param.grad.detach()).cpu() )
                
            elif len(param_sq.shape)==2:
                ax[0].imshow(param.grad.detach().cpu(),
                    cmap='jet',aspect='auto',vmin=-1e-3,vmax=1e-3)
                
            else:
                ax[0].imshow(param.grad.view(-1,auxs.near_sqrt_divisor(param.numel())).detach().cpu(),
                    cmap='jet',aspect='auto',vmin=-1e-3,vmax=1e-3)
            
        if len(param_sq.shape)==1:
            ax[1].bar( list(range(0,len(param_sq))),t.squeeze(param.detach()).cpu() )
            ax[1].set_title(f"Param values, size:{list(param.shape)}")
            
        elif len(param_sq.shape)==2:
            ax[1].imshow(param.detach().cpu(),
                cmap='jet',aspect='auto')
            
        else:
            ax[1].imshow(param.view(-1,auxs.near_sqrt_divisor(param.numel())).detach().cpu(),
                cmap='jet',aspect='auto')
        return fig
        

#----------------------------------------------#
def out_2d(out,y,size=(7,7)): 
    
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
    fig,ax = plt.subplots()
    fig.set_size_inches(*size)
    ax.set_title('2D projection of model output',fontsize=13)
    ax.scatter(z[:,0], z[:,1], s=70, c=y.detach().cpu().numpy(), cmap="jet")
    return fig

