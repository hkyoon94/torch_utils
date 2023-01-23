######################################################################################
#----------------- MONITORING TASKS FOR BUILDING LEARNING PIPELINES -----------------#
######################################################################################


import torch as t
import torch_utils as utils
from torch_utils import auxs as auxs
import numpy as np
from math import inf
import matplotlib.pyplot as plt



#-------------------------- LEARNING PROGRESS MONITORS ---------------------------#
class Prog():
  
    #------------------------------------------------------
    def print_epoch(self):
        print(f"Epoch: {self.epoch}")
      
    
    #------------------------------------------------------
    def loss_log(self): # logging basic loss
        s = f"Epoch: {self.epoch}\t| Loss |   Tr: {self.train_loss:.6f}"
        if self.forward_val is True:
            s = "   ".join([s,f"Val: {self.val_loss:.6f}"])
        if self.forward_test is True:
            s = "   ".join([s,f"Test: {self.test_loss:.6f}"])
        print(s)
        
        if self.save_dir is not False:
            self.log = open(f"{self.save_dir}/_logs.txt",'a')
            self.log.write(f"{s}\n"); self.log.close() 
      
  
  #------------------------------------------------------
    def acc_log(self): # logging basic maskwise accuracy 
        s = f"\t\t|  Acc |   Tr: {self.train_acc:.4f}%"
        if self.forward_val is True:
            s = "   ".join([s,f"Val: {self.val_acc:.4f}%"])
        if self.forward_test is True:
            s = "   ".join([s,f"Test: {self.test_acc:.4f}%"])
        print(s)
        
        if self.save_dir is not False:
            self.log = open(f"{self.save_dir}/_logs.txt",'a')
            self.log.write(f"{s}\n"); self.log.close() 
      
            
    #------------------------------------------------------
    ##### method drawing learning statistics progress
    def LA_plot(self,overlap = 4):
        
        ##### nested routines
        def draw_LA(inches, L_ind, A_ind):
            fig, ax = plt.subplots(1,2)
            fig.set_size_inches(*inches)
            fig.suptitle("Learning Statistics", fontsize = 15)
            i=0
            for key in [("Train",'b'),("Validation",'g'),("Test",'r')]:
                ax[0].plot( self.loss_epoch[L_ind], self.loss_arr[i,L_ind], 
                            color=key[1], linewidth=1.5, label = key[0] )
                ax[1].plot( self.acc_epoch[A_ind], self.acc_arr[i,A_ind], 
                            color=key[1], linewidth=1.5, label = key[0] )
                i+=1
            ax[0].legend()
            ax[0].grid(True)
            ax[0].set_xlabel("Epoch")
            ax[0].set_title(f"{self.criterion}")
                
            ax[1].legend()
            ax[1].grid(True)
            ax[1].set_xlabel("Epoch")
            ax[1].set_title("Accuracy(%)")         
            return fig
            
        def draw_L(inches, L_ind):
            fig, ax = plt.subplots()
            fig.set_size_inches(*inches)
            i=0
            for key in [("Train",'b'),("Validation",'g'),("Test",'r')]:
                ax.plot( self.loss_epoch[L_ind], self.loss_arr[i,L_ind], 
                          color=key[1], linewidth=1.5, label = key[0] )
                i+=1
            ax.legend()
            ax.grid(True)
            ax.set_xlabel("Epoch")
            ax.set_title(f"{self.criterion}")
            return fig
            
        ##### MAIN
        if self.status is None: # initializing
            self.LA_plot_ct = -1 
            fig = None
        
        elif self.status is True and self.L_ct >=1:
            self.LA_plot_ct += 1 # initiating monitor
            per = self.task_frequency # finding monitoring indices
            L_ind = (self.loss_epoch >= per*(self.LA_plot_ct - overlap)) \
                * (self.loss_epoch <= per*(self.LA_plot_ct + 1)) # bolean 'and' operation
            
            if hasattr(self, 'acc_arr'):
                A_ind = (self.acc_epoch >= per*(self.LA_plot_ct - overlap)) \
                    * (self.acc_epoch <= per*(self.LA_plot_ct + 1))
                fig = draw_LA((12,4), L_ind, A_ind)
            else: #! if acc is not saved (or, learning task is NOT classification)
                fig = draw_L((6,4), L_ind)
                    
        else: # after mainstream
            self.LA_plot_ct += 1
            
            L_ind = self.loss_epoch < inf # taking whole indices
            if hasattr(self, 'acc_arr'):
                A_ind = self.acc_epoch < inf
                fig = draw_LA((16,5.5), L_ind, A_ind)        
            else:
                fig = draw_L((8,5.5), L_ind)
                
        if fig is not None: 
            if self.streaming is True:
                plt.show()
            plt.close()
            if self.save_dir is not False:
                fig.savefig(f"{self.save_dir}/L_plot_final.jpg")
        
        
            
            

#--------------------------- MODEL PARAMETER MONITORS ----------------------------#
class Param():
    
    #------------------------------------------------------
    def P_max_mean_log(self): # logging parameter max, mean value
        s1="---------------------------------------------------------------------"
        s2="\tsize\tmax|grad|\tmean|grad|\tmax|val|\tmean|val|".expandtabs(12)
        s3="---------------------------------------------------------------------"
        s = "\n".join([s1,s2,s3])
        i=0
        for param in self.params:
            try: 
                tag = param.tag
            except: # if p has no tag
                tag = [" ",list(param.shape)]
            try:
                s_=f"Param {tag[0]}\t{tag[1]}\t{self.P_Mag[i]:.6f}\t{self.P_mag[i]:.6f}\t{self.P_Mav[i]:.6f}\t{self.P_mav[i]:.6f}".expandtabs(12)
                i+=1
            except: # if p's gradient is None
                s_=f"Param {tag[0]}\t{tag[1]}\t{self.P_Mag[i]}\t{self.P_mag[i]}\t{self.P_Mav[i]:.6f}\t{self.P_mav[i]:.6f}".expandtabs(12)
                i+=1
            s = "\n".join([s,s_])
        s_="---------------------------------------------------------------------"
        s = "\n".join([s,s_])
        
        if self.streaming is True:
            print(s)
        if self.save_dir is not False:
            self.log = open(f"{self.save_dir}/_logs.txt",'a')
            self.log.write(f"{s}\n"); self.log.close() 
    

    #------------------------------------------------------
    def P_max_mean_window(self,overlap=1): #method drawing parameter statistics progress
        
        ##### nested routine
        def draw(inches, ind, fmt, elinewidth, capsize):
            fig, ax = plt.subplots(1,2)
            fig.set_size_inches(*inches)
            fig.suptitle("Parameter Statistics",fontsize = 15)
            
            for i in range(0,self.n_params):
                ax[0].errorbar( x = self.P_Mm_epoch[ind], 
                                y = self.P_mag_arr[i,ind], fmt=fmt,
                                yerr = np.array([np.zeros(sum(ind)), np.squeeze(yerr0[i,ind])]), 
                                elinewidth=elinewidth, capsize=capsize, label = f"P{i}")
                ax[0].set_title("Max, Mean |grad|")
                ax[0].set_xlabel("Epoch")
                ax[0].legend()
                ax[0].grid(True)
                
                ax[1].errorbar( x = self.P_Mm_epoch[ind], 
                                y = self.P_mav_arr[i,ind], fmt=fmt,
                                yerr = np.array([np.zeros(sum(ind)), np.squeeze(yerr1[i,ind])]),
                                elinewidth=elinewidth, capsize=capsize )
                ax[1].set_title("Max, Mean |value|")
                ax[1].set_xlabel("Epoch")
                ax[1].grid(True)
            return fig
        
        ##### MAIN
        if self.status is None: # initializing
            self.P_Mm_window_ct = -1
            fig = None
        
        elif self.status is True and self.P_Mm_ct > 1: 
            self.P_Mm_window_ct += 1 # initiating monitor
            yerr0 = self.P_Mag_arr - self.P_mag_arr
            yerr1 = self.P_Mav_arr - self.P_mav_arr
            per = self.task_frequency # finding monitoring indices
            ind = (self.P_Mm_epoch >= per*(self.P_Mm_window_ct - overlap)) \
                    * (self.P_Mm_epoch <= per*(self.P_Mm_window_ct + 1)) # boolean 'and' operation
            
            fig = draw((12,4), ind, fmt='-.o', elinewidth=2, capsize=7)

        else: # after mainstream
            self.P_Mm_window_ct += 1
            yerr0 = self.P_Mag_arr - self.P_mag_arr
            yerr1 = self.P_Mav_arr - self.P_mav_arr
            ind = self.P_Mm_epoch < inf # taking whole indices
            fig = draw((16,5.5), ind, fmt='-.', elinewidth=2, capsize=7)
        if fig is not None:
            if self.streaming is True:
                plt.show()
            plt.close()
            if self.save_dir is not False:
                fig.savefig(f"{self.save_dir}/P_Mm_window_{self.P_Mm_window_ct}.jpg")
    
    
    #------------------------------------------------------
    def P_show(self):
        if self.status is None: # initializing
            self.P_show_ct = 0; size = (14,7)
        if self.status is True:
            size = (8,4)
        if self.status is False:
            size = (14,7)
        fig = utils.P_draw(self.params,size)
        self.P_show_ct += 1
        
        if self.streaming is True:
            plt.show()
        plt.close()
        if self.save_dir is not False:
            fig.savefig(f"{self.save_dir}/P_{self.P_show_ct}.jpg")



#----------------------------- MODEL STATE MONITORS ------------------------------#
class X(): 
    
    #------------------------------------------------------
    def out_2d(self): # Drawing TSNE 2D projection of whole output
        if self.status is None: # initializing
            self.out_2d_ct = 0
            size = (7,7)
            
        if self.status is True: # during mainstream
            size = (5,5)
            
        else:
            size = (7,7) # after mainstream
            
        if (self.forward_val is True) and self.forward_test is True:
            fig = utils.out_2d(t.cat((self.train_out,self.val_out,self.test_out)), 
                               t.cat((self.y_train,self.y_val,self.y_test)), size)
        elif (self.forward_val is False) and (self.forward_test is True):
            fig = utils.out_2d(t.cat((self.train_out,self.test_out)), 
                               t.cat((self.y_train,self.y_test)), size)
        elif (self.forward_val is True) and (self.forward_test is False):
            fig = utils.out_2d(t.cat((self.train_out,self.val_out)), 
                               t.cat((self.y_train,self.y_test)), size)
        else:
            fig = utils.out_2d(self.train_out,self.y_train, size)
            
        self.out_2d_ct += 1
        
        if self.streaming is True:
            plt.show()
        plt.close()
        if self.save_dir is not False:
            fig.savefig(f"{self.save_dir}/P_{self.out_2d_ct}.jpg")
            
            
    #------------------------------------------------------
    def out_2d_GNN(self): # Drawing TSNE 2D projection of whole output
        if self.status is None: # initializing
            self.out_2d_ct = 0
            size = (7,7)
        if self.status is True: # during mainstream
            size = (5,5)
        else:
            size = (7,7) # after mainstream
            
        fig = utils.out_2d(
            t.cat((self.out[self.D.train_mask],self.out[self.D.val_mask],self.out[self.D.test_mask])), 
            t.cat((self.D.y[self.D.train_mask],self.D.y[self.D.val_mask],self.D.y[self.D.test_mask])), size)

        self.out_2d_ct += 1
        
        if self.streaming is True:
            plt.show()
        plt.close()
        if self.save_dir is not False:
            fig.savefig(f"{self.save_dir}/P_{self.out_2d_ct}.jpg")