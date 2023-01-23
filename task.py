######################################################################################
#----------------------- CALCULATING ROUTINESES FOR MONITORS ------------------------#
######################################################################################

import torch as t
from torch_utils import get_acc, get_P_max_mean
from math import nan
import numpy as np


#----------------------- PROGRESS INFO COMPUTING ROUTINES ------------------------#
class Prog():
  
    def idle(self): # do nothing
        pass

    def train_batch(self): # basic single-loss minimizing routine
        
        for self.x_train_batch, self.y_train_batch in self.trainloader: # grab batch & computing loss
            self.x_train_batch, self.y_train_batch = \
                self.x_train_batch.to(self.device), self.y_train_batch.to(self.device) 
            self.model.train()
            self.train_out_batch = self.model(self.x_train_batch)
            self.train_loss_batch = self.criterion(self.train_out_batch,self.y_train_batch)
            
            if self.status is True: # computing gradients for each mini-batch
                self.optimizer.zero_grad()
                self.train_loss_batch.backward(retain_graph = self.retain_graph) #! BACKWARD ROUTINE
                self.optimizer.step()
    
    
    def loss(self):
        
        if self.status is None: # initializing
            per = self.task_frequency
            self.loss_epoch = np.array(list(range(0,self.total_epoch+per+1,per)))
            self.loss_length = len(self.loss_epoch)
            self.loss_arr = nan*np.ones((3,self.loss_length))
            self.L_ct = -1 # set task counter
        
        # during mainstream & end
        self.L_ct += 1
        
        self.model.eval()
        self.train_out = self.model(self.x_train.to(self.device))
        self.train_loss = self.criterion(self.train_out,self.y_train.to(self.device)).item()
        self.loss_arr[0,self.L_ct] = self.train_loss
        if self.forward_val is True: # get loss for validation set
            self.model.eval()
            self.val_out = self.model(self.x_val.to(self.device))
            self.val_loss = self.criterion(self.val_out,self.y_val.to(self.device)).item()
            self.loss_arr[1,self.L_ct] = self.val_loss
        if self.forward_test is True: # get loss for train set
            self.model.eval()
            self.test_out = self.model(self.x_test.to(self.device))
            self.test_loss = self.criterion(self.test_out,self.y_test.to(self.device)).item()
            self.loss_arr[2,self.L_ct] = self.test_loss
    
    
    def acc_maskwise(self): # computing maskwise accuracy if current task is classification        
        
        if self.status is None: # initializing
            per = self.task_frequency
            self.acc_epoch = np.array(list(range(0,self.total_epoch+per+1,per)))
            self.acc_length = len(self.acc_epoch)
            self.acc_arr = nan*np.ones((3,self.acc_length))
            self.A_ct = -1    # set task counter
        
        # during mainstream & end
        self.A_ct += 1
        self.train_acc = get_acc(self.train_out,self.y_train.to(self.device))
        self.acc_arr[0,self.A_ct] = self.train_acc
        if self.forward_val is True: # get accuracy for validation set
            self.val_acc = get_acc(self.val_out,self.y_val.to(self.device))
            self.acc_arr[1,self.A_ct] = self.val_acc
        if self.forward_test is True: # get accuracy for validation set
            self.test_acc = get_acc(self.test_out,self.y_test.to(self.device))
            self.acc_arr[2,self.A_ct] = self.test_acc
            
            
    def acc_confusion(self):
        pass
            
            
    #-----------------------------------------------------------------------------------------------#
    def loss_GNN_node_classification(self):

        if self.status is None: # initializing
            per = self.task_frequency
            self.loss_epoch = np.array(list(range(0,self.total_epoch+per+1,per)))
            self.loss_length = len(self.loss_epoch)
            self.loss_arr = nan*np.ones((3,self.loss_length))
            self.L_ct = -1 # set task counter

        self.L_ct += 1
        self.out = self.model.forward(self.x,self.G)
        
        self.train_loss = self.criterion(self.out[self.trm],self.y[self.trm])
        self.train_loss_item = self.train_loss.item()
        self.loss_arr[0,self.L_ct] = self.train_loss.item()
        self.val_loss = self.criterion(self.out[self.vm],self.y[self.vm])
        self.loss_arr[1,self.L_ct] = self.val_loss
        self.test_loss = self.criterion(self.out[self.tm],self.y[self.tm])
        self.loss_arr[2,self.L_ct] = self.test_loss
        
        if self.status is True: # computing gradient during mainstream
            self.optimizer.zero_grad()
            self.train_loss.backward(retain_graph = self.retain_graph) #! BACKWARD ROUTINE
            self.optimizer.step()
 
 
    def acc_maskwise_GNN_node_classification(self): # computing maskwise accuracy if current task is classification        
        
        if self.status is None: # initializing
            per = self.task_frequency
            self.acc_epoch = np.array(list(range(0,self.total_epoch+per+1,per)))
            self.acc_length = len(self.acc_epoch)
            self.acc_arr = nan*np.ones((3,self.acc_length))
            self.A_ct = -1    # set task counter
        
        # during mainstream & end
        self.A_ct += 1
        self.train_acc = get_acc(self.out[self.trm],self.y[self.trm])
        self.acc_arr[0,self.A_ct] = self.train_acc
        self.val_acc = get_acc(self.out[self.vm],self.y[self.vm])
        self.acc_arr[1,self.A_ct] = self.val_acc
        self.test_acc = get_acc(self.out[self.tm],self.y[self.tm])
        self.acc_arr[2,self.A_ct] = self.test_acc 
        
        
    def acc_confusion_GNN_node_classification(self):
        pass



#------------------- PARAMETER STATISTICS COMPUTING ROUTINES --------------------#
class Param():
    
    #------------------------------------------------------
    def P_max_mean(self): # saving parameter max & mean abolute values
        if self.status is None: # initializing
            per = self.task_frequency
            self.P_Mm_epoch = np.array(list(range(0,self.total_epoch+per+1,per)))
            self.P_Mm_length = len(self.P_Mm_epoch)
            self.P_Mm_ct = -1 # set task counter
            
            self.P_Mag_arr = nan*np.ones((self.n_params,self.P_Mm_length))
            self.P_mag_arr = nan*np.ones((self.n_params,self.P_Mm_length))
            self.P_Mav_arr = nan*np.ones((self.n_params,self.P_Mm_length))
            self.P_mav_arr = nan*np.ones((self.n_params,self.P_Mm_length))
            
        # during mainstream & end
        self.P_Mm_ct += 1
        self.P_Mag, self.P_mag, self.P_Mav, self.P_mav = get_P_max_mean(self.params)
        self.P_Mag_arr[:,self.P_Mm_ct] = self.P_Mag
        self.P_mag_arr[:,self.P_Mm_ct] = self.P_mag
        self.P_Mav_arr[:,self.P_Mm_ct] = self.P_Mav
        self.P_mav_arr[:,self.P_Mm_ct] = self.P_mav
        self.P_Mm_epoch[self.P_Mm_ct] = self.epoch
        
        
            


#------------------------------ UTILITY ROUTINES ------------------------------#
class utils():
    
    def search_pausefile(self): ##### determining pause and go (or stop)

        pass