######################################################################################
#-------------------------- LEARNING PIPELINES FOR PYTORCH --------------------------#
######################################################################################

from torch.utils.data import DataLoader
import torch_utils as utils
from torch_utils import monitor, task
import torch
import dill
import matplotlib.pyplot as plt
import os
from time import time
import warnings; warnings.filterwarnings(action='ignore')


#------------------------------------------ PIPELINE BUILDER ----------------------------------------#
def Builder( data, model ):
    
    if data.tasktype == 'simple_classification':
        return Simple(data,model)
    
    elif data.tasktype == 'GNN_node_classification':
        return GNN_node_classification(data,model)
    
    elif data.tasktype == 'GNN_graph_classification':
        return GNN_graph_classification(data,model)


#-------------------------------------- BASIC LEARNING PIPELINE -------------------------------------#
class Simple:
    def __init__( self, data, model ):
        print(f"\nPipeline type: {data.tasktype}")
        self.data = data; self.model = model
        self.params = list(model.parameters())
        self.n_params = len(self.params)
        for i in range(0,len(self.params)):
            setattr(self.params[i],"tag",(i,list(self.params[i].shape)))
        
        self.map_task()
        self.loss_log = monitor.Prog.loss_log
        self.acc_log = monitor.Prog.acc_log
        self.LA_plot = monitor.Prog.LA_plot
        self.P_show = monitor.Param.P_show
        self.P_max_mean = task.Param.P_max_mean
        self.P_max_mean_log = monitor.Param.P_max_mean_log
        self.P_max_mean_window = monitor.Param.P_max_mean_window
        
    
    def learn(self, criterion, optimizer, total_epoch, tasklist=[],
              batch_size=False, shuffle=True, gpu=False,
              is_regression=False, forward_val=False, forward_test=False,
              streaming=True, save_dir=False, search_pause_per=False, retain_graph=False):
        
        self.criterion = criterion; self.optimizer = optimizer
        self.total_epoch = total_epoch; self.tasklist = tasklist
        self.batch_size = batch_size; self.shuffle = shuffle; self.gpu = gpu
        self.is_regression = is_regression
        self.forward_val = forward_val; self.forward_test = forward_test
        self.streaming = streaming; self.save_dir = save_dir
        self.search_pause_per = search_pause_per; self.retain_graph = retain_graph
        self.device = torch.device("cuda:0" if self.gpu is True else "cpu")
        self.set_data()
        
        ##### initializing block ---------------------------------------------------------
        self.status = None; self.epoch = 0
        if self.save_dir is not False: # initializing log file
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        self.write_log('init')
        self.model = self.model.to(self.device)
        for self.task, self.task_frequency in self.tasklist: # initializing all tasks
            self.task(self)
        
        ##### mainstream block -----------------------------------------------------------
        self.status = True
        self.write_log('start')
        try:
            for self.epoch in range(1,self.total_epoch+1):
                self.train_batch(self)
                for self.task, self.task_frequency in self.tasklist: # executing main and tasks
                    if self.epoch % self.task_frequency == 0:
                        self.task(self)
            self.write_log('end')
            
        except KeyboardInterrupt: # stopping learning with ctrl+c
            self.write_log('interrupt')

        ##### final monitoring block -----------------------------------------------------
        self.status = False
        self.write_log('final')
        for self.task, self.task_frequency in self.tasklist: # executing terminal tasks
            self.task(self)
        
        if self.save_dir is not False: # saving whole structure
            with open(f"{self.save_dir}/_result.pickle","wb") as fw:
                dill.dump(self, fw)
            torch.save({'model': self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'criterion': self.criterion,
                                    'epoch': self.epoch,
                                    'loss': self.train_loss,
                                    }, f"{self.save_dir}/_checkpoint.tar")
                
    def write_log(self,key):
        if key=='init':
            print("\n---------------------------- INITIAL INFO ---------------------------\n")
            if self.save_dir is not False:
                self.log = open(f"{self.save_dir}/_logs.txt",'w')
                self.log.write("\n---------------------------- INITIAL INFO ---------------------------\n")
                self.log.write(f"Total Epochs: {self.total_epoch}\tIterations per Epoch: {self.dataloader.iter_num}")
                self.log.close()
        if key=='start':
            print("\n------------------------- STARTED LEARNING --------------------------\n")
            if self.save_dir is not False:
                self.log = open(f"{self.save_dir}/_logs.txt",'a')
                self.log.write("\n------------------------- STARTED LEARNING --------------------------\n") 
                self.log.close()
            self.start_time = time()
        if key=='end':
            T = time() - self.start_time
            print("\n------------------------- FINISHED LEARNING -------------------------\n")
            print(f"Total elapsed time is {T:.6f} (per epoch: {T/(self.epoch+1):.6f}) seconds.\n")
            if self.save_dir is not False:
                self.log = open(f"{self.save_dir}/_logs.txt",'a')
                self.log.write("\n------------------------- FINISHED LEARNING -------------------------\n")
                self.log.close()
        if key=='interrupt':
            T = time() - self.start_time
            print(f"\n---------------------- TERMINATED AT EPOCH {self.epoch} ---------------------\n")
            print(f"Total elapsed time is {T:.6f} (per epoch: {T/(self.epoch+1):.6f}) seconds.\n")
            if self.save_dir is not False:
                self.log = open(f"{self.save_dir}/_logs.txt",'a')
                self.log.write(f"\n---------------------- TERMINATED AT EPOCH {self.epoch} ---------------------\n")
                self.log.close()
        if key=='final':
            print("\n--------------------------- FINAL RESULTS ---------------------------\n")
            if self.save_dir is not False:
                self.log = open(f"{self.save_dir}/_logs.txt",'a')
                self.log.write("\n\n\n--------------------------- FINAL RESULTS ---------------------------\n")
                self.log.close()    
        
        
#! SHOULD BE OVERRIDED FOR DIFFERENT LEARNING STRUCTURE
    def map_task(self):
        self.train_batch = task.Prog.train_batch
        self.loss = task.Prog.loss
        self.acc_maskwise = task.Prog.acc_maskwise
        self.acc_confusion = task.Prog.acc_confusion
        self.out_2d = monitor.X.out_2d
    
    def set_data(self):
        if self.batch_size is not False:
            self.batch_size = self.batch_size
        else:
            self.batch_size = self.data.num_train
        self.trainloader =\
            DataLoader(self.data,batch_size=self.batch_size,shuffle=self.shuffle,pin_memory=self.gpu)
        self.x_train = self.data.x_train; self.y_train = self.data.y_train
        self.x_val = self.data.x_val; self.y_val = self.data.y_val
        self.x_test = self.data.x_test; self.y_test = self.data.y_test
        
    def infer(self,x,y): # BASIC INFERENCE
        out = self.model(x.to(self.device))
        loss = self.criterion(out,y.to(self.device))
        if self.is_regression is False:
            acc = utils.get_acc(out,y.to(self.device))
            print(f"Loss: {loss.item():.6f}, Accuracy: {acc:.4f}%")
            utils.out_2d(out,y.to(self.device)); plt.show(); plt.close()
        else:
            print(f"Loss: {loss.item():.6f}")
#! SHOULD BE OVERRIDED FOR DIFFERENT LEARNING STRUCTURE




#-------------------------------------- FOR GNN NODE CLASSIFICATION -------------------------------------#
class GNN_node_classification(Simple):
    
    def __init__(self,*args):
        super().__init__(*args)
        
    def map_task(self):
        self.train_batch = task.Prog.idle
        self.loss = task.Prog.loss_GNN_node_classification
        self.acc_maskwise = task.Prog.acc_maskwise_GNN_node_classification
        self.acc_confusion = task.Prog.acc_confusion_GNN_node_classification
        self.out_2d = monitor.X.out_2d_GNN
    
    def set_data(self):
        self.x = self.data.x.to(self.device); self.y = self.data.y.to(self.device)
        self.G = self.data.G.to(self.device)
        self.trm = self.data.trm.to(self.device); self.vm = self.data.vm.to(self.device)
        self.tm = self.data.tm.to(self.device)    




#-------------------------------------- FOR GNN GRAPH CLASSIFICATION ------------------------------------#
class GNN_graph_classification(Simple):
    def __init__(self,*args):
        super().__init__(*args)
    
    def map_task(self):
        pass
    
    def set_data(self):
        pass