# -*- coding: utf-8 -*-
# tool box for pytorch

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import trange
from torch.utils.data import DataLoader
from datetime import datetime
import os
import pandas as pd

def p_structure(md):
    """Print out the parameter structure"""
    for par in md.parameters():
        print(par.size())

def p_count(md):
    """count the parameters in side a pytorch module"""
    allp=0
    for p in md.parameters():allp+=np.product(p.data.numpy().shape)
    return allp

class Trainer:
    def __init__(self,dataset,val_dataset = None,batch_size=16,print_on=20,fields=None,is_log=True):
        """
        Pytorch trainer
        fields: the fields you choose to print out
        is_log: writing a log？
        
        Training:
        
        write action funtion for a step of training,
        assuming a generator will spit out tuple x,y,z in each:
        
        def action(*args,**kwargs):
            x,y,z = args[0]
            x,y,z = Variable(x).cuda(),Variable(y).cuda(),Variable(z).cuda()
            
            #optimizer is a global variable, or many different optimizers if you like
            sgd.zero_grad()
            adam.zero_grad()
            
            # model is a global variable, or many models if you like
            y_ = model(x)
            y2_ = model_2(z)
            
            ...... more param updating details here
            
            return {"loss":loss.data[0],"acc":accuracy.data[0]}
            ...
        
        then pass the function to object 
        trainer=Trainer(...)
        trainer.action=action
        trainer.train(epochs = 30)
        
        same work for validation:trainer.val_action = val_action
        """
        self.batch_size=batch_size
        self.dataset = dataset
        self.train_data = DataLoader(self.dataset,batch_size=self.batch_size, shuffle=True)
        self.train_len=len(self.train_data)
        self.val_dataset=val_dataset
        self.print_on = print_on
        
        if self.val_dataset:
            self.val_dataset = val_dataset
            self.val_data = DataLoader(self.val_dataset,batch_size=self.batch_size)
            self.val_len=len(self.val_data)
            self.val_track=dict()
            
        self.track=dict()
        self.fields=fields
        self.is_log=is_log
        
    def train(self,epochs,name=None,log_addr=None):
        """
        Train the model
        """
        if name==None:
            name="torch_train_"+datetime.now().strftime("%y%m%d_%H%M%S")
        if log_addr==None:
            log_addr=name
            
        if log_addr[-1]!="/": log_addr += "/"
            
        for epoch in range(epochs):
            self.track[epoch]=list()
            self.run(epoch)
        if self.is_log:
            os.system("mkdir -p %s"%(log_addr))
            trn_track = pd.DataFrame(list(v for v in self.track.items()))
            trn_track = trn_track.to_csv(log_addr+"trn_"+datetime.now().strftime("%y%m%d_%H%M%S")+".csv")
            
            if self.val_dataset:
                
                val_track = pd.DataFrame(list(v for v in self.val_track.items()))
                val_track.to_csv(log_addr+"trn_"+datetime.now().strftime("%y%m%d_%H%M%S")+".csv")
    
    def run(self,epoch):
        t=trange(self.train_len)
        self.train_gen = iter(self.train_data)
        
        for i in t:
            
            ret = self.action(next(self.train_gen),epoch=epoch,ite=i)
            ret.update({"epoch":epoch,"iter":i})
            self.track[epoch].append(ret)
            
            if i%self.print_on==self.print_on-1:
                self.update_descrition(epoch,i,t)
                
        if self.val_dataset:
            
            self.val_track[epoch]=list()
            self.val_gen = iter(self.val_data)
            val_t = trange(self.val_len)
            
            for i in val_t:
                ret = self.val_action(next(self.val_gen),epoch=epoch,ite=i)
                ret.update({"epoch":epoch,"iter":i})
                self.val_track[epoch].append(ret)
                
                #print(self.val_track)
                self.update_descrition_val(epoch,i,val_t)
                    
    def update_descrition(self,epoch,i,t):
        window_dict = dict(pd.DataFrame(self.track[epoch][max(i-self.print_on,0):i]).mean())
        
        del window_dict["epoch"]
        del window_dict["iter"]
        
        desc = "⭐[ep_%s_i_%s]"%(epoch,i)
        if self.fields!=None:
            desc += "✨".join(list("\t%s\t%.3f"%(k,v) for k,v in window_dict.items() if k in self.fields))
        else:
            desc += "✨".join(list("\t%s\t%.3f"%(k,v) for k,v in window_dict.items() ))
        t.set_description(desc)
        
    def update_descrition_val(self,epoch,i,t):
        window_dict = dict(pd.DataFrame(self.val_track[epoch]).mean())
        #print(pd.DataFrame(self.val_track[epoch]))
        del window_dict["epoch"]
        del window_dict["iter"]
        
        desc = "😎[val_ep_%s_i_%s]"%(epoch,i)
        if self.fields!=None:
            desc += "😂".join(list("\t%s\t%.3f"%(k,v) for k,v in window_dict.items() if k in self.fields))
        else:
            desc += "😂".join(list("\t%s\t%.3f"%(k,v) for k,v in window_dict.items()))
        t.set_description(desc)
            
    def todataframe(self,dict_):
        """return a dataframe on the train log dictionary"""
        tracks=[]
        for i in range(len(dict_)):
            tracks+=dict_[i]

        return pd.DataFrame(tracks)
    
    def save_track(self,filepath,val_filepath=None):
        """
        Save the track to csv files in a path you designated,
        :param filepath: a file path ended with .csv
        :return: None
        """
        self.todataframe(self.track).to_csv(filepath,index=False)
        if val_filepath:
            self.todataframe(self.val_track).to_csv(val_filepath,index=False)

def clip_weight(model,clamp_lower=-1e-2,clamp_upper=1e-2):
    for p in model.parameters():
        p.data.clamp_(clamp_lower, clamp_upper)


def argmax(x):
    """
    Arg max of a torch tensor (2 dimensional, dim=1)
    :param x:  torch tensor
    :return: index the of the max
    """
    return torch.max(x, dim=1)[1]


def accuracy(y_pred, y_true):
    """

    :param y_pred: predition of y (will be argmaxed)
    :param y_true: true label of y (index)
    :return:
    """
    return (argmax(y_pred) == y_true).float().mean()


def save_model(model, path):
    """
    model:pytorch model
    path:save to path, end with pkl
    """
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))

