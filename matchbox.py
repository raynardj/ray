# -*- coding: utf-8 -*-
# tool box for pytorch

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math
from datetime import datetime
import os
import pandas as pd
from functools import  reduce
import math
from collections import Counter

# JUPYTER = True if "jupyter" in os.environ["_"] else False
JUPYTER = False

if JUPYTER:from tqdm import tqdm_notebook as tn

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
    def __init__(self,dataset,val_dataset = None,batch_size=16,
                 print_on=20,fields=None,is_log=True, shuffle = True, 
                 conn = None, modelName = "model", tryName = "try",time = "timestamp"):
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
        
        conn: a sql table connection, (sqlalchemy). if assigned value, save the record in the designated sql database;
        """
        self.batch_size=batch_size
        self.dataset = dataset
        self.conn = conn
        self.modelName = modelName
        self.tryName = tryName
        self.train_data = DataLoader(self.dataset,batch_size=self.batch_size, shuffle = shuffle)
        self.train_len=len(self.train_data)
        self.val_dataset=val_dataset
        self.print_on = print_on
        
        if self.val_dataset:
            self.val_dataset = val_dataset
            self.val_data = DataLoader(self.val_dataset,batch_size=self.batch_size ,shuffle = shuffle)
            self.val_len=len(self.val_data)
            self.val_track=dict()
            
        self.track=dict()
        self.fields=fields
        self.is_log=is_log
        
    def get_time(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def train(self,epochs,name=None,log_addr=None):
        """
        Train the model
        """
        if name==None:
            name="torch_train_"+datetime.now().strftime("%y%m%d_%H%M%S")
        if log_addr==None:
            log_addr=".log_%s"%(name)
            
        if log_addr[-1]!="/": log_addr += "/"
            
        for epoch in range(epochs):
            self.track[epoch]=list()
            self.run(epoch)
        if self.is_log:
            os.system("mkdir -p %s"%(log_addr))
            trn_track = pd.DataFrame(reduce((lambda x,y:x+y),list(self.track.values())))
            trn_track = trn_track.to_csv(log_addr+"trn_"+datetime.now().strftime("%y_%m_%d__%H_%M_%S")+".csv",index=False)
            
            if self.val_dataset:
                
                val_track = pd.DataFrame(reduce((lambda x,y:x+y),list(self.val_track.values())))
                val_track.to_csv(log_addr+"val_"+datetime.now().strftime("%y_%m_%d__%H_%M_%S")+".csv",index=False)
    
    def run(self,epoch):
        if JUPYTER:
            t=tn(range(self.train_len))
        else:
            t=trange(self.train_len)
        self.train_gen = iter(self.train_data)
        
        for i in t:
            
            ret = self.action(next(self.train_gen),epoch=epoch,ite=i)
            ret.update({"epoch":epoch,
                        "iter":i,
                        "ts":self.get_time()})
            self.track[epoch].append(ret)
            
            if i%self.print_on==self.print_on-1:
                self.update_descrition(epoch,i,t)
                
        if self.val_dataset:
            
            self.val_track[epoch]=list()
            self.val_gen = iter(self.val_data)
            if JUPYTER:
                val_t=tn(range(self.val_len))
            else:
                val_t=trange(self.val_len)
            
            for i in val_t:
                ret = self.val_action(next(self.val_gen),epoch=epoch,ite=i)
                ret.update({"epoch":epoch,
                            "iter":i,
                            "ts":self.get_time()})
                self.val_track[epoch].append(ret)
                
                #print(self.val_track)
                self.update_descrition_val(epoch,i,val_t)
                    
    def update_descrition(self,epoch,i,t):
        window_df = pd.DataFrame(self.track[epoch][max(i-self.print_on,0):i])

        if self.conn: # if saving to a SQL database
            window_df["split_"] = "train"
            window_df["tryName"] = self.tryName+"_train"
            window_df.to_sql("track_%s"%(self.modelName),con = self.conn, if_exists = "append", index = False)
        window_dict = dict(window_df.mean())
        del window_dict["epoch"]
        del window_dict["iter"]
        
        desc = "⭐[ep_%s_i_%s]"%(epoch,i)
        if JUPYTER:
            t.set_postfix(window_dict)
        else:
            if self.fields!=None:
                desc += "✨".join(list("\t%s\t%.3f"%(k,v) for k,v in window_dict.items() if k in self.fields))
            else:
                desc += "✨".join(list("\t%s\t%.3f"%(k,v) for k,v in window_dict.items() ))
        t.set_description(desc)

        
    def update_descrition_val(self,epoch,i,t):
        if self.conn: # if saving to a SQL database
            window_df = pd.DataFrame(self.val_track[epoch][max(i-self.print_on,0):i])
            window_df["split_"] = "valid"
            window_df["tryName"] = self.tryName+"_valid"
            window_df.to_sql("track_%s"%(self.modelName),con = self.conn, if_exists = "append", index = False)
        window_dict = dict(pd.DataFrame(self.val_track[epoch]).mean())
        #print(pd.DataFrame(self.val_track[epoch]))
        del window_dict["epoch"]
        del window_dict["iter"]
        
        desc = "😎[val_ep_%s_i_%s]"%(epoch,i)
        if JUPYTER:
            t.set_postfix(window_dict)
        else:
            if self.fields!=None:
                desc += "😂".join(list("\t%s\t%.3f"%(k,v) for k,v in window_dict.items() if k in self.fields))
            else:
                desc += "😂".join(list("\t%s\t%.3f"%(k,v) for k,v in window_dict.items()))
        t.set_description(desc)
            
    def todataframe(self,dict_):
        """return a dataframe on the train log dictionary"""
        tracks=[]
        for i in range(len(dict_)):
            tracks += dict_[i]

        return pd.DataFrame(to)
    
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
    
def load_model(model,path):
    model.load_state_dict(torch.load(path))

def supermean(x):
    """return mean, incase denomenator is 0 return 1e-6 (a very small number)"""
    if x.size()[0]==0:
        rt = torch.FloatTensor([1e-6])
        if CUDA:
            rt.cuda()
    else:
        rt = x.mean()
    return rt

def f1_score(y_pred,y_true):
    """
    f1 score for multi label classification problem
    y_pred will be the same shape with y_true
    y_true will prefer long tensor format
    """
    y_pred_tf = (F.sigmoid(y_pred)>.5)
    guess_right = (y_pred_tf.float()==y_true).float()
    
    accuracy = supermean(guess_right)
    recall = supermean(guess_right[y_true.byte()])
    precision = supermean(guess_right[y_pred_tf.byte()])
    
    f1  = 2*(recall*precision)/(recall+precision)
    return accuracy,recall,precision,f1

class DF_Dataset(Dataset):
    def __init__(self,df, bs,prepro=lambda x:x.values,shuffle=False):
        """
        df_dataset, a dataset for slicing pandas dataframe,instead of single indexing and collate
        Please use batch_size=1 for dataloader, and define the batch size here

        eg.
        ```
        ds = DF_Dataset(data_df,bs=1024)
        ```
        """
#         super(DF_Dataset, self).__init__()
        
        if shuffle: 
            print("shuffling")
            df = df.sample(frac=1.).reset_index().drop("index",axis=1)
            print("shuffled") # shuffle the data here
        
        self.df = df
        self.prepro = prepro
        self.bs = bs

    def __len__(self):
        return math.ceil(len(self.df) / self.bs)

    def __getitem__(self, idx):
        start = idx * self.bs
        end = (idx + 1) * self.bs
#         print(type(self.x_prepro(self.df[start:end])),type(self.y_prepro(self.df[start:end])))
        return self.prepro(self.df[start:end])

class Arr_Dataset(Dataset):
    def __init__(self, *args, bs):
        """
        arr_dataset, a dataset for slicing numpy array,instead of single indexing and collate
        Please use batch_size=1 for dataloader, and define the batch size here

        eg.
        ```
        ds = Arr_Dataset(arr_1,arr_2,arr_3,bs = 512)
        ```
        """
        super(Arr_Dataset, self).__init__()
        self.arrs = args
        self.bs = bs

    def __len__(self):
        return math.ceil(len(self.arrs[0]) / self.bs)

    def __getitem__(self, idx):
        start = idx * self.bs
        end = (idx + 1) * self.bs
        return tuple(self.arrs[i][start:end] for i in range(len(self.arrs)))
    
class Seq_Dataset(Dataset):
    """
    a mechanism for reading sequence data, the preparation stage of nlp learning
    """
    def __init__(self,
                 seqname,seq,seq_len=None,vocab_path=None,
                 bs=16,vocab_size=10000,build_vocab=False,sep_tok = "<tok>",discard_side="right",element_type="int"):
        '''
        seq: enumerable sequence
        vocab_path: path to vocabulary json file
        threads:process number
        element_type: "int" or "float"
        '''
        self.seqname = seqname
        print(seqname,"sequence type:",type(seq))
        self.process_funcs = []
        self.seq = list(seq)
        self.seq_len = seq_len
        self.vocab_path = vocab_path
        self.vocab_size = int(vocab_size)

        self.bs = bs
        self.N = len(self.seq)
        self.BN = math.ceil(self.N/self.bs)
        print(seqname,"sequence total_length type:",self.N)
        
        self.sep_tok = sep_tok if sep_tok!=None else ""
        self.make_breaker()
        self.discard_side = discard_side
        if self.seq_len:
            self.make_discard()
        
        if vocab_path:
            if build_vocab==False and os.path.exists(vocab_path)==False:
                print("vocab path not found, building vocab path anyway")
                build_vocab=True
            if build_vocab:
                self.vocab = self.build_vocab(self.seq)
                self.vocab.to_json(self.vocab_path)
            else:
                self.vocab = pd.read_json(self.vocab_path)
            self.char2idx,self.idx2char = self.get_mapping(self.vocab)
            self.make_translate()
        self.element_type = element_type
        self.make_totorch()

    def __len__(self):
        return self.BN
    
    def __getitem__(self,idx):
        seq_crop = self.seq[idx:idx+self.bs]
        return self.process_batch(seq_crop)
    
    def make_breaker(self):
        if self.sep_tok!="":
            def breaker(self,line):
                return str(line).split(self.sep_tok)
        else:
            def breaker(self,line):
                return list(str(line))
        self.breaker=breaker
        self.process_funcs.append(self.breaker)
        
    def make_discard(self):      
        if self.discard_side=="right":
            def discard(self,seqlist):
                return seqlist[:self.seq_len]
        if self.discard_side=="left":
            def discard(self,seqlist):
                return seqlist[-self.seq_len:]
        self.discard = discard
        self.process_funcs.append(self.discard)
        
    def make_translate(self):
        def translate(self,x):
            return np.vectorize(self.mapfunc)(x)
        self.translate = translate
        self.process_funcs.append(self.translate)
        
    def make_totorch(self):
        if self.element_type == "int":
            def totorch(self,seq):
                return torch.LongTensor(np.array(seq).astype(np.int))
        if self.element_type == "float":
            def totorch(self,seq):
                return torch.FloatTensor(np.array(seq).astype(np.float32))
        self.totorch = totorch
        self.process_funcs.append(self.totorch)
    
    def process_batch(self,batch):
        return nn.utils.rnn.pad_sequence(np.vectorize(self.seq2idx,otypes=[list])(batch),batch_first = True)
    
    def seq2idx(self,x):
        for f in self.process_funcs: x = f(self,x)
        return x
    
    def get_mapping(self,vocab_df):
        char2idx=dict(zip(vocab_df["token"],vocab_df["idx"]))
        idx2char=dict(zip(vocab_df["idx"],vocab_df["token"]))
        return char2idx,idx2char
    
    def mapfunc(self,x):
        """
        from token to index number
        """
        try:
            return self.char2idx[x]
        except:
            return 1
        
    def get_token_count_dict(self,full_token):
        """count the token to a list"""
        return Counter(full_token)    
    
    def get_full_token(self,list_of_tokens):
        """
        From a list of list of tokens, to a long list of tokens, duplicate tokens included
        """
        return self.breaker(self,self.sep_tok.join(list_of_tokens))
    
    def build_vocab(self,seq_list):
        ct_dict = self.get_token_count_dict(self.get_full_token(seq_list))
        ct_dict["SOS_TOKEN"] = 9e9
        ct_dict["EOS_TOKEN"] = 8e9
        ct_dict[" "] = 7e9
        tk,ct = list(ct_dict.keys()),list(ct_dict.values())
        
        token_df=pd.DataFrame({"token":tk,"count":ct}).sort_values(by="count",ascending=False)
        return token_df.reset_index().drop("index",axis=1).reset_index().rename(columns={"index":"idx"}).fillna("")[:self.vocab_size]

class fuse(Dataset):
    def __init__(self,*datasets):
        self.datasets = datasets
        bs_s = set(list(d.bs for d in self.datasets))
        length_s = set(list(len(d) for d in self.datasets))
        assert len(bs_s) ==1, "batch sized not matched"
        assert len(length_s) ==1, "dataset lenth not matched"
        self.bs = list(bs_s)[0]
        self.length = list(length_s)[0]
        
    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        return tuple(d.__getitem__(idx) for d in self.datasets)
    
def collate(batch):
    return tuple(i[0] for i in zip(*batch))