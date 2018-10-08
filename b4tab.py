import os
import pandas as pd
import numpy as np

# A tool for tabulated data preprocess

class col_core:
    def __init__(self,col_name,save_dir = ".matchbox/fields",debug=False):
        os.system("mkdir -p %s"%(save_dir))
        self.col_name = col_name
        self.debug = debug
        self.save_dir = save_dir
        if self.save_dir[-1]!="/": self.save_dir+="/"
            
        self.meta = dict()
    
    def save_meta(self):
        np.save(self.save_dir+str(self.col_name)+".npy",self.meta)
        
    def set_meta(self,meta=None):
        """
        pass meta dict values to obj attributes
        """
        if meta == None: meta = self.meta
        for k,v in meta.items():
            if self.debug: print("setting:\t%s\t%s"%(k,v))
            setattr(self,k,v)
        
    def load_meta(self,path=None):
        if path==None:
            path = self.save_dir+str(self.col_name)+".npy"
        self.meta = np.load(path).tolist()
        self.set_meta(self.meta)
        
        if self.meta["coltype"] == "tabulate": self.make_sub() # make sub-objects out of meta
        
    def make_meta(self):
        for attr in self.make_meta_list:
            self.meta[attr]  = getattr(self,attr)
            
    def check_dim(self,data):
        return pd.DataFrame(data,columns=self.dim_names)
    
class categorical(col_core):
    def __init__(self,col_name,save_dir = ".matchbox/fields"):
        super(categorical,self).__init__(col_name, save_dir)
        self.coltype = "categorical"
        self.make_meta_list = ["col_name","coltype","idx2cate","cate2idx","width","eye","dim_names"]
        
    def build(self,pandas_s,max_ = 20):
        assert max_>1, "max should be bigger than 1"
        
        vcount = pd.DataFrame(pandas_s.value_counts())
        
        print(vcount)
        
        self.cate_full = list(vcount.index.tolist())
        self.cate_list = self.cate_full[:max_-1]
        
        # build dictionary
        self.idx2cate = dict((k,v) for k,v in enumerate(self.cate_list))
        self.idx2cate.update({len(self.cate_list):"_other"})
        
        self.cate2idx = dict((v,k) for k,v in self.idx2cate.items())
        self.eye = np.eye(len(self.cate2idx))
        
        self.width = len(self.cate2idx)
        
        self.dim_names = list("%s -> %s"%(self.col_name,k) for k in self.cate2idx.keys())
        self.make_meta()
        
    def trans2idx(self,cate):
        """Translate category to index
        """
        try:
            return self.cate2idx[cate]
        except:
            return self.cate2idx["_other"]
        
    def prepro_idx(self,pandas_s):
        return pandas_s.apply(self.trans2idx)
    
    def prepro(self,pandas_s):
        return self.eye[self.prepro_idx(pandas_s).values.astype(int)]
    
class categorical_idx(col_core):
    def __init__(self,col_name,save_dir = ".matchbox/fields"):
        super(categorical_idx,self).__init__(col_name, save_dir)
        self.coltype = "categorical_idx"
        self.dim_names = [self.col_name]
        self.width = 1
        self.make_meta_list = ["col_name","coltype","idx2cate","cate2idx","width","dim_names"]
        
    def build(self,pandas_s,max_ = 20):
        assert max_>1, "max should be bigger than 1"
        
        vcount = pd.DataFrame(pandas_s.value_counts())
        
        print(vcount)
        
        self.cate_full = list(vcount.index.tolist())
        self.cate_list = self.cate_full[:max_-1]
        
        # build dictionary
        self.idx2cate = dict((k,v) for k,v in enumerate(self.cate_list))
        self.idx2cate.update({len(self.cate_list):"_other"})
        
        self.cate2idx = dict((v,k) for k,v in self.idx2cate.items())
        
        self.make_meta()
        
    def trans2idx(self,cate):
        try:
            return self.cate2idx[cate]
        except:
            return self.cate2idx["_other"]
        
    def prepro(self,pandas_s, expand=True):
        x = pandas_s.apply(self.trans2idx).values
        if expand:x = np.expand_dims(x,-1)
        return x
    
class minmax(col_core):
    def __init__(self,col_name,fillna=0.0,save_dir = ".matchbox/fields"):
        """minmax scaler: scale to 0~1"""
        super(minmax,self).__init__(col_name, save_dir)
        self.coltype = "minmax"
        self.fillna = fillna
        self.dim_names = [self.col_name]
        self.width = 1
        self.make_meta_list = ["col_name","coltype","min_","max_","range","width","dim_names"]
        
    def build(self,pandas_s=None,min_=None,max_=None):
        if type(pandas_s) != pd.core.series.Series:
            assert (min_!=None) and (max_!=None), "If no pandas series is set you have to set min_,max_ value"
            self.min_ = min_
            self.max_ = max_
            
        else:
            pandas_s = pandas_s.fillna(self.fillna)
            if min_ == None:
                self.min_ = pandas_s.min()
            else:
                self.min_ = min_
            if max_ == None:
                self.max_ = pandas_s.max()
            else:
                self.max_ = max_
                
        self.range = self.max_-self.min_
        assert self.range!=0, "the value range is 0"
        print("min_:%.3f \tmax_:%.3f\t range:%.3f"%(self.min_,self.max_,self.range))
        self.make_meta()
        
    def prepro(self,data,expand=True):
        x = (np.clip(data.values.astype(np.float64),self.min_,self.max_)-self.min_)/self.range
        if expand:x = np.expand_dims(x,-1)
        return x
        
class tabulate(col_core):
    def __init__(self,table_name,save_dir = ".matchbox/fields"):
        super(tabulate,self).__init__(table_name, save_dir)
        self.coltype = "tabulate"
        self.cols=dict()
        
        self.save_dir = save_dir
        if self.save_dir[-1] != "/":
            self.save_dir = "%s/"%(self.save_dir)
        
        self.make_meta_list = ["col_name","coltype","cols","dim_names"]
        
    def build_url(self,metalist):
        for url in metalist:
            meta_dict = np.load(url).tolist()
            self.cols[meta_dict["col_name"]] = meta_dict
        self.make_dim()
        self.make_meta()
        
    def build(self,*args):
        for obj in args:
            self.cols[obj.col_name] = obj.meta
        self.make_sub()
        self.make_dim()
        self.make_meta()
            
    def make_col(self,meta):
        """
        creat sub obj according to sub meta
        """
        col_name = meta["col_name"]
        
        setattr(self,col_name,eval(meta["coltype"])(col_name))
        getattr(self,col_name).set_meta(meta)
        if meta["coltype"] == "tabulate": 
            getattr(self,col_name).make_sub()
            getattr(self,col_name).meta = meta
        
    def make_sub(self):
        """
        create sub-objects according to meta
        """
        for k,meta in self.cols.items():
            self.make_col(meta)
        
    def make_dim(self):
        self.dim_names = []
        
        for k,meta in self.cols.items():
            for sub_dim in meta["dim_names"]:
                self.dim_names.append("%s -> %s"%(self.col_name, sub_dim))
            
        self.width = len(self.dim_names)
        
    def prepro(self,data):
        """
        data being a pandas dataframe
        """
        data_list = []
        
        for k,v in self.meta["cols"].items():
            # preprocess the data for every column
            col = getattr(self,k)
            if v["coltype"] == "tabulate":
                data_list.append(col.prepro(data))
            else:
                data_list.append(col.prepro(data[k]))
        return np.concatenate(data_list,axis = 1)

class mapper:
    def __init__(self,conf_path, original = None, old_default = None, new_default = None):
        """
        Handling mapping mechanism, all index mapping should be saved as config file
        [kwargs]
        
        conf_path: path of the configuration file, end with npy
        original: a list, will remove the duplicates
        old_default: defualt original value if the interpretaion failed
        new_default: defualt new value if the interpretaion failed
        
        user_map = mapper("conf/user_map.npy", user_Id_List )
        user_map.o2n will be the dictionary for original index to the new index
        user_map.n2o will be the dictionary for new index to the original index
        
        user_map = mappper("conf/user_map.npy") to load the conf file
        """
        self.conf_path = conf_path # config file path
        self.old_default = old_default
        self.new_default = new_default
        if original:
            self.original = list(set(original))
            self.mapping()
        else:
            self.load_map()
            
    def mapping(self):
        self.n2o = dict((k,v) for k,v in enumerate(self.original))
        self.o2n = dict((v,k) for k,v in enumerate(self.original))
        np.save(self.conf_path,{"n2o":self.n2o,"o2n":self.o2n})
        
    def load_map(self):
        dicts = np.load(self.conf_path).tolist()
        self.n2o = dicts["n2o"]
        self.o2n = dicts["o2n"]
        
    def spit_new(self,o_idx):
        try:
            return self.o2n[o_idx]
        except:
            return self.new_default
    
    def spit_old(self,n_idx):
        try:
            return self.n2o[n_idx]
        except:
            return self.old_default