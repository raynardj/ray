# -*- coding: utf-8 -*-
# Ray's private ML/DL code armory
from utils2 import *
import pandas as pd
from IPython.display import display
from PIL import Image
from pymongo import MongoClient

class prepro(object):
    def __init__(self,df):
        """
        Initialize the object with dataframe
        ==================
        Try self.check_col(Column_name)
        ==================
        """
        self.df=df
        self.cols=list(df.columns)
        self.cols_types=dict()
        self.length=len(self.df)
        print("%s columns in total"%(len(self.df.columns)))
        print("DataFrame Loaded")
    
    def set_label(self,label_col):
        """
        Set the label column name
        Input a str or list of str
        """
        if type(label_col).__name__=="str":
            self.label=list([label_col])
        elif type(label_col).__name__=="list":
            self.label=label_col
    def check_col(self,col_name):
        """
        Check the column by column name
        """
        if col_name in self.cols:
            col_s=self.df[col_name]
            col_df=pd.DataFrame(col_s)
            col_min=col_s.min()
            col_max=col_s.max()
            col_uniq=list(set(col_s))
            col_uniq_ct=len(col_uniq)
            uni_list=[["Unique Value","Count"]]
            # Column Unique DataFrame
            col_uniq_df=pd.DataFrame(col_s.value_counts())
            col_uniq_df["perc"]=col_uniq_df[col_name]/self.length
            
            col_uniq_df["type_"]=map(lambda x:type(x).__name__,list(col_uniq_df.index.values))
            col_uniq_df=col_uniq_df.head(15)
            nan_ct=col_s.isnull().sum()
            print("%s nan values, %.2f percent of the total rows"%(nan_ct,nan_ct*100/self.length))
            print("Min Value:%s\t Max Value:%s\t Uniq Count:%s"%(col_min, col_max,col_uniq_ct))
            print("Top 20 values:")
            display(col_uniq_df)
        else:
            print("Column name not in the dataframe")
    def get_df(self):
        return self.df
def check_img_folder(img_dir):
    """
    Check if we can open all the images under a folder, 
    if not rename the file to *.damage,
    input the image folder directory
    check_img_folder("/data/isface/pics/faces/")
    """
    if list(img_dir)[-1]=="/":
        endsym="*" 
    else:
        endsym="/*"
    img_path=glob(img_dir+endsym)
    print("="*50)
    print("Samples of folders")
    for j in img_path[:20]:
        print("-"*45)
        print(j)
    print("-"*45)
    right_count=0;fault_count=0
    for i in img_path:
        # from PIL import Image
        try:
            tmpimg=Image.open(i)
            tmpimg.close()
            right_count+=1
        except:
            os.system("mv %s %s"%(i,i+".damaged"))
            fault_count+=1
    print("="*50)
    print("%s images ok, %s images not ok, %s images in total under folder %s"%(right_count,
                                                                                fault_count,
                                                                                right_count+fault_count,
                                                                               img_dir))
    print("="*50)
def folder_split(folder_dir,percent=.7):
    """
    Perform train/ test split under a folder
    folder_split(folder_dir,train_percent=.7)
    Input absolute path as parameter。
    Use system command "cp"
    """
    if list(folder_dir)[-1]!="/":
        folder_dir=u"%s/"%(folder_dir)
    subfolder_dir=u"%s*"%(folder_dir)
    allcats=glob(subfolder_dir)
    print("="*60)
    print("Path %s Loaded"%(subfolder_dir))
    # Get the categories
    cats=map(lambda x:x.split("/")[-1],allcats)
    print("="*60)
    print("%s Categories found:"%(len(cats)))
    print("="*60)
    cat_path_list=dict()
    cat_div_list=dict()
    # make train/test folders
    os.system("mkdir %strain;mkdir %stest"%(folder_dir,folder_dir))
    for i in cats:
        cat_path_list[i]=glob(folder_dir+i+"/*")
        print ("[%s]:\t%s entries"%(i,len(cat_path_list[i])))
        print("-"*60)
        os.system("mkdir %s"%(folder_dir+"train/"+i))
        os.system("mkdir %s"%(folder_dir+"test/"+i))
    print ("="*60)
    for cate_, paths_ in cat_path_list.iteritems():
        print("[%s] split:"%(cate_))
        print("="*60)
        # make random array (draft True or False for each pic)
        rdlist=list(np.random.rand(len(paths_))>percent)
        cat_div_list[cate_]={"test":list(paths_[i] for i in range(len(rdlist)) if rdlist[i])}
        cat_div_list[cate_]["train"]=list(set(paths_)-set(cat_div_list[cate_]["test"]))
        for tt in cat_div_list[cate_].iterkeys():
            print("\t<%s> has \t%s entries"%(tt,len(cat_div_list[cate_][tt])))
            print("\t"+"-"*35)
            # make before and after absolute path pair
            cat_div_list[cate_][tt]=list([oripath,folder_dir+tt+"/"+cate_+"/"+oripath.split("/")[-1]] for oripath in cat_div_list[cate_][tt])
            map(lambda pair:os.system("cp %s %s"%(pair[0],pair[1])),cat_div_list[cate_][tt])
    return cat_div_list

def get_db(database=u"foundation"):
	"""
	A quick way to get MongoDb Client link
	"""
	clientmg=MongoClient()
	db=clientmg[database]
	return db
