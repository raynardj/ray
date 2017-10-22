import sys
from keras.preprocessing import image
from p3self import lprint
import bcolz
import numpy as np

class bcolzer:
    """
    A class to make image dir into bcolz array
    """
    def __init__(self,datatype="img"):
        """
        if you use self.empety_bcolz("bcolz_dir","b_shape")
        you'll have to setup the generator, self.set_gen(your_generator)
        
        if you use self.empety_image_bcolz
        you'll have to set you img dir using self.img_gen()
        
        bz=bcolzer()
        bz.img_gen("/dir_to_your_img_folder",with_class=True)
        bz.empety_img_bcolz("/dir_to_your_img_bcolz","/dir_to_your_label_bcolz") # insert the empety bcolz array
        bz.eat() # transfer all the data from gen to bcolz array
        datatype:default "img"
        """
        self.l=lprint.lprint(task="bcolzer_task")
        self.datatype=datatype
        self.with_class=False
        self.l.p("starting","for data type [%s]"%(self.datatype))
        if self.datatype:
            self.l.p("type_set","type set to \"img\", use self.img_gen(image_dir) to create the generator")
        else:
            self.l.p("type_error","no bcolzer type specified\nIt could be \"img\" for image")
        self.func()
            
    def img_gen(self,img_dir,with_class=False,
                batch_size=32,img_shape=(224,224),shuffle=True,seed=0):
        """
        Create Image data generator
        img_dir:Image directory (pictures should be under 2 directories lower)
        with_class: gen return the class label or not, default False
        batch_size:default 32
        img_shape:(x,y)
        shuffle:True False, default True
        """
        self.with_class=with_class
        self.img_shape=img_shape
        self.batch_size=batch_size
        img_gen0=image.ImageDataGenerator()
        self.gen=img_gen0.flow_from_directory(directory=img_dir,
                                        seed=seed,
                                        batch_size=batch_size,
                                        target_size=img_shape,
                                        shuffle=shuffle)
        return self.gen
    
    def set_gen(self,gen):
        self.gen=gen
        self.batch_size=self.gen.batch_size
    
    def empety_img_bcolz(self,b_dir,cls_b_dir=None):
        """
        Create empety bcolz array
        According to the image input size
        b_dir: Bcolz directory
        """
        if self.gen.data_format=="channel_first":
            self.bc=self.empety_bcolz(b_dir,b_shape=(0,3,self.img_shape[0],self.img_shape[1]))
        else:
            self.bc=self.empety_bcolz(b_dir,b_shape=(0,self.img_shape[0],self.img_shape[1],3))
        if self.with_class:
            if cls_b_dir:
                self.cls_bc=self.empety_bcolz(cls_b_dir,b_shape=(0,#self.gen.n,
                    self.gen.num_class))
            else:
                self.l.p("class_label_dir","class label dir not set, run self.empety_img_bcolz()")
    
    def empety_bcolz(self,b_dir,b_shape):
        """
        Empety bcolz array
        b_dir: Saving dir for bcolz array
        b_shape: Saving shape for bcolz array, entire shape, include the rows
        """
        bc=bcolz.carray(np.zeros(b_shape),rootdir=b_dir)
        bc.flush()
        self.l.p("create_empety_bcolz",b_shape)
        self.l.p("bcolz_saved_on",b_dir)
        return bc
    
    def func(self,roll_func=None):
        """
        process_func: New function
        """
        if roll_func:
            self.roll_func=roll_func
        else:
            if self.with_class:
                def default_func(x):
                    """
                    default roll_func
                    directly return x
                    return the 2 elements of the set
                    x:input data
                    """
                    return x
            else:
                def default_func(x):
                    """
                    default roll_func
                    return the 1st element of the set
                    x:input data
                    """
                    return x[0]
            self.roll_func=default_func
        self.l.p("set_roll_process_func",self.roll_func.__doc__)
    
    def eat(self):
        """
        Shipping data from the generator,
        to the bcolz
        """
        idx=0
        if not self.roll_func:
            self.func()
        while idx<self.gen.n:
            new=self.gen.next()
            #print(self.bc.shape)
            #print(new[0].shape)
            if self.with_class:
                self.bc.append(self.roll_func(new[0])) # append with same shape(besides shape 0 )
                self.cls_bc.append(new[1])
                self.cls_bc.flush() # store the label to hd
            else:
                self.bc.append(self.roll_func(new))
            self.bc.flush() # store the output to hd
            sys.stdout.write("\r[index/total]\t%s/%s\t%s"%(idx,self.gen.n,self.bc.shape))
            idx+=self.batch_size
        self.l.p("gen2bcolz","The transformation is done")