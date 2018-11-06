import numpy as np
import os
from glob import glob
from PIL import Image
from multiprocessing import Pool
from sys import getsizeof

def preproc(img, rgb_mean=[123.68, 116.779, 103.939]):
    """
    preprocessing
    Turn rgb into bgr on picture
    img:input image, numpy array
    rgb_mean:a list of r,g,b value, default [123.68, 116.779, 103.939]
    """
    rn_mean = np.array(rgb_mean, dtype=np.float32)
    return (img - rn_mean)[:, :, :, ::-1]


def deproc(img, shp=None, rgb_mean=[123.68, 116.779, 103.939]):
    """
    deprocessing
    Turn bgr into rgb on picture
    img:input image, numpy array
    shp:output shape, default None and using the input shape
    rgb_mean:a list of r,g,b value, default [123.68, 116.779, 103.939]
    """
    rn_mean = np.array(rgb_mean, dtype=np.float32)
    if not shp:
        shp = img.shape
    return np.clip(img.reshape(shp)[:, :, :, ::-1] + rn_mean, 0, 255)


def check_img_folder(img_dir,img_path=False):
    """
    Check if we can open all the images under a folder,
    if not rename the file to *.damage,
    input the image folder directory
    check_img_folder("/data/isface/pics/faces/")
    """
    if list(img_dir)[-1] == "/":
        endsym = "*"
    else:
        endsym = "/*"
    if img_path==False:
        img_path = glob(img_dir + endsym)
    print("=" * 50)
    print("Samples of folders")
    for j in img_path[:20]:
        print("-" * 45)
        print(j)
        # print("-"*45)
    right_count = 0;
    fault_count = 0
    for i in img_path:
        # from PIL import Image
        try:
            tmpimg = Image.open(i)
            tmpimg.close()
            right_count += 1
        except:
            os.system("mv %s %s" % (i, i + ".damaged"))
            fault_count += 1
    print("=" * 50)
    print("%s images ok, %s images not ok, %s images in total under folder %s" % (right_count,
                                                                                  fault_count,
                                                                                  right_count + fault_count,
                                                                                  img_dir))
    print("=" * 50)

def tryimg(i):
    # from PIL import Image
    try:
        tmpimg = Image.open(i)
        tmpimg.resize((100,100),Image.ANTIALIAS)
        tmpimg.close()
        #right_count += 1
        #print("Img Ok")
        return 1
    except:
        os.system("mv %s %s" % (i, i + ".damaged"))
        print(i,"Damaged")
        #fault_count += 1
        return 0

def check_img_folder_multi(img_dir,img_path=False,threads=5):
    """
    Check if we can open all the images under a folder,
    Support multi-threads
    if not rename the file to *.damage,
    input the image folder directory
    check_img_folder("/data/isface/pics/faces/",threads=5)
    """
    if list(img_dir)[-1] == "/":
        endsym = "*"
    else:
        endsym = "/*"
    if img_path==False:
        img_path = glob(img_dir + endsym)
    print("=" * 50)
    print("Samples of folders")
    for j in img_path[:20]:
        print("-" * 45)
        print(j)
        # print("-"*45)
    # right_count = 0;
    # fault_count = 0
    p=Pool(threads)
    result=p.map(tryimg,img_path)
    right_count=result.count(1)
    fault_count=result.count(0)
    print("=" * 50)
    print("%s images ok, %s images not ok, %s images in total under folder %s" % (right_count,
                                                                                  fault_count,
                                                                                  right_count + fault_count,
                                                                                  img_dir))
    print("=" * 50)

def folder_split(folder_dir, percent=.7):
    """
    Perform train/ test split under a folder
    folder_split(folder_dir,train_percent=.7)
    Input absolute path as parameter。
    Use system command "cp"
    :type folder_dir: object
    """
    if list(folder_dir)[-1] != "/":
        folder_dir = u"%s/" % (folder_dir)
    subfolder_dir = u"%s*" % (folder_dir)
    allcats = glob(subfolder_dir)
    print("=" * 60)
    print("Path %s Loaded" % (subfolder_dir))
    # Get the categories
    cats = map(lambda x: x.split("/")[-1], allcats)
    print("=" * 60)
    print("%s Categories found:" % (len(cats)))
    print("=" * 60)
    cat_path_list = dict()
    cat_div_list = dict()
    # make train/test folders
    os.system("mkdir %strain;mkdir %stest" % (folder_dir, folder_dir))
    for i in cats:
        cat_path_list[i] = glob(folder_dir + i + "/*")
        print("[%s]:\t%s entries" % (i, len(cat_path_list[i])))
        print("-" * 60)
        os.system("mkdir %s" % (folder_dir + "train/" + i))
        os.system("mkdir %s" % (folder_dir + "test/" + i))
    print("=" * 60)
    for cate_, paths_ in cat_path_list.iteritems():
        print("[%s] split:" % (cate_))
        print("=" * 60)
        # make random array (draft True or False for each pic)
        rdlist = list(np.random.rand(len(paths_)) > percent)
        cat_div_list[cate_] = {"test": list(paths_[i] for i in range(len(rdlist)) if rdlist[i])}
        cat_div_list[cate_]["train"] = list(set(paths_) - set(cat_div_list[cate_]["test"]))
        for tt in cat_div_list[cate_].iterkeys():
            print("\t<%s> has \t%s entries" % (tt, len(cat_div_list[cate_][tt])))
            print("\t" + "-" * 35)
            # make before and after absolute path pair
            cat_div_list[cate_][tt] = list(
                [oripath, folder_dir + tt + "/" + cate_ + "/" + oripath.split("/")[-1]] for oripath in
                cat_div_list[cate_][tt])
            map(lambda pair: os.system("cp %s %s" % (pair[0], pair[1])), cat_div_list[cate_][tt])
    return cat_div_list

def mem(v):
    """
    Get the variable size
    v: the variable
    """
    s = getsizeof(v)
    if s< 1e3:
        return "%s B"%(s)
    elif s< 1e6: 
        return "%.2f KB"%(s/1e3)
    elif s< 1e9: 
        return "%.2f MB"%(s/1e6)
    else: 
        return "%.2f GB"%(s/1e9)

def one_hot(array,num_classes=None):
    """
    one_hot encoding the numpy array
    :param array: input array, rank should be 1
    :param num_classes: number of classes, if not specify, we'll use the max index +1 as class number
    :return: return the onehot encoded array
    """
    assert len(array.shape) ==1, "In put dimemsion not right, should be 1."
    if num_classes == None:
        num_classes=array.max()+1
    eye=np.eye(num_classes)
    return eye[array]
