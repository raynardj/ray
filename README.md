# Python 3 Program

## lprint.py

A program to print log in a very neat format
````python
from rayself import lprint
l=lprint.lprint("newtask")

# Do something time consuming
l.p("data loaded","data avatar images loaded")

# or you can just pass one string
l.p("data processed")
````

## bcolzer.py

Turns big bulk array to bcolz file.

For a numpy, if you create some thing big, like a.shape=(1000000,224,224,3)

You memory won't indulge this simplicity. With bcolz, you can flush array to hard drive, and still use bcolz array as a single variable.

````python
from rayself import lprint
bzr=bcolzer.bcolzer("img")
# create an image generator, use bzr.gen as the generator
bzr.img_gen("/dir_to_your_img_folder",with_class=True)

bzr.empety_img_bcolz("/dir_to_your_img_bcolz","/dir_to_your_label_bcolz")
````

## metrics.py

Augmented metrics for keras

Added ```ratio```,```precision``` and ```recall```

This will calculate the ratio,precision and recall of a specific category for classification problem.

````python
from rayself.metrics import precision

# Precision is the fraction of detections
# reported by the model that were correct.

def dog_precision(y_true,y_pred):
    # assuming dog is your second category
    return precision(1,y_true,y_pred)

# while compiling a keras model
model.compile(loss='categorical cross',metrics=["accuracy",dog_precision],optimizer="Adam")
````

## armory.py

### preproc

Preprocess before entering the cnn/resnet

Notice: it does not scale element to ````[0,1]````

### check_img_folder_multi

check img folder with multi processing
```python
check_img_folder_multi('/data/cats/img/')
```
### folder_split

Split folder to train/valid

```python
from rayself.armory import folder_split

# The path we input contains sub-folders(categories)

folder_split("/data/animals",percent=.8)

# percent : the percentage of train data

```

