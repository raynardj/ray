# Hacks and Tools on machine learning

## matchbox.py

A tool box for pytorch, A wrapper around trainer

Full introduction [here](https://raynardj.github.io/p3self/docs/matchbox)

## kmean_torch.py

The core class for cuda accelerated kmeans by batch

Full introduction [here](https://raynardj.github.io/p3self/docs/kmean_torch)

## lprint.py

A program to print log in a very neat format
````python
from p3self import lprint
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
from p3self import bcolzer
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
from p3self.metrics import precision

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
from p3self.armory import folder_split

# The path we input contains sub-folders(categories)

folder_split("/data/animals",percent=.8)

# percent : the percentage of train data

```
### one_hot

Turn index array to one hot encoded array

```python

from p3self.armory import one_hot

# say if we have a array which labels 5 classes
one_hot(label_array, num_classes=5)

```

