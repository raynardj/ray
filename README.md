# Python 3 self made program for deep learning

## matchbox.py

A tool box for pytorch

```python
from p3self.matchbox import Trainer
# train_set is a pytorch dateset class
# if you don't set the arg: val_dataset, it won't run validation.
trainer = Trainer(train_set,batch_size=8,print_on=20)
```

#### Here we define a step, takes dataset output as args.

```python

# for a simple structured model
import torch
from torch.autograd import Variable

def action(*args,**kwargs):
    x,y = args[0]
    # model is a pytorch nn.Module object
    # model, loss_func,optimizer are all global variable , not a part of this class
    x,y= Variable(x),Variable(y)
    if torch.cuda.is_available:
        x,y=x.cuda(),y.cuda()
    optimizer.zero_grad()
    y_ = model(x)
    
    loss = loss_func(y,y_)
    score = some_calculation_about_score(y,y_)
    score_corrupt = another_kind_of_score(y,y_)
    loss.backward()
    optimizer.step()
    return {"loss":loss.data[0],
            "score":score.data[0],
            "score_corrupt":score_corrupt.data[0],
            }
```
#### Now assign the function to the trainer
```python
trainer.action=action
```
#### Train & Tracking 
```python
trainer.train(4) # for training 4 epochs
```

Then you will see progress bars running for each epoch like following
```
⭐[ep_0_i_199]	loss	1.341✨	score	1.725✨	score_corrupt	2.387: 100%|██████████| 200/200 [00:23<00:00,  8.47it/s]
😎[val_ep_0_i_22]	loss	1.255😂	score	1.722😂	score_corrupt	2.475: 100%|██████████| 23/23 [00:00<00:00, 24.03it/s]
⭐[ep_1_i_199]	loss	0.539✨	score	2.010✨	score_corrupt	3.999: 100%|██████████| 200/200 [00:25<00:00,  7.82it/s]
😎[val_ep_1_i_22]	loss	0.640😂	score	2.100😂	score_corrupt	3.948: 100%|██████████| 23/23 [00:00<00:00, 24.53it/s]
⭐[ep_2_i_199]	loss	0.311✨	score	2.315✨	score_corrupt	4.931: 100%|██████████| 200/200 [00:25<00:00,  7.94it/s]
😎[val_ep_2_i_22]	loss	0.527😂	score	2.484😂	score_corrupt	4.774: 100%|██████████| 23/23 [00:01<00:00, 18.98it/s]
⭐[ep_3_i_199]	loss	0.198✨	score	2.631✨	score_corrupt	5.640: 100%|██████████| 200/200 [00:24<00:00,  8.26it/s]
😎[val_ep_3_i_22]	loss	0.487😂	score	2.852😂	score_corrupt	5.420: 100%|██████████| 23/23 [00:00<00:00, 24.15it/s]
```

To save the training/validation record
```python
trainer.save_track(your_file_path_here,val_filepath=another_csv_path)
```
All the epochs will be saved to the csv file, or 2 csv files if you have validation record to save.

```val_filepath``` is optional

You can easily read the csv file to plot any charts you like.

#### Validation data
Define a val_action for a validation step
```python
def val_action(*args,**kwargs):
    ...
    return {"loss":loss.data[0],
            "some_metric":metric.data[0],
            "some_other_metric":metric2.data[0]}
trainer.val_action=val_action
```
#### More complicated models
The reason I don't embed the model,optimizer and loss function is:

There might be more than 1 model, with 3 loss function, run on 2 separate optimizers.

The following is an example:
```python
import torch
from torch.autograd import Variable

def action(*args,**kwargs):
    x1,x2,y = args[0]
    # model is a pytorch nn.Module object
    # model, loss_func,optimizer are all global variable , not a part of this class
    x1,x2,y= Variable(x1),Variable(x2),Variable(y)
    if torch.cuda.is_available:
        x1,x2,y=x1.cuda(),x2.cuda(),y.cuda()
    adam.zero_grad()
    sgd.zero_grad()

    y1_ = model(x1)
    y2_ = model2(x2)
    y3_ = model3(x1,x2,y)
    
    loss = loss_func(y1_,y)
    loss += another_loss_func(y2_,y)
    loss += yet_another_loss_func(y3_,y)
    
    accuracy = some_calculation_about_accuracy(y1_,y2_,y3_,y)
    loss.backward()
    adam.step()
    sgd.step()
    return {"loss":loss.data[0],
            "acc":accuracy.data[0]}
```
### What if I want to save the model?

**matchbox** deosn't over worry about this, but it allows you to save in a frequency or condition you can customize, easy as this:

```
def action(*args,**kwargs):
    ...other codes...
    i = kwargs["ite"] # iteration index
    e = kwargs["epoch"]
    if i %30 ==29: # for each 30 iterations
        if loss < 0.2: # save model if certain condition is made, if you like
            torch.save(model.state_dict(),"model1.0.%s.%s.pkl"%(e,i))
    ...other codes...
```

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

