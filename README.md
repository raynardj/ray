# Python 3 Program

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
from p3self import lprint
bzr=bcolzer.bcolzer("img")
bzr.img_gen("/dir_to_your_img_folder",with_class=True)
bzr.empety_img_bcolz("/dir_to_your_img_bcolz","/dir_to_your_label_bcolz")
````

