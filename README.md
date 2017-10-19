# Python 3 Program

## lprint.py

A program to print log in a very neat format
````
from p3self import lprint
l=lprint.lprint("newtask")

# Do something time consuming
l.p("data loaded","data avatar images loaded")

# or you can just pass one string
l.p("data processed")
````

## bcolzer.py

Turns big bulk data to bcolz file

````
from p3self import lprint
bzr=bcolzer.bcolzer("img")
bzr.img_gen("/dir_to_your_img_folder",with_class=True)
bzr.empety_img_bcolz("/dir_to_your_img_bcolz","/dir_to_your_label_bcolz")
````

