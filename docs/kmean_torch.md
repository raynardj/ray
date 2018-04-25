# Kmean torch

A cuda accelerated kmeans by batch


## Problem:
Assume we have a large array A, will be clustered in a big centeroid number: K

```python
import numpy as np
A = np.random.rand(1e6,4)
K = 200
```
As the calculation grow exponentially with the centeroids number.


## Solution:

In this case we can use this pytorch to harvest the power of cuda GPU to accelerate the calculation

If you use sklearn's kmeans, you could have waited for hours

```python
from p3self.kmean_torch import kmeans_core
km = kmeans_core(k=K,data_array=A,batch_size=3000,epochs=200)
km.run()
```
Here we have the cluster index

```python
idx = km.idx
```
## Side Configure and Hacking

Suppose you are clustering anchor box for object detetion.

You use a very specific distance function calculated from IOU(Intersection Over Union)

You can tweak the kmeans like the following:
```python
class iou_km(kmeans_core):
    def __init__(self, k, data_array, batch_size=1000, epochs=200):
        super(iou_km,self).__init__(k, data_array, batch_size=batch_size, epochs=epochs)
    
    def calc_distance(self,dt):
        # calculation steps here
        # dt is the data batch , size = (batch_size * dimension)
        # self.cent is the centeroid, size = (k,dim)
        # the return distance, size = (batch_size * k), from each point's distance to centeroids
        return distance
    
```

Then use the inherited class iou_km