# KMeans on batch, accelerated on Pytorch

When you have a hammer, every problem looks like nail to you.

When you have a torch, you burn every thing...

You can also [check here](https://raynardj.github.io/p3self/docs/gowalla_preprocess) for a use case on gowalla dataset

## Problem:
Assume we have a large array A, will be clustered in a big centroid number: K

```python
import numpy as np
A = np.random.rand(1e6,4)
K = 200
```
As the calculation grow exponentially with the centroids number.


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
## Self defined distance function

Suppose you are clustering anchor box for object detetion.

We want use a very specific distance function calculated from IOU(Intersection Over Union)

So we tweak the kmeans like the following:

```python
class iou_km(kmeans_core):
    def __init__(self, k, data_array, batch_size=1000, epochs=200):
        super(iou_km, self).__init__(k, data_array, batch_size=batch_size, epochs=epochs)

    def calc_distance(self, dt):
        """
        calculation steps here
        dt is the data batch , size = (batch_size , dimension)
        self.cent is the cent
        centroid, size = (k,dim)
        the return distance, size = (batch_size , k), from each point's distance to centroids
        """
        bs = dt.size()[0]
        box = dt.unsqueeze(1).repeat(1, self.k, 1)
        anc = self.cent.unsqueeze(0).repeat(bs, 1, 1)

        outer = torch.max(box[..., 2:4], anc[..., 2:4])
        inner = torch.min(box[..., 2:4], anc[..., 2:4])

        inter = inner[..., 0] * inner[..., 1]
        union = outer[..., 0] * outer[..., 1]

        distance = 1 - inter / union

        return distance
```

Then use the inherited class iou_km

```python
kiou = iou_km(k = 9,data_array = bounding_box_label_array)

kiou.run()
```

You can define your own kmeans by batch run on pytorch with other distance function in above way.

Notice: When design the distance function, use matrix operations, avoid python loop as much as possible to best harnesting the power of cuda


