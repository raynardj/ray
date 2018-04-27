
# Gowalla Preprocess

### Use Kmeans to cut slots on geo information

Load the gowalla data, with geo location


```python
import numpy as np
import pandas as pd

df = pd.read_csv("/data/gowalla_admin.csv")
```


```python
df.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>time</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>loc_id</th>
      <th>name</th>
      <th>admin1</th>
      <th>admin2</th>
      <th>cc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>771873</th>
      <td>3279</td>
      <td>2010-10-16T06:25:17Z</td>
      <td>37.560015</td>
      <td>126.975453</td>
      <td>683910</td>
      <td>Seoul</td>
      <td>Seoul</td>
      <td>NaN</td>
      <td>KR</td>
    </tr>
    <tr>
      <th>3441388</th>
      <td>48625</td>
      <td>2010-10-18T15:14:06Z</td>
      <td>57.690823</td>
      <td>11.901670</td>
      <td>204029</td>
      <td>Majorna</td>
      <td>Vaestra Goetaland</td>
      <td>Goteborg</td>
      <td>SE</td>
    </tr>
    <tr>
      <th>1053005</th>
      <td>4819</td>
      <td>2010-07-23T01:09:45Z</td>
      <td>47.610026</td>
      <td>-122.342623</td>
      <td>9758</td>
      <td>Seattle</td>
      <td>Washington</td>
      <td>King County</td>
      <td>US</td>
    </tr>
    <tr>
      <th>1220472</th>
      <td>6022</td>
      <td>2010-10-17T12:46:50Z</td>
      <td>33.100306</td>
      <td>-96.824493</td>
      <td>17296</td>
      <td>Frisco</td>
      <td>Texas</td>
      <td>Collin County</td>
      <td>US</td>
    </tr>
    <tr>
      <th>4966389</th>
      <td>109740</td>
      <td>2010-08-30T02:05:42Z</td>
      <td>42.797312</td>
      <td>-83.219029</td>
      <td>1435504</td>
      <td>Lake Orion</td>
      <td>Michigan</td>
      <td>Oakland County</td>
      <td>US</td>
    </tr>
    <tr>
      <th>2489368</th>
      <td>25669</td>
      <td>2010-03-07T03:52:53Z</td>
      <td>37.798418</td>
      <td>-122.430559</td>
      <td>28105</td>
      <td>San Francisco</td>
      <td>California</td>
      <td>San Francisco County</td>
      <td>US</td>
    </tr>
    <tr>
      <th>5844680</th>
      <td>142712</td>
      <td>2010-06-23T07:42:57Z</td>
      <td>57.919452</td>
      <td>13.856635</td>
      <td>931719</td>
      <td>Mullsjoe</td>
      <td>Joenkoeping</td>
      <td>Mullsjo Kommun</td>
      <td>SE</td>
    </tr>
    <tr>
      <th>4961129</th>
      <td>109690</td>
      <td>2010-08-13T21:34:46Z</td>
      <td>21.278857</td>
      <td>-157.825953</td>
      <td>161491</td>
      <td>Honolulu</td>
      <td>Hawaii</td>
      <td>Honolulu County</td>
      <td>US</td>
    </tr>
    <tr>
      <th>1820855</th>
      <td>14443</td>
      <td>2010-06-27T08:34:57Z</td>
      <td>47.615645</td>
      <td>7.661247</td>
      <td>178917</td>
      <td>Loerrach</td>
      <td>Baden-Wuerttemberg</td>
      <td>Freiburg Region</td>
      <td>DE</td>
    </tr>
    <tr>
      <th>1576630</th>
      <td>10857</td>
      <td>2010-05-31T00:54:23Z</td>
      <td>47.828276</td>
      <td>-122.275107</td>
      <td>276886</td>
      <td>Alderwood Manor</td>
      <td>Washington</td>
      <td>Snohomish County</td>
      <td>US</td>
    </tr>
  </tbody>
</table>
</div>



Shuffle the data with ```df.sample(frac=1.)```.

Retrieve the geo information to numpy array


```python
df = df.sample(frac=1.)
geo_array = df[["latitude","longitude"]].as_matrix()
```

Use The kmeans_core.

Set k=200, put in the array, set batch size to 800,000


```python
from p3self.kmean_torch import kmeans_core

km = kmeans_core(200,geo_array,batch_size=8e5)
```

Run the calculation.

Each epoch takes 3 seconds, on a NVIDIA GTX 1070 GPU


```python
km.run()
```

    🔥[epoch:0	 iter:8]🔥 	🔥k:200	🔥distance:4665.021: 100%|██████████| 9/9 [00:03<00:00,  2.91it/s]
    🔥[epoch:1	 iter:8]🔥 	🔥k:200	🔥distance:4668.177: 100%|██████████| 9/9 [00:03<00:00,  2.96it/s]
    🔥[epoch:2	 iter:8]🔥 	🔥k:200	🔥distance:5084.151: 100%|██████████| 9/9 [00:03<00:00,  2.95it/s]
    🔥[epoch:3	 iter:8]🔥 	🔥k:200	🔥distance:5084.281: 100%|██████████| 9/9 [00:03<00:00,  2.94it/s]
    🔥[epoch:4	 iter:8]🔥 	🔥k:200	🔥distance:5084.367: 100%|██████████| 9/9 [00:03<00:00,  2.95it/s]
    🔥[epoch:5	 iter:8]🔥 	🔥k:200	🔥distance:5084.366: 100%|██████████| 9/9 [00:03<00:00,  2.96it/s]
    🔥[epoch:6	 iter:8]🔥 	🔥k:200	🔥distance:5084.366: 100%|██████████| 9/9 [00:03<00:00,  2.96it/s]
    🔥[epoch:7	 iter:8]🔥 	🔥k:200	🔥distance:5084.366: 100%|██████████| 9/9 [00:03<00:00,  2.97it/s]
    🔥[epoch:8	 iter:8]🔥 	🔥k:200	🔥distance:5084.366: 100%|██████████| 9/9 [00:03<00:00,  2.95it/s]
    🔥[epoch:9	 iter:8]🔥 	🔥k:200	🔥distance:5084.366: 100%|██████████| 9/9 [00:03<00:00,  2.96it/s]
    🔥[epoch:10	 iter:8]🔥 	🔥k:200	🔥distance:5084.366: 100%|██████████| 9/9 [00:03<00:00,  2.94it/s]
    🔥[epoch:11	 iter:8]🔥 	🔥k:200	🔥distance:5084.365: 100%|██████████| 9/9 [00:03<00:00,  2.95it/s]
    🔥[epoch:12	 iter:8]🔥 	🔥k:200	🔥distance:5084.366: 100%|██████████| 9/9 [00:03<00:00,  2.90it/s]
      0%|          | 0/9 [00:00<?, ?it/s]

    Centroids is not shifting anymore


    100%|██████████| 9/9 [00:03<00:00,  2.97it/s]





    tensor([  74,  102,   63,  ...,  158,  108,  172], device='cuda:0')




```python
from p3self.kmean_torch import kmeans_core
```

For comparison, 1 epoch took 38 seconds to run on cpu


```python
km = kmeans_core(200,geo_array,batch_size=8e5,epochs=1)
km.run()
```

    🔥[epoch:0	 iter:8]🔥 	🔥k:200	🔥distance:4665.269: 100%|██████████| 9/9 [00:38<00:00,  4.23s/it]
    100%|██████████| 9/9 [00:36<00:00,  4.07s/it]





    tensor([ 171,  152,   27,  ...,  117,   14,   46])




```python
idx_array = km.idx.cpu().numpy()
```


```python
cen = km.cent.cpu().numpy()
```


```python
np.save("/data/centroids.npy",cen)
```

## Mapping It Back to DataFrame


```python
df["idx"]=idx_array
```


```python
cen_lati = cen[:,:1].tolist()
cen_longi = cen[:,1:].tolist()
```


```python
centroids = list("lati:%s longi%s"%(la[0],lo[0]) for la,lo in zip(cen_lati,cen_longi))
```


```python
df["cent"] = df["idx"].apply(lambda x:centroids[x])
```


```python
df.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>time</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>loc_id</th>
      <th>name</th>
      <th>admin1</th>
      <th>admin2</th>
      <th>cc</th>
      <th>idx</th>
      <th>cent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3355801</th>
      <td>46101</td>
      <td>2010-07-29T16:08:11Z</td>
      <td>52.513544</td>
      <td>13.319991</td>
      <td>127070</td>
      <td>Hansaviertel</td>
      <td>Berlin</td>
      <td>NaN</td>
      <td>DE</td>
      <td>90</td>
      <td>lati:51.475975036621094 longi13.54339599609375</td>
    </tr>
    <tr>
      <th>4753318</th>
      <td>105757</td>
      <td>2010-05-22T12:46:45Z</td>
      <td>55.602003</td>
      <td>13.024511</td>
      <td>1153250</td>
      <td>Malmoe</td>
      <td>Skane</td>
      <td>Malmo</td>
      <td>SE</td>
      <td>74</td>
      <td>lati:55.60551452636719 longi13.091133117675781</td>
    </tr>
    <tr>
      <th>2008736</th>
      <td>17272</td>
      <td>2010-03-12T10:19:50Z</td>
      <td>50.809053</td>
      <td>8.810520</td>
      <td>679546</td>
      <td>Marburg an der Lahn</td>
      <td>Hesse</td>
      <td>Regierungsbezirk Giessen</td>
      <td>DE</td>
      <td>53</td>
      <td>lati:49.7291374206543 longi8.50769329071045</td>
    </tr>
    <tr>
      <th>3944538</th>
      <td>67736</td>
      <td>2010-04-06T18:43:10Z</td>
      <td>39.958952</td>
      <td>-86.009593</td>
      <td>171442</td>
      <td>Fishers</td>
      <td>Indiana</td>
      <td>Hamilton County</td>
      <td>US</td>
      <td>118</td>
      <td>lati:39.82094955444336 longi-86.19813537597656</td>
    </tr>
    <tr>
      <th>1504032</th>
      <td>10388</td>
      <td>2009-11-13T22:40:32Z</td>
      <td>28.518201</td>
      <td>-81.344147</td>
      <td>50220</td>
      <td>Conway</td>
      <td>Florida</td>
      <td>Orange County</td>
      <td>US</td>
      <td>164</td>
      <td>lati:28.502212524414062 longi-81.45722198486328</td>
    </tr>
    <tr>
      <th>5586680</th>
      <td>129560</td>
      <td>2010-06-30T11:40:20Z</td>
      <td>59.319182</td>
      <td>18.076099</td>
      <td>1359960</td>
      <td>Stockholm</td>
      <td>Stockholm</td>
      <td>Stockholms Kommun</td>
      <td>SE</td>
      <td>50</td>
      <td>lati:59.325557708740234 longi18.073034286499023</td>
    </tr>
    <tr>
      <th>82823</th>
      <td>282</td>
      <td>2010-10-16T14:11:36Z</td>
      <td>32.786529</td>
      <td>-96.794764</td>
      <td>50853</td>
      <td>Dallas</td>
      <td>Texas</td>
      <td>Dallas County</td>
      <td>US</td>
      <td>28</td>
      <td>lati:32.839134216308594 longi-96.77941131591797</td>
    </tr>
    <tr>
      <th>2903954</th>
      <td>36019</td>
      <td>2010-05-10T01:02:01Z</td>
      <td>40.588843</td>
      <td>-111.940675</td>
      <td>27137</td>
      <td>West Jordan</td>
      <td>Utah</td>
      <td>Salt Lake County</td>
      <td>US</td>
      <td>192</td>
      <td>lati:40.87124252319336 longi-111.69757080078125</td>
    </tr>
    <tr>
      <th>2191735</th>
      <td>19338</td>
      <td>2010-06-14T15:47:05Z</td>
      <td>25.334211</td>
      <td>51.467593</td>
      <td>854133</td>
      <td>Ar Rayyan</td>
      <td>Baladiyat ar Rayyan</td>
      <td>NaN</td>
      <td>QA</td>
      <td>189</td>
      <td>lati:25.092052459716797 longi48.042118072509766</td>
    </tr>
    <tr>
      <th>1766141</th>
      <td>13081</td>
      <td>2010-07-06T17:36:14Z</td>
      <td>56.350999</td>
      <td>13.733063</td>
      <td>95606</td>
      <td>Bjarnum</td>
      <td>Skane</td>
      <td>Hassleholms Kommun</td>
      <td>SE</td>
      <td>13</td>
      <td>lati:56.30799865722656 longi14.415539741516113</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.to_csv("/data/gowalla_admin_slotted.csv")
```


```python

```
