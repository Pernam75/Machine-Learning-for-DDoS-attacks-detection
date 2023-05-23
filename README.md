# <a id='toc1_'></a>[Machine-Learning for DDOS attack detection](#toc0_)
*Ayman BEN HAJJAJ & Jules RUBIN*

## <a id='toc1_1_'></a>[Description](#toc0_)
This study is the final project of the Machine-Learning II course at EFREI Paris (Master 1 Data Science & AI 2023).
This project aims to detect DDOS attacks in a network using machine learning. The dataset used is the [CIC-DDoS2019](https://www.kaggle.com/datasets/dhoogla/cicddos2019) dataset. This dataset contains 78 features and 430K rows. The dataset contains 18 types of attacks. The goal is to detect the attacks using machine learning.
The taxonomy of attacks present in the dataset are described in the research paper [DDoS Evaluation Dataset (CIC-DDoS2019)](https://www.unb.ca/cic/datasets/ddos-2019.html).

**Taxonomy of attacks present in the dataset:**

![Taxonomy of attacks](https://www.unb.ca/cic/_assets/images/ddostaxonomy.png)

As we have the choice on the type of attack to detect, we will focus on the detection of 3 types of attacks:
- UDP : UDP Flood attack
- Syn : Syn Flood attack
- DrDoS DNS : DNS amplification attack

We will also provide a method that can classify the benign traffic from the malicious traffic.

**Table of contents**<a id='toc0_'></a>    
- [Machine-Learning for DDOS attack detection](#toc1_)    
  - [Description](#toc1_1_)    
  - [Preprocessing](#toc1_2_)    
    - [Importing libraries and dataset](#toc1_2_1_)    
    - [Data cleaning](#toc1_2_2_)    
      - [Dealing with missing values](#toc1_2_2_1_)    
      - [Dealing with outliers](#toc1_2_2_2_)    
      - [Dealing with categorical variables](#toc1_2_2_3_)    
    - [Balancing the dataset](#toc1_2_3_)    
    - [Handling outliers](#toc1_2_4_)    
  - [Data visualization](#toc1_3_)    
    - [Distribution according to the attack type](#toc1_3_1_)    
    - [Correlation matrix](#toc1_3_2_)    
  - [Feature Dimensionality Reduction](#toc1_4_)    
    - [PCA](#toc1_4_1_)    
    - [Kernel PCA](#toc1_4_2_)    
    - [t-SNE](#toc1_4_3_)    
  - [Supervised learning for multiclassification](#toc1_5_)    
    - [LDA](#toc1_5_1_)    
    - [QDA](#toc1_5_2_)    
    - [PCA + LDA](#toc1_5_3_)    
    - [PCA + QDA](#toc1_5_4_)    
    - [K-PCA + LDA](#toc1_5_5_)    
    - [K-PCA + QDA](#toc1_5_6_)    
      - [T-SNE + LDA](#toc1_5_6_1_)    
    - [t-SNE + QDA](#toc1_5_7_)    
  - [Unsupervised learning for clustering](#toc1_6_)    
    - [k-Means](#toc1_6_1_)    
    - [GMM](#toc1_6_2_)    
    - [DBSCAN](#toc1_6_3_)    
    - [Hierarchical clustering](#toc1_6_4_)    
  - [Other approaches](#toc1_7_)    
    - [Decision tree](#toc1_7_1_)    
    - [K-Nearst Neighbors](#toc1_7_2_)    
    - [Random Forest](#toc1_7_3_)    
  - [Conclusion](#toc1_8_)    
- [References](#toc2_)    
- [License](#toc3_)    

<!-- vscode-jupyter-toc-config
	numbering=false
	anchor=true
	flat=false
	minLevel=1
	maxLevel=6
	/vscode-jupyter-toc-config -->
<!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

## <a id='toc1_2_'></a>[Preprocessing](#toc0_)

### <a id='toc1_2_1_'></a>[Importing libraries and dataset](#toc0_)


```python
# making the necessary imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import warnings
warnings.filterwarnings('ignore')
```


```python
df = pd.DataFrame()
df = pd.concat([df, pd.read_parquet('data/Syn-training.parquet')])
df = pd.concat([df, pd.read_parquet('data/DNS-testing.parquet')])
df = pd.concat([df, pd.read_parquet('data/UDP-training.parquet')])
df['Label'].value_counts()
```




    Syn          43302
    Benign       32901
    UDP          14792
    DrDoS_DNS     3669
    MSSQL          145
    Name: Label, dtype: int64




```python
# We can remove the MSSQL data as it is not required for our analysis
df = df[df['Label'] != 'MSSQL']
```


```python
df['Label'].value_counts()
```




    Syn          43302
    Benign       32901
    UDP          14792
    DrDoS_DNS     3669
    Name: Label, dtype: int64



### <a id='toc1_2_2_'></a>[Data cleaning](#toc0_)

#### <a id='toc1_2_2_1_'></a>[Dealing with missing values](#toc0_)


```python
# count the number of missing values in each column
df.isnull().sum().sort_values(ascending=False)
```




    Protocol                0
    CWE Flag Count          0
    Fwd Avg Packets/Bulk    0
    Fwd Avg Bytes/Bulk      0
    Avg Bwd Segment Size    0
                           ..
    Bwd IAT Total           0
    Fwd IAT Min             0
    Fwd IAT Max             0
    Fwd IAT Std             0
    Label                   0
    Length: 78, dtype: int64



#### <a id='toc1_2_2_2_'></a>[Dealing with outliers](#toc0_)


```python
# check for duplicate rows
df.duplicated().sum()
```




    194




```python
# remove duplicate rows
df.drop_duplicates(inplace=True)
```

We do not have any missing values in the dataset. However, we have some duplicates. We will remove them.

#### <a id='toc1_2_2_3_'></a>[Dealing with categorical variables](#toc0_)


```python
# count the number of unique values in each column
df.nunique().sort_values(ascending=True).head(25)
```




    Bwd Avg Bulk Rate          1
    Bwd Avg Packets/Bulk       1
    Bwd Avg Bytes/Bulk         1
    Fwd Avg Bulk Rate          1
    Fwd Avg Packets/Bulk       1
    Fwd Avg Bytes/Bulk         1
    ECE Flag Count             1
    Fwd URG Flags              1
    Bwd PSH Flags              1
    Bwd URG Flags              1
    PSH Flag Count             1
    FIN Flag Count             1
    URG Flag Count             2
    Fwd PSH Flags              2
    RST Flag Count             2
    ACK Flag Count             2
    CWE Flag Count             2
    SYN Flag Count             2
    Protocol                   3
    Label                      4
    Down/Up Ratio             15
    Bwd IAT Min              105
    Bwd Packet Length Min    195
    Fwd Act Data Packets     212
    Total Fwd Packets        263
    dtype: int64



As some columns has only one unique value, they do not bring any information. We will remove them.
We can also see that some columns are categorical variables. We will convert them to numerical variables using the OneHotEncoder.


```python
one_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
df = df.drop(one_value_cols, axis=1)
```


```python
three_value_cols = [col for col in df.columns if df[col].nunique() <= 3]
# One Hot Encoding
df = pd.get_dummies(df, columns=three_value_cols)
```

### <a id='toc1_2_3_'></a>[Balancing the dataset](#toc0_)


```python
df['Label'].value_counts()
```




    Syn          43302
    Benign       32707
    UDP          14792
    DrDoS_DNS     3669
    Name: Label, dtype: int64




```python
# As the dataset is imbalanced, we will balance it by taking 3000 samples from each class
balanced = pd.DataFrame()
balanced = pd.concat([balanced, df[df['Label'] == 'Syn'].sample(n=3500)])
balanced = pd.concat([balanced, df[df['Label'] == 'DrDoS_DNS'].sample(n=3500)])
balanced = pd.concat([balanced, df[df['Label'] == 'UDP'].sample(n=3500)])
balanced = pd.concat([balanced, df[df['Label'] == 'Benign'].sample(n=3500)])
df = balanced.copy()
# free up memory
del balanced
```

### <a id='toc1_2_4_'></a>[Handling outliers](#toc0_)


```python
df.describe()
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
      <th>Flow Duration</th>
      <th>Total Fwd Packets</th>
      <th>Total Backward Packets</th>
      <th>Fwd Packets Length Total</th>
      <th>Bwd Packets Length Total</th>
      <th>Fwd Packet Length Max</th>
      <th>Fwd Packet Length Min</th>
      <th>Fwd Packet Length Mean</th>
      <th>Fwd Packet Length Std</th>
      <th>Bwd Packet Length Max</th>
      <th>...</th>
      <th>SYN Flag Count_0</th>
      <th>SYN Flag Count_1</th>
      <th>RST Flag Count_0</th>
      <th>RST Flag Count_1</th>
      <th>ACK Flag Count_0</th>
      <th>ACK Flag Count_1</th>
      <th>URG Flag Count_0</th>
      <th>URG Flag Count_1</th>
      <th>CWE Flag Count_0</th>
      <th>CWE Flag Count_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.400000e+04</td>
      <td>14000.000000</td>
      <td>14000.000000</td>
      <td>1.400000e+04</td>
      <td>1.400000e+04</td>
      <td>14000.000000</td>
      <td>14000.000000</td>
      <td>14000.000000</td>
      <td>14000.000000</td>
      <td>14000.000000</td>
      <td>...</td>
      <td>14000.000000</td>
      <td>14000.000000</td>
      <td>14000.000000</td>
      <td>14000.000000</td>
      <td>14000.000000</td>
      <td>14000.000000</td>
      <td>14000.000000</td>
      <td>14000.000000</td>
      <td>14000.000000</td>
      <td>14000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.390554e+07</td>
      <td>8.189786</td>
      <td>4.455357</td>
      <td>2.721783e+03</td>
      <td>3.217427e+03</td>
      <td>401.637726</td>
      <td>339.272644</td>
      <td>353.859070</td>
      <td>21.215477</td>
      <td>97.797997</td>
      <td>...</td>
      <td>0.999429</td>
      <td>0.000571</td>
      <td>0.968786</td>
      <td>0.031214</td>
      <td>0.691143</td>
      <td>0.308857</td>
      <td>0.898214</td>
      <td>0.101786</td>
      <td>0.945786</td>
      <td>0.054214</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.733182e+07</td>
      <td>62.703896</td>
      <td>75.798956</td>
      <td>1.291497e+05</td>
      <td>1.202525e+05</td>
      <td>580.521790</td>
      <td>484.627106</td>
      <td>481.843567</td>
      <td>76.411736</td>
      <td>444.340881</td>
      <td>...</td>
      <td>0.023899</td>
      <td>0.023899</td>
      <td>0.173903</td>
      <td>0.173903</td>
      <td>0.462039</td>
      <td>0.462039</td>
      <td>0.302377</td>
      <td>0.302377</td>
      <td>0.226448</td>
      <td>0.226448</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e+00</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.900000e+01</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>6.000000e+01</td>
      <td>0.000000e+00</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.057425e+05</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>7.660000e+02</td>
      <td>0.000000e+00</td>
      <td>357.000000</td>
      <td>47.000000</td>
      <td>135.102020</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.128170e+06</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>2.088000e+03</td>
      <td>2.400000e+01</td>
      <td>428.000000</td>
      <td>330.000000</td>
      <td>359.500000</td>
      <td>22.516661</td>
      <td>6.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.199879e+08</td>
      <td>5063.000000</td>
      <td>8029.000000</td>
      <td>1.526642e+07</td>
      <td>1.289243e+07</td>
      <td>32120.000000</td>
      <td>1729.000000</td>
      <td>3015.290527</td>
      <td>2221.556152</td>
      <td>3571.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 73 columns</p>
</div>




```python
# check for the outliers
fig, ax = plt.subplots(4, 2, figsize=(15, 15))
# create a list of columns to check for outliers
outliers_col = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Fwd Packet Length Max',
                'Bwd Packet Length Max', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std']
                

# create a for loop to iterate over the columns
for i in range(0, 4):
    for j in range(0, 2):
        col = outliers_col[i * 2 + j]
        # create a boxplot for each column
        sns.boxplot(x=df[col], ax=ax[i, j])
plt.show()
```


    
![png](README_files/README_26_0.png)
    



```python
# remove the outliers using IsolationForest
from sklearn.ensemble import IsolationForest

# create an instance of the IsolationForest class
iso = IsolationForest(n_estimators=1000, max_samples='auto', contamination=float(0.05), max_features=1.0,
                        bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
# fit the model
yhat = iso.fit_predict(df.drop('Label', axis=1))
# select all rows that are not outliers
mask = yhat != -1
df = df[mask]
df.shape
```




    (13300, 74)




```python
df['Label'].value_counts()
```




    UDP          3500
    DrDoS_DNS    3489
    Syn          3485
    Benign       2826
    Name: Label, dtype: int64



We can see that there is a few outliers among the attacks and around 20% of outliers among the benign traffic. We have removed them.


```python
# put the Label colum in first place
df = pd.concat([df['Label'], df.drop('Label', axis=1)], axis=1)
df
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
      <th>Label</th>
      <th>Flow Duration</th>
      <th>Total Fwd Packets</th>
      <th>Total Backward Packets</th>
      <th>Fwd Packets Length Total</th>
      <th>Bwd Packets Length Total</th>
      <th>Fwd Packet Length Max</th>
      <th>Fwd Packet Length Min</th>
      <th>Fwd Packet Length Mean</th>
      <th>Fwd Packet Length Std</th>
      <th>...</th>
      <th>SYN Flag Count_0</th>
      <th>SYN Flag Count_1</th>
      <th>RST Flag Count_0</th>
      <th>RST Flag Count_1</th>
      <th>ACK Flag Count_0</th>
      <th>ACK Flag Count_1</th>
      <th>URG Flag Count_0</th>
      <th>URG Flag Count_1</th>
      <th>CWE Flag Count_0</th>
      <th>CWE Flag Count_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34849</th>
      <td>Syn</td>
      <td>62754538</td>
      <td>12</td>
      <td>8</td>
      <td>72.0</td>
      <td>48.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31946</th>
      <td>Syn</td>
      <td>44090418</td>
      <td>8</td>
      <td>2</td>
      <td>48.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11554</th>
      <td>Syn</td>
      <td>62485953</td>
      <td>12</td>
      <td>4</td>
      <td>72.0</td>
      <td>24.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3025</th>
      <td>Syn</td>
      <td>144</td>
      <td>2</td>
      <td>2</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35987</th>
      <td>Syn</td>
      <td>32300690</td>
      <td>6</td>
      <td>2</td>
      <td>36.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>69898</th>
      <td>Benign</td>
      <td>4578941</td>
      <td>3</td>
      <td>2</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>65736</th>
      <td>Benign</td>
      <td>47488</td>
      <td>2</td>
      <td>2</td>
      <td>70.0</td>
      <td>126.0</td>
      <td>35.0</td>
      <td>35.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53787</th>
      <td>Benign</td>
      <td>61212</td>
      <td>2</td>
      <td>2</td>
      <td>90.0</td>
      <td>146.0</td>
      <td>45.0</td>
      <td>45.0</td>
      <td>45.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16431</th>
      <td>Benign</td>
      <td>47060577</td>
      <td>18</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>62084</th>
      <td>Benign</td>
      <td>280</td>
      <td>1</td>
      <td>3</td>
      <td>6.0</td>
      <td>18.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>13300 rows × 74 columns</p>
</div>



## <a id='toc1_3_'></a>[Data visualization](#toc0_)

### <a id='toc1_3_1_'></a>[Distribution according to the attack type](#toc0_)


```python
# Plot the distribution of the Flow Duration for each class
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
sns.distplot(df[df['Label'] == 'Syn']['Flow Duration'], ax=ax[0], color='r')
ax[0].set_title('Syn')
sns.distplot(df[df['Label'] == 'DrDoS_DNS']['Flow Duration'], ax=ax[1], color='b')
ax[1].set_title('DrDoS_DNS')
sns.distplot(df[df['Label'] == 'UDP']['Flow Duration'], ax=ax[2], color='g')
ax[2].set_title('UDP')
sns.distplot(df[df['Label'] == 'Benign']['Flow Duration'], ax=ax[3], color='y')
ax[3].set_title('Benign')
plt.show()
```


    
![png](README_files/README_33_0.png)
    



```python
# Plot the distribution of the Total Fwd Packets for each class
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
sns.distplot(df[df['Label'] == 'Syn']['Total Fwd Packets'], ax=ax[0], color='r')
ax[0].set_title('Syn')
sns.distplot(df[df['Label'] == 'DrDoS_DNS']['Total Fwd Packets'], ax=ax[1], color='b')
ax[1].set_title('DrDoS_DNS')
sns.distplot(df[df['Label'] == 'UDP']['Total Fwd Packets'], ax=ax[2], color='g')
ax[2].set_title('UDP')
sns.distplot(df[df['Label'] == 'Benign']['Total Fwd Packets'], ax=ax[3], color='y')
ax[3].set_title('Benign')
plt.show()
```


    
![png](README_files/README_34_0.png)
    



```python
# Plot the distribution of the Total Backward Packets for each class
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
sns.distplot(df[df['Label'] == 'Syn']['Total Backward Packets'], ax=ax[0], color='r')
ax[0].set_title('Syn')
sns.distplot(df[df['Label'] == 'DrDoS_DNS']['Total Backward Packets'], ax=ax[1], color='b')
ax[1].set_title('DrDoS_DNS')
sns.distplot(df[df['Label'] == 'UDP']['Total Backward Packets'], ax=ax[2], color='g')
ax[2].set_title('UDP')
sns.distplot(df[df['Label'] == 'Benign']['Total Backward Packets'], ax=ax[3], color='y')
ax[3].set_title('Benign')
plt.show()
```


    
![png](README_files/README_35_0.png)
    



```python
# Plot the distribution of the Flow Bytes/s for each class
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
sns.distplot(df[df['Label'] == 'Syn']['Flow Bytes/s'], ax=ax[0], color='r')
ax[0].set_title('Syn')
sns.distplot(df[df['Label'] == 'DrDoS_DNS']['Flow Bytes/s'], ax=ax[1], color='b')
ax[1].set_title('DrDoS_DNS')
sns.distplot(df[df['Label'] == 'UDP']['Flow Bytes/s'], ax=ax[2], color='g') 
ax[2].set_title('UDP')
sns.distplot(df[df['Label'] == 'Benign']['Flow Bytes/s'], ax=ax[3], color='y')
ax[3].set_title('Benign')
plt.show()
```


    
![png](README_files/README_36_0.png)
    



```python
# Plot the distribution of the Flow Packets/s for each class
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
sns.distplot(df[df['Label'] == 'Syn']['Flow Packets/s'], ax=ax[0], color='r')
ax[0].set_title('Syn')
sns.distplot(df[df['Label'] == 'DrDoS_DNS']['Flow Packets/s'], ax=ax[1], color='b')
ax[1].set_title('DrDoS_DNS')
sns.distplot(df[df['Label'] == 'UDP']['Flow Packets/s'], ax=ax[2], color='g')
ax[2].set_title('UDP')
sns.distplot(df[df['Label'] == 'Benign']['Flow Packets/s'], ax=ax[3], color='y')
ax[3].set_title('Benign')
plt.show()
```


    
![png](README_files/README_37_0.png)
    


We can see that the distribution of the Syn attack is very different from the other classes. The values can be very high when regarding the number of packets sent or the flow duration.
However, the DNS attack have some really different charasterisitcs to benign traffic. We can see that the Flow Bytes/s and Flow Packets/s are very high for the DNS attack.

These visualizations help us to have a better understanding of the characteristics of the attacks.

### <a id='toc1_3_2_'></a>[Correlation matrix](#toc0_)


```python
# Plot the correlation matrix
corr = df.drop('Label', axis=1).corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.show()
```


    
![png](README_files/README_40_0.png)
    


We can see that we have high correlation between some features such as Packets Min/Max/Mean and Fwd Packets Min/Max/Mean, Idle Mean/Std/Min/Max and Flow IAT Mean/Std/Min/Max, etc. The OneHotEncoder has created some columns that are perfectly anti-correlated as we can see with the last columns.

The correlations will be handled by the PCA after the Scaling.

## <a id='toc1_4_'></a>[Feature Dimensionality Reduction](#toc0_)

### <a id='toc1_4_1_'></a>[PCA](#toc0_)


```python
# we standardize the features
from sklearn.preprocessing import StandardScaler
df.reset_index(inplace=True, drop=True)
# separate the features from the labels
X = df.drop('Label', axis=1)
y = df['Label']

# standardize the features
X = StandardScaler().fit_transform(X)
```


```python
# find the optimal number of components
from sklearn.decomposition import PCA
df_pca = PCA().fit(X)
plt.plot(np.cumsum(df_pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title('Explained variance vs number of components')
plt.show()
```


    
![png](README_files/README_45_0.png)
    


We see that to get 80% of the explained variance, we need to keep 12 components.


```python
pca = PCA(n_components=12)
principalComponents = pca.fit_transform(X)

# create a dataframe with the principal components
df_pca = pd.DataFrame(data=principalComponents, columns=['PC' + str(i) for i in range(1, 13)])

# concatenate the labels to the dataframe
df_pca = pd.concat([df_pca, df[['Label']]], axis=1)

# print the first 5 rows of the dataframe
df_pca.head()
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
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>PC5</th>
      <th>PC6</th>
      <th>PC7</th>
      <th>PC8</th>
      <th>PC9</th>
      <th>PC10</th>
      <th>PC11</th>
      <th>PC12</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.721210</td>
      <td>-1.175171</td>
      <td>1.191116</td>
      <td>-0.105998</td>
      <td>0.468198</td>
      <td>0.657741</td>
      <td>0.068088</td>
      <td>-1.278519</td>
      <td>-0.005922</td>
      <td>-0.057667</td>
      <td>-0.033371</td>
      <td>0.505432</td>
      <td>Syn</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.378424</td>
      <td>-1.749425</td>
      <td>-0.015752</td>
      <td>-0.240077</td>
      <td>-0.362503</td>
      <td>-1.557207</td>
      <td>-2.472501</td>
      <td>1.099247</td>
      <td>-0.021523</td>
      <td>-0.129386</td>
      <td>0.117597</td>
      <td>0.010918</td>
      <td>Syn</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.951613</td>
      <td>-1.542314</td>
      <td>0.610912</td>
      <td>0.080970</td>
      <td>0.207170</td>
      <td>-0.025175</td>
      <td>-0.350369</td>
      <td>-1.132098</td>
      <td>0.002249</td>
      <td>-0.016915</td>
      <td>-0.011656</td>
      <td>0.172930</td>
      <td>Syn</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.887585</td>
      <td>1.502519</td>
      <td>-1.590420</td>
      <td>-0.120690</td>
      <td>-1.285992</td>
      <td>0.041307</td>
      <td>0.189657</td>
      <td>0.127617</td>
      <td>-0.043467</td>
      <td>-0.101589</td>
      <td>-0.456610</td>
      <td>2.111943</td>
      <td>Syn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.232298</td>
      <td>-2.271800</td>
      <td>0.145053</td>
      <td>-0.605002</td>
      <td>-0.278456</td>
      <td>-1.818116</td>
      <td>-2.955724</td>
      <td>1.266808</td>
      <td>-0.027538</td>
      <td>-0.144523</td>
      <td>0.235049</td>
      <td>-0.498365</td>
      <td>Syn</td>
    </tr>
  </tbody>
</table>
</div>



in order to plot the PCA, we will keep 2 components.


```python
# plot 
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('PC1', fontsize=15)
ax.set_ylabel('PC2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
targets = ['Syn', 'DrDoS_DNS', 'UDP', 'Benign']
colors = ['r', 'b', 'g', 'y']
for target, color in zip(targets, colors):
    indicesToKeep = df_pca['Label'] == target
    ax.scatter(df_pca.loc[indicesToKeep, 'PC1'], df_pca.loc[indicesToKeep, 'PC2'], c=color, s=50 , alpha=0.5)

ax.legend(targets)
ax.grid()
plt.show()

```


    
![png](README_files/README_49_0.png)
    



```python
# plot 
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('PC1', fontsize=15)
ax.set_ylabel('PC2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
targets = ['Syn', 'DrDoS_DNS', 'UDP', 'Benign']
colors = ['r', 'b', 'g', 'y']
for target, color in zip(targets, colors):
    indicesToKeep = df_pca['Label'] == target
    ax.scatter(df_pca.loc[indicesToKeep, 'PC1'], df_pca.loc[indicesToKeep, 'PC2'], c=color, s=50 , alpha=0.5)
for i, txt in enumerate(df.drop('Label', axis=1).columns):
    plt.arrow(0, 0, 200*pca.components_[0][i], 200*pca.components_[1][i], color='black', alpha=0.2, head_width=0.3, width=.1)
    plt.annotate(txt, (200*pca.components_[0][i], 200*pca.components_[1][i]), size=7, alpha=0.7)

ax.legend(targets)
ax.grid()
plt.show()
```


    
![png](README_files/README_50_0.png)
    


### <a id='toc1_4_2_'></a>[Kernel PCA](#toc0_)


```python
from sklearn.decomposition import KernelPCA
```


```python
# check if the file exists
if not os.path.exists('data/results/df_kpca.parquet'):
    # we do the same thing for a kernel PCA
    kpca = KernelPCA(n_components=4, kernel='rbf')
    # create a dataframe with the principal components
    df_kpca = pd.DataFrame(data=kpca.fit_transform(X), columns=['PC' + str(i) for i in range(1, 5)])

    # concatenate the labels to the dataframe
    df_kpca = pd.concat([df_kpca, df[['Label']]], axis=1)

    # save the dataframe
    df_kpca.to_parquet('data/results/df_kpca.parquet')
else:
    df_kpca = pd.read_parquet('data/results/df_kpca.parquet')

# print the first 5 rows of the dataframe
df_kpca.head()
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
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.510527</td>
      <td>-0.281851</td>
      <td>-0.379878</td>
      <td>-0.125800</td>
      <td>Syn</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.211209</td>
      <td>-0.083731</td>
      <td>0.219308</td>
      <td>-0.005801</td>
      <td>Syn</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.697282</td>
      <td>-0.054895</td>
      <td>-0.282159</td>
      <td>-0.106529</td>
      <td>Syn</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.015130</td>
      <td>-0.329838</td>
      <td>0.062997</td>
      <td>-0.091189</td>
      <td>Syn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.015233</td>
      <td>-0.330022</td>
      <td>0.062888</td>
      <td>-0.091146</td>
      <td>Syn</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot 
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('PC1', fontsize=15)
ax.set_ylabel('PC2', fontsize=15)
ax.set_title('2 component K-PCA', fontsize=20)
targets = ['Syn', 'DrDoS_DNS', 'UDP', 'Benign']
colors = ['r', 'b', 'g', 'y']
for target, color in zip(targets, colors):
    indicesToKeep = df_kpca['Label'] == target
    ax.scatter(df_kpca.loc[indicesToKeep, 'PC1'], df_kpca.loc[indicesToKeep, 'PC2'], c=color, s=50 , alpha=0.5)

ax.legend(targets)
ax.grid()
plt.show()

```


    
![png](README_files/README_54_0.png)
    


Both the PCA and the kernel-PCA did a good job at reducing the dimensionality of the dataset to a point where we can visualy see the clusters.

but we can see that its hard to distinguish the UDP attacks from the DrDoS_DNS attacks

### <a id='toc1_4_3_'></a>[t-SNE](#toc0_)


```python
from sklearn.manifold import TSNE
# scatter plot the data
def plot_tsne(df_tsne):
    sns.scatterplot(x='PC1', y='PC2', hue='Label', data=df_tsne)
    plt.show()
```


```python
tsne = TSNE(n_components=2, verbose=1, random_state=123, perplexity=5)
df_tsne = pd.DataFrame(tsne.fit_transform(df.drop('Label', axis=1)), columns=['PC1', 'PC2'])
df_tsne['Label'] = df['Label'].values
plot_tsne(df_tsne)
```

    [t-SNE] Computing 16 nearest neighbors...
    [t-SNE] Indexed 13300 samples in 0.005s...
    [t-SNE] Computed neighbors for 13300 samples in 0.750s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 13300
    [t-SNE] Computed conditional probabilities for sample 2000 / 13300
    [t-SNE] Computed conditional probabilities for sample 3000 / 13300
    [t-SNE] Computed conditional probabilities for sample 4000 / 13300
    [t-SNE] Computed conditional probabilities for sample 5000 / 13300
    [t-SNE] Computed conditional probabilities for sample 6000 / 13300
    [t-SNE] Computed conditional probabilities for sample 7000 / 13300
    [t-SNE] Computed conditional probabilities for sample 8000 / 13300
    [t-SNE] Computed conditional probabilities for sample 9000 / 13300
    [t-SNE] Computed conditional probabilities for sample 10000 / 13300
    [t-SNE] Computed conditional probabilities for sample 11000 / 13300
    [t-SNE] Computed conditional probabilities for sample 12000 / 13300
    [t-SNE] Computed conditional probabilities for sample 13000 / 13300
    [t-SNE] Computed conditional probabilities for sample 13300 / 13300
    [t-SNE] Mean sigma: 0.000000
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 84.962517
    [t-SNE] KL divergence after 1000 iterations: 1.056271
    


    
![png](README_files/README_58_1.png)
    


We can see that the t-SNE algorithm is able to separate the different classes. However, the classes are not enough separated. We need to adjust the perplexity parameter in order to have a better separation.

According to this [article](https://towardsdatascience.com/how-to-tune-hyperparameters-of-tsne-7c0596a18868#:~:text=How%20to%20select%20optimal%20perplexity%3F), The optimal perplexity parameter depends on the number of samples in the dataset. As we have around 13k samples, we can try to set the perplexity to 100.

In order to save_time and avoid to run the t-SNE algorithm for a long time, we will load the data that are already computed and have 


```python
# check if the file 'data/results/df_tsne.parquet' exists
if os.path.isfile('data/results/df_tsne.parquet'):
    df_tsne = pd.read_parquet('data/results/df_tsne.parquet')
else:
    tsne = TSNE(n_components=2, verbose=1, random_state=123, perplexity=100)
    df_tsne = pd.DataFrame(tsne.fit_transform(df.drop('Label', axis=1)), columns=['PC1', 'PC2'])
    df_tsne['Label'] = df['Label'].values
plot_tsne(df_tsne)

```


    
![png](README_files/README_61_0.png)
    


The result looks very good with a perplexity of 100. We can see that the classes are well separated. We only have more difficulty to separate the UDP attacks and Benign traffic. And the UDP attacks are not well separated from the DrDos DNS attacks. The Syn attacks are well separated from the other classes.

Without the Benign traffic, we can see that the clusters are well separated.


```python
df_tsne_no_benign = df_tsne[df_tsne['Label'] != 'Benign']
plot_tsne(df_tsne_no_benign)
```


    
![png](README_files/README_63_0.png)
    



```python
if os.path.isfile('data/results/df_tsne.parquet'):
    df_tsne.to_parquet('data/results/df_tsne.parquet')
if os.path.isfile('data/results/df_tsne_no_benign.parquet'):
    df_tsne_no_benign.to_parquet('data/results/df_tsne_no_benign.parquet')
```

## <a id='toc1_5_'></a>[Supervised learning for multiclassification](#toc0_)

### <a id='toc1_5_1_'></a>[LDA](#toc0_)


```python
# copy the dataframe to a new one
df_lda = df.copy()

# mpa each label to a number
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df_lda['Label'])
df_lda['Label'] = le.transform(df_lda['Label'])
```


```python
# perform LDA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_lda.drop('Label', axis=1), df_lda['Label'], test_size=0.2, random_state=42)

# perform LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# make predictions
y_pred = lda.predict(X_test)

# calculate accuracy
accuracy_score(y_test, y_pred)
```




    0.9740601503759398



the overall accuracy is verry high, we can calculate the precision and recall and f1 score for each class


```python
# calculate the accuracy for each class
from sklearn.metrics import precision_score, recall_score, f1_score

print('Precision: ', precision_score(y_test, y_pred, average=None))
print('Recall: ', recall_score(y_test, y_pred, average=None))
print('F1: ', f1_score(y_test, y_pred, average=None))
```

    Precision:  [0.9963964  0.93697479 0.99159664 0.97636632]
    Recall:  [0.97359155 0.98818316 0.99578059 0.93892045]
    F1:  [0.98486198 0.96189792 0.99368421 0.95727734]
    


```python
# plot the confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
```




    <Axes: >




    
![png](README_files/README_71_1.png)
    


we can comapre it to an Qaudratic Discriminant Analysis
### <a id='toc1_5_2_'></a>[QDA](#toc0_)


```python
# perform QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

lda = QuadraticDiscriminantAnalysis()
lda.fit(X_train, y_train)

# make predictions
y_pred = lda.predict(X_test)

# calculate accuracy
accuracy_score(y_test, y_pred)
```




    0.9609022556390977




```python
# calculate the accuracy for each class
from sklearn.metrics import precision_score, recall_score, f1_score

print('Precision: ', precision_score(y_test, y_pred, average=None))
print('Recall: ', recall_score(y_test, y_pred, average=None))
print('F1: ', f1_score(y_test, y_pred, average=None))
```

    Precision:  [0.99125874 0.99657534 1.         0.87859825]
    Recall:  [0.99823944 0.85967504 0.99156118 0.99715909]
    F1:  [0.99473684 0.92307692 0.99576271 0.93413174]
    

both overall and per class accuracy are lower than in an LDA


```python
# plot the confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
```




    <Axes: >




    
![png](README_files/README_76_1.png)
    


### <a id='toc1_5_3_'></a>[PCA + LDA](#toc0_)


```python
# perform LDA on the pca data
# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_pca.drop('Label', axis=1), df_pca['Label'], test_size=0.2, random_state=42)

# perform LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# make predictions
y_pred = lda.predict(X_test)

accuracy_PCA_LDA = accuracy_score(y_test, y_pred)

# calculate the accuracy for each class
print('Precision: ', precision_score(y_test, y_pred, average=None))
print('Recall: ', recall_score(y_test, y_pred, average=None))
print('F1: ', f1_score(y_test, y_pred, average=None))
```

    Precision:  [0.99600798 0.97163121 0.91920732 0.73269436]
    Recall:  [0.87852113 0.80945347 0.84810127 0.97727273]
    F1:  [0.93358279 0.88315874 0.88222385 0.83749239]
    


```python
# plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
```




    <Axes: >




    
![png](README_files/README_79_1.png)
    


### <a id='toc1_5_4_'></a>[PCA + QDA](#toc0_)


```python
# perform QDA on the pca data
# perform QDA
lda = QuadraticDiscriminantAnalysis()
lda.fit(X_train, y_train)

# make predictions
y_pred = lda.predict(X_test)

accuracy_PCA_QDA = accuracy_score(y_test, y_pred)

# calculate the accuracy for each class
print('Precision: ', precision_score(y_test, y_pred, average=None))
print('Recall: ', recall_score(y_test, y_pred, average=None))
print('F1: ', f1_score(y_test, y_pred, average=None))
```

    Precision:  [0.99469027 0.97376543 0.99297753 0.93605442]
    Recall:  [0.98943662 0.93205318 0.99437412 0.97727273]
    F1:  [0.99205649 0.95245283 0.99367533 0.9562196 ]
    


```python
# plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
```




    <Axes: >




    
![png](README_files/README_82_1.png)
    


### <a id='toc1_5_5_'></a>[K-PCA + LDA](#toc0_)


```python
# perform LDA on the kpca data
# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_kpca.drop('Label', axis=1), df_kpca['Label'], test_size=0.2, random_state=42)

# perform LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# make predictions
y_pred = lda.predict(X_test)

accuracy_KPCA_LDA = accuracy_score(y_test, y_pred)

# calculate the accuracy for each class
print('Precision: ', precision_score(y_test, y_pred, average=None))
print('Recall: ', recall_score(y_test, y_pred, average=None))
print('F1: ', f1_score(y_test, y_pred, average=None))
```

    Precision:  [0.70849176 0.98888889 0.99068901 0.74773756]
    Recall:  [0.98415493 0.65731167 0.74824191 0.93892045]
    F1:  [0.8238762  0.78970719 0.8525641  0.8324937 ]
    


```python
# plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
```




    <Axes: >




    
![png](README_files/README_85_1.png)
    


### <a id='toc1_5_6_'></a>[K-PCA + QDA](#toc0_)


```python
# perform QDA on the kpca data
# perform QDA

lda = QuadraticDiscriminantAnalysis()
lda.fit(X_train, y_train)

# make predictions
y_pred = lda.predict(X_test)

accuracy_KPCA_QDA = accuracy_score(y_test, y_pred)

# calculate the accuracy for each class
print('Precision: ', precision_score(y_test, y_pred, average=None))
print('Recall: ', recall_score(y_test, y_pred, average=None))
print('F1: ', f1_score(y_test, y_pred, average=None))
```

    Precision:  [0.94661922 0.9084507  0.95485636 0.97108067]
    Recall:  [0.93661972 0.95273264 0.98171589 0.90625   ]
    F1:  [0.94159292 0.93006489 0.96809986 0.93754592]
    


```python
# plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
```




    <Axes: >




    
![png](README_files/README_88_1.png)
    


#### <a id='toc1_5_6_1_'></a>[T-SNE + LDA](#toc0_)


```python
# perform LDA on the t-SNE data
# copy the dataframe to a new one
df_tsne_lda = df_tsne.copy()

# mpa each label to a number
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df_tsne_lda['Label'])
df_tsne_lda['Label'] = le.transform(df_tsne_lda['Label'])
```


```python
# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_tsne_lda.drop('Label', axis=1), df_tsne_lda['Label'], test_size=0.2, random_state=42)

# perform LDA   
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# make predictions
y_pred = lda.predict(X_test)

accuracy_tsne_LDA = accuracy_score(y_test, y_pred)

# calculate the accuracy for each class
print('Precision: ', precision_score(y_test, y_pred, average=None))
print('Recall: ', recall_score(y_test, y_pred, average=None))
print('F1: ', f1_score(y_test, y_pred, average=None))
```

    Precision:  [0.68235294 0.83310902 0.81342282 0.74038462]
    Recall:  [0.4084507  0.91432792 0.84992987 0.87749288]
    F1:  [0.51101322 0.87183099 0.83127572 0.80312907]
    


```python
# plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
```




    <Axes: >




    
![png](README_files/README_92_1.png)
    


### <a id='toc1_5_7_'></a>[t-SNE + QDA](#toc0_)


```python
# perform QDA on the t-SNE data
# perform QDA
lda = QuadraticDiscriminantAnalysis()
lda.fit(X_train, y_train)

# make predictions
y_pred = lda.predict(X_test)

accuracy_tsne_QDA = accuracy_score(y_test, y_pred)

# calculate the accuracy for each class
print('Precision: ', precision_score(y_test, y_pred, average=None))
print('Recall: ', recall_score(y_test, y_pred, average=None))
print('F1: ', f1_score(y_test, y_pred, average=None))
```

    Precision:  [0.77468354 0.88841202 0.84246575 0.76315789]
    Recall:  [0.53873239 0.91728213 0.86255259 0.90883191]
    F1:  [0.63551402 0.90261628 0.85239085 0.82964889]
    


```python
# plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
```




    <Axes: >




    
![png](README_files/README_95_1.png)
    


the accuracy is very bad as we are not supposed to use t-SNE for classification

## <a id='toc1_6_'></a>[Unsupervised learning for clustering](#toc0_)

### <a id='toc1_6_1_'></a>[k-Means](#toc0_)


```python
from sklearn.cluster import KMeans
```


```python
# elbow method to choose the number of clusters
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=123)
    kmeanModel.fit(df_tsne.drop('Label', axis=1))
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method to find the optimal k')
plt.show()
```


    
![png](README_files/README_100_0.png)
    


According to the elbow method, the optimal number of clusters is 4, which is coherent with the number of classes in the dataset.


```python
# create kmeans object
N = 4
kmeans = KMeans(n_clusters=N, random_state=123)

# fit kmeans object to data
kmeans.fit(df_tsne[['PC1', 'PC2']])

# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)
```

    [[ 46.657986   -9.370605 ]
     [  5.588283   43.945107 ]
     [-46.13855     3.4791222]
     [ -4.443057  -40.495094 ]]
    


```python
# plot the data and the clusters learned
df_tsne['kmeans'] = kmeans.labels_
fig = plt.figure(figsize=(10, 10))
plt.subplot(221)
sns.scatterplot(x='PC1', y='PC2', hue='Label', data=df_tsne)
plt.title('Actual Labels')
plt.subplot(222)
sns.scatterplot(x='PC1', y='PC2', hue='kmeans', data=df_tsne)
plt.title('KMeans with {} Clusters'.format(N))
plt.show()
```


    
![png](README_files/README_103_0.png)
    


We can see that k-means is not able to separate the data into the correct clusters. This is because the shape of the k-means' clusters is always spherical, and it looks for clusters of equal variance, which in this case, is not the case.

We will try to use the k-means algorithm with the t-SNE results without the Benign traffic.


```python
# create a plot grid of 2x2
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
# plot the pie charts of distributions of labels for each cluster
for i in range(4):
    df_tsne[df_tsne['kmeans'] == i]['Label'].value_counts().plot.pie(ax=ax[i//2][i%2], autopct='%.2f', fontsize=12)
    ax[i//2][i%2].set_title('Cluster {}'.format(i))
plt.show()
```


    
![png](README_files/README_105_0.png)
    


We can see that 3 of the 4 clusters are almost fully constituted of one attack class but we still have trouble to separate the Benign traffic from the attacks as the Benign traffic is mixed among the 4 clusters.


```python
# evaluate the performance of the clustering using the silouette score
from sklearn.metrics import silhouette_score
print(silhouette_score(df_tsne[['PC1', 'PC2']], kmeans.labels_))

# plot the silhouette for the various clusters
from yellowbrick.cluster import SilhouetteVisualizer
visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
visualizer.fit(df_tsne[['PC1', 'PC2']])
visualizer.show()
```

    0.42762336
    


    
![png](README_files/README_107_1.png)
    





    <Axes: title={'center': 'Silhouette Plot of KMeans Clustering for 13300 Samples in 4 Centers'}, xlabel='silhouette coefficient values', ylabel='cluster label'>



The silhouette plot is a graphical tool presenting how well our data points fit into the clusters they’ve been assigned to and how well they would fit into other clusters. The silhouette coefficient is a measure of cluster cohesion and separation.

### <a id='toc1_6_2_'></a>[GMM](#toc0_)


```python
from sklearn.mixture import GaussianMixture
```


```python
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=123)
gmm.fit(df_tsne[['PC1', 'PC2']])
df_tsne['gmm'] = gmm.predict(df_tsne[['PC1', 'PC2']])
```


```python
# plot the results
fig = plt.figure(figsize=(10, 10))
plt.subplot(221)
sns.scatterplot(x='PC1', y='PC2', hue='Label', data=df_tsne, palette='Set1')
plt.title('Original labels')
plt.subplot(222)
sns.scatterplot(x='PC1', y='PC2', hue='gmm', data=df_tsne)
plt.title('GMM with {} components'.format(df_tsne['gmm'].nunique()))
plt.show()
```


    
![png](README_files/README_112_0.png)
    



```python
# create a plot grid of 2x2
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
# plot the pie charts of distributions of labels for each cluster
for i in range(4):
    df_tsne[df_tsne['gmm'] == i]['Label'].value_counts().plot.pie(ax=ax[i//2][i%2], autopct='%.2f', fontsize=12)
    ax[i//2][i%2].set_title('Cluster {}'.format(i))
plt.show()
```


    
![png](README_files/README_113_0.png)
    


### <a id='toc1_6_3_'></a>[DBSCAN](#toc0_)


```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.7, min_samples= 15).fit(df_tsne[['PC1', 'PC2']])
df_tsne['dbscan'] = dbscan.labels_
```


```python
# plot the data and the clusters learned
fig = plt.figure(figsize=(10, 10))
plt.subplot(221)
sns.scatterplot(x='PC1', y='PC2', hue='Label', data=df_tsne)
plt.title('Actual Labels')
plt.subplot(222)
sns.scatterplot(x='PC1', y='PC2', hue='dbscan', data=df_tsne)
plt.title('DBSCAN')
plt.show()
```


    
![png](README_files/README_116_0.png)
    


We can see that the DBSCAN algorithm creates more than 100 clusters, wich is not what we want. We need to check for the best eps parameter in order to have a realistic number of clusters.


```python
EPS = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
fig, ax = plt.subplots(4, 2, figsize=(20, 20))
for eps in EPS:
    dbscan = DBSCAN(eps=eps, min_samples= 20).fit(df_tsne[['PC1', 'PC2']])
    df_tsne['dbscan'] = dbscan.labels_
    # plot the data and the clusters learned
    sns.scatterplot(x='PC1', y='PC2', hue='dbscan', data=df_tsne, ax=ax[EPS.index(eps)//2][EPS.index(eps)%2])
    ax[EPS.index(eps)//2][EPS.index(eps)%2].set_title('DBSCAN with eps={}'.format(eps))
plt.show()
```


    
![png](README_files/README_118_0.png)
    


### <a id='toc1_6_4_'></a>[Hierarchical clustering](#toc0_)


```python
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
```


```python
def euclidean_distance(*args):
    return np.sqrt(np.sum((args[0] - args[1]) ** 2))
```

For the hierarchical clustering, we will first use a subsample of the dataset in order to have a better visualization of the dendrogram.


```python
samples = df_tsne.sample(100)
```


```python
# compute the distance matrix between all the points
distance_matrix = np.zeros((samples[['PC1', 'PC2']].shape[0], samples[['PC1', 'PC2']].shape[0]))
for i in range(samples[['PC1', 'PC2']].shape[0]):
    for j in range(samples[['PC1', 'PC2']].shape[0]):
        distance_matrix[i, j] = euclidean_distance(samples[['PC1', 'PC2']].iloc[i], samples[['PC1', 'PC2']].iloc[j])

sns.heatmap(distance_matrix)
```




    <Axes: >




    
![png](README_files/README_124_1.png)
    



```python
# compute the linkage matrix
Z = linkage(samples[['PC1', 'PC2']], method='average', metric='euclidean')

# plot the dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z, leaf_rotation=90., leaf_font_size=8., labels=samples['Label'].values)
plt.show()
```


    
![png](README_files/README_125_0.png)
    


We can see that the hierarchical clustering is able to separate the different types of attacks, however, the benign traffic is mixed among the different clusters.

We can try to perform the hierarchical clustering algorthm after having classified the Benign traffic from the attacks thanks to the LDA model.


```python
samples_no_benign = df_tsne_no_benign.sample(100)
```


```python
Z_no_benign = linkage(samples_no_benign[['PC1', 'PC2']], method='average', metric='euclidean')

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z_no_benign, leaf_rotation=90., leaf_font_size=8., labels=samples_no_benign['Label'].values)
plt.show()
```


    
![png](README_files/README_128_0.png)
    


We can see that without the Benign class, the hierarchical clustering with euclidean distance has very good results on the 3 classes. We clearly see 3 clusters (orange, green and red) corresponding to each class. However, we still have One big cluster with a mix of UDP, DrDoS_DNS and a little bit of Syn.


```python
Z_full = linkage(df_tsne_no_benign[['PC1', 'PC2']], method='average', metric='euclidean')

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z_full, leaf_rotation=90., leaf_font_size=8., no_labels=True)
plt.show()
```


    
![png](README_files/README_130_0.png)
    


We can see that we have 4 main clusters (orange, green, red and blue) whereas we should get only 3 classes. Let's see the proportions of each class in each cluster.


```python
df_tsne_no_benign['hierarchical'] = cut_tree(Z_full, n_clusters=4)
```


```python
# plot the pie charts of distributions of labels for each cluster
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
for i in range(4):
    df_tsne_no_benign[df_tsne_no_benign['hierarchical'] == i]['Label'].value_counts().plot.pie(ax=ax[i // 2, i % 2], title='Cluster {}'.format(i), autopct='%.2f')
plt.show()

```


    
![png](README_files/README_133_0.png)
    


We can see that we have 3 clusters that are almost only composed of one class. The DrDoS DNS and UDP attacks can't be sparated in the same cluster. However, the Syn attacks are well separated from the other classes.


```python
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
for cluster in range(0, df_tsne_no_benign['hierarchical'].nunique()):
    sns.scatterplot(x='PC1', y='PC2', hue='Label', data=df_tsne_no_benign[df_tsne_no_benign['hierarchical'] == cluster], ax=ax[cluster // 2, cluster % 2])
    ax[cluster // 2, cluster % 2].set_title('Cluster {}'.format(cluster))
# set the title of the plot
plt.suptitle('Hierarchical Clustering')
plt.show()
```


    
![png](README_files/README_135_0.png)
    


## <a id='toc1_7_'></a>[Other approaches](#toc0_)


```python
X_train, X_test, y_train, y_test = train_test_split(df.drop('Label', axis=1), df['Label'], test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
```

### <a id='toc1_7_1_'></a>[Decision tree](#toc0_)


```python
# train a decision tree classifier on the training set
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# predict the labels of the test set
y_pred = clf.predict(X_test)

# compute the accuracy of the predictions
accuracy_score(y_test, y_pred)
```




    0.9887218045112782




```python
# plot confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
```




    <Axes: >




    
![png](README_files/README_140_1.png)
    


### <a id='toc1_7_2_'></a>[K-Nearst Neighbors](#toc0_)


```python
# train a knn classifier on the training set
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

# predict the labels of the test set
y_pred = clf.predict(X_test)

# compute the accuracy of the predictions
accuracy_score(y_test, y_pred)
```




    0.9654135338345865




```python
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
```




    <Axes: >




    
![png](README_files/README_143_1.png)
    


### <a id='toc1_7_3_'></a>[Random Forest](#toc0_)


```python
# train a random forest classifier on the training set
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=123)
clf.fit(X_train, y_train)

# predict the labels of the test set
y_pred = clf.predict(X_test)

# compute the accuracy of the predictions
accuracy_score(y_test, y_pred)
```




    0.9913533834586467




```python
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
```




    <Axes: >




    
![png](README_files/README_146_1.png)
    


## <a id='toc1_8_'></a>[Conclusion](#toc0_)


```python
accuracy_LDA = 0.9763246899661782
accuracy_QDA = 0.9676813228109733
accuracy_DT = 0.9898496240601504
accuracy_KNN = 0.9616541353383459
accuracy_RF = 0.9902255639097745

x=['LDA', 'QDA', 'PCA/LDA', 'PCA/QDA', 'KPCA/LDA', 'KPCA/QDA', 't-SNE/LDA', 't-SNE/QDA', 'DT', 'KNN', 'RF']
y=[accuracy_LDA, accuracy_QDA, accuracy_PCA_LDA, accuracy_PCA_QDA, accuracy_KPCA_LDA, accuracy_KPCA_QDA, accuracy_tsne_LDA, accuracy_tsne_QDA, accuracy_DT, accuracy_KNN, accuracy_RF]

accuracies = {
    label: accuracy for label, accuracy in zip(x, y)
}
accuracies = {k: v for k, v in sorted(accuracies.items(), key=lambda item: item[1])}

plt.figure(figsize=(15, 7))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.75, 1)
plt.show()
```


    
![png](README_files/README_148_0.png)
    


According to our study, we can get some conclusions:

- Dimensionality reduction with PCA has not been very efficient. The PCA algorithm is not able to separate the different classes. This is because the PCA algorithm is a linear algorithm and the classes are not linearly separable. The Kernel PCA algortihm and the t-SNE algorithm have been more efficient at separating the classes. They are non-linear algorithms.
- Supervised Machine-Learning algortihms we learned in class have been very good at classifying the attacks types on the raw datas (without Dimensionality reduction).
- When we tried to apply the supervised algorithms on the dimensionality reduced data, the results were very different with first PCA/LDA and PCA/QDA that were not as good as the simple LDA and QDA. The KPCA has slightly decreased the accuracy of the QDA model compared to the simple one and has decreased the accuracy of the LDA model. Finally, the t-SNE model has decreased the accuracy of both LDA and QDA models. This is because the t-SNE algorithm is not supposed to be used for classification but is just a visualization tool.
- The results of the different unsupervised algorithms have been very different. The k-means algorithm has not been able to separate the classes as the k-means' clusters are spherical and the data are not linearly separable. The GMM algorithm has been able to separate the classes but not as good as the hierarchical clustering algorithm. The DBSCAN algorithm has not been able to separate the classes as it has created more than 100 clusters. Finally, the hierarchical clustering algorithm has been able to separate the classes very well.
- When we tried other supervised algorithms such as Decision Tree, K-Nearest Neighbors and Random Forest, we got medium results with KNN and some very good results with Random Forest and Decision Tree. This is because the Decision Tree and Random Forest are perfect for this kind of study as they are able to handle non-linear data and are able to classify the data with a hierarchical approach.

# <a id='toc2_'></a>[References](#toc0_)

- [Instructions](files/instructions.pdf)
- [Kaggle dataset](https://www.kaggle.com/devendra416/ddos-datasets)
- I. Sharafaldin, A. H. Lashkari, S. Hakak and A. A. Ghorbani, "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy," 2019 International Carnahan Conference on Security Technology (ICCST), Chennai, India, 2019, pp. 1-8, doi: 10.1109/CCST.2019.8888419.

# <a id='toc3_'></a>[License](#toc0_)
[MIT](LICENSE)
