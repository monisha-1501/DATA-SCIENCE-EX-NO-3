## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation

  # 2. POWER TRANSFORMATION
• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd 
df=pd.read_csv("Encoding Data.csv") 
df
```
<img width="565" height="463" alt="Screenshot 2026-11-24 081122" src="https://github.com/user-attachments/assets/2decb6b3-04e0-44dd-83a8-828e109d1e32" />




```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold'] 
e1=OrdinalEncoder(categories=[pm]) 
e1.fit_transform(df[["ord_2"]])
```

<img width="295" height="237" alt="image" src="https://github.com/user-attachments/assets/dc26747f-1ed8-4e7f-95f4-1a3be4539d4b" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

<img width="457" height="385" alt="image" src="https://github.com/user-attachments/assets/c5d90b9e-bdea-47ca-98d2-ce807b388bc4" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

```

<img width="390" height="352" alt="image" src="https://github.com/user-attachments/assets/709d69b6-381c-40cc-85c2-4199264dc0fd" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2

```

<img width="492" height="348" alt="image" src="https://github.com/user-attachments/assets/c664cae0-d3a1-4016-b96e-0f431fb5d3e5" />

```
pd.get_dummies(df2,columns=["nom_0"])

```
<img width="772" height="395" alt="image" src="https://github.com/user-attachments/assets/7c691317-3067-47d5-9ee1-ae2f20072b8f" />

```
pip install --upgrade category_encoders

```

<img width="755" height="431" alt="image" src="https://github.com/user-attachments/assets/fe17b564-e762-466d-8f9f-994f94855ba9" />

```

import pandas as pd 
from category_encoders import BinaryEncoder 
df=pd.read_csv("data.csv") 
df


```

<img width="547" height="357" alt="image" src="https://github.com/user-attachments/assets/d694bd32-9f87-41c5-961d-d0b0bc3f8ffe" />


```

be=BinaryEncoder() 
nd=be.fit_transform(df['Ord_2']) 
dfb=pd.concat([df,nd],axis=1) 
dfb


```

<img width="732" height="386" alt="image" src="https://github.com/user-attachments/assets/3533ae5d-7e2f-45dd-976a-d9f9fcb9a569" />

```
from category_encoders import TargetEncoder 
te=TargetEncoder() 
CC=df.copy() 
new=te.fit_transform(X=CC["City"],y=CC["Target"]) 
CC=pd.concat([CC,new],axis=1) 
CC

```

<img width="652" height="390" alt="image" src="https://github.com/user-attachments/assets/946ccb10-4ed6-42ce-8079-6eced393d594" />


```
import pandas as pd 
from scipy import stats 
import numpy as np 
df=pd.read_csv("Data_to_Transform.csv") 
df

```

<img width="735" height="467" alt="image" src="https://github.com/user-attachments/assets/0964be7c-ed72-4007-8323-3a48e0efb642" />

```
df.skew()

```

<img width="357" height="113" alt="image" src="https://github.com/user-attachments/assets/186af95c-97d9-4e53-9e0a-7c4f959652d1" />

```


np.log(df["Highly Positive Skew"])

```
<img width="582" height="265" alt="image" src="https://github.com/user-attachments/assets/524c30f7-a026-453d-a984-4d419fe5f3e1" />

```

np.reciprocal(df["Moderate Positive Skew"])


````


<img width="676" height="288" alt="image" src="https://github.com/user-attachments/assets/97168538-9e4f-41f4-b972-88214cc28b81" />


```

np.sqrt(df["Highly Positive Skew"])

```



<img width="610" height="280" alt="image" src="https://github.com/user-attachments/assets/e75bb974-cc52-4bea-9f85-7cf1cb75bb3d" />

```
np.square(df["Highly Positive Skew"])


```


<img width="618" height="272" alt="image" src="https://github.com/user-attachments/assets/5b7273fb-cb16-4d4e-a1c7-49b215f99b44" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"]) 
df
```



<img width="731" height="460" alt="image" src="https://github.com/user-attachments/assets/47a87201-2476-4b92-9f5e-9baa89e969b1" />


```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"]) 
df.skew()

```


<img width="503" height="186" alt="image" src="https://github.com/user-attachments/assets/11d92901-76b5-47be-9393-19fccb12e878" />



````

from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal') 
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
df

````


<img width="755" height="471" alt="image" src="https://github.com/user-attachments/assets/62c7ed92-ba42-4d54-ae44-4b4dd04dad22" />


````
import seaborn as sns 
import statsmodels.api as sm # STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION 
import matplotlib.pyplot as plt 
sm.qqplot(df["Moderate Negative Skew"],line='45') # QQ - QUANTILE QUANTILE PLOT 
plt.show()

````

<img width="793" height="547" alt="image" src="https://github.com/user-attachments/assets/930bf375-44bd-4e23-8bfb-8e06eb5f98ee" />



```

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL 
plt.show()

```

<img width="688" height="537" alt="image" src="https://github.com/user-attachments/assets/7b762f9c-8c69-40ae-8e1b-b03e345c424b" />

````

from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891) 
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
sm.qqplot(df["Moderate Negative Skew"],line='45') 
plt.show()

````

<img width="758" height="580" alt="image" src="https://github.com/user-attachments/assets/f48b2b2a-c858-4fc8-bb15-b42bb46e6ab9" />





## Summary:

This experiment focuses on preparing a dataset for analysis by using **feature encoding** and **data transformation**. First, the data is read and cleaned. Categorical information, such as labels or categories, is then converted into numbers so that a computer can process it.

Different encoding methods are used for different types of categories. **Ordinal encoding** assigns numbers to categories that have a natural order, such as *Cold, Warm,* and *Hot*. **Label encoding** assigns a number to each category. **One-hot encoding** creates a separate column for each category and uses 0 and 1 to show whether it is present. **Binary encoding** represents categories using binary digits and can reduce the number of columns needed.

The experiment also transforms numerical data. Methods such as logarithm, reciprocal, square root, and square transformation change the values in a column. Power transformations, including **Box-Cox** and **Yeo-Johnson**, can help adjust skewed data. A quantile transformation is also applied, and skewness values and Q-Q plots are used to examine the data distribution.

Finally, the encoded and transformed data can be saved to a file for later use. These techniques help organize data into a form that is more suitable for statistical analysis and machine-learning tasks.







      
# RESULT:       
To read the given data and perform Feature Encoding and Transformation process and save the data to a file is successfully  completed
