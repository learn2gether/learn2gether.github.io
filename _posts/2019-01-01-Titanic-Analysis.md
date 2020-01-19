---
title: "Titanic Analysis"
date: 2019-01-01
tags: [machine learning, data science]
# header:
#     image: "images/posts/titanic.jpeg"
excerpt: "Titanic Analysis by using machine learning algorithm such as Logistic Regression, Naive Bayes, Support Vector Machine, Decision Tree and Random Forest."
---

# Dataset
You can download the data from [Kaggle competition website](https://www.kaggle.com/c/titanic/data).

```python
# read the training data
all_train = pd.read_csv('titanic_train.csv')
```

```python
# look at the shape of the data
all_train.shape
```

```python
# split the training set into two parts.
train = pd.read_csv('titanic_train.csv', sep=',').loc[0:712,:]
# The validation set can be used to tune models.
validation = pd.read_csv('titanic_train.csv', sep=',').loc[713:890,:]
test = pd.read_csv('titanic_test.csv', sep=',')
```

```python
# have a look at the data
train.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>


# Feature Selection and Feature Engineering


```python
# looking at missing values
train.isnull().any().sum()
```

```python
# msno is a missing data visualization module for Python.
msno.bar(train)
```
![alt text](https://learn2gether.github.io/images/posts/titanic/missing_values.png "missing values")


```python
# Cabin, PassengerId, Ticket, Name, Embarked is an irrelevant feature, and need to be removed.
train1 = train.drop(columns=['PassengerId', 'Ticket','Name','Embarked'])
test1 = test.drop(columns=['PassengerId', 'Ticket','Name','Embarked'])
validation1 = validation.drop(columns=['PassengerId', 'Ticket','Name','Embarked'])
```


```python
# look at the correlation between features
corr1 = train1.corr()

plt.figure(figsize=(6,3))
corr_heatmap1 = sns.heatmap(corr1, annot=True)
bottom, top = corr_heatmap1.get_ylim()
corr_heatmap1.set_ylim(bottom + 0.5, top - 0.5)
```
![alt text](https://learn2gether.github.io/images/posts/titanic/corr1.png "correlation")


```python
train1.head()
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
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C85</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>C123</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The Fare should have a strong positive relationship with the Pclass. However, some people in Pclass 1 paid less
# fare than some people in Pclass 2. Therefore, there may be two reasons. First, there are some typos in the record.
# Otherwise, some people got discounts. Thus, Fare cannot be used as a feature for the prediction.
plt.figure(figsize=(6,3))
train1[['Pclass','Fare']].plot(kind='scatter', x='Pclass',y='Fare', color='blue',alpha=0.5, figsize=(10,7))
```
![alt text](https://learn2gether.github.io/images/posts/titanic/pclass_fare.png "pclass vs fare")



```python
train2 = train1.drop(columns=['Fare'])
test2 = test1.drop(columns=['Fare'])
validation2 = validation1.drop(columns=['Fare'])
```



```python
# convert gender from categorical to numeric
train2['gender'] = train2['Sex'].apply(lambda x : 1 if x=='male' else 2)
validation2['gender'] = validation2['Sex'].apply(lambda x : 1 if x=='male' else 2)
test2['gender'] = test2['Sex'].apply(lambda x : 1 if x=='male' else 2)
```


not everyone has Cabin, so we assign NaN as 0, people who have cabin as 1


```python
def isNaN(num):
    if num != num:
        return 0
    else:
        return 1
```


```python
train2['Has_cabin']=train2['Cabin'].apply(isNaN)
test2['Has_cabin']=test2['Cabin'].apply(isNaN)
validation2['Has_cabin']=validation2['Cabin'].apply(isNaN)
```


```python
validation2.head()
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
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Cabin</th>
      <th>gender</th>
      <th>Has_cabin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>713</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>29.0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>714</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>52.0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>715</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>F G73</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>716</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>C45</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>717</th>
      <td>1</td>
      <td>2</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>E101</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# we can combine some feature together such as the family size.
train2['familySize']=train2['SibSp']+train2['Parch']
test2['familySize']=test2['SibSp']+test2['Parch']
validation2['familySize']=validation2['SibSp']+validation2['Parch']
```


```python
train3 = train2.drop(columns=['SibSp','Parch','Cabin'])
test3 = test2.drop(columns=['SibSp','Parch','Cabin'])
validation3 = validation2.drop(columns=['SibSp','Parch','Cabin'])
```


```python
sns.boxplot(x=train3.Pclass, y=train3.Age)
```
![alt text](https://learn2gether.github.io/images/posts/titanic/age_box_plot.png "age box plot")



```python
# There are missing values for ages
train3[train3['Age'].isnull()]['Pclass'].value_counts()
```




    3    113
    1     24
    2     10
    Name: Pclass, dtype: int64




```python
# We can fill up by the mean value of each class
print(train3[train3.Pclass==1]['Age'].mean())
print(train3[train3.Pclass==2]['Age'].mean())
print(train3[train3.Pclass==3]['Age'].mean())
```

    38.52907894736842
    30.294379562043794
    25.3014440433213


```python
import math
def fill_age(d):
    if isNaN(d['Age'])==0:
        if d['Pclass']==1:
            return math.ceil(train3[train3.Pclass==1]['Age'].mean())-1
        elif d['Pclass']==2:
            return math.ceil(train3[train3.Pclass==2]['Age'].mean())-1
        elif d['Pclass']==3:
            return math.ceil(train3[train3.Pclass==3]['Age'].mean())-1
    else:
        return d['Age']
```


```python
train3['filledAge']=train3[['Pclass','Age']].apply(fill_age, axis=1)
test3['filledAge']=test3[['Pclass','Age']].apply(fill_age, axis=1)
validation3['filledAge']=validation3[['Pclass','Age']].apply(fill_age, axis=1)
```


```python
train4 = train3.drop(columns=['Age'])
test4 = test3.drop(columns=['Age'])
validation4 = validation3.drop(columns=['Age'])
```


```python
train5 = train4.drop(columns=['Sex'])
test5 = test4.drop(columns=['Sex'])
validation5 = validation4.drop(columns=['Sex'])
```

# prediction


```python
train5.head()
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
      <th>Survived</th>
      <th>Pclass</th>
      <th>gender</th>
      <th>Has_cabin</th>
      <th>familySize</th>
      <th>filledAge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>35.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test5.head()
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
      <th>Pclass</th>
      <th>gender</th>
      <th>Has_cabin</th>
      <th>familySize</th>
      <th>filledAge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>34.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train = train5.iloc[:, 1:]
```


```python
y_train = train5.iloc[:, 0:1]
```


```python
# double check whether there are missing values.
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 713 entries, 0 to 712
    Data columns (total 5 columns):
    Pclass        713 non-null int64
    gender        713 non-null int64
    Has_cabin     713 non-null int64
    familySize    713 non-null int64
    filledAge     713 non-null float64
    dtypes: float64(1), int64(4)
    memory usage: 28.0 KB






```python
X=X_train.to_numpy()
```


```python
y=y_train.to_numpy()
```


```python
X_valid = validation5.iloc[:, 1:].to_numpy()
y_valid = validation5.iloc[:, 0:1].to_numpy()
```


## logistic regression


```python
from sklearn.linear_model import LogisticRegression
```


```python
lr = LogisticRegression().fit(X,y)
```


```python
lr_predTest = lr.predict(X_valid)
```


```python
from sklearn.metrics import classification_report
```


```python
print(classification_report(y_valid,lr_predTest))
```

                  precision    recall  f1-score   support
    
               0       0.86      0.88      0.87       115
               1       0.77      0.75      0.76        63
    
        accuracy                           0.83       178
       macro avg       0.82      0.81      0.81       178
    weighted avg       0.83      0.83      0.83       178
    



```python
# submission
```


```python
lr_pred = lr.predict(test5.to_numpy())
```


## random forest


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
rfm = RandomForestClassifier(n_estimators=70, oob_score=True, n_jobs=-1, random_state=101, max_features=None, min_samples_leaf=30)
```


```python
rfm.fit(X,y)
```


```python
rfm_predTest = rfm.predict(X_valid)
```


```python
print(classification_report(y_valid,rfm_predTest))
```

                  precision    recall  f1-score   support
    
               0       0.82      0.95      0.88       115
               1       0.87      0.62      0.72        63
    
        accuracy                           0.83       178
       macro avg       0.84      0.78      0.80       178
    weighted avg       0.84      0.83      0.82       178
    



```python
#submission
```


```python
rfm_pred = rfm.predict(test5.to_numpy())
```



## Naive Bayes


```python
from sklearn.naive_bayes import GaussianNB
```


```python
nb = GaussianNB().fit(X,y)
```

```python
nb_predTest = nb.predict(X_valid)
```


```python
print(classification_report(y_valid,nb_predTest))
```

                  precision    recall  f1-score   support
    
               0       0.88      0.82      0.85       115
               1       0.70      0.79      0.75        63
    
        accuracy                           0.81       178
       macro avg       0.79      0.81      0.80       178
    weighted avg       0.82      0.81      0.81       178
    



```python
# submission
nb_pred = nb.predict(test5.to_numpy())
```


## decision tree


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
dtree = DecisionTreeClassifier(max_depth=10, random_state=101, max_features=None, min_samples_leaf=15)
```


```python
dtree.fit(X,y)
```


```python
dtree_predTest = dtree.predict(X_valid)
```


```python
print(classification_report(y_valid,dtree_predTest))
```

                  precision    recall  f1-score   support
    
               0       0.85      0.88      0.86       115
               1       0.76      0.71      0.74        63
    
        accuracy                           0.82       178
       macro avg       0.81      0.80      0.80       178
    weighted avg       0.82      0.82      0.82       178
    



```python
# submission
dtree_pred = dtree.predict(test5.to_numpy())
```


## support vector machine


```python
from sklearn.svm import SVC
```


```python
svm = SVC(kernel='linear', C=0.025, random_state=101)
```


```python
svm.fit(X,y)
```

```python
svm_predTest = svm.predict(X_valid)
```


```python
print(classification_report(y_valid,svm_predTest))
```

                  precision    recall  f1-score   support
    
               0       0.83      0.87      0.85       115
               1       0.74      0.68      0.71        63
    
        accuracy                           0.80       178
       macro avg       0.79      0.78      0.78       178
    weighted avg       0.80      0.80      0.80       178
    



```python
# submission
svm_pred = svm.predict(test5.to_numpy())
```

