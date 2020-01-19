---
title: "Credit Default Analysis"
date: 2018-02-01
tags: [machine learning, data science, Credit Default Analysis]
# header:
#     image: "images/posts/titanic.jpeg"
excerpt: "A case study on Credit Default Analysis."
---

# Introduction
The objective of this report is to build a credit default model for a retail bank based on provided customer dataset. This dataset consists of 13444 records with 14 features. The main goal is to identify behaviours of defaulters. Then, the bank could provide credit card services based on
customers’ characteristics to reduce losses. The analysis is based on Python and a few supported libraries.

# Data Exploration

```python
# Missing values have a lot of format.
missing_values = ["n/a", "na", "--", " ","  "]
data = pd.read_csv('credit_data.txt', sep=",", na_values=missing_values)
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
      <th>CARDHLDR</th>
      <th>DEFAULT</th>
      <th>AGE</th>
      <th>ACADMOS</th>
      <th>ADEPCNT</th>
      <th>MAJORDRG</th>
      <th>MINORDRG</th>
      <th>OWNRENT</th>
      <th>INCOME</th>
      <th>SELFEMPL</th>
      <th>INCPER</th>
      <th>EXP_INC</th>
      <th>SPENDING</th>
      <th>LOGSPEND</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>27.250000</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1200.000000</td>
      <td>0</td>
      <td>18000.0</td>
      <td>0.000667</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>40.833332</td>
      <td>111</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4000.000000</td>
      <td>0</td>
      <td>13500.0</td>
      <td>0.000222</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>37.666668</td>
      <td>54</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3666.666667</td>
      <td>0</td>
      <td>11300.0</td>
      <td>0.033270</td>
      <td>121.989677</td>
      <td>4.803936</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>42.500000</td>
      <td>60</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2000.000000</td>
      <td>0</td>
      <td>17250.0</td>
      <td>0.048427</td>
      <td>96.853621</td>
      <td>4.573201</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>21.333334</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2916.666667</td>
      <td>0</td>
      <td>35000.0</td>
      <td>0.016523</td>
      <td>48.191670</td>
      <td>3.875186</td>
    </tr>
  </tbody>
</table>
</div>

# Handling Missing Values

```python
# There are two features with missing values.
data.isnull().sum()
```




    CARDHLDR        0
    DEFAULT         0
    AGE             0
    ACADMOS         0
    ADEPCNT         0
    MAJORDRG        0
    MINORDRG        0
    OWNRENT         0
    INCOME          0
    SELFEMPL        0
    INCPER          0
    EXP_INC         0
    SPENDING     2945
    LOGSPEND     2945
    dtype: int64


```python
# Visualization of missing values.
msno.matrix(data)
```

![alt text](https://learn2gether.github.io/images/posts/creditDefault/missingValue.png "missing values")

We can know that all missing values come from people who do not own credit card service, so they do not have credit card transaction records. Thus, we should remove these observations to reflect the truth. However, if these two features are not significant on truncated data. I will consider drop these two variables and build a model based on the whole population sample later. 

```python
# Remove all observations with missing values.
truncatedData = data[~data['SPENDING'].isnull()]
```

# EDA

The following diagram reflects the distribution of Default. By observing the whole sample population, we can see that defaulter is the minority which is less than ten percents. This figure is representative in the real world. However, the data is highly imbalanced, which may make the model overfitting the majority.

```python
sns.countplot(x=truncatedData['DEFAULT'],data=truncatedData)
print(truncatedData['DEFAULT'].value_counts())
truncatedData['DEFAULT'].value_counts()/truncatedData['DEFAULT'].count()
```

    0    9503
    1     996
    Name: DEFAULT, dtype: int64





    0    0.905134
    1    0.094866
    Name: DEFAULT, dtype: float64

![alt text](https://learn2gether.github.io/images/posts/creditDefault/defaulter.png "Credit Defaulters")

## Card Holders

We can see that if people have major derogatory reports greater than 5, it is hard to be accepted by credit card service. Thus, derogatory report is the marjor indicator to apply credit card successfully.

```python
pd.crosstab(data['DEFAULT'],data['MAJORDRG'])
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
      <th>MAJORDRG</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>21</th>
      <th>22</th>
    </tr>
    <tr>
      <th>DEFAULT</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10017</td>
      <td>1207</td>
      <td>512</td>
      <td>239</td>
      <td>137</td>
      <td>109</td>
      <td>58</td>
      <td>38</td>
      <td>32</td>
      <td>28</td>
      <td>17</td>
      <td>22</td>
      <td>5</td>
      <td>10</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>866</td>
      <td>99</td>
      <td>22</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[data['MAJORDRG']==5]['CARDHLDR'].value_counts()
```




    0    108
    1      2
    Name: CARDHLDR, dtype: int64

## Age


```python
truncatedData[truncatedData['AGE']<18]['AGE'].count()
```




    38

There are 38 customers under 18 years old. According to my research, people must be at least 18 years of age to apply for a credit card in Australia. However, there are 38 people under 18 years old who owns credit card in this dataset, which is not appropriate. This may be caused by system errors. 

```python
# split into different age group
#['18-22','23-34','35-40','41-60','61-80','81-100']
def ageGroup(self):
    if self < 22:
        return 'under 22'
    elif self < 35:
        return '22-34'
    elif self < 41:
        return '35-40'
    elif self < 66:
        return '41-65'
    else:
        return 'above 66'
```


```python
truncatedData['Age_labeled'] = truncatedData['AGE'].apply(ageGroup)
```

```python
# The proportion of defaulter at each group.
plt.figure(figsize=(16,9))
sns.countplot(x='Age_labeled',data=truncatedData, hue='DEFAULT')
```

![alt text](https://learn2gether.github.io/images/posts/creditDefault/ageGroup.png "Age Group")

## Majordrg: Number of major derogatory reports (loan payments that are 60 days overdue)

```python
truncatedData['MAJORDRG'].value_counts()
```

    0    9361
    1     855
    2     220
    3      47
    4      13
    5       2
    6       1
    Name: MAJORDRG, dtype: int64

```python
print(truncatedData[truncatedData['MAJORDRG']==0]['DEFAULT'].value_counts()/truncatedData[truncatedData['MAJORDRG']==0]['DEFAULT'].count())
print(truncatedData[truncatedData['MAJORDRG']==1]['DEFAULT'].value_counts()/truncatedData[truncatedData['MAJORDRG']==1]['DEFAULT'].count())
print(truncatedData[truncatedData['MAJORDRG']==2]['DEFAULT'].value_counts()/truncatedData[truncatedData['MAJORDRG']==2]['DEFAULT'].count())
print(truncatedData[truncatedData['MAJORDRG']==3]['DEFAULT'].value_counts()/truncatedData[truncatedData['MAJORDRG']==3]['DEFAULT'].count())
print(truncatedData[truncatedData['MAJORDRG']==4]['DEFAULT'].value_counts()/truncatedData[truncatedData['MAJORDRG']==4]['DEFAULT'].count())
print(truncatedData[truncatedData['MAJORDRG']==5]['DEFAULT'].value_counts()/truncatedData[truncatedData['MAJORDRG']==5]['DEFAULT'].count())
print(truncatedData[truncatedData['MAJORDRG']==6]['DEFAULT'].value_counts()/truncatedData[truncatedData['MAJORDRG']==6]['DEFAULT'].count())
```

    0    0.907489
    1    0.092511
    Name: DEFAULT, dtype: float64
    0    0.884211
    1    0.115789
    Name: DEFAULT, dtype: float64
    0    0.9
    1    0.1
    Name: DEFAULT, dtype: float64
    0    0.893617
    1    0.106383
    Name: DEFAULT, dtype: float64
    0    0.769231
    1    0.230769
    Name: DEFAULT, dtype: float64
    1    0.5
    0    0.5
    Name: DEFAULT, dtype: float64
    0    1.0
    Name: DEFAULT, dtype: float64


According to analysis above, default and the number of major derogatory report has a positive correlation. People are more likely to default along with the increasing number of major derogatory report. Thus, once customer has the first major derogatory report, the bank should re-evaluate the customer to determine whether suspend or terminate his/her credit card service.

## MINORDRG: Number of minor derogatory reports (loan payments that are less than 60 days overdue)

```python
truncatedData['MINORDRG'].value_counts()
```

    0    8960
    1    1046
    2     314
    3     117
    4      31
    5      20
    6       9
    7       2
    Name: MINORDRG, dtype: int64

```python
print(truncatedData[truncatedData['MINORDRG']==0]['DEFAULT'].value_counts()/truncatedData[truncatedData['MINORDRG']==0]['DEFAULT'].count())
print(truncatedData[truncatedData['MINORDRG']==1]['DEFAULT'].value_counts()/truncatedData[truncatedData['MINORDRG']==1]['DEFAULT'].count())
print(truncatedData[truncatedData['MINORDRG']==2]['DEFAULT'].value_counts()/truncatedData[truncatedData['MINORDRG']==2]['DEFAULT'].count())
print(truncatedData[truncatedData['MINORDRG']==3]['DEFAULT'].value_counts()/truncatedData[truncatedData['MINORDRG']==3]['DEFAULT'].count())
print(truncatedData[truncatedData['MINORDRG']==4]['DEFAULT'].value_counts()/truncatedData[truncatedData['MINORDRG']==4]['DEFAULT'].count())
print(truncatedData[truncatedData['MINORDRG']==5]['DEFAULT'].value_counts()/truncatedData[truncatedData['MINORDRG']==5]['DEFAULT'].count())
print(truncatedData[truncatedData['MINORDRG']==6]['DEFAULT'].value_counts()/truncatedData[truncatedData['MINORDRG']==6]['DEFAULT'].count())
```

    0    0.909821
    1    0.090179
    Name: DEFAULT, dtype: float64
    0    0.881453
    1    0.118547
    Name: DEFAULT, dtype: float64
    0    0.869427
    1    0.130573
    Name: DEFAULT, dtype: float64
    0    0.871795
    1    0.128205
    Name: DEFAULT, dtype: float64
    0    0.903226
    1    0.096774
    Name: DEFAULT, dtype: float64
    0    0.85
    1    0.15
    Name: DEFAULT, dtype: float64
    0    0.888889
    1    0.111111
    Name: DEFAULT, dtype: float64



The result of the analysis of minor derogatory report is quite similar with the major derogatory report. Thus, the bank also should re-evaluate customers if they have minor derogatory report as well.

## Income: Monthly income (divided by 10,000)

```python
# Distribution of income
plt.figure(figsize=(8,6))
sns.distplot(truncatedData['INCOME'])
```

![alt text](https://learn2gether.github.io/images/posts/creditDefault/income.png "Income Distribution")

```python
# Most people has income between 1500 and 3000
print(truncatedData[truncatedData['INCOME']<1500]['DEFAULT'].value_counts())
print(truncatedData[truncatedData['INCOME']<2000]['DEFAULT'].value_counts())
print(truncatedData[truncatedData['INCOME']<2500]['DEFAULT'].value_counts())
print(truncatedData[truncatedData['INCOME']<3000]['DEFAULT'].value_counts())
```

    0    1079
    1     225
    Name: DEFAULT, dtype: int64
    0    3113
    1     513
    Name: DEFAULT, dtype: int64
    0    5112
    1     715
    Name: DEFAULT, dtype: int64
    0    6848
    1     852
    Name: DEFAULT, dtype: int64



```python
def income(self):
    if self<2000:
        return 'under 2000'
    elif self<3000:
        return '2000-3000'
    else:
        return 'above 3000'
```


```python
# Create a categoricial feature for income.
truncatedData['income level'] = truncatedData['INCOME'].apply(income)
```

```python
truncatedData['income level'].value_counts()
```




    2000-3000     4074
    under 2000    3626
    above 3000    2799
    Name: income level, dtype: int64




```python
sns.countplot(x='income level', data=truncatedData, hue='DEFAULT')
```
![alt text](https://learn2gether.github.io/images/posts/creditDefault/income_group.png "Income Group")

According to above analysis, we can conclude that customers with low level of income are most likely to default.

## Exp_Inc: Ratio of monthly credit card expenditure to yearly income

The majority has ratio of monthly credit card expenditure to yearly income less than one twelfth, which means that their income and spending are balanced. However, they are still a large number of people who could not afford their bills, and these people are more likely to default. Thus, the bank should pay more attention on these people to avoid loss.

```python
plt.figure(figsize=(8,6))
sns.distplot(truncatedData['EXP_INC'])
```

![alt text](https://learn2gether.github.io/images/posts/creditDefault/Exp_Inc.png "Exp Inc")



```python
truncatedData[truncatedData['EXP_INC']>(1/12)]['DEFAULT'].value_counts()
```

    0    3550
    1     348
    Name: DEFAULT, dtype: int64


# Feature Selection




