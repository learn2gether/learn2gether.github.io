---
title: "Credit Default Analysis"
date: 2018-02-01
tags: [machine learning, data science, Credit Default Analysis]
# header:
#     image: "images/posts/titanic.jpeg"
excerpt: "A case study on Credit Default Analysis."
---

![alt text](https://learn2gether.github.io/images/posts/creditDefault/credit_card.jpg "Credit Card")


- [Introduction](#introduction)
- [Data Exploration](#data-exploration)
- [Handling Missing Values](#handling-missing-values)
- [EDA](#eda)
  * [Card Holders](#card-holders)
  * [Age](#age)
  * [MAJORDRY](#majordry)
  * [MINORDRG](#minordrg)
  * [Income](#income)
  * [Exp_Inc](#exp_inc)
- [Feature Selection](#feature-selection)
- [Data Cleaning](#data-cleaning)
- [Building Models](#building-models)
  * [Logit Regression](#logit-regression)
  * [Predicted probabilities and goodness of fit measures](#predicted-probabilities-and-goodness-of-fit-measures)


# Introduction
<div style="text-align: justify"> The objective of this report is to build a credit default model for a retail bank based on provided customer dataset. This dataset consists of 13444 records with 14 features. The main goal is to identify behaviours of defaulters. Then, the bank could provide credit card services based on customers’ characteristics to reduce losses. The analysis is based on Python and a few supported libraries. </div>
<br />

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
<br />

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

<div style="text-align: justify"> We can know that all missing values come from people who do not own credit card service, so they do not have credit card transaction records. Thus, we should remove these observations to reflect the truth. However, if these two features are not significant on truncated data. I will consider drop these two variables and build a model based on the whole population sample later. </div>


```python
# Remove all observations with missing values.
truncatedData = data[~data['SPENDING'].isnull()]
```
<br />

# EDA

<div style="text-align: justify"> The following diagram reflects the distribution of Default. By observing the whole sample population, we can see that defaulter is the minority which is less than ten percents. This figure is representative in the real world. However, the data is highly imbalanced, which may make the model overfitting the majority. </div>

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

<div style="text-align: justify"> We can see that if people have major derogatory reports greater than 5, it is hard to be accepted by credit card service. Thus, derogatory report is the marjor indicator to apply credit card successfully. </div>

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

<div style="text-align: justify"> people must be at least 18 years of age to apply for a credit card in Australia. However, there are 38 people under 18 years old who owns credit card in this dataset, which is not appropriate. This may be caused by system errors. </div>

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

## MAJORDRY

Majordrg: Number of major derogatory reports (loan payments that are 60 days overdue)

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


<div style="text-align: justify"> According to analysis above, default and the number of major derogatory report has a positive correlation. People are more likely to default along with the increasing number of major derogatory report. Thus, once customer has the first major derogatory report, the bank should re-evaluate the customer to determine whether suspend or terminate his her credit card service. </div>

## MINORDRG

MINORDRG: Number of minor derogatory reports (loan payments that are less than 60 days overdue)

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



<div style="text-align: justify"> The result of the analysis of minor derogatory report is quite similar with the major derogatory report. Thus, the bank also should re-evaluate customers if they have minor derogatory report as well. </div>

## Income

Income: Monthly income (divided by 10,000)

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

## Exp_Inc

Exp_Inc: Ratio of monthly credit card expenditure to yearly income

<div style="text-align: justify"> The majority has ratio of monthly credit card expenditure to yearly income less than one twelfth, which means that their income and spending are balanced. However, they are still a large number of people who could not afford their bills, and these people are more likely to default. Thus, the bank should pay more attention on these people to avoid loss. </div>

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

<br />

# Feature Selection

<div style="text-align: justify"> According to the correlation heatmap, we can see that LOGSPEND and CARDHLDR are highly positively correlated (0.85). ADEPCNT and INCPER are relatively negatively correlated (-0.55). EXP_INC and SPENDING are higly positively correlated (0.86). EXP_INC and LOGSPEND are relatively positively correlated (0.62). SPENDING and LOGSPEND are relatively positively correlated (0.62). </div>

```python
# Calculate correlations
corr = data.corr()
plt.figure(figsize=(16,9))
plt.tight_layout()
# Heatmap
sns.heatmap(corr, annot=True, linewidths=0.5, cmap='coolwarm')
```

![alt text](https://learn2gether.github.io/images/posts/creditDefault/corr.png "correlation")

<br />

# Data Cleaning

<div style="text-align: justify"> We should split our dataset into response variable and predictor variables before building models. We will use DEFAULT as our response variable and all the remaining variable as predictors. We should drop CARDHLDR as there is only one category. First, we will build models based on data with original features. </div>

```python
Y = truncatedData['DEFAULT']
```

```python
X = truncatedData.drop(['CARDHLDR','Age_labeled','acadmos','income level','DEFAULT'], 1)
```


<div style="text-align: justify"> Then we use the statsmodels function to fit our models with our response variable and design matrix. The statsmodels package is unique from other languages and packages as it does not include an intercept term by default. This needs to be manually set. </div>



```python
import statsmodels.api as sm
```


```python
X = sm.add_constant(X)
```

```python
X.head()
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
      <th>const</th>
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
      <th>2</th>
      <td>1.0</td>
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
      <td>1.0</td>
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
      <td>1.0</td>
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
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>20.833334</td>
      <td>78</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1750.000000</td>
      <td>0</td>
      <td>11750.0</td>
      <td>0.031323</td>
      <td>54.815956</td>
      <td>4.003981</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>62.666668</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5250.000000</td>
      <td>0</td>
      <td>36500.0</td>
      <td>0.039269</td>
      <td>206.162467</td>
      <td>5.328664</td>
    </tr>
  </tbody>
</table>
</div>




```python
X.astype(float).info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 10499 entries, 2 to 13442
    Data columns (total 13 columns):
    const        10499 non-null float64
    AGE          10499 non-null float64
    ACADMOS      10499 non-null float64
    ADEPCNT      10499 non-null float64
    MAJORDRG     10499 non-null float64
    MINORDRG     10499 non-null float64
    OWNRENT      10499 non-null float64
    INCOME       10499 non-null float64
    SELFEMPL     10499 non-null float64
    INCPER       10499 non-null float64
    EXP_INC      10499 non-null float64
    SPENDING     10499 non-null float64
    LOGSPEND     10499 non-null float64
    dtypes: float64(13)
    memory usage: 1.1 MB

<br />

# Building models
## Logit Regression

```python
from statsmodels.discrete.discrete_model import Logit, Probit
```


```python
logitModel = Logit(Y, X.astype(float))
logit_model = logitModel.fit()
```

    Optimization terminated successfully.
             Current function value: 0.299394
             Iterations 7



```python
logit_model.summary2()
```




<table class="simpletable">
<tr>
        <td>Model:</td>              <td>Logit</td>      <td>Pseudo R-squared:</td>    <td>0.045</td>  
</tr>
<tr>
  <td>Dependent Variable:</td>      <td>DEFAULT</td>           <td>AIC:</td>         <td>6312.6721</td>
</tr>
<tr>
         <td>Date:</td>        <td>2019-11-03 12:08</td>       <td>BIC:</td>         <td>6407.0395</td>
</tr>
<tr>
   <td>No. Observations:</td>        <td>10499</td>       <td>Log-Likelihood:</td>    <td>-3143.3</td> 
</tr>
<tr>
       <td>Df Model:</td>             <td>12</td>            <td>LL-Null:</td>        <td>-3293.1</td> 
</tr>
<tr>
     <td>Df Residuals:</td>          <td>10486</td>        <td>LLR p-value:</td>    <td>6.1656e-57</td>
</tr>
<tr>
      <td>Converged:</td>           <td>1.0000</td>           <td>Scale:</td>         <td>1.0000</td>  
</tr>
<tr>
    <td>No. Iterations:</td>        <td>7.0000</td>              <td></td>               <td></td>     
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>       <th>Coef.</th>  <th>Std.Err.</th>    <th>z</th>     <th>P>|z|</th> <th>[0.025</th>  <th>0.975]</th> 
</tr>
<tr>
  <th>const</th>     <td>-0.2827</td>  <td>0.1860</td>  <td>-1.5203</td> <td>0.1284</td> <td>-0.6472</td> <td>0.0818</td> 
</tr>
<tr>
  <th>AGE</th>       <td>-0.0130</td>  <td>0.0041</td>  <td>-3.1913</td> <td>0.0014</td> <td>-0.0210</td> <td>-0.0050</td>
</tr>
<tr>
  <th>ACADMOS</th>   <td>0.0004</td>   <td>0.0006</td>  <td>0.7168</td>  <td>0.4735</td> <td>-0.0007</td> <td>0.0016</td> 
</tr>
<tr>
  <th>ADEPCNT</th>   <td>0.0936</td>   <td>0.0441</td>  <td>2.1241</td>  <td>0.0337</td> <td>0.0072</td>  <td>0.1800</td> 
</tr>
<tr>
  <th>MAJORDRG</th>  <td>0.2469</td>   <td>0.0704</td>  <td>3.5088</td>  <td>0.0005</td> <td>0.1090</td>  <td>0.3848</td> 
</tr>
<tr>
  <th>MINORDRG</th>  <td>0.2251</td>   <td>0.0483</td>  <td>4.6574</td>  <td>0.0000</td> <td>0.1304</td>  <td>0.3198</td> 
</tr>
<tr>
  <th>OWNRENT</th>   <td>-0.2713</td>  <td>0.0794</td>  <td>-3.4189</td> <td>0.0006</td> <td>-0.4269</td> <td>-0.1158</td>
</tr>
<tr>
  <th>INCOME</th>    <td>-0.0003</td>  <td>0.0001</td>  <td>-4.6838</td> <td>0.0000</td> <td>-0.0004</td> <td>-0.0002</td>
</tr>
<tr>
  <th>SELFEMPL</th>  <td>0.0725</td>   <td>0.1632</td>  <td>0.4445</td>  <td>0.6567</td> <td>-0.2472</td> <td>0.3923</td> 
</tr>
<tr>
  <th>INCPER</th>    <td>-0.0000</td>  <td>0.0000</td>  <td>-1.0107</td> <td>0.3121</td> <td>-0.0000</td> <td>0.0000</td> 
</tr>
<tr>
  <th>EXP_INC</th>   <td>2.7837</td>   <td>0.7913</td>  <td>3.5177</td>  <td>0.0004</td> <td>1.2327</td>  <td>4.3347</td> 
</tr>
<tr>
  <th>SPENDING</th>  <td>-0.0008</td>  <td>0.0004</td>  <td>-1.9270</td> <td>0.0540</td> <td>-0.0016</td> <td>0.0000</td> 
</tr>
<tr>
  <th>LOGSPEND </th> <td>-0.2188</td>  <td>0.0292</td>  <td>-7.4861</td> <td>0.0000</td> <td>-0.2761</td> <td>-0.1615</td>
</tr>
</table>

<div style="text-align: justify"> According to above table, we can interpret some outcome outputs. There is a list of log likelihoods at each iteration. The first iteration is the log likelihood of the empty model without any predictors. At the next iteration, the predictor is considered in the model. At Each iteration, the log likelihood increases untill the maximum value. Pseudo R square reflects how well the model fit the data. If this value is closed to 1, it means that the model fit the data very well. However, pseduo R square values in both our probit and logit models are quite low, and we do not have a very goodness-of-fit for these models. </div>
<br />
<div style="text-align: justify"> It appears that there are some predictors which are statistically significant (p-value is less than 0.05) including AGE, ADEPCNT, MAJORDRG, MINORDRG, OWNRENT, INCOME, EXP_INC and LOGSPEND. </div>

![alt text](https://learn2gether.github.io/images/posts/creditDefault/logitModel.png "Logit Regression")
<br />
<div style="text-align: justify"> By interpreting the coefficient, we normally interpret the sign of the coefficient but not the magnitude. The magnitude cannot be interpreted using the coefficient because different models have differnt scales of coeffcients. If the sign of the coefficient is positive, instead of saying higher predictor variable will lead to higher response variable, we will interpret that the response variable is more likely to be the category of 1. On the contrary, we will say that the response variable is less likely to be the category of 1 if the sign of the coefficient is negative. For example, as people get older, they are less likely to default. </div>

###  Predicted probabilities and goodness of fit measures

<br />
<div style="text-align: justify"> In order to find out how well we predict, we can look at predictions probabilities produced by this model. The accuracy is above 90 percent for both probit and logit models which is actually perfect. However, it predicts almost all people who do not default but not a single defaulter. In fact, the data is highly imbalanced. Thus, the result reflects overfitting for both probit and logit models. </div>
<br />

```python
9502/truncatedData['DEFAULT'].count()
```

    0.9050385751023907




```python
logit_model.pred_table()
```

    array([[9.502e+03, 1.000e+00],
           [9.960e+02, 0.000e+00]])

<br />
<div style="text-align: justify"> However, the magnitude of marginal effects can be interpreted. The marginal effects reflect the change in the probability of y=1 given a 1 unit change in an independent variable x. An increase in x increases (decreases) the probability that y=1 by the marginal effect expressed as a percent. For dummy independent variables, the marginal effect is expressed in comparision to the base category (x=0). For example, if someone is retired, they could be three percent more likely to have a insurance compared to those who are not retired. For continuous independent variable, the marginal effect is expressed for a one-unit change in x. For example, for each additional year of education, people are so many percent more likely to have insurance. Thus, we interpret both the sign and magnitude of the marginal effects. </div>
<br />

```python
me_logit = logit_model.get_margeff()
print(me_logit.summary()) 
```

            Logit Marginal Effects       
    =====================================
    Dep. Variable:                DEFAULT
    Method:                          dydx
    At:                           overall
    ==============================================================================
                    dy/dx    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    AGE           -0.0011      0.000     -3.187      0.001      -0.002      -0.000
    ACADMOS     3.528e-05   4.92e-05      0.717      0.474   -6.12e-05       0.000
    ADEPCNT        0.0078      0.004      2.123      0.034       0.001       0.015
    MAJORDRG       0.0206      0.006      3.504      0.000       0.009       0.032
    MINORDRG       0.0188      0.004      4.648      0.000       0.011       0.027
    OWNRENT       -0.0226      0.007     -3.414      0.001      -0.036      -0.010
    INCOME      -2.23e-05   4.78e-06     -4.665      0.000   -3.17e-05   -1.29e-05
    SELFEMPL       0.0060      0.014      0.445      0.657      -0.021       0.033
    INCPER     -4.077e-07   4.03e-07     -1.011      0.312    -1.2e-06    3.83e-07
    EXP_INC        0.2320      0.066      3.515      0.000       0.103       0.361
    SPENDING   -6.479e-05   3.36e-05     -1.926      0.054      -0.000    1.14e-06
    LOGSPEND      -0.0182      0.002     -7.464      0.000      -0.023      -0.013
    ==============================================================================


