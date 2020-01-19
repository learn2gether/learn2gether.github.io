---
title: "Titanic Analysis"
date: 2019-01-01
tags: [machine learning, data science]
# header:
#     image: "images/posts/titanic.jpeg"
excerpt: "Titanic Analysis by using machine learning algorithm such as Logistic Regression, Naive Bayes, Support Vector Machine, Decision Tree and Random Forest."
---

## Dataset
You can download the data from [Kaggle competition website](https://www.kaggle.com/c/titanic/data).
```python
# Download titanic train dataset
output = 'titanic_train.csv'
file = wget.download('https://drive.google.com/uc?authuser=0&id=12_P1znF91bWIM3ymS3PNslxpAOC7eUj_&export=download', output)
# Overwrite file if already exists
if os.path.exists(output):
    shutil.move(file,output)
# Download titanic test dataset (without labels)
output = 'titanic_test.csv'
file = wget.download('https://drive.google.com/uc?authuser=0&id=1Qn8R2Zm2U8-r07fT51bhozMUaOyKRkfS&export=download', output)
# Overwrite file if already exists
if os.path.exists(output):
    shutil.move(file,output)
```
