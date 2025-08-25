# Bank-Subscription-Prediction-with-XGBoost
Leveraging XGBoost to predict whether a customer will subscribe to a bank's term deposit

<img width="626" height="626" alt="image" src="https://github.com/user-attachments/assets/b932b1c9-6fe3-4f38-b9c3-5d4e26a486a8" />

---

### About the Project

This project has been completed as part of Kaggle's Playground Series Competition and as part of my continued practice and learning of data science techniques.

The goal of this competition and project is to predict whether or not a client will subscribe to a bank term deposit. A term deposit is a fixed-interest savings tool where money is deposited into an account at a financial institution and can only be withdrawn after a pre-determined term ends.

This represents a binary classification task. More specifically, a client either WILL or WONT subscribe to the term deposit.

In order to solve this task, I will be using XGBoost (eXtreme Gradient Boosting). XGBoost is a regularizing gradient boosting framework that utilizes decision trees and ensemble learning. This particular type of model is typically very effective in classification tasks.

I will first build and fit a basic XGBoost model to the raw data. This will provide a baseline performance. Following this, I will preprocess the data by removing outliers and creating new features via feature engineering.

The parameters of a second XGBoost model will be tuned with Optuna, which is an open source hyperparameter optimization framework.

All models will be evaluated using the area under the curve (AUC) of the receiver operating characteristic curve (ROC curve). The ROC curve is essentially the plot of the true positive rate versus the false positive rate. In the case of this model and competition, a 'perfect' model would yield an AUC of 1.0, while a terrible model would likely result in an AUC of 0.5 (basically random guess). However, obtaining an AUC of exactly 1 is typically impossible, so the objective will be to get the AUC as close to 1.0 as possible.

In order to speed up this notebook, I will also use NVIDIA's RAPIDS. By loading in 'cudf.pandas', all pandas operations will be accelerated on the GPU.

If you learned something new from this project or have suggestions on how to improve my model, please let me know! I'm also open to constructive criticism and learning new approaches.

---

### About the Data

This project utilizes the two datasets provided by Kaggle for this competition. The data is split into a 'train' and a 'test' set. The 'train' set includes 75% of the total data, or 750000 rows/entries. The 'test' set includes 25% of the total data, or 250000 rows/entries.

Target Variable:
- y: 0/1 --> Whether or not the client will subscribe to the term deposit

Numeric Features:
- id
- age
- balance
- day
- duration
- campaign
- pdays
- previous

Categorical Features
- job
- marital
- education
- default
- housing
- loan
- contact
- month
- poutcome

This data is publicly available for use under the Apache 2.0 license.

---

### Libraries Used

- pandas (cudf)
- numpy
- seaborn
- matplotlib
- sklearn
- catboost
- xgboost
- optuna
- shap

---

### Results
<img width="701" height="556" alt="image" src="https://github.com/user-attachments/assets/40a495d9-8bee-4426-9a2d-df5903170ce6" />

<img width="671" height="463" alt="image" src="https://github.com/user-attachments/assets/47b17c97-c75d-418c-a011-b89c0b3a71ed" />
