# Predict future sales 

This is the final project for coursera online course: "How to Win a Data Science Competition: Learn from Top Kagglers".It is also a public challenge on [Kaggle platform](https://www.kaggle.com/c/competitive-data-science-predict-future-sales).

### Goal

In this competition you will work with a challenging time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - 1C Company. We are asking you to predict total sales for every product and store in the next month. By solving this competition you will be able to apply and enhance your data science skills.

### Exploratory Data Analysis

* The overall sale show clear trend: decreasing with time, as well as seasonality: with peaks in November each year.
* There are no missing values in this dataset. However, training set includes item-shop pairs with non-zero sale, while test set includes all possible item-shop pairs. Thus we generate all possible item-shop pairs in the training set and assign the item count to be zero.
* The distribution of monthly sale is largely skewed, with many items sold 0-1 times, and a few with very large sale. The competition asks prediction of sales clipped to (0,20) range, this may alleviate potential problem of unbalanced targets.

### Feature Engineering

* The feature engineering part is built on the notebook by [Denis Larionov: eature engineering, xgboost](https://www.kaggle.com/dlarionov/feature-engineering-xgboost), with optimization on lag features and trend features. We also added group statistics of recent months (e.g. average of sale for each shop during past the 12 months).

### Modeling

* The validation strategy is to use month 12-32 as training set, month 33 as validation, and predict for month 34. We do not use month<12 * because we have used features involving statistics of the past 12 months.
* Three single models are fitted: LGBMRegressor, XGBRegressor, multilayer neural network.
Model ensembling is performed with weighted average and linear regression. The second level models are fitted using the prediction of validation set as features.

### public scores:
* singel model:  
lgb: 0.90501  
xgb: 0.91600  
nn:  0.99123  
* after ensembling:  
weighted average: 0.90338  
linear regression:0.90255  

As of Sept 24/2018, this score ranks 113/1458 on public leaderboard.

### Note on source code

prerequisite: python3, matplotlib, seaborn, pandas, sklearn, lightgbm, xgboost, keras, tensorflow.  
The source code should be run in the following order:  
1. feature_engineering.ipynb: some EDA and feature engineering, save the results for model fitting.
1. single model training:
    1. lgb.ipynb: LGBMRegressor
    1. xgb.ipynb: XGBRegressor
    1. nn.ipynb: multilayer neural network.
    The results of single models are saved to files for ensembling.
3. ensembling.ipynb: use weighted average and linear regression for ensampling.

The file data.pkl (engineered features) was too large (~1G) to be uploaded, so it is not included. One has to rerun feature_engineering.ipynb to generate the file. Other intermediate and final results can be found in the output folder. 


