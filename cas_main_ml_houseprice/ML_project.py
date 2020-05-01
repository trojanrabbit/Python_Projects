# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:23:02 2020

@author: Trojan Rabbit
"""

###### PACKAGES ######

# load packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn import linear_model, preprocessing, svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_regression, SelectPercentile, VarianceThreshold


###### FUNCTIONS ######

def load_data():
    """
    load training and test data.    
    """
    df_train = pd.read_csv("data/train.csv")
    df_train = df_train.drop(columns=['Id']) # no need
    
    df_test = pd.read_csv("data/test.csv")
    df_test = df_test.drop(columns=['Id']) # no need
    df_test['SalePrice'] = pd.read_csv("data/sample_submission.csv")['SalePrice']
    return df_train, df_test


def missing_data(df, cols):
    """
    get total and percentage of missing data.
    """
    missing_total = df[cols].isnull().sum().sort_values(ascending=True)
    missing_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=True)
    missing_data = pd.concat([missing_total, missing_percent], axis=1, keys=['missing_total', 'missing_percent'])
    missing_data = missing_data[missing_data.missing_total > 0]
    return missing_data


def get_col_type(df):
    """
    separate numeric and categorical variables.
    """
    col_quant = [f for f in df.columns if df.dtypes[f] != 'object']
    col_quant.remove('SalePrice') # target variable not needed in this selection
    col_qual = [f for f in df.columns if df.dtypes[f] == 'object']
    return col_quant, col_qual

    
def data_preprocessing(df):
    """
    really basic and naive preprocess data.
    """
    col_quant, col_qual = get_col_type(df)
    
    #---- missing data ----#
    #---- Quantitative
    print("missing numeric var:\n", missing_data(df, col_quant))    
    # replace NA with median
    df[col_quant] = df[col_quant].fillna(df[col_quant].median())
    
    #---- Qualitative
    print("missing categorical var:\n", missing_data(df, col_qual))    
    # replace NA with missing
    for c in col_qual:
        if df[c].isnull().any():
            df[c] = df[c].fillna('missing')
            
    #---- tranformations ----#
    # log transform sale price
    df["SalePrice"] = np.log1p(df["SalePrice"])        
    # log transform skewed features    
    skewed_feats = df[col_quant].apply(lambda x: skew(x)) # compute skewness per feature
    skewed_feats_sel = skewed_feats[skewed_feats > 0.75] # transform only features with skewness > 0.75
    skewed_feats_sel = skewed_feats_sel.append(skewed_feats[skewed_feats < -0.75]) # add and transform only features with skewness < -0.75
    skewed_feats_sel = skewed_feats_sel.index  
    df[skewed_feats_sel] = np.log1p(df[skewed_feats_sel])           
    return df


def equal_category(df_train, df_test):
    """
    add missing categories to test set based on categories from traininng data.
    Otherwise dummy creation leads to different amount of features due to not every category present in test-set.
    """
    col_quant, col_qual = get_col_type(df_train)
    
    for col in col_qual:
        # get categories from train set
        df_train[col] = df_train[col].astype('category')        
        cats = df_train[col].cat.categories
        # create dummy variables for test set
        dummies = pd.get_dummies(df_test[col], drop_first=True)
        dummies = dummies.reindex(columns=cats[1:], fill_value=0)        
        df_test = df_test.drop(columns=col)
        df_test[dummies.columns] = dummies
        # create dummy variables for train set
        dummies = pd.get_dummies(df_train[col], drop_first = True)
        df_train = df_train.drop(columns=col)
        df_train[dummies.columns] = dummies    
    return df_train, df_test


def plot_sale_price(df_train, df_test, plt_title):
    """
    plot sale price from train and test in same plot.
    """
    tr_sale_price = df_train.SalePrice
    te_sale_price = df_test.SalePrice
    sns.distplot(tr_sale_price, color="skyblue", label="Training")
    sns.distplot(te_sale_price, color="red", label="Test")
    plt.legend()
    plt.title(plt_title)
 
    
def plot_quant_var(df):
    """
    plot numeric variables in a grid.
    """
    col_quant, col_qual = get_col_type(df)

    f = pd.melt(df, value_vars=col_quant)
    g = sns.FacetGrid(f, col="variable",  col_wrap=8, sharex=False, sharey=False)
    g = g.map(sns.distplot, "value", kde = False)


def model_evaluation(test_y, train_y, pred_test_y, pred_train_y, model, df_summary, fit = None):
    """    
    print and plot some diagnostics for linear model:
        - coefficients (not applicable for SVR)
        - RMSE
        - R2
        - Residuals (not applicable for SVR)
        - Prediction vs. Truth
    """
    if model != 'SVR':
        # coefficients
        print('Coefficients: \n', fit.coef_)
    
    print('############# TEST SET #############')
    # mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(test_y, pred_test_y))
    # coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f' % r2_score(test_y, pred_test_y))
    
    print('############# TRAINING SET #############')
    # mean squared error
    print('Mean squared error: %.2f' % mean_squared_error(train_y, pred_train_y))
    # coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f' % r2_score(train_y, pred_train_y))       
    
    if model != 'SVR':
        # Plot residuals
        plt.scatter(pred_train_y, pred_train_y - train_y, c = "blue", label = "Training data")
        plt.scatter(pred_test_y, pred_test_y - test_y, c = "lightgreen", label = "Test data")
        plt.title("Residuals")
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        plt.legend(loc = "upper left")
        plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
        plt.show()    
    
    # Plot predictions
    plt.scatter(np.exp(pred_train_y)-1, np.exp(train_y)-1, c = "blue", label = "Training data")
    plt.scatter(np.exp(pred_test_y)-1, np.exp(test_y)-1, c = "lightgreen", label = "Test data")
    plt.title("Y True vs. Y Pred")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc = "upper left")
    plt.show()
    
    df_summary = df_summary.append({'Model':model, 
                                      'RMSE_train':mean_squared_error(train_y, pred_train_y), 
                                      'RMSE_test':mean_squared_error(test_y, pred_test_y),
                                      'R2_train':r2_score(train_y, pred_train_y),
                                      'R2_test':r2_score(test_y, pred_test_y)}, ignore_index=True)
    return df_summary


def scale_input(df):
    """
    scale model inputs.
    """
    col_quant, col_qual = get_col_type(df)
    df[col_quant] = preprocessing.scale(df[col_quant])
    #df['SalePrice'] = preprocessing.scale(df['SalePrice'])
    return df


def get_input_dataset(df_train, df_test):
    """
    split datasets in predictors and target.
    """
    train_y = df_train['SalePrice']
    train_x = df_train.drop(columns=['SalePrice'])
    
    test_y = df_test['SalePrice']
    test_x = df_test.drop(columns=['SalePrice'])
    return train_x, train_y, test_x, test_y


###### MAKE DATASETS ML READY ######
#---- load data ----#
df_train, df_test = load_data()

#---- plot numeric vars ----#
#plot_sale_price(df_train, df_test, 'SalePrice')

print("skewness train data:", skew(df_train['SalePrice']))
print("skewness test data:", skew(df_test['SalePrice']))
print("kurtosis train data:", kurtosis(df_train['SalePrice']))
print("kurtosis test data:", kurtosis(df_test['SalePrice']))

#plot_quant_var(df_train)

#---- preprocess data ----#
df_train = data_preprocessing(df_train)
df_test = data_preprocessing(df_test)
# make categories equal
df_train, df_test = equal_category(df_train, df_test)

#---- plot target after transformation ----#
#plot_sale_price(df_train, df_test, 'SalePrice Log')


#---- create dataframe for performance measures for each model ----#
df_ml_summary = pd.DataFrame(columns = ['Model', 'RMSE_train', 'RMSE_test', 'R2_train', 'R2_test'])


###### MODELS ######
#---- Linear Model All Features ----#
train_x, train_y, test_x, test_y = get_input_dataset(df_train, df_test)

# fit model
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

# make predictions
pred_train_y = regr.predict(train_x)
pred_test_y = regr.predict(test_x)

df_ml_summary = model_evaluation(test_y, train_y, pred_test_y, pred_train_y, 'Linear All Features', df_ml_summary, regr)

#---- Linear Model Features Selection ----#
train_x, train_y, test_x, test_y = get_input_dataset(df_train, df_test)

# remove features with low variance
selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
selector.fit_transform(train_x)
selector_cols_id = selector.get_support(indices=True)
train_x = train_x.iloc[:,selector_cols_id]
test_x = test_x.iloc[:,selector_cols_id]

# remove features based on univariate linear regression tests
selector = SelectPercentile(f_regression)
selector.fit(train_x, train_y)
selector_cols_id = selector.get_support(indices=True)
train_x = train_x.iloc[:,selector_cols_id]
test_x = test_x.iloc[:,selector_cols_id]

print("verbleibende Spalten:", list(train_x.columns) )

# fit model
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

# make predictions
pred_train_y = regr.predict(train_x)
pred_test_y = regr.predict(test_x)

df_ml_summary = model_evaluation(test_y, train_y, pred_test_y, pred_train_y, 'Linear Feature Selection', df_ml_summary, regr)

#---- Ridge Regression ----#
train_x, train_y, test_x, test_y = get_input_dataset(df_train, df_test)

# set alphas
alphas = alphas = np.arange(1, 50)

# grid search and fit
ridge = linear_model.RidgeCV(alphas = alphas, scoring='neg_root_mean_squared_error', cv=5)
ridge.fit(train_x, train_y)

print("bestes Alpha:", ridge.alpha_)

# make predictions
pred_train_y = ridge.predict(train_x)
pred_test_y = ridge.predict(test_x)

df_ml_summary = model_evaluation(test_y, train_y, pred_test_y, pred_train_y, 'Ridge Regression', df_ml_summary, ridge)


#---- SVR ----#
# scale datasets
df_train_s = scale_input(df_train)
df_test_s = scale_input(df_test)

train_x, train_y, test_x, test_y = get_input_dataset(df_train_s, df_test_s)

# set tuning parameters
tuned_parameters = [{'kernel': ['rbf', 'poly'],
                    'gamma': [0.001, 0.0001],
                    'C': [0.1, 0.3, 0.5, 0.8, 1, 2, 4, 10, 100]}]

# set scores
scores = ['neg_root_mean_squared_error', 'r2']

# grid search and fit
svr = GridSearchCV(svm.SVR(cache_size=2000), param_grid=tuned_parameters, scoring=scores, cv=5, refit='neg_root_mean_squared_error', return_train_score=True)
svr.fit(train_x, train_y)

print("SVR beste Parameter:", svr.best_params_)

# make predictions
pred_train_y = svr.predict(train_x)
pred_test_y = svr.predict(test_x)

df_ml_summary = model_evaluation(test_y, train_y, pred_test_y, pred_train_y, 'SVR', df_ml_summary)      


###### SUMMARY ######

print("Summary oof models ordered by 'RMSE'")
df_ml_summary.sort_values(by=['RMSE_test'])
print("Summary oof models ordered by 'R2'")
df_ml_summary.sort_values(by=['R2_test'], ascending=False)

