#!/usr/bin/env python
# coding: utf-8

# ### 1. Problem Statement
# The fatigue dataset has the chemical composition, process condition and fatigue strength of metals. The goal of this project is to predict fatigue strength based on chemical composition and process condition.
# 
# Success will be measured with r2 score and the objective is to get a r2 score > 0.9
# 
# ### 2. Data extraction
# data resource from Exploration of data science techniques to predict fatigue strength of steel from composition and processing parameters
# 
# ### 3. Exploratory Data Analysis
# ### 4. Feature Engineering
# ### 5. Feature Selection
# ### 6. Model Selection
# ### 7. Model tuning
# ### 8. Model serving (if applicable)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# ### 2. Data extraction

# In[2]:


data_set = pd.read_excel("Fatigue.xlsx")


# In[3]:


data_set.info()


# In[4]:


# split in train and test
#from sklearn.model_selection import train_test_split

#train_set, test_set = train_test_split(data_set, test_size=0.2, random_state=42)
#data_set = train_set

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data_set, data_set["CT"]):
    strat_train_set = data_set.loc[train_index]
    strat_test_set = data_set.loc[test_index]
    
data_set = strat_train_set


# In[5]:


strat_train_set['CT'].value_counts() / len(strat_train_set)


# In[6]:


strat_test_set["CT"].value_counts() / len(strat_test_set)


# ### 3. Exploratory Data Analysis
# 
# - Data Shape
# - Data Types
# - Target variable
# - Missing Values
# - Constant features
# - Cardinality
# - Duplicate features
# - Correlations
# - Scale
# - Distributions (skewness)
# - Outliers

# In[7]:


data_set.head(3)


# In[8]:


data_set.info(memory_usage='deep')


# This dataset has:
# - 27 features, all are numerical
# - 349 observations
# - no missing values
# 
# The target variable is ['Fatigue']

# #### Feature Details
# - C: % Carbon
# - Si: % Silicon
# - Mn: % Manganese
# - P: % Phosphorus
# - S: % Sulphur
# - Ni: % Nickel
# - Cr: % Chromlum
# - Cu: % Copper
# - Mo: % Molybdenum
# - NT: Nomalizing Temperature
# - THT: Through Hardening Temperature
# - THt: Through Hardening Time
# - THQCr: Cooling Rate for Through Hardening
# - CT: Carburization Temperature
# - Ct: Carburization Time
# - DT: Diffusion Temperature
# - QmT: Quenching Media Temperature (for Carburization)
# - TT: Tempering Temperature
# - Tt: Tempering Time
# - TCr: Cooling Rate for Tempering
# - RedRatio: Reduction Ratio (Ingot to Bar)
# - dA: Area Proportion of Inclusions Deformed by Plastic Work
# - dB: Area Proportion of Inclusions Occuring in Discontinuous Array
# - dC: Area Proportion of Isolated Inclusions
# - Fatigue: Rotating Bending Fatigue Strength (10^7 cycles)

# In[9]:


#constant features
data_set.nunique().sort_values(ascending=False)


# This dataset has:
# 
# - 2 binary features: ['CT'], ['THt']
# - ['Sl. No.'] has as many unique values as observations in our dataset. Judging by the name, it may be ordinal and we may drop this feature.

# In[10]:


# high level statistics for numerical variables
desc = data_set.describe()
desc


# In[11]:


desc.iloc[[1,5, 3, 7], 1::].T


# - ['SI.No.'] mean and median are the same making the symmetrical normal and it seems to be ordinal with a unique value per observation, making the variable irrelevant.
# - left tail (median>mean): ['THT'], ['THt'], ['TT'], ['TCr'], ['C'], ['P'], ['S'], ['Ni'], ['Cr']
# - right tail (mean>median): ['NT'], ['THQCr'], ['CT'], ['Ct'], ['DT'], ['Dt'], ['QmT'], ['Tt'], ['Si'], ['Mn'], ['Cu'], ['Mo'], ['RedRatio'], ['dA'], ['dB'], ['dC'], ['Fatigue']  
# - 4 order of magnitude: ['RedRatio'], ['Fatigue']
# - 3 order of magnitude: ['NT'], ['THT'], ['CT'], ['Ct'], ['DT'], ['QmT'], ['TT'], ['Tt']
# - 2 order of magnitude: ['THt'], ['THQCr'], ['Dt'], ['TCr']
# - 1 order of magnitude: ['C'], ['Si'], ['Mn'], ['P'], ['S'], ['Ni'], ['Cr'], ['Cu'], ['Mo'], ['dA'], ['dB'], ['dC']
# - scaling is required

# In[12]:


pearcorr = data_set.corr()
plt.figure(figsize=(15,15), dpi=100)
sns.heatmap(pearcorr, annot=True)


# - There are many features have correlation > 0.9, which can be dropped. 
# - For example, ['THT'], ['THt'], ['TCr'], ['TT'] and ['CT'], ['Ct'], ['DT'], ['Tt']

# In[13]:


data_set.corr(method='pearson')['Fatigue'].sort_values(key=lambda x:abs(x))


# - Processing parameters are more relevant
# - Chemical compositions and reduction ratio are less relevant
# - The most relevant chemical compositions are Cr, C and Mo

# In[14]:


sns.set_theme(style="ticks")
sns.pairplot(data_set[['Dt', 'Ct', 'DT', 'CT', 'Tt', 'Fatigue']])


# ### 4. Feature Engineering
# 
# - Split data in train and validation
# - Missing values imputation
# - Deal with outliers
# - Encoding
# - Variance Stabilizing Transformations
# - Scaling

# In[15]:


# Split in train and validation
y = strat_train_set['Fatigue']
X = strat_train_set.drop(columns=['Fatigue', 'Sl. No.'])


# In[16]:


# Numerical imputation
from feature_engine.imputation import MeanMedianImputer

num_imputer = MeanMedianImputer()


# In[17]:


# Deal with outliers
from feature_engine.outliers import Winsorizer

capper = Winsorizer(capping_method="iqr", tail="both")


# In[18]:


# Variance Stabilizing Transformations
from feature_engine.transformation import YeoJohnsonTransformer

yeo_trans = YeoJohnsonTransformer()


# In[19]:


# Scale
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# ### 5. Feature Selection
# - Drop Duplicate
# - Drop Constant/Quasi constant
# - Drop correlated
# - Embedded methods

# In[20]:


from feature_engine.selection import DropConstantFeatures, SmartCorrelatedSelection, DropDuplicateFeatures

# Drop constant
drop_const = DropConstantFeatures()

# Drop duplicates
drop_dup = DropDuplicateFeatures()

# Drop correlated
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

drop_corr = SmartCorrelatedSelection(method='pearson', threshold = 0.95,
                                     selection_method='model_performance',
                                     estimator=linreg)


# In[21]:


from sklearn.pipeline import make_pipeline

pipe = make_pipeline(num_imputer,
                   #  capper,
                  #   yeo_trans,
                     scaler,
                     drop_const,
                     drop_dup,
                     drop_corr
                    )
X_tr = pipe.fit_transform(X, y)


# In[22]:


X_tr


# ### 6. Model Selection

# In[23]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from xgboost import XGBRFRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import GridSearchCV


# In[24]:


lr = LinearRegression()
svr = SVR(kernel="linear", C=10, gamma=0.1, epsilon=0.1)
knr = KNeighborsRegressor()
dt = DecisionTreeRegressor()
rfr = RandomForestRegressor(max_depth=8,random_state=42)
xgb = XGBRegressor(objective = 'reg:squarederror', random_state =27)


# In[25]:


models = [lr, svr, knr, dt, rfr, xgb]


# In[26]:


from sklearn.model_selection import cross_val_score


# In[27]:


for model in models:
    
    scores = cross_val_score(model, 
                         X_tr, 
                         y, 
                         scoring="r2", 
                         cv = 5)
    
    print(scores.mean())


# ### 7. Model tuning
# - SVR
# - RandomForest
# - XGBoost

# In[ ]:


parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear', 'reg:squarederror'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [4, 5, 6, 7],
              'min_child_weight': [4, 5, 6],
              'silent': [1],
              'subsample': [0.6, 0.7, 0,8],
              'colsample_bytree': [0.6, 0.7, 0.8],
              'n_estimators': [100, 300, 500]}

xgb_reg = XGBRegressor()

grid_search = GridSearchCV(xgb_reg, parameters,
                           scoring='r2',
                           n_jobs = 5,
                           verbose=True,
                           return_train_score=True,
                           cv=5,
                          )

grid_search.fit(X_tr, y)


# In[63]:


grid_search.best_params_


# In[64]:


grid_search.best_score_


# In[65]:


# feature importances 
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[66]:


attrs = list(X)
sorted(zip(attrs, feature_importances), reverse=True)


# In[28]:


# final model 
xgb_model = XGBRegressor(colsample_bytree = 0.6,
                         learning_rate = 0.03,
                         max_depth = 7,
                         min_child_weight = 4,
                         n_estimators = 500,
                         nthread = 4,
                         objective = 'reg:linear', 
                         silent = 1,
                         subsample = 0.7,
                         random_state =27 
                         )

xgb_model_deploy = xgb_model.fit(X_tr, y)
prediction  = xgb_model_deploy.predict(X_tr)

#st_dev = (mean_squared_error(prediction, y) ** 0.5).round(-3)
st_dev = mean_squared_error(prediction, y) ** 0.5
xgb_model_deploy.st_dev = st_dev
    
plt.rcParams["figure.figsize"] = (5,15)
plot_importance(xgb_model_deploy)
plt.show()


# ### Evaluate on test data

# In[29]:


#final_model = grid_search.best_estimator_

X_test = strat_test_set.drop(columns =["Fatigue", "Sl. No."], axis=1)
y_test = strat_test_set["Fatigue"].copy()

X_test_tr = pipe.transform(X_test)

y_test_pred = xgb_model_deploy.predict(X_test_tr)
final_r2 = r2_score(y_test, y_test_pred)
final_r2


# In[30]:


#plot
plt.figure(figsize=(6,6))
    
mx = max(max(y_test),max(y_test_pred))
mi = min(min(y_test),min(y_test_pred))
arr = np.linspace(mi, mx, 20)
plt.plot(arr, arr, c='b')
plt.scatter(y_test, y_test_pred, c='r')
plt.xlabel("Actual Fatigue Strength")
plt.ylabel("Predicted Fatigue Strength")


# ### 8. Model serving

# In[31]:


full_pipeline_with_predictor = make_pipeline(pipe,
                                            xgb_model)
full_pipeline_with_predictor.fit(X, y)


# In[32]:


# validate 
y_test_pred = full_pipeline_with_predictor.predict(X_test)
final_r2 = r2_score(y_test, y_test_pred)
final_r2


# In[34]:


import pickle


# In[43]:


xgb_model_deploy = full_pipeline_with_predictor
xgb_model_deploy.st_dev = st_dev
with open('xgb_model_deploy.pickle', 'wb') as f:
    pickle.dump(xgb_model_deploy, f)


# In[44]:


X_test.iloc[0]


# In[45]:


y_test_pred[0]


# In[46]:


st_dev


# In[ ]:




