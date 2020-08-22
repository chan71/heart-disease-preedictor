#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[167]:


#Loading data from Github repository

filename = 'https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter16/Dataset/processed.cleveland.data'


# In[132]:


# Loading the data using pandas

heartData = pd.read_csv(filename,sep=",",header = None,na_values = "?")
heartData.head()


# In[133]:


heartData.columns = ['age','sex', 'cp', 'trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','label']
heartData.head()


# In[134]:


# Changing the Classes to 1 & 0
heartData.loc[heartData['label'] > 0 , 'label'] = 1

heartData.head()


# In[135]:


# Dropping all the rows with na values
newheart = heartData.dropna(axis = 0)
newheart.shape


# In[136]:


# Seperating X and y variables

y = newheart.pop('label')
y.shape


# In[137]:


X = newheart
X.head()


# In[138]:


import mlflow

mlflow.set_experiment("heart_disease")
with mlflow.start_run():
    print("dummy")


    # In[139]:


    from sklearn.model_selection import train_test_split

    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


    # **Creating processing Engine**

    # In[140]:


    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler


    # In[141]:


    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])


    # In[142]:


    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns


    # In[143]:


    from sklearn.compose import ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)])


    # **Spot checking different models**

    # In[144]:


    # Importing necessary libraries
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


    # In[145]:


    # Creating a list of the classifiers
    classifiers = [
        KNeighborsClassifier(),     
        RandomForestClassifier(random_state=123),
        AdaBoostClassifier(random_state=123),
        LogisticRegression(random_state=123)
        ]


    # In[146]:


    # Looping through classifiers to get the best model
    for classifier in classifiers:
        estimator = Pipeline(steps=[('preprocessor', preprocessor),
                        ('dimred', PCA(10)),
                            ('classifier',classifier)])
        estimator.fit(X_train, y_train)   
        print(classifier)
        print("model score: %.2f" % estimator.score(X_test, y_test))
        mlflow.log_metric(classifier.__class__.__name__, estimator.score(X_test, y_test))


    # **Grid Search**

    # In[147]:


    # Creating a pipeline with Logistic Regression
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                        ('dimred', PCA()),
                        ('classifier',LogisticRegression(random_state=123))])


    # In[148]:


    param_grid = {'dimred__n_components':[10,11,12,13],'classifier__penalty' : ['l1', 'l2'],'classifier__C' : [1,3, 5],'classifier__solver' : ['liblinear']}


    # In[149]:


    from sklearn.model_selection import GridSearchCV
    # Fitting the grid search
    estimator = GridSearchCV(pipe, cv=10, param_grid=param_grid)


    # In[150]:


    # Fitting the estimator on the training set
    estimator.fit(X_train,y_train)


    # In[151]:


    # Printing the best score and best parameters
    print("Best: %f using %s" % (estimator.best_score_, 
        estimator.best_params_))
    mlflow.log_metric("best score", estimator.best_score_)


    # In[152]:


    # Predicting with the best estimator
    pred = estimator.predict(X_test)


    # In[153]:


    # Printing the classification report
    from sklearn.metrics import classification_report

    print(classification_report(pred, y_test))


    # In[154]:


    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Evaluate metrics
    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    (rmse, mae, r2) = eval_metrics(y_test, pred)

    # Print out model metrics
    print("RMSE: %s" % rmse)
    print("MAE: %s" % mae)
    print("R2: %s" % r2)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)


    # In[155]:


    import mlflow.sklearn
    mlflow.sklearn.log_model(estimator, "model")


    # In[156]:


    mlflow.end_run()


    # In[ ]:




