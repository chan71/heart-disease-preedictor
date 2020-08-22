import pandas as pd

#Loading data from Github repository
filename = 'https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter16/Dataset/processed.cleveland.data'

# Loading the data using pandas
heartData = pd.read_csv(filename,sep=",",header = None,na_values = "?")
heartData.head()

heartData.columns = ['age','sex', 'cp', 'trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','label']
heartData.head()

# Changing the Classes to 1 & 0
heartData.loc[heartData['label'] > 0 , 'label'] = 1
heartData.head()

# Dropping all the rows with na values
newheart = heartData.dropna(axis = 0)
newheart.shape

# Seperating X and y variables\
y = newheart.pop('label')
y.shape

X = newheart
X.head()

import mlflow

mlflow.set_experiment("heart_disease")

with mlflow.start_run():

    from sklearn.model_selection import train_test_split

    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    # **Creating processing Engine**
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

    from sklearn.compose import ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)])

    # **Spot checking different models**

    # Importing necessary libraries
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

    # Creating a list of the classifiers
    classifiers = [
        KNeighborsClassifier(),     
        RandomForestClassifier(random_state=123),
        AdaBoostClassifier(random_state=123),
        LogisticRegression(random_state=123)
        ]

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

    # Creating a pipeline with Logistic Regression
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                        ('dimred', PCA()),
                        ('classifier',LogisticRegression(random_state=123))])

    param_grid = {'dimred__n_components':[10,11,12,13],'classifier__penalty' : ['l1', 'l2'],'classifier__C' : [1,3, 5],'classifier__solver' : ['liblinear']}

    from sklearn.model_selection import GridSearchCV

    # Fitting the grid search
    estimator = GridSearchCV(pipe, cv=10, param_grid=param_grid)

    # Fitting the estimator on the training set
    estimator.fit(X_train,y_train)

    # Printing the best score and best parameters
    print("Best: %f using %s" % (estimator.best_score_, 
        estimator.best_params_))
    mlflow.log_metric("best score", estimator.best_score_)

    # Predicting with the best estimator
    pred = estimator.predict(X_test)

    # Printing the classification report
    from sklearn.metrics import classification_report
    print(classification_report(pred, y_test))

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

    import mlflow.sklearn
    mlflow.sklearn.log_model(estimator, "model")

    mlflow.end_run()

