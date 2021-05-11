import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import numpy as np
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress scientific notation in pandas
pd.set_option('display.float_format', '{:.2f}'.format)

#===================================================================================================================================================================#

''' USER INPUTS - MUST BE SPECIFIED '''


# ASSIGN THE TRAINING DATASET TO A VARIABLE
data = dataset[:round(len(dataset)*9/10)]

# SPECIFY THE PREDICTION OR VALIDATION DATA HERE
actual = dataset[round(len(dataset)*9/10):]

# SPECIFY THE DEPENDENT VARIABLE BY ITS COLUMN NAME
dependent = 'outcome_type'

# SPECIFY WHETHER YOU WANT TO 'predict' OR 'classify' THE DEPENDENT VARIABLE
analysis_type = 'classify'.lower()

# IF YOU ARE CLASSIFYING, SET resample TO 'yes' IF YOU HAVE SKEWED INPUTS AND WANT TO BALANCE THE TRAINING DATA TO 50/50
resample = 'yes'.lower()

# SPECIFY IF YOU WANT TO RUN PCA 'yes' OR KEEP YOUR INPUT VALUES INTACT FOR INTERPRETABILITY 'no'
run_pca = 'yes'

#===================================================================================================================================================================#

''' The rest of the script is automated - just let it run '''

''' Convert text fields to numeric '''
def preprocess(data):
    
    # For any binary variables (e.g. True/False, Yes/No), convert them to integers
    for i in list(data):
        vals = data[i].dropna().unique()
        if len(vals) == 2:
            try:
                data[i] = data[i].str.replace(vals[0],'0')
                data[i] = data[i].str.replace(vals[1],'1')
                data[i] = data[i].astype(float)
                data[i] = data[i].fillna(data[i].mean())
                print('{} converted to binary'.format(i))
            except:
                pass

    # Gather non-numeric columns so that we can convert them to a format the makes sense to the model
    text = data.select_dtypes('object').columns
    
    # Create a list of column names that are successfully converted (we will need this to convert the text columns in the prediction data later)
    converted_text_columns = []
    
    # Loop through each of the text columns
    for i in text:
        
        try:
        # If the column does not contain dashes/colons, the number of unique values is less than 50, and it doesn't contain '20' (since all dates have 20 in them) then convert it to numeric values
            if data[i].str.contains('20').any() == False and data[i].str.contains(':').any() == False and data[i].nunique() < 50 and data[i].nunique() > 1: 
                
                # Convert the column to a dummy dataframe (each value becomes a column in the new dataframe with a binary value for each record)
                dummies = pd.get_dummies(data[i])
                # Merge the dummy dataset with the main dataset
                data = data.merge(dummies, left_index = True, right_index = True)
                
                for i in list(dummies):
                    data[i] = data[i].astype(float)
                
                # Append the column names to the converted text columns list
                converted_text_columns.append(i)
                print('converted {} to dummy columns'.format(str(i)))
        except:
            pass
        
        else:
            print(i, 'is not valid')
        
    # Subset the data to just the numeric columns
    for i in list(data):
        try:
            data[i] = data[i].astype(float)
        except:
            pass
    data = data[list(data.select_dtypes('int').columns) + list(data.select_dtypes('float').columns) + [dependent]]
    data = data.T.drop_duplicates().T
    return(data)

# Run the preprocessing function over the training data and the actual data that we are predicting
data = preprocess(data)
actual = preprocess(actual)

# Drop fields from the data where more than 75% of the records in that field are null, as that will adversely affect the model
data = data.replace([np.inf, -np.inf], np.nan)
for i in list(data):
    try:
        if sum(data[i].isna())/len(data) > .75:
            data = data.drop(i, axis = 1)
    except:
        pass

#===================================================================================================================================================================#

''' Run Regression Over Each Variable '''

# Create a dataframe to populate regression results into
regressions = pd.DataFrame()
row = 0

# Run regressions over the factors
for i in list(data):
    try:
        print(i)
        
        # Create a subset of a single x & y variable. Remove all infinite/nan values
        model = data[[i, dependent]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop = True)
        model = model.T.drop_duplicates().T
                
        # Resample the data for binary variables to ensure greater accuracy        
        if analysis_type == 'classify' and resample == 'yes' and model[dependent].nunique() == 2: 
            try:
                 # Split the binary data into separate groups
                group_zero = shuffle(model[model[dependent]== 0])
                group_one = shuffle(model[model[dependent] == 1])
                
                # Reduce the larger of the two groups to the same number of rows as the smaller one
                if len(group_zero) > len(group_one):
                    group_zero = group_zero[:len(group_one)]
                else:
                    group_one = group_one[:len(group_zero)]
                
                # Recombine the groups into the model dataframe
                model = pd.concat([group_zero, group_one]).reset_index(drop = True)
                
            except:
                pass
                
        # Convert the variables to arrays
        x = model[model.columns[:-1]].values
        y = model[model.columns[-1]].values
        
        # Run cross validations for the RMSE and R2
        scores = cross_val_score(LinearRegression(), x, y, cv = 10, n_jobs = 1, scoring = 'neg_mean_squared_error')
        r2_scores = cross_val_score(LinearRegression(), x, y, cv = 10, n_jobs = 1, scoring = 'r2')
        if analysis_type == 'classify':
            try:
                accuracy_scores = cross_val_score(LogisticRegression(max_iter = 1000), x, y, cv = 10, n_jobs = 1,scoring = 'accuracy')
            except:
                pass
        
        # Calculate overall cross validation scores
        rmse = sqrt(abs(scores.mean()))
        r2 = r2_scores.mean()
        accuracy = 0
        try:
            accuracy = accuracy_scores.mean()
        except:
            pass
                
        # Add the values from the cross validation to the regressions dataframe
        regressions.loc[row, 'FACTOR'] = i
        regressions.loc[row, 'CORRELATION_COEFFICIENT'] = r2
        regressions.loc[row, 'RMS_ERROR'] = rmse
        regressions.loc[row, 'ACCURACY'] = accuracy

        
        # Add 1 to the row variable for each iteration
        row = row + 1
        
    except:
        pass

# If it is a binary classification problem, sort by accuracy, otherwise sort by error rate
if regressions['ACCURACY'].sum() > 0 and analysis_type == 'classify': 
    try:
        regressions = regressions.sort_values('ACCURACY', ascending = False)
    except:
        regressions = regressions.sort_values('RMS_ERROR', ascending = True) 
else:
    regressions = regressions.sort_values('RMS_ERROR', ascending = True)

# Remove the dependent variable from the regressions data and reindex it
regressions = regressions[~regressions['FACTOR'].str.contains(dependent)].reset_index(drop = True)

# Drop anything with very high R squared and/or accuracy becuase it is probably overfit the model - THIS IS OPTIONAL
regressions = regressions[regressions['CORRELATION_COEFFICIENT'] < .95].reset_index(drop = True)
if analysis_type == 'clasify':
    regressions = regressions[regressions['ACCURACY'] < .95].reset_index(drop = True)

# Print the final result
print(regressions)

#===================================================================================================================================================================#

''' Find the optimal input mix '''

# Create a list variable for the model inputs
model_inputs = [dependent]

if run_pca != 'yes':
    # Loop through each variable in the regressions, starting with the most highly correlated
    for i in range(len(regressions)):
        
        try:
            # For each iteration, add another variable to the model
            variables = model_inputs
            variables.append(regressions['FACTOR'][i])
            
            # Create a subset of the dataframe with only the relevant variables
            model = data[variables].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop = True)
            model = model.T.drop_duplicates().T
            
            if analysis_type == 'classify' and resample == 'yes' and model[dependent].nunique() == 2: 
                try:
                    
                    # Split the binary data into separate groups
                    group_zero = shuffle(model[model[dependent]== 0])
                    group_one = shuffle(model[model[dependent] == 1])
                    
                    # Reduce the larger of the two groups to the same number of rows as the smaller one
                    if len(group_zero) > len(group_one):
                        group_zero = group_zero[:len(group_one)]
                    else:
                        group_one = group_one[:len(group_zero)]
                    
                    # Recombine the groups into the model dataframe
                    model = pd.concat([group_zero, group_one]).reset_index(drop = True)
                        
                except:
                    pass        
            
            # Convert the variables to arrays
            x = model.drop(dependent, axis = 1).values
            y = model[dependent].values
            
            # Run the cross validations
            scores = cross_val_score(LinearRegression(), x, y, cv = 10, n_jobs = 1, scoring = 'neg_mean_squared_error')
            r2_scores = cross_val_score(LinearRegression(), x, y, cv = 10, n_jobs = 1, scoring = 'r2')
            if analysis_type == 'classify':
                try:
                    accuracy_scores = cross_val_score(LogisticRegression(max_iter = 1000), x, y, cv = 10, n_jobs = 1, scoring = 'accuracy')
                    accuracy = accuracy_scores.mean()
                except:
                    accuracy = 0
                    pass
            
            # Calculate overall cross validation results
            rmse = sqrt(abs(scores.mean()))
            r2 = r2_scores.mean()
            
            # Check to see if this is the first run through
            if i == 0:
                # calculate the mean r2 and root mean squared error for the model with the given variables
                mean_r2 = r2
                mean_rmse = rmse
                mean_accuracy = accuracy
                print('\n', 'R2:', mean_r2, '\n', 'RMSE:', mean_rmse, '\n', 'Accuracy:', mean_accuracy, '\n', 'Dependend Variables:', variables[:-1], '\n')
        
            # If it is not the first run, see if the new RMSE is less than the current one    
            else:
                # If the rmse improves by at least 1% between runs, print the output and update the list of dependent variables to use
                if (accuracy > mean_accuracy) and analysis_type == 'classify' and accuracy > 0:
                    mean_r2 = r2
                    mean_accuracy = accuracy
                    print('\n', 'R2:', mean_r2, '\n', 'Accuracy:', mean_accuracy, '\n', 'Independend Variables:', model_inputs, '\n')
                
                elif rmse < (mean_rmse)*.99 and analysis_type == 'predict':
                    mean_r2 = r2
                    mean_rmse = rmse    
                    print('\n', 'R2:', mean_r2, '\n', 'RMSE:', mean_rmse, '\n', 'Independend Variables:', model_inputs, '\n')
                    
                else:
                    print('Dropping', regressions['FACTOR'][i], 'from model')
                    try:
                        variables.remove(regressions['FACTOR'][i])
                    except:
                        pass
            
        except:
            pass

else:
    model_inputs = list(regressions['FACTOR']) + [dependent]

#===================================================================================================================================================================#


''' Preprocess the prediction data '''

# Drop the model inputs that are not available in the prediction dataset
model_inputs = [x for x in model_inputs if x in list(actual)] 
try:
    model_inputs.remove(dependent) 
except:
    pass

# Create a prediction dataset 
prediction = actual[model_inputs]

# Fill null values with the column means
prediction.fillna(prediction.mean(), inplace = True)

# Replace infinite values with nan, then drop nan
prediction = prediction[prediction.columns[0:]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop = True).astype(float)

# Drop duplicated columns
prediction = prediction.T.drop_duplicates().T

# Reset the model inputs to ensure that it only consists of available fields in the prediction data
model_inputs = list(prediction)

# Create a results dataframe to populate the predictions into
results = actual[model_inputs]


''' Preprocess the training data '''

# Create a model variable using the training dataset to train the predictive models and replace the null values
model_inputs.append(dependent)
model = data[model_inputs]
model = model.fillna(model.mean()).reset_index(drop = True).astype(float)
model = model.T.drop_duplicates().T
model = model.dropna().reset_index(drop = True)

if analysis_type == 'classify' and resample == 'yes' and model[dependent].nunique() == 2:  
    try:
        
        # Split the binary data into separate groups
        group_zero = shuffle(model[model[dependent]== 0])
        group_one = shuffle(model[model[dependent] == 1])
        
        # Reduce the larger of the two groups to the same number of rows as the smaller one
        if len(group_zero) > len(group_one):
            group_zero = group_zero[:len(group_one)]
        else:
            group_one = group_one[:len(group_zero)]
        
        # Recombine the groups into the model dataframe
        model = pd.concat([group_zero, group_one]).reset_index(drop = True)
        
    except:
        pass    
            
# Convert the training variables to arrays
x = model.drop(dependent, axis = 1).values
y = model[dependent].values

# Create a dataframe to populate model scores into
model_results = pd.DataFrame()

# If run_pca was set to yes, then create a new X using PCA
if run_pca == 'yes':
    
    # Create a list of factors in the regression results (since this will contain all valid factors)
    factors = list(regressions['FACTOR'])
    
    # Remove anything that is not in the model dataset
    factors = [x for x in factors if x in list(model)]
    
    # Create and fit the PCA model to the training, testing, and prediction data
    pca = PCA(n_components = 10)
    x = pca.fit_transform(model[factors].fillna(model.mean()).values)
    prediction = pca.fit_transform(prediction[factors].dropna())

#===================================================================================================================================================================#


''' Train, Test, and Run the REGRESSION Model '''


# Build the model
if analysis_type == 'predict':
        
    # Create the model
    reg = LinearRegression()
    
    # Compute the RMSE and R2    
    scores = cross_val_score(reg, x, y, cv = 10, n_jobs = 3, scoring = 'neg_mean_squared_error')
    r2_scores = cross_val_score(reg, x, y, cv = 10,n_jobs = 3,  scoring = 'r2')
    rmse = sqrt(abs(scores.mean()))
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Linear Regression R2: {}'.format(r2))
    print('Linear Regression RMSE: {}'.format(rmse), '\n')
    
    model_results.loc[0, 'Model'] = 'Linear Regression'
    model_results.loc[0, 'RMSE'] = rmse
    model_results.loc[0, 'Column'] = 'Regression_Prediction'
    
elif analysis_type == 'classify':
    try:
        # Create and fit the model
        reg = LogisticRegression(max_iter = 1000)
        
        # Compute the RMSE and R2
        scores = cross_val_score(reg, x, y, cv = 10, n_jobs = 3, scoring = 'accuracy')
        r2_scores = cross_val_score(reg, x, y, cv = 10, n_jobs = 3, scoring = 'r2')
        accuracy = scores.mean()
        r2 = r2_scores.mean()
        
        # Assess accuracy
        print('\n', 'Logistic Regression R2: {}'.format(r2))
        print('Logistic Accuracy: {}'.format(accuracy), '\n')
        
        model_results.loc[0, 'Model'] = 'Logistic Regression'
        model_results.loc[0, 'Accuracy'] = accuracy
        model_results.loc[0, 'Column'] = 'Regression_Prediction'
        
    except:
        model_results.loc[0, 'Model'] = 'Logistic Regression'
        model_results.loc[0, 'Accuracy'] = np.nan
        model_results.loc[0, 'RMSE'] = np.nan
        model_results.loc[0, 'Column'] = 'Regression_Prediction'
        
else:
    print("YOU NEED TO SPECIFY THE ANALYSIS TYPE AS EITHER 'predict' OR 'classify' BEFORE RUNNING THE MODEL")

# Fit the model
reg.fit(x, y)

# Add a column with the predicted values to the dataframe
results['Regression_Prediction'] = reg.predict(prediction)


#===================================================================================================================================================================#


''' Train, Test, and Run the Support Vector Machine Model '''

# Build the model
if analysis_type == 'predict':
    
    # Create the model
    svm = SVR()
    
    # Compute the RMSE and R2
    scores = cross_val_score(svm, x, y, cv = 10, n_jobs = 3, scoring = 'neg_mean_squared_error')
    r2_scores = cross_val_score(svm, x, y, cv = 10, n_jobs = 3, scoring = 'r2')
    rmse = sqrt(abs(scores.mean()))
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Support Vector R2: {}'.format(r2))
    print('Support Vector RMSE: {}'.format(rmse), '\n')
    
    model_results.loc[1, 'Model'] = 'Support Vector Machine'
    model_results.loc[1, 'RMSE'] = rmse
    model_results.loc[1, 'Column'] = 'Support_Vector_Prediction'
    
elif analysis_type == 'classify':
    
    # Create and fit the model
    svm = SVC()
    
    # Compute the RMSE and R2
    scores = cross_val_score(svm, x, y, cv = 10, n_jobs = 3, scoring = 'accuracy')
    r2_scores = cross_val_score(svm, x, y, cv = 10, n_jobs = 3, scoring = 'r2')
    accuracy = scores.mean()
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Support Vector R2: {}'.format(r2))
    print('Support Vector Accuracy: {}'.format(accuracy), '\n')
    
    model_results.loc[1, 'Model'] = 'Support Vector Machine'
    model_results.loc[1, 'Accuracy'] = accuracy
    model_results.loc[1, 'Column'] = 'Support_Vector_Prediction'

else:
    print("YOU NEED TO SPECIFY THE ANALYSIS TYPE AS EITHER 'predict' OR 'classify' BEFORE RUNNING THE MODEL")

# Fit the model
svm.fit(x, y)

# Add a column with the predicted values to the dataframe
results['Support_Vector_Prediction'] = svm.predict(prediction)


#===================================================================================================================================================================#


''' Train, Test, and Run the DECISION TREE Model '''

# Build the model
if analysis_type == 'predict':

    # Create and fit the variable
    tree = DecisionTreeRegressor()
    
    # Compute the RMSE and R2
    scores = cross_val_score(tree, x, y, cv = 10, n_jobs = 3, scoring = 'neg_mean_squared_error')
    r2_scores = cross_val_score(tree, x, y, cv = 10, n_jobs = 3, scoring = 'r2')
    rmse = sqrt(abs(scores.mean()))
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Decision Tree R2: {}'.format(r2))
    print('Decision Tree RMSE: {}'.format(rmse), '\n')   

    model_results.loc[2, 'Model'] = 'Decision Tree'
    model_results.loc[2, 'RMSE'] = rmse    
    model_results.loc[2, 'Column'] = 'Tree_Prediction'
    
# If the prediction variable is continuous, use the Regressor
elif analysis_type == 'classify':

    # Create and fit the model
    tree = DecisionTreeClassifier()
    
    # Compute the RMSE and R2
    scores = cross_val_score(tree, x, y, cv = 10, n_jobs = 3, scoring = 'accuracy')
    r2_scores = cross_val_score(tree, x, y, cv = 10, n_jobs = 3, scoring = 'r2')
    accuracy = scores.mean()
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Decision Tree R2: {}'.format(r2))
    print('Decision Tree Accuracy: {}'.format(accuracy), '\n')
    
    model_results.loc[2, 'Model'] = 'Decision Tree'
    model_results.loc[2, 'Accuracy'] = accuracy  
    model_results.loc[2, 'Column'] = 'Tree_Prediction'
    
else:
    print("YOU NEED TO SPECIFY THE ANALYSIS TYPE AS EITHER 'predict' OR 'classify' BEFORE RUNNING THE MODEL")
    
# Fit the model
tree.fit(x, y)

# Add a column with the predicted values to the dataframe
results['Tree_Prediction'] = tree.predict(prediction)


#===================================================================================================================================================================#


''' Train, Test, and Run the RANDOM FOREST Model '''

# Build the model
if analysis_type == 'predict':

    # Create and fit the variable
    forest = RandomForestRegressor(n_estimators = 100)
    
    # Compute the RMSE and R2
    scores = cross_val_score(forest, x, y, cv = 10, n_jobs = 3, scoring = 'neg_mean_squared_error')
    r2_scores = cross_val_score(forest, x, y, cv = 10, n_jobs = 3, scoring = 'r2')
    rmse = sqrt(abs(scores.mean()))
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Random Forest R2: {}'.format(r2))
    print('Random Forest RMSE: {}'.format(rmse), '\n')
    
    model_results.loc[3, 'Model'] = 'Random Forest'
    model_results.loc[3, 'RMSE'] = rmse
    model_results.loc[3, 'Column'] = 'Forest_Prediction'

# If the prediction variable is continuous, use the Regressor
elif analysis_type == 'classify':
    
    # Create and fit the model
    forest = RandomForestClassifier(n_estimators = 100)
    
    # Compute the RMSE and R2
    scores = cross_val_score(forest, x, y, cv = 10, n_jobs = 3, scoring = 'accuracy')
    r2_scores = cross_val_score(forest, x, y, cv = 10, n_jobs = 3, scoring = 'r2')
    accuracy = scores.mean()
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Random Forest R2: {}'.format(r2))
    print('Random Forest Accuracy: {}'.format(accuracy), '\n')
    
    model_results.loc[3, 'Model'] = 'Random Forest'
    model_results.loc[3, 'Accuracy'] = accuracy 
    model_results.loc[3, 'Column'] = 'Forest_Prediction'
   
else:
    print("YOU NEED TO SPECIFY THE ANALYSIS TYPE AS EITHER 'predict' OR 'classify' BEFORE RUNNING THE MODEL")
    
# Fit the model
forest.fit(x, y)

# Add a column with the predicted values to the dataframe
results['Forest_Prediction'] = forest.predict(prediction)


#===================================================================================================================================================================#


''' Prep the data for predictive models THAT REQUIRE DATA NORMALIZATION'''

if run_pca != 'yes':
    
    # Assign a scaler variable
    scaler = preprocessing.MinMaxScaler()
    
    # Convert the independent/dependent variables to separate arrays
    x = scaler.fit_transform(model.drop(dependent, axis = 1).values)
    y = model[dependent].values
       
    # Split the data into train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .8)
    
    # Scale the prediction dataset
    prediction_scaled = scaler.fit_transform(prediction)
    
else:
    prediction_scaled = prediction


#===================================================================================================================================================================#


''' Train, Test, and Run the K-NEAREST NEIGHBORS Model '''

# Build the model
if analysis_type == 'predict':
    
    # Create and fit the model
    KNN = KNeighborsRegressor(n_neighbors = 3)

    # Compute the RMSE and R2
    scores = cross_val_score(KNN, x, y, cv = 10, n_jobs = 3, scoring = 'neg_mean_squared_error')
    r2_scores = cross_val_score(KNN, x, y, cv = 10, n_jobs = 3, scoring = 'r2')
    rmse = sqrt(abs(scores.mean()))
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'KNN R2: {}'.format(r2))
    print('KNN RMSE: {}'.format(rmse), '\n')
    
    model_results.loc[4, 'Model'] = 'KNN'
    model_results.loc[4, 'RMSE'] = rmse
    model_results.loc[4, 'Column'] = 'KNN_Prediction'
    
elif analysis_type == 'classify':

    # Create and fit the model
    KNN = KNeighborsClassifier(n_neighbors = 3)
    
    # Compute the RMSE and R2
    scores = cross_val_score(KNN, x, y, cv = 10, n_jobs = 3, scoring = 'accuracy')
    r2_scores = cross_val_score(KNN, x, y, cv = 10, n_jobs = 3, scoring = 'r2')
    accuracy = scores.mean()
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'KNN R2: {}'.format(r2))
    print('KNN Accuracy: {}'.format(accuracy), '\n')
    
    model_results.loc[4, 'Model'] = 'KNN'
    model_results.loc[4, 'Accuracy'] = accuracy    
    model_results.loc[4, 'Column'] = 'KNN_Prediction'

else:
    print("YOU NEED TO SPECIFY THE ANALYSIS TYPE AS EITHER 'predict' OR 'classify' BEFORE RUNNING THE MODEL")
    
# Fit the model
KNN.fit(x, y)
    
# Add a column with the predicted values to the dataframe
results['KNN_Prediction'] = KNN.predict(prediction_scaled)


#===================================================================================================================================================================#


''' Train, Test, and Run the NEURAL NETWORK Model '''

# Build the model
if analysis_type == 'predict':
    
    # Create and fit the model
    net = MLPRegressor(hidden_layer_sizes = (1000, 100, 100, 100, 10), max_iter = 100000)
    
    # Compute the RMSE and R2
    scores = cross_val_score(net, x, y, cv = 10, n_jobs = 3, scoring = 'neg_mean_squared_error')
    r2_scores = cross_val_score(net, x, y, cv = 10, n_jobs = 3, scoring = 'r2')
    rmse = sqrt(abs(scores.mean()))
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Neural Network R2: {}'.format(r2))
    print('Neural Network RMSE: {}'.format(rmse), '\n')
    
    model_results.loc[5, 'Model'] = 'Neural Network'
    model_results.loc[5, 'RMSE'] = rmse
    model_results.loc[5, 'Column'] = 'Neural_Net_Prediction'
    
elif analysis_type == 'classify':
    
    # Create and fit the model
    net = MLPClassifier(hidden_layer_sizes = (1000, 100, 100, 100, 10), max_iter = 100000)    
    
    # Compute the RMSE and R2
    scores = cross_val_score(net, x, y, cv = 10, n_jobs = 3, scoring = 'accuracy')
    r2_scores = cross_val_score(net, x, y, cv = 10, n_jobs = 3, scoring = 'r2')
    accuracy = scores.mean()
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Neural Network R2: {}'.format(r2))
    print('Neural Network Accuracy: {}'.format(accuracy), '\n')
    
    model_results.loc[5, 'Model'] = 'Neural Network'
    model_results.loc[5, 'Accuracy'] = accuracy
    model_results.loc[5, 'Column'] = 'Neural_Net_Prediction'

else:
    print("YOU NEED TO SPECIFY THE ANALYSIS TYPE AS EITHER 'predict' OR 'classify' BEFORE RUNNING THE MODEL")
    
# Fit the model
net.fit(x, y)
    
# Add a column with the predicted values to the dataframe
results['Neural_Net_Prediction'] = net.predict(prediction_scaled)


#===================================================================================================================================================================#

#===================================================================================================================================================================#

# Generate the modeling report
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------')

if analysis_type == 'predict':
    
    # Sort the values in the dataframe so that the best model is first
    model_results.sort_values('RMSE', ascending = True, inplace = True)
    model_results.reset_index(inplace = True, drop = True)
    
    # Print the results of the analysis
    print('\n Your optimal model is {} with an average root mean squared error of {}'.format(model_results.loc[0,'Model'], model_results.loc[0,'RMSE']))
    
elif analysis_type == 'classify':
    
    # Sort the values in the dataframe so that the best model is first
    model_results.sort_values('Accuracy', ascending = False, inplace = True)
    model_results.reset_index(inplace = True, drop = True)
    
    # Print the results of the analysis
    print('\n Your optimal model is {} with an overall accuracy of {}'.format(model_results.loc[0,'Model'], model_results.loc[0,'Accuracy']))
    
# Assign a variable for the name of the column with the prediction that we want to use, then remove the 'Column' field from the model_results dataframe
column = model_results.loc[0,'Column']
model_results.drop('Column', axis = 1)
    
# Print the independent variables
if run_pca != 'yes':
    print('\n\n The following factors should be included in your model as independent variables: {}'.format(model_inputs))

# Check the results
print('\n\n The results for each model are as follows: \n', model_results)

# Show the predictions vs the actual
review = results[['Regression_Prediction', 'Support_Vector_Prediction', 'Tree_Prediction', 'Forest_Prediction', 'KNN_Prediction', 'Neural_Net_Prediction' ]]

# If you are validating against data that already has the actual value, this will append it to the review dataframe
try:
    review['Actual'] = actual[dependent]
except:
    pass

# Reindex the results
review.reset_index(inplace = True, drop = True)

# Check the predictive accuracy of the model on the validation data
try:
    if analysis_type == 'classify':
        print('\n The accuracy of your model against the validation set is {}'.format(sum(review[model_results.loc[0,'Column']] == review['Actual'])/len(review)))
    else:
        print('\n The mean error against the validation set is {}. The standard deviation of the training set is {}'.format(round(abs(review['Actual'] - review[column]).mean(),2), round(data[dependent].std(),2)))
except:
    pass

try:
    # Plot the distribution of predictions
    sns.set(rc = {'figure.figsize' : (15,6)})
    ax = sns.kdeplot(review[column], label = column)
    try:
        ax = sns.kdeplot(review['Actual'], label = 'Validation Data Actual')
    except:
        pass
    ax = sns.kdeplot(data[dependent],  label = 'Training Data Actual')
    ax.legend(bbox_to_anchor = (1, 1), loc = 2)
    ax.set(xlabel = dependent)
    ax.set(yscale="symlog")
    plt.show()
except: 
    pass

# Try printing the confusion matrix and classification report, if the model is a classification model
if analysis_type == 'classify':
    
    # Create a model variable using the training dataset to train the predictive models and replace the null values
    model_inputs.append(dependent)
    model = data[model_inputs]
    model = model.fillna(model.mean()).reset_index(drop = True).astype(float)
    model = model.T.drop_duplicates().T
    model = model.dropna().reset_index(drop = True)
    
    # Resample if necessary
    if resample == 'yes' and model[dependent].nunique() == 2:  
        try:
            
            # Split the binary data into separate groups
            group_zero = shuffle(model[model[dependent]== 0])
            group_one = shuffle(model[model[dependent] == 1])
            
            # Reduce the larger of the two groups to the same number of rows as the smaller one
            if len(group_zero) > len(group_one):
                group_zero = group_zero[:len(group_one)]
            else:
                group_one = group_one[:len(group_zero)]
            
            # Recombine the groups into the model dataframe
            model = pd.concat([group_zero, group_one]).reset_index(drop = True)
            
        except:
            pass    
         
    # Scale the inputs if it is a KNN or Neural Network model
    if model_results.loc[0,'Model'] == 'KNN' or model_results.loc[0,'Model'] == 'Neural Network':
        
        # Convert the independent/dependent variables to separate, scaled arrays
        x = scaler.fit_transform(model.drop(dependent, axis = 1).values)
        y = model[dependent].values
    
    else:
        
        # Convert the training variables to arrays
        x = model.drop(dependent, axis = 1).values
        y = model[dependent].values
    
    # Split the new dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .8)
        
    # # Format the model inputs as numpy arrays
    x_train = x_train
    y_train = y_train.reshape(-1,1)
    
    # Determine which model was used, then create a variable for that model that can be trained on a subset of the input data
    if model_results.loc[0,'Model'] == 'Logistic Regression':
        selected_model = LogisticRegression(max_iter = 1000)
    elif model_results.loc[0,'Model'] == 'Support Vector Machine':
        selected_model = SVC()
    elif model_results.loc[0,'Model'] == 'Decision Tree':
        selected_model = DecisionTreeClassifier()
    elif model_results.loc[0,'Model'] == 'Random Forest':
        selected_model = RandomForestClassifier(n_estimators = 100)
    elif model_results.loc[0,'Model'] == 'KNN':
        selected_model = KNeighborsClassifier(n_neighbors = 3)
    elif model_results.loc[0,'Model'] == 'Neural Network':
        selected_model = MLPClassifier(hidden_layer_sizes = (1000, 100, 100, 100, 100), max_iter = 100000)
    
    # Fit the model on the training data
    selected_model.fit(x_train,y_train)
    y_pred = selected_model.predict(x_test)
    
    # Check the results
    print('\n Your confusion matrix and classification reports are as follows: \n')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    
try:
    # Plot features by importance for random forest
    if model_results.loc[0,'Model'] == 'Random Forest':
        print('\n\n The relative importance of your model inputs are as follows: \n')
        feat_importances = pd.Series(forest.feature_importances_, index=model.drop(dependent, axis = 1).columns)
        feat_importances.nlargest(20).plot(kind='barh')
        
    # Plot features by importance for the decision tree
    elif model_results.loc[0,'Model'] == 'Decision Tree':
        print('\n\n The relative importance of your model inputs are as follows: \n')
        feat_importances = pd.Series(tree.feature_importances_, index=model.drop(dependent, axis = 1).columns)
        feat_importances.nlargest(20).plot(kind='barh')
        
        # Start a new plot and plot the tree
        print('\n Your decision tree: \n')
        plt.clf()
        plot_tree(tree, max_depth = 2, feature_names = model.drop(dependent, axis = 1).columns, filled = True)
        
    # Plot the coefficients for the regression models
    elif model_results.loc[0,'Model'] == 'Linear Regression' or model_results.loc[0,'Model'] == 'Logistic Regression':
        print('\n\n Your regression coefficients are as follows: \n')
        variables = pd.DataFrame(model.drop(dependent, axis = 1).columns)
        coefficients = pd.DataFrame(np.transpose(reg.coef_)).set_index(variables[0])
        coefficients.plot(kind = 'barh')
        
    # Plot the coefficients for the SVM models
    elif model_results.loc[0,'Model'] == 'Support Vector Machine':
        print('\n\n Your support vector coefficients are as follows: \n')
        variables = pd.DataFrame(model.drop(dependent, axis = 1).columns)
        coefficients = pd.DataFrame(np.transpose(svm.coef_)).set_index(variables[0])
        coefficients.plot(kind = 'barh')
        
except:
    pass


print('\n--------------------------------------------------------------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------')
