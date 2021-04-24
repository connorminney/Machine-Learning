import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, classification_report
from sklearn import preprocessing
import numpy as np
from math import sqrt

# Suppress scientific notation in pandas
pd.set_option('display.float_format', '{:.2f}'.format)

#===================================================================================================================================================================#

''' USER INPUTS - MUST BE SPECIFIED '''

# Supply the dataset
data = YOUR_DATA
# Specify the dependent variable
dependent = 'DEPENDENT VARIABLE COLUMN NAME'
# Specificy whether you want to 'predict' or 'classify' the data
analysis_type = 'classify'

# Specify the prediction/classification dataset
actual = THE_DATA_YOU_WANT_TO_PREDICT

#===================================================================================================================================================================#


''' Convert text fields to numeric '''

def preprocess(data):
    
    # Start by trying to convert everything to numeric - this will convert True/False values to binary values
    for i in list(data):
        try:
            data[i] = data[i].astype(int)
            print('{} converted to integer'.format(i))
        except:
            pass
        
    # For any remaining binary variables (e.g. True/False, Yes/No), convert them to integers
    for i in list(data):
        vals = data[i].unique()
        if len(vals) == 2:
            try:
                data[i] = data[i].str.replace(vals[0],'0')
                data[i] = data[i].str.replace(vals[1],'1').astype(int)
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
                    
                    # Append the column names to the converted text columns list
                    converted_text_columns.append(i)
                    print('converted {} to dummy columns'.format(i))
        except:
            pass
        
        else:
            print(i, 'is not valid')
        
    # Subset the data to just the numeric columns
    data = data[list(data.select_dtypes('int').columns) + list(data.select_dtypes('float').columns)]

# Run the preprocessing function over the training data and the actual data that we are predicting
preprocess(data)
preprocess(actual)

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
                
        # Resample the data for binary variables to ensure greater accuracy        
        if analysis_type == 'classify': 
            try:
                # Split the recombined data into adoption/euthanasia sets
                underperformed = model[model[dependent] == 0]
                outperformed = model[model[dependent] == 1]
                
                # Upsample the number of euthanasia records to match the length of the adoption records
                pop_upsampled = resample(outperformed, replace = True, n_samples = len(underperformed))
                
                # Merge into one dataset
                model = pd.concat([pop_upsampled, underperformed])
                
            except:
                pass
                
        # Convert the variables to arrays
        x = model[model.columns[:-1]].values
        y = model[model.columns[-1]].values
        
        # Run cross validations for the RMSE and R2
        scores = cross_val_score(LinearRegression(), x, y, cv = 10, scoring = 'neg_mean_squared_error')
        r2_scores = cross_val_score(LinearRegression(), x, y, cv = 10, scoring = 'r2')
        try:
            accuracy_scores = cross_val_score(LogisticRegression(max_iter = 1000), x, y, cv = 10, scoring = 'accuracy')
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

# Drop anything with very high R squared becuase it is probably not useful for predictions
regressions = regressions[regressions['CORRELATION_COEFFICIENT'] < .98].reset_index(drop = True)

# Print the final result
print(regressions)

#===================================================================================================================================================================#

''' Find the optimal input mix '''

# Create a list variable for the model inputs
model_inputs = [dependent]

# Loop through each variable in the regressions, starting with the most highly correlated
for i in range(len(regressions)):
    
    try:
        # For each iteration, add another variable to the model
        variables = model_inputs
        variables.append(regressions['FACTOR'][i])
        
        # Create a subset of the dataframe with only the relevant variables
        model = data[variables].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop = True)
        
        if analysis_type == 'classify': 
            try:
                # Split the recombined data into adoption/euthanasia sets
                underperformed = model[model[dependent] == 0]
                outperformed = model[model[dependent] == 1]
                
                # Upsample the number of euthanasia records to match the length of the adoption records
                pop_upsampled = resample(underperformed, replace = True, n_samples = len(ou))
                
                # Merge into one dataset
                model = pd.concat([pop_upsampled, underperformed])
                
            except:
                pass        
        
        # Convert the variables to arrays
        x = model.drop(dependent, axis = 1).values
        y = model[dependent].values
        
        # Run the cross validations
        scores = cross_val_score(LinearRegression(), x, y, cv = 10, scoring = 'neg_mean_squared_error')
        r2_scores = cross_val_score(LinearRegression(), x, y, cv = 10, scoring = 'r2')
        try:
            accuracy_scores = cross_val_score(LogisticRegression(max_iter = 1000), x, y, cv = 10, scoring = 'accuracy')
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
                variables.remove(regressions['FACTOR'][i])
        
    except:
        pass

#===================================================================================================================================================================#


''' Preprocess the prediction data '''

# Drop the model inputs that are not available in the prediction dataset
model_inputs = [x for x in model_inputs if x in list(actual)]

################################
# Drop model inputs where there are null values in the prediction dataset
#for i in model_inputs:
#    try:
#        if actual[i].isna().sum() > 0:
#            model_inputs.remove(i)
#    except:
#        model_inputs.remove(i)
################################      

# Create a prediction dataset 
prediction = actual[model_inputs]

# Fill null values with the column means
prediction.fillna(prediction.mean(), inplace = True)

# Replace infinite values with nan, then drop nan
prediction = prediction[prediction.columns[1:]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop = True).astype(float)

# Drop duplicated columns
prediction = prediction.loc[:,~prediction.columns.duplicated()]

# Reset the model inputs to ensure that it only consists of available fields in the prediction data
model_inputs = list(prediction)

# Create a results dataframe to populate the predictions into
results = actual[model_inputs]


''' Preprocess the training data '''

# Create a model variable using the training dataset to train the predictive models
model_inputs.append(dependent)
model = data[model_inputs].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop = True).astype(float)
model = model.loc[:,~model.columns.duplicated()]

# Convert the training variables to arrays
x = model.drop(dependent, axis = 1).values
y = model[dependent].values

# Create a dataframe to populate model scores into
model_results = pd.DataFrame()

#===================================================================================================================================================================#


''' Train, Test, and Run the REGRESSION Model '''


# Build the model
if analysis_type == 'predict':
        
    # Create the model
    reg = LinearRegression()
    
    # Compute the RMSE and R2    
    scores = cross_val_score(reg, x, y, cv = 10, scoring = 'neg_mean_squared_error')
    r2_scores = cross_val_score(reg, x, y, cv = 10, scoring = 'r2')
    rmse = sqrt(abs(scores.mean()))
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Linear Regression R2: {}'.format(r2))
    print('Linear Regression RMSE: {}'.format(rmse), '\n')
    
    model_results.loc[0, 'Model'] = 'Linear Regression'
    model_results.loc[0, 'RMSE'] = rmse
    
elif analysis_type == 'classify':
    try:
        # Create and fit the model
        reg = LogisticRegression(max_iter = 1000)
        
        # Compute the RMSE and R2
        scores = cross_val_score(reg, x, y, cv = 10, scoring = 'accuracy')
        r2_scores = cross_val_score(reg, x, y, cv = 10, scoring = 'r2')
        accuracy = scores.mean()
        r2 = r2_scores.mean()
        
        # Assess accuracy
        print('\n', 'Logistic Regression R2: {}'.format(r2))
        print('Logistic Accuracy: {}'.format(accuracy), '\n')
        
        model_results.loc[0, 'Model'] = 'Logistic Regression'
        model_results.loc[0, 'Accuracy'] = accuracy
    except:
        model_results.loc[0, 'Model'] = 'Logistic Regression'
        model_results.loc[0, 'Accuracy'] = np.nan
        model_results.loc[0, 'RMSE'] = np.nan
        
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
    scores = cross_val_score(svm, x, y, cv = 10, scoring = 'neg_mean_squared_error')
    r2_scores = cross_val_score(svm, x, y, cv = 10, scoring = 'r2')
    rmse = sqrt(abs(scores.mean()))
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Support Vector R2: {}'.format(r2))
    print('Support Vector RMSE: {}'.format(rmse), '\n')
    
    model_results.loc[1, 'Model'] = 'Support Vector Machine'
    model_results.loc[1, 'RMSE'] = rmse
    
elif analysis_type == 'classify':
    
    # Create and fit the model
    svm = SVC()
    
    # Compute the RMSE and R2
    scores = cross_val_score(svm, x, y, cv = 10, scoring = 'accuracy')
    r2_scores = cross_val_score(svm, x, y, cv = 10, scoring = 'r2')
    accuracy = scores.mean()
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Support Vector R2: {}'.format(r2))
    print('Support Vector Accuracy: {}'.format(accuracy), '\n')
    
    model_results.loc[1, 'Model'] = 'Support Vector Machine'
    model_results.loc[1, 'Accuracy'] = accuracy

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
    scores = cross_val_score(tree, x, y, cv = 10, scoring = 'neg_mean_squared_error')
    r2_scores = cross_val_score(tree, x, y, cv = 10, scoring = 'r2')
    rmse = sqrt(abs(scores.mean()))
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Decision Tree R2: {}'.format(r2))
    print('Decision Tree RMSE: {}'.format(rmse), '\n')   

    model_results.loc[2, 'Model'] = 'Decision Tree'
    model_results.loc[2, 'RMSE'] = rmse    
    
# If the prediction variable is continuous, use the Regressor
elif analysis_type == 'classify':

    # Create and fit the model
    tree = DecisionTreeClassifier()
    
    # Compute the RMSE and R2
    scores = cross_val_score(tree, x, y, cv = 10, scoring = 'accuracy')
    r2_scores = cross_val_score(tree, x, y, cv = 10, scoring = 'r2')
    accuracy = scores.mean()
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Decision Tree R2: {}'.format(r2))
    print('Decision Tree Accuracy: {}'.format(accuracy), '\n')
    
    model_results.loc[2, 'Model'] = 'Decision Tree'
    model_results.loc[2, 'Accuracy'] = accuracy    
    
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
    scores = cross_val_score(forest, x, y, cv = 10, scoring = 'neg_mean_squared_error')
    r2_scores = cross_val_score(forest, x, y, cv = 10, scoring = 'r2')
    rmse = sqrt(abs(scores.mean()))
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Random Forest R2: {}'.format(r2))
    print('Random Forest RMSE: {}'.format(rmse), '\n')
    
    model_results.loc[3, 'Model'] = 'Random Forest'
    model_results.loc[3, 'RMSE'] = rmse

# If the prediction variable is continuous, use the Regressor
elif analysis_type == 'classify':
    
    # Create and fit the model
    forest = RandomForestClassifier(n_estimators = 100)
    
    # Compute the RMSE and R2
    scores = cross_val_score(forest, x, y, cv = 10, scoring = 'accuracy')
    r2_scores = cross_val_score(forest, x, y, cv = 10, scoring = 'r2')
    accuracy = scores.mean()
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Random Forest R2: {}'.format(r2))
    print('Random Forest Accuracy: {}'.format(accuracy), '\n')
    
    model_results.loc[3, 'Model'] = 'Random Forest'
    model_results.loc[3, 'Accuracy'] = accuracy    
   
else:
    print("YOU NEED TO SPECIFY THE ANALYSIS TYPE AS EITHER 'predict' OR 'classify' BEFORE RUNNING THE MODEL")
    
# Fit the model
forest.fit(x, y)

# Add a column with the predicted values to the dataframe
results['Forest_Prediction'] = forest.predict(prediction)


#===================================================================================================================================================================#


''' Prep the data for predictive models THAT REQUIRE DATA NORMALIZATION'''

# Create a model variable to use to train the predictive models
model = data[model_inputs].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop = True).astype(float)[:5000]

# Assign a scaler variable
scaler = preprocessing.MinMaxScaler()

# Convert the independent/dependent variables to separate arrays
x = scaler.fit_transform(model.drop(dependent, axis = 1).values)
y = model[dependent].values

# Split the data into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .8)

# Create a prediction dataframe with only the relevant variables that the model will use
#prediction = actual[model_inputs].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop = True).astype(float).drop(dependent, axis = 1)


#===================================================================================================================================================================#


''' Train, Test, and Run the K-NEAREST NEIGHBORS Model '''

# Build the model
if analysis_type == 'predict':
    
    # Create and fit the model
    KNN = KNeighborsRegressor(n_neighbors = 3)

    # Compute the RMSE and R2
    scores = cross_val_score(KNN, x, y, cv = 10, scoring = 'neg_mean_squared_error')
    r2_scores = cross_val_score(KNN, x, y, cv = 10, scoring = 'r2')
    rmse = sqrt(abs(scores.mean()))
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'KNN R2: {}'.format(r2))
    print('KNN RMSE: {}'.format(rmse), '\n')
    
    model_results.loc[4, 'Model'] = 'KNN'
    model_results.loc[4, 'RMSE'] = rmse
    
elif analysis_type == 'classify':

    # Create and fit the model
    KNN = KNeighborsClassifier(n_neighbors = 3)
    
    # Compute the RMSE and R2
    scores = cross_val_score(KNN, x, y, cv = 10, scoring = 'accuracy')
    r2_scores = cross_val_score(KNN, x, y, cv = 10, scoring = 'r2')
    accuracy = scores.mean()
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'KNN R2: {}'.format(r2))
    print('KNN Accuracy: {}'.format(accuracy), '\n')
    
    model_results.loc[4, 'Model'] = 'KNN'
    model_results.loc[4, 'Accuracy'] = accuracy    

else:
    print("YOU NEED TO SPECIFY THE ANALYSIS TYPE AS EITHER 'predict' OR 'classify' BEFORE RUNNING THE MODEL")
    
# Fit the model
KNN.fit(x, y)
    
# Add a column with the predicted values to the dataframe
results['KNN_Prediction'] = KNN.predict(prediction)


#===================================================================================================================================================================#


''' Train, Test, and Run the NEURAL NETWORK Model '''

# Build the model
if analysis_type == 'predict':
    
    # Create and fit the model
    net = MLPRegressor(hidden_layer_sizes = (1000, 100, 100, 100, 10), max_iter = 100000)
    
    # Compute the RMSE and R2
    scores = cross_val_score(net, x, y, cv = 10, scoring = 'neg_mean_squared_error')
    r2_scores = cross_val_score(net, x, y, cv = 10, scoring = 'r2')
    rmse = sqrt(abs(scores.mean()))
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Neural Network R2: {}'.format(r2))
    print('Neural Network RMSE: {}'.format(rmse), '\n')
    
    model_results.loc[5, 'Model'] = 'Neural Network'
    model_results.loc[5, 'RMSE'] = rmse
    
elif analysis_type == 'classify':
    
    # Create and fit the model
    net = MLPClassifier(hidden_layer_sizes = (1000, 100, 100, 100, 10), max_iter = 100000)    
    
    # Compute the RMSE and R2
    scores = cross_val_score(net, x, y, cv = 10, scoring = 'accuracy')
    r2_scores = cross_val_score(net, x, y, cv = 10, scoring = 'r2')
    accuracy = scores.mean()
    r2 = r2_scores.mean()
    
    # Assess accuracy
    print('\n', 'Neural Network R2: {}'.format(r2))
    print('Neural Network Accuracy: {}'.format(accuracy), '\n')
    
    model_results.loc[5, 'Model'] = 'Neural Network'
    model_results.loc[5, 'Accuracy'] = accuracy

else:
    print("YOU NEED TO SPECIFY THE ANALYSIS TYPE AS EITHER 'predict' OR 'classify' BEFORE RUNNING THE MODEL")
    
# Fit the model
net.fit(x, y)
    
# Add a column with the predicted values to the dataframe
results['Neural_Net_Prediction'] = net.predict(prediction)


#===================================================================================================================================================================#

# Sort the model based on the analysis type, and select the best performing model

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
    
# Print the independent variables
print('\n\n The following factors should be included in your model as independent variables: {}'.format(model_inputs))

# Check the results
print('\n\n The results for each model are as follows: \n', model_results)

print('\n--------------------------------------------------------------------------------------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------')


# Show the predictions vs the actual
review = results[[dependent, 'Regression_Prediction', 'Support_Vector_Prediction', 'Tree_Prediction', 'Forest_Prediction', 'KNN_Prediction', 'Neural_Net_Prediction' ]]
