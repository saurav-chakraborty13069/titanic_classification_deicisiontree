import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from logger import App_Logger
import pickle
import matplotlib.pyplot as plt

def check_data(data, log_writer, file_object):
    print(data.head())
    print(data.columns)
    print(data.info())
    print(data.describe())
    print(type(data))
    print(data.shape)
    print(data.isnull().sum())

def preprocess_data(data, log_writer, file_object):

    X = data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    y = data['Survived']
    gender = {'male': 0, 'female': 1}
    X.Sex = [gender[item] for item in X.Sex]
    X['Age'] = X['Age'].replace(float('nan'), round(X['Age'].mean(), 2))
    return X,y


def grid_search_data(clf, x_train, y_train, log_writer, file_object):
    log_writer.log(file_object, 'Starting the grid seach')
    log_writer.log(file_object, 'Setting up parameters')
    grid_param = {

        'max_depth': range(2, 32, 1),
        'min_samples_leaf': range(1, 10, 1),
        'min_samples_split': range(2, 10, 1),
        'splitter': ['best', 'random']
    }

    grid_search = GridSearchCV(estimator=clf,
                               param_grid=grid_param,
                               cv=5,
                               n_jobs=-1)
    log_writer.log(file_object, 'Fitting the Grid search Model')
    grid_search.fit(x_train, y_train)
    best_parameters = grid_search.best_params_
    log_writer.log(file_object, 'Best parameter for max depth is {}'.format(best_parameters['max_depth']))
    log_writer.log(file_object, 'Best parameter for min_samples_leaf is {}'.format(best_parameters['min_samples_leaf']))
    log_writer.log(file_object, 'Best parameter for min_samples_split is {}'.format(best_parameters['min_samples_split']))
    log_writer.log(file_object, 'Best parameter for splitter is {}'.format(best_parameters['splitter']))
    print("Best Grid Search Score is:   ", grid_search.best_score_)
    return best_parameters


def save_model(clf, log_writer, file_object):

    # Writing different model files to file
    log_writer.log(file_object, 'Saving the models at location')
    with open('models/modelForPrediction.sav', 'wb') as f:
        pickle.dump(clf, f)

def metrics_data(clf,y_test, y_pred,file_object,log_writer):
    conf_mat = confusion_matrix(y_test, y_pred)
    print(conf_mat)
    true_positive = conf_mat[0][0]
    false_positive = conf_mat[0][1]
    false_negative = conf_mat[1][0]
    true_negative = conf_mat[1][1]
    log_writer.log(file_object, 'true_positive:     {}'.format(true_positive))
    log_writer.log(file_object, 'false_positive:    {}'.format(false_positive))
    log_writer.log(file_object, 'false_negative:    {}'.format(false_negative))
    log_writer.log(file_object, 'true_negative:     {}'.format(true_negative))
    precision = true_positive / (true_positive + false_positive)
    print("Precision is:    ",precision)
    log_writer.log(file_object, 'Precision is:  {}'.format(precision))
    recall = true_positive / (true_positive + false_negative)
    print("Recall is:   ",recall)
    log_writer.log(file_object, 'Recall is:  {}'.format(recall))
    f1_score = 2 * precision * recall / (precision + recall)
    print("F1 Score is: ",f1_score)
    log_writer.log(file_object, 'F1 Score is:  {}'.format(f1_score))
    auc = roc_auc_score(y_test, y_pred)
    print("AUC is:  ",auc)
    log_writer.log(file_object, 'AUC is:  {}'.format(auc))



def train_data(log_writer):

    file_object = open("logs/TrainingLogs.txt", 'a+')
    log_writer.log(file_object, 'Start of Training')
    log_writer.log(file_object, 'Getting the data from url')
    Url = 'https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
    data = pd.read_csv(Url)

    log_writer.log(file_object, 'Received the data')
    check_data(data, log_writer, file_object)

    X,y = preprocess_data(data, log_writer, file_object)
    log_writer.log(file_object, 'Preprocessing of data completed')



    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=355)
    log_writer.log(file_object, 'Splitting of data into train and test completed')
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    log_writer.log(file_object, 'Training of Decision Tree classifier without best parameters')
    best_params = grid_search_data(clf, x_train, y_train, log_writer, file_object)
    log_writer.log(file_object, 'Grid search completed and received the best parameters ')
    clf = DecisionTreeClassifier(criterion='gini', max_depth=best_params['max_depth'],
                                 min_samples_leaf=best_params['min_samples_leaf'],
                                 min_samples_split=best_params['min_samples_split'],
                                 splitter=best_params['splitter'])
    clf.fit(x_train, y_train)
    log_writer.log(file_object, 'Fitting the Decision Tree Classifier with best parameters ')
    y_pred = clf.predict(x_test)
    metrics_data(clf,y_test, y_pred,file_object,log_writer)
    log_writer.log(file_object, 'Training complete')
    save_model(clf,log_writer,file_object)
    log_writer.log(file_object, '====================================')
    print('Current Classifier score is: ', clf.score(x_test, y_test))