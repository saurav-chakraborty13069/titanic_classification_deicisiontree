import pickle
import pandas as pd




def load_models(log_writer,file_object):
    log_writer.log(file_object, 'Starting to load models')
    with open("models/modelForPrediction.sav", 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_data(final_df, log_writer,file_object):
    gender = {'male': 0, 'female': 1}
    final_df.Sex = [gender[item] for item in final_df.Sex]
    log_writer.log(file_object, 'Coverted Sex to float object')
    return  final_df


def predict_data(dict_pred, log_writer):

    #validate the data entered
    #preprocess to get X in sme format
    #then apply models to predict
    file_object = open("logs/PredictionLogs.txt", 'a+')
    log_writer.log(file_object, 'Starting the predict data')

    model = load_models(log_writer,file_object)
    log_writer.log(file_object, 'Loading of models completed')
    final_df = pd.DataFrame(dict_pred, index = [1,])
    final_df = preprocess_data(final_df, log_writer,file_object)
    log_writer.log(file_object, 'Prepared the final dataframe')
    log_writer.log(file_object, 'Predicting the result')
    predict = model.predict(final_df)

    print('Class is:    ', predict[0])
    log_writer.log(file_object, 'Prediction completed')
    log_writer.log(file_object, '=================================================')
    return predict[0]


