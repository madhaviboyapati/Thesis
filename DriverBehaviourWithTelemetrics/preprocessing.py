import numpy as np
import pandas as pd
import os
import sys

def preprocess(execution_type):
    labelpath = ''
    csv_dir = ''
    input_file_name = ''
    output_file_name = ''
    if(execution_type == 'evaluate'):
        labelpath = 'evaluate_labels\\label_data_set.csv'
        csv_dir = 'evaluate_features'  # specify folder for features csv file
        input_file_name = 'evaluate_input.npy'
        output_file_name = 'evaluate_output.npy'
    elif(execution_type == 'train'):
        labelpath = 'labels\\Dataset_Labels.csv'  # specify path to label file
        csv_dir = 'features'  # specify folder for features csv file
        input_file_name = 'input.npy'
        output_file_name = 'output.npy'
    elif (execution_type == 'predict'):
        labelpath = 'predict_labels\\Predict_Labels.csv'  # specify path to label file
        csv_dir = 'predict_features'  # specify folder for features csv file
        input_file_name = 'predict_input.npy'
        output_file_name = 'predict_output.npy'

    # create empty dataframe to store features and label
    features_df = pd.DataFrame()
    label_df = pd.DataFrame()
    print("Processing Started")
    # read features files
    for csv_file in os.listdir(csv_dir):
        if csv_file == '.DS_Store':
            continue
        filepath = csv_dir+'\\'+csv_file
        print(filepath)
        features_df = pd.concat([features_df, pd.read_csv(filepath)])

    # read label files
    label_df = pd.read_csv(labelpath)

    # calculate total number of booking and label
    num_booking = features_df['bookingID'].nunique()
    num_label = label_df['bookingID'].nunique()

    # get minimum number of timestep for each booking
    max_second = list()
    for ID in features_df['bookingID'].unique():
        values = list()
        df = features_df[features_df['bookingID']==ID]
        for value in df['second'].unique():
            values.append(value)
        max_second.append(max(values))
    num_timestep = int(min(max_second))
    print(num_timestep)
    #num_timestep = 80
    num_timestep = 119  # Set the number of timestep to 119 since the lowest timestep in the given data is 119
    m = num_booking    # number of samples
    n = 9 # number of fetures

    # convert features to numpy array
    i=0
    num_neg = 0
    num_pos = 0
    x = np.empty([m, num_timestep, n])
    y = np.empty([m, 1])
    for ID in features_df['bookingID'].unique():
        df = label_df[label_df['bookingID']==ID]
        label = int(df['label'].values[0])
        # # Down sampling negative examples to balance datasets
        if (execution_type == 'train'):
            if label == 0:
               num_neg = num_neg+1
               if num_neg > 4990:
                  continue
        y[i,:] = label
        df1 = features_df[features_df['bookingID']==ID]
        del df1['bookingID']
        if 'Path' in df1.columns:
            del df1['Path']
        if 'Accuracy' in df1.columns:
            del df1['Accuracy']
        # if 'Bearing' in df1.columns:
        #    del df1['Bearing']
        x[i,:,:] = df1.values[0:num_timestep,:]
        if (execution_type == 'train'):
            i=i+1
            if i > m-1:
               break

    print(x.shape)
    print(y.shape)
    np.save('preprocessed_data\\' + input_file_name, x)
    np.save('preprocessed_data\\' + output_file_name, y)
    print("Processing Completed")

if __name__ == "__main__":
    params = sys.argv
    # options that can be passed to params: python preprocessing.py train
    # options that can be passed to params: python preprocessing.py evaluate
    # options that can be passed to params: python preprocessing.py predict
    print(params[1])
    preprocess(str(params[1]))