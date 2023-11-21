import re
import os
import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

import csv


# Configuration Parameters
DATA_DIR = '/vol/actrec/SisFall_dataset/' 
README_FILE_PATH = os.path.join(DATA_DIR, 'Readme.txt')
SUBJECT= ['SA01', 'SA02', 'SA03', 'SA04', 'SA05', 'SA06', 'SA07', 'SA08', 'SA09', 'SA10', 'SA11', 'SA12', 
               'SA13', 'SA14', 'SA15', 'SA16', 'SA17', 'SA18', 'SA19', 'SA20', 'SA21', 'SA22', 'SA23', 'SE01', 'SE02',
               'SE03', 'SE04', 'SE05', 'SE06', 'SE07', 'SE08', 'SE09', 'SE10', 'SE11', 'SE12', 'SE13', 'SE14', 'SE15']
Subject_id= {'SA01':0, 'SA02':1, 'SA03':2, 'SA04':3, 'SA05':4, 'SA06':5, 'SA07':6, 'SA08':7, 'SA09':8, 'SA10':9, 'SA11':10, 'SA12':11, 
               'SA13':12, 'SA14':13, 'SA15':14, 'SA16':15, 'SA17':16, 'SA18':17, 'SA19':18, 'SA20':19, 'SA21':20, 'SA22':21, 'SA23':22, 'SE01':23, 'SE02':24,
               'SE03':25, 'SE04':26, 'SE05':27, 'SE06':28, 'SE07':29, 'SE08':30, 'SE09':31, 'SE10':32, 'SE11':33, 'SE12':34, 'SE13':35, 'SE14':36, 'SE15':37}
WINDOW_SIZE = 200
STRIDE = 50
SOFT_BIOMETRICS = ['age', 'height', 'weight', 'gender']
sensor_cols = ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x', 'MMA8451Q_y', 'MMA8451Q_z']
    

def get_person_info():
    # Open the readme file and read the lines
    subjects=[]
    file_path = 'sisfall_subject.csv'
    print(file_path)
    with open(file_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter= ';')
        next(spamreader)
        for row in spamreader:
            subjects.append(list(map(float, row)))
    list(np.float_(subjects))

    return subjects  # Returning the DataFrame directly

def process_file(file_path):
    sensor_data = [] 
    with open(file_path, 'r') as txtfile:
        spamreader = csv.reader(txtfile, delimiter=';')
        for row in spamreader:
            row=row[0].strip()
            res = []
            for elem in row.split(','):
                res.append(float(elem))
            sensor_data.append(res)
        print(sensor_data)
        '''
        for i in range(0, len(chunk) - WINDOW_SIZE + 1, STRIDE):
                segment = chunk.iloc[i:i + WINDOW_SIZE].copy() 
                segments.append(segment) 
            person_info = get_person_info(subject_id)  
        for segment in segments:
            segment['person_id'] = subject_id
            for label in SOFT_BIOMETRICS:
                segment[label] = person_info[label].values[0]  

    except Exception as e:
        print("Error processing {file_path}:",file_path )
        return None
    return segments
    '''

def process_subject(subject_id):
    print('Processing subject', subject_id)
    subject_dir = os.path.join(DATA_DIR, subject_id)
    file_list = os.listdir(subject_dir)
    all_segments = []
    for file_name in file_list:
        if file_name.endswith('.csv'):
            file_path = os.path.join(subject_dir, file_name)
            segments = process_file(file_path, subject_id)
            if segments is not None:
                all_segments.extend(segments)
    subject_dir = os.path.join(DATA_DIR, subject_id) 
    return all_segments

def normalize_and_encode(all_data):
    try:
        scaler = StandardScaler()
        cols_to_normalize = all_data.iloc[:, :-5].columns
        

        all_data[cols_to_normalize] = scaler.fit_transform(all_data[cols_to_normalize])

        # Encode the person IDs and soft biometric labels
        print('Encoding labels...')
        label_encoders = {}  
        for col in ['person_id'] + SOFT_BIOMETRICS:
            le = LabelEncoder()
            all_data[col] = le.fit_transform(all_data[col])
            label_encoders[col] = le  
            
    except Exception as e:
        print("Error in normalize_and_encode")
        return None

def extract_features(segment):
    features = []
    sensor_cols = ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x', 'MMA8451Q_y', 'MMA8451Q_z']
    for col in segment.columns:
        if col in  sensor_cols:
            features.append(segment[col].mean())
            features.append(segment[col].std())
            features.append(segment[col].max())
            features.append(segment[col].min())
            features.append(np.sqrt(np.mean(segment[col]**2)))
        feature_names = ['{col}_{stat}' for col in sensor_cols for stat in ['min', 'max','mean', 'std','rms']]
    return pd.Series(features, index=feature_names).astype('float32')  

def remove_original_sensor_data(df):
    sensor_cols = ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x', 'MMA8451Q_y', 'MMA8451Q_z']
    return df.drop(columns=sensor_cols)

def rearrange_columns(df):
    cols = list(df.columns)
    cols = [col for col in cols if col not in ['person_id', 'age', 'height', 'weight', 'gender']]
    cols.extend(['person_id', 'age', 'height', 'weight', 'gender'])
    return df[cols]

def split_and_save_data(X, y):
    try:
        
    
        # Split the data into training and validation sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify= y['person_id'])

        # Split the temp data into validation and test sets
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp['person_id'])
        
        # Now, you can concatenate the X and y DataFrames for each split and save them to CSV
        train_data = pd.concat([X_train, y_train], axis=1)
        valid_data = pd.concat([X_valid, y_valid], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        train_data.to_csv('Sis_train_data.csv', index=False)
        valid_data.to_csv('Sis_valid_data.csv', index=False)
        test_data.to_csv('Sis_test_data.csv', index=False)
    except Exception as e:
        print("Error in split_and_save_data:")
        
def generate_data(ids, activities, sliding_window_length, sliding_window_step, data_dir=None,
                  identity_bool=False, usage_modus='train'):
    '''
    creates files for each of the sequences, which are extracted from a file
    following a sliding window approach

    returns
    Sequences are stored in given path

    @param ids: ids for train, val or test
    @param sliding_window_length: length of window for segmentation
    @param sliding_window_step: step between windows for segmentation
    @param data_dir: path to dir where files will be stored
    @param identity_bool: selecting for identity experiment
    @param usage_modus: selecting Train, Val or testing
    '''
    
    all_segments = []
    #counter_seq = 0
    #hist_classes_all = np.zeros((NUM_CLASSES))
    recordings= ['R01', 'R02', 'R03', 'R04', 'R05', 'R06']
    
    for subject_id in ids:
       print('Processing subject', subject_id)
       subject_dir = os.path.join(DATA_DIR, subject_id)
       file_list = os.listdir(subject_dir)
       for act in activities:
           print(act)
           for R in recordings:
               try:
                   file_name= "{}_{}_{}.txt".format(act, subject_id, R)
                   print(file_name)
                   file_path = os.path.join(subject_dir, file_name)
                   print(file_path)
                   segments = process_file(file_path)
               except: 
                   print('no file path with name', file_name)
                   if segments is not None:
                       all_segments.extend(segments)
      


def main():
    person_info = get_person_info()
    train_ids= ['SA01', ]#'SA02', 'SA03', 'SA04', 'SA05', 'SA06', 'SA07', 
               #'SA08', 'SA09', 'SA10', 'SA11', 'SA12', 'SA13']
    activities= ['D01', 'D02', 'D03',]# 'D04', 'D05', 'D07', 'D08', 'D09', 
                 #'D10', 'D011', 'D12', 'D14', 'D15', 'D16', 'D17']
    
    base_directory='/data/nnair/idimuall/'
    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'
    
    generate_data(train_ids, activities, sliding_window_length=200, sliding_window_step=50, data_dir=data_dir_train, usage_modus='train')
    '''
    generate_data(val_ids, activties, sliding_window_length=200, sliding_window_step=50, data_dir=data_dir_val, usage_modus='val')
    generate_data(test_ids, activities, sliding_window_length=200, sliding_window_step=50, data_dir=data_dir_test, usage_modus='test')

    generate_CSV(base_directory, "train.csv", data_dir_train)
    generate_CSV(base_directory, "val.csv", data_dir_val)
    generate_CSV(base_directory, "test.csv", data_dir_test)
    generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)

    
    all_segments = []
    for subject_id in SUBJECT_IDS:
        subject_segments = process_subject(subject_id)
        
        if subject_segments:
            
            all_segments.extend(subject_segments)

    try:
        all_data = pd.concat(all_segments, axis=0, ignore_index=True)
        float_cols = all_data.select_dtypes(include=['float64']).columns
        all_data[float_cols] = all_data[float_cols].astype('float32')
        cat_cols = all_data.select_dtypes(include=['category']).columns
        all_data[cat_cols] = all_data[cat_cols].astype('category')
        print(all_data.head())
        all_data['MMA8451Q_z'] = all_data['MMA8451Q_z'].str.replace(';', '').astype(float)
       
        feature_df = all_data.groupby('person_id').apply(lambda segment: extract_features(segment))
        feature_df.reset_index(inplace=True)
       
        all_data = pd.merge(all_data, feature_df, on='person_id', how='left')
        all_data = remove_original_sensor_data(all_data) 
        all_data = rearrange_columns(all_data)
        
        normalize_and_encode(all_data)
        #print(all_data.head())
        X = all_data.iloc[:, :-5]
        y = all_data[['person_id', 'age', 'height', 'weight', 'gender']]
      
        split_and_save_data(X, y)

    except ValueError as e:
        print(f"Error: {e}")  
    '''

if __name__ == "__main__":
    main()