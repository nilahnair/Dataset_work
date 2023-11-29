import re
import os
import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from sliding_window_sf import sliding_window

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
ws = 200 #WINDOW_SIZE
ss = 50 #STRIDE 
SOFT_BIOMETRICS = ['age', 'height', 'weight', 'gender']
sensor_cols = ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x', 'MMA8451Q_y', 'MMA8451Q_z']
    

def find_stats(ids, activities, data_dir=None):
    recordings= ['R01', 'R02', 'R03', 'R04', 'R05', 'R06']
    all_segments = np.empty((0, 9))
    for subject_id in ids:
       print('Processing subject', subject_id)
       subject_dir = os.path.join(DATA_DIR, subject_id)
       file_list = os.listdir(subject_dir)
       for act in activities:
           #print(act)
           for R in recordings:
               segments=[]
               try:
                   file_name= "{}_{}_{}.txt".format(act, subject_id, R)
                   print(file_name)
                   file_path = os.path.join(subject_dir, file_name)
                   #print(file_path)
                   segments = process_file(file_path)
                   #print(len(segments))
                   segments=np.array([np.array(i) for i in segments])
                   #print(segments.shape)
                   #print(segments)
                   all_segments=np.concatenate((all_segments, segments), axis=0)
                   #print('len of all segments')
                   #print(all_segments.shape)
                   
               except: 
                   print('no file path with name', file_name)
                   #if segments is not None:
                    #   all_segments.extend(segments)
                    
    #print(all_segments.shape)
    #numpy.array([numpy.array(xi) for xi in x])
    
    max_values = np.max(all_segments, axis=0)
    print("Max values")
    print(max_values)
    min_values = np.min(all_segments, axis=0)
    print("Min values")
    print(min_values)
    mean_values = np.mean(all_segments, axis=0)
    print("Mean values")
    print(mean_values)
    std_values = np.std(all_segments, axis=0)
    print("std values")
    print(std_values)
    
    return

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
            
    #print(len(sensor_data)) 
    return sensor_data
       
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

def normalize(data):
    
    #max_values=[1192.0, 1032.0, 2582.0, 16098.0, 12124.0, 12585.0, 4035.0, 5511.0, 6673.0]
    #min_values=[-986.0, -2333.0, -1785.0, -15988.0, -12732.0, -11882.0, -3757.0, -8192.0, -7582.0]
    mean_values=np.array([3.16212380, -220.821147, -37.2032848, -4.97325388, 34.9530823, -7.05977257, -0.394311490, -864.960077, -98.0097123])
    std_values=np.array([76.42413571, 133.73065249, 108.80401481, 664.20882435, 503.17930668, 417.85844231, 296.16101639, 517.27540723, 443.30238268])
    
    mean_values = np.reshape(mean_values, [1, 9])
    std_values = np.reshape(std_values, [1, 9])
    try:
        mean_array = np.repeat(mean_values, data.shape[0], axis=0)
        std_array = np.repeat(std_values, data.shape[0], axis=0)

        max_values = mean_array + 2 * std_array
        min_values = mean_array - 2 * std_array

        data_norm = (data - min_values) / (max_values - min_values)

        data_norm[data_norm > 1] = 1
        data_norm[data_norm < 0] = 0
    except:
        raise("Error in normalisation")

    return data_norm

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
    
    if usage_modus=='trainval':
        X_train = np.empty((0, 9))
        act_train = np.empty((0))
        id_train = np.empty((0))
    
        X_val = np.empty((0, 9))
        act_val = np.empty((0))
        id_val = np.empty((0))
    
    elif usage_modus=='test':
        X_test = np.empty((0, 9))
        act_test = np.empty((0))
        id_test = np.empty((0))
    
    #counter_seq = 0
    #hist_classes_all = np.zeros((NUM_CLASSES))
    recordings= ['R01', 'R02', 'R03', 'R04', 'R05', 'R06']
    all_segments = np.empty((0, 9))
    for subject_id in ids:
       print('Processing subject', subject_id)
       subject_dir = os.path.join(DATA_DIR, subject_id)
       file_list = os.listdir(subject_dir)
       for act in activities:
           #print(act)
           for R in recordings:
               segments=[]
               try:
                   file_name= "{}_{}_{}.txt".format(act, subject_id, R)
                   print(file_name)
                   file_path = os.path.join(subject_dir, file_name)
                   #print(file_path)
                   segments = process_file(file_path)
                   #print(len(segments))
                   segments=np.array([np.array(i) for i in segments])
                   #print(segments.shape)
                   #print(segments)
                   all_segments=np.concatenate((all_segments, segments), axis=0)
                   #print('len of all segments')
                   #print(all_segments.shape)
                   
               except: 
                   print('no file path with name', file_name)
                   #if segments is not None:
                    #   all_segments.extend(segments)
           try:
               data_x = normalize(all_segments)
           except:
               print("\n3  In normalising, issues found.")
               continue
           frames=all_segments.shape[0]
           if frames != 0:
               train_no=round(0.64*frames)
               val_no=round(0.18*frames)
               tv= train_no+val_no
               
           if usage_modus=='trainval':
               X_train = np.vstack((X_train, all_segments[0:train_no,:]))
               #act_train = np.append(act_train, [lbls[0:train_no,0]])
               #id_train = np.append(id_train, [lbls[0:train_no,1]])
               print('done train')
                            
               X_val = np.vstack((X_val, all_segments[train_no:tv,:]))
               #act_val = np.append(act_val, [lbls[train_no:tv,0]])
               #id_val = np.append(id_val, [lbls[train_no:tv,1]])
               print('done val')
           elif usage_modus=='test':
                X_test = np.vstack((X_test, all_segments[tv:frames,:]))
                #act_test = np.append(act_test, [lbls[tv:frames,0]])
                #id_test = np.append(id_test, [lbls[tv:frames,1]])
                print('done test')
                    
           print('frames')
           print(frames)
           if usage_modus=='trainval':
               print('X_train')
               print(X_train.shape)
               print('X_val')
               print(X_val.shape)
           elif usage_modus=='test':
               print('X_test')
               print(X_test.shape)
                    
               
           try: 
               if usage_modus=='trainval':
                   print('Sliding window')
                   X_train = sliding_window(X_train, (ws, X_train.shape[1]), (ss, 1))
                   print(X_train.shape)
                   X_val = sliding_window(X_val, (ws, X_val.shape[1]), (ss, 1))
                   print(X_val.shape)
               elif usage_modus=='test':
                   print('Sliding window')
                   X_test= sliding_window(X_test, (ws, X_test.shape[1]), (ss, 1))
                   print(X_test.shape)
           except:
               print("error in sliding window")
    

def main():
    person_info = get_person_info()
    train_ids= ['SA01','SA02', 'SA03', 'SA04', 'SA05', 'SA06', 'SA07', 
                'SA08', 'SA09', 'SA10', 'SA11', 'SA12', 'SA13', 'SA14', 
                'SA15', 'SA16', 'SA17', 'SA18', 'SA19', 'SA20', 'SA21', 
                'SA22', 'SA23', 'SE01', 'SE02', 'SE03', 'SE04', 'SE05', 
                'SE06', 'SE07', 'SE08', 'SE09', 'SE10', 'SE11', 'SE12', 'SE13', 'SE14', 'SE15']
    activities= ['D01', 'D02', 'D03', 'D04', 'D05', 'D07', 'D08', 'D09', 
                 'D10', 'D011', 'D12', 'D14', 'D15', 'D16', 'D17']
    
    base_directory ='/data/nnair/idimuall/'
    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'
    
    generate_data(train_ids, activities, sliding_window_length=200, sliding_window_step=50, data_dir=data_dir_train, usage_modus='trainval')
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