import re
import os
import gc

import numpy as np
import pandas as pd
from sliding_window_sf import sliding_window

import sys
import pickle


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
activities_id= {'D01':0, 'D02':1, 'D03':2, 'D04':3, 'D05':4, 'D07':5, 'D08':6, 'D09':7, 
                 'D10':8, 'D011':9, 'D12':10, 'D14':11, 'D15':12, 'D16':13, 'D17':14}
NUM_ACT_CLASSES= 15
NUM_CLASSES = 37
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

def opp_sliding_window(data_x, data_y, data_z, label_pos_end=True):
    '''
    Performs the sliding window approach on the data and the labels

    return three arrays.
    - data, an array where first dim is the windows
    - labels per window according to end, middle or mode
    - all labels per window

    @param data_x: ids for train
    @param data_y: ids for train
    @param ws: ids for train
    @param ss: ids for train
    @param label_pos_end: ids for train
    '''

    print("Sliding window: Creating windows {} with step {}".format(ws, ss))

    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    print('IMU data split')
    # Label from the end
    if label_pos_end:
        data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
        data_z = np.asarray([[i[-1]] for i in sliding_window(data_z, ws, ss)])
    else:
        if False:
            # Label from the middle
            # not used in experiments
            data_y_labels = np.asarray(
                [[i[i.shape[0] // 2]] for i in sliding_window(data_y, ws, ss)])
            data_z_labels = np.asarray(
                [[i[i.shape[0] // 2]] for i in sliding_window(data_z, ws, ss)])
        else:
            # Label according to mode
            try:
                data_y_labels = []
                data_z_labels = []
                
                for sw in sliding_window(data_y, ws, ss):
                    count_l = np.bincount(sw.astype(int), minlength=NUM_ACT_CLASSES)
                    idy = np.argmax(count_l)
                    data_y_labels.append(idy)
                data_y_labels = np.asarray(data_y_labels)
                for sz in sliding_window(data_z, ws, ss):
                    count_l = np.bincount(sz.astype(int), minlength=NUM_CLASSES)
                    idy = np.argmax(count_l)
                    data_z_labels.append(idy)
                data_z_labels = np.asarray(data_z_labels)


            except:
                print("Sliding window: error with the counting {}".format(count_l))
                print("Sliding window: error with the counting {}".format(idy))
                return np.Inf

            # All labels per window
            data_y_all = np.asarray([i[:] for i in sliding_window(data_y, ws, ss)])
            data_z_all = np.asarray([i[:] for i in sliding_window(data_z, ws, ss)])
            print('sliding window complete')
    
    return data_x.astype(np.float32), data_y_labels.astype(np.uint8), data_y_all.astype(np.uint8), data_z_labels.astype(np.uint8), data_z_all.astype(np.uint8)
    

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
    
    data_dir_train = data_dir + 'sequences_train/'
    data_dir_val = data_dir + 'sequences_val/'
    data_dir_test = data_dir + 'sequences_test/'
    
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
    for subject_id in ids:
       print('Processing subject', subject_id)
       subject_dir = os.path.join(DATA_DIR, subject_id)
       file_list = os.listdir(subject_dir)
       for act in activities:
           #print(act)
           all_segments = np.empty((0, 9))
           for R in recordings:
               try:
                   file_name= "{}_{}_{}.txt".format(act, subject_id, R)
                   print(file_name)
                   file_path = os.path.join(subject_dir, file_name)
                   segments = process_file(file_path)
                   segments=np.array([np.array(i) for i in segments])
                   all_segments=np.concatenate((all_segments, segments), axis=0)
                   
               except: 
                   print('no file path with name', file_name)
                   #if segments is not None:
                    #   all_segments.extend(segments)
           try:
               print('normalise')
               data_x = normalize(all_segments)
           except:
               print("\n3  In normalising, issues found.")
               continue
           frames=all_segments.shape[0]
           if frames != 0:
               train_no=round(0.70*frames)
               val_no=round(0.15*frames)
               tv= train_no+val_no
               
               if usage_modus=='trainval':
                    X_train = np.vstack((X_train, all_segments[0:train_no,:]))
                    act_train = np.append(act_train, np.full((train_no), activities_id[act]))
                    id_train = np.append(id_train, np.full((train_no), Subject_id[subject_id]))
                    print('done train')
                            
                    X_val = np.vstack((X_val, all_segments[train_no:tv,:]))
                    act_val = np.append(act_val, np.full((val_no), activities_id[act]))
                    id_val = np.append(id_val, np.full((val_no), Subject_id[subject_id]))
                    print('done val')
               elif usage_modus=='test':
                    X_test = np.vstack((X_test, all_segments[tv:frames,:]))
                    act_test = np.append(act_test, np.full((frames-tv), activities_id[act]))
                    id_test = np.append(id_test, np.full((frames-tv), Subject_id[subject_id]))
                    print('done test')
           else:
                continue
                    
    try: 
        if usage_modus=='trainval':
            data_train, act_train, act_all_train, labelid_train, labelid_all_train = opp_sliding_window(X_train, act_train, id_train, label_pos_end = False)
            data_val, act_val, act_all_val, labelid_val, labelid_all_val = opp_sliding_window(X_val, act_val, id_val, label_pos_end = False)
        elif usage_modus=='test':
            data_test, act_test, act_all_test, labelid_test, labelid_all_test = opp_sliding_window(X_test, act_test, id_test, label_pos_end = False)
    except:
        print("error in sliding window")
        
    try:
        
        print("window extraction begining")
        
        print("data save")
        if usage_modus=='trainval':
            print("target file name")
            print(data_dir_train)
            counter_seq = 0
            for f in range(data_train.shape[0]):
                try:
                    sys.stdout.write('\r' + 'Creating sequence file '
                                     'number {} with id {}'.format(f, counter_seq))
                    sys.stdout.flush()
                    
                    # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                    seq = np.reshape(data_train[f], newshape = (1, data_train.shape[1], data_train.shape[2]))
                    seq = np.require(seq, dtype=np.float)
                    # Storing the sequences
                    #obj = {"data": seq, "label": labelid}
                    obj = {"data": seq, "act_label": act_train[f], "act_labels_all": act_all_train[f], "label": labelid_train[f]}
                    
                    f = open(os.path.join(data_dir_train, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()

                    counter_seq += 1
                except:
                    raise('\nError adding the seq')
                
            print("val data save")
            print("target file name")
            print(data_dir_val)
            counter_seq = 0
            for f in range(data_val.shape[0]):
                try:
                    sys.stdout.write('\r' + 'Creating sequence file '
                                     'number {} with id {}'.format(f, counter_seq))
                    sys.stdout.flush()

                    # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                    seq = np.reshape(data_val[f], newshape = (1, data_val.shape[1], data_val.shape[2]))
                    seq = np.require(seq, dtype=np.float)
                    # Storing the sequences
                    #obj = {"data": seq, "label": labelid}
                    obj = {"data": seq, "act_label": act_val[f], "act_labels_all": act_all_val[f], "label": labelid_val[f]}
                
                    f = open(os.path.join(data_dir_val, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()

                    counter_seq += 1
                except: 
                     raise('\nError adding the seq')
        elif usage_modus=='test':         
            print("test data save")
            print("target file name")
            print(data_dir_test)
            counter_seq = 0
            for f in range(data_test.shape[0]):
                try:
                    sys.stdout.write('\r' + 'Creating sequence file '
                                     'number {} with id {}'.format(f, counter_seq))
                    sys.stdout.flush()

                    # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                    seq = np.reshape(data_test[f], newshape = (1, data_test.shape[1], data_test.shape[2]))
                    seq = np.require(seq, dtype=np.float)
                    # Storing the sequences
                    #obj = {"data": seq, "label": labelid}
                    obj = {"data": seq, "act_label": act_test[f], "act_labels_all": act_all_test[f], "label": labelid_test[f]}
                
                    f = open(os.path.join(data_dir_test, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()

                    counter_seq += 1
                except:
                    raise('\nError adding the seq')
    except:
        print("error in saving") 
        
    return

def generate_CSV(csv_dir, data_dir):
    '''
    Generate CSV file with path to all (Training) of the segmented sequences
    This is done for the DATALoader for Torch, using a CSV file with all the paths from the extracted
    sequences.

    @param csv_dir: Path to the dataset
    @param data_dir: Path of the training data
    '''
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for n in range(len(filenames)):
            f.append(data_dir + 'seq_{0:06}.pkl'.format(n))

    #np.savetxt(csv_dir + type_file, f, delimiter="\n", fmt='%s')
    np.savetxt(csv_dir, f, delimiter="\n", fmt='%s')
    
    return

def generate_CSV_final(csv_dir, data_dir1, data_dir2):
    '''
    Generate CSV file with path to all (Training and Validation) of the segmented sequences
    This is done for the DATALoader for Torch, using a CSV file with all the paths from the extracted
    sequences.

    @param csv_dir: Path to the dataset
    @param data_dir1: Path of the training data
    @param data_dir2: Path of the validation data
    '''
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir1):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:06}.pkl'.format(n))

    for dirpath, dirnames, filenames in os.walk(data_dir2):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:06}.pkl'.format(n))

    np.savetxt(csv_dir, f, delimiter="\n", fmt='%s')

    return


def main():
    person_info = get_person_info()
    train_ids= ['SA01','SA02', 'SA03', 'SA04', 'SA05', 'SA06', 'SA07', 
                'SA08', 'SA09', 'SA10', 'SA11', 'SA12', 'SA13', 'SA14', 
                'SA15', 'SA16', 'SA17', 'SA18', 'SA19', 'SA20', 'SA21', 
                'SA22', 'SA23', 'SE01', 'SE02', 'SE03', 'SE04', 'SE05', 
                'SE06', 'SE07', 'SE08', 'SE09', 'SE10', 'SE11', 'SE12', 'SE13', 'SE14', 'SE15']
    activities= ['D01', 'D02', 'D03', 'D04', 'D05', 'D07', 'D08', 'D09', 
                 'D10', 'D011', 'D12', 'D14', 'D15', 'D16', 'D17']
    
    base_directory ='/data/nnair/sisfall_prepros/'
    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'
    
    generate_data(train_ids, activities, sliding_window_length=200, sliding_window_step=50, data_dir=data_dir_train, usage_modus='trainval')
    generate_data(train_ids, activities, sliding_window_length=200, sliding_window_step=50, data_dir=data_dir_test, usage_modus='test')

    generate_CSV(base_directory, data_dir_train)
    generate_CSV(base_directory, data_dir_val)
    generate_CSV(base_directory, data_dir_test)
    generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)

    return

if __name__ == "__main__":
    main()