'''
Created on Oct 02, 2019

@author: fmoya


'''

import numpy as np
import csv
import os
import sys
import matplotlib.pyplot as plt
import datetime
#from sliding_window import sliding_window
import pickle
import time
import pandas as pd

# folder path
FOLDER_PATH = "/vol/actrec/MobiAct_Dataset/"
SUBJECT_INFO_FILE = '/vol/actrec/MobiAct_Dataset/Readme.txt'


WINDOW_SIZE = 200
STRIDE = 50

activities_id={'STD':0, 'WAL':1, 'JOG':2, 'JUM':3, 'STU':4, 'STN':5, 'SCH':6, 'CSI':7, 'CSO':8}
subject_id={'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8, '10':9, 
            '11':10, '12':11, '16':12, '18':13, '19':14, '20':15, 
            '21':16, '22':17, '23':18, '24':19, '25':20, '26':21, '27':22, '29':23,
            '32':24, '33':25, '35':26, '36':27, '37':28, '38':29, '39':30, '40':31, 
            '41':32, '42':33, '43':34, '44':35, '45':36, '46':37, '47':38, '48':39, '49':40, '50':41, 
            '51':42, '52':43, '53':44, '54':45, '55':46, '56':47, '58':48, '59':49, '60':50,
            '61':51, '62':52, '63':53, '64':54, '65':55, '66':56, '67':57}
act_record={'STD':1, 'WAL':1, 'JOG':3, 'JUM':3, 'STU':6, 'STN':6, 'SCH':6, 'CSI':6, 'CSO':6}

def read_subject_info(file_path):
    """
    Reads subject information from a file and returns a pandas DataFrame.

    Args:
        file_path (str): The path to the file containing the subject information.

    Returns:
        pandas.DataFrame: A DataFrame containing the subject information, with columns for subject ID, age, height, weight, and gender.
    """
    with open(file_path, 'r', encoding='latin1') as file:
        strings = file.readlines()
    file.close()
    person_list = []
    for s in strings:
        if 'sub' in s and '|' in s:
            temp = s.split('|')
            temp = [x.strip() for x in temp]
            if len(temp) == 9:
                person_list.append(temp[3:-1])
    columns = ['subject', 'age', 'height', 'weight', 'gender']
    person_info = pd.DataFrame(person_list, columns=columns)
    person_info[['age', 'height', 'weight']] = person_info[['age', 'height', 'weight']].apply(pd.to_numeric)
    person_info['gender'] = pd.Categorical(person_info['gender'], categories=['M', 'F','I'])
    return person_info


def reader_data(path):
    '''
    gets data from csv file
    data contains 30 columns
    the first column corresponds to sample
    the second column corresponds to class label
    the rest 30 columns corresponds to all of the joints (x,y,z) measurements

    returns:
    A dict with the sequence, time and label

    @param path: path to file
    '''
    #annotated file structure: timestamp,rel_time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,azimuth,pitch,roll,label
    print('Getting data from {}'.format(path))
    counter = 0
    IMU_test = []
    time_test = []
    label_test=[]
    with open(path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            try:
                if spamreader.line_num == 1:
                    # print('\n')
                    print(', '.join(row))
                else:
                    time=list(map(float, row[0:2]))
                    time_test.append(time)
                    
                    IMU=list(map(float, row[2:11]))
                    IMU_test.append(IMU)
                    
                    label=[row[11]]
                    label_test.append(label)
                    
            except:
                    print("Error in line {}".format(row))
                    break
    print('shape of the IMU_test')
    print(len(IMU_test))
    imu_data = {'IMU': IMU_test, 'time': time_test, 'label': label_test}
        
    return imu_data



def opp_sliding_window(data_x, data_y, ws, ss, label_pos_end=True):
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
'''
    print("Sliding window: Creating windows {} with step {}".format(ws, ss))

    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    count_l = 0
    idy = 0
    # Label from the end
    if label_pos_end:
        data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])
    else:
        if False:
            # Label from the middle
            # not used in experiments
            data_y_labels = np.asarray(
                [[i[i.shape[0] // 2]] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])
        else:
            # Label according to mode
            try:
                data_y_labels = []
                for sw in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1)):
                    labels = np.zeros((1)).astype(int)
                    count_l = np.bincount(sw[:, 0], minlength=NUM_CLASSES)
                    idy = np.argmax(count_l)
                   
                    labels[0] = idy
                   
                    data_y_labels.append(labels)
                data_y_labels = np.asarray(data_y_labels)
            except:
                print("Sliding window: error with the counting {}".format(count_l))
                print("Sliding window: error with the counting {}".format(idy))
                return np.Inf

            # All labels per window
            data_y_all = np.asarray([i[:] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])

    return data_x.astype(np.float32), data_y_labels.astype(np.uint8), data_y_all.astype(np.uint8)

'''
def divide_x_y(data):
    """
    Segments each sample into features and label

    :param data: numpy integer matrix
        Sensor data
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """
    data_t = data[:, 0]
    data_y = data[:, 1]
    data_x = data[:, 2:]

    return data_t, data_x, data_y


################
# Generate data
#################
def generate_data(ids, sliding_window_length, sliding_window_step, base_directory=None,
                  identity_bool=False, usage_modus='trainval'):
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
    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'
    
    if usage_modus == 'train':
           activities = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'CSI', 'CSO']
           #activities = ['STD',]
    #elif usage_modus == 'val':
     #      activities = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'CSI', 'CSO']
    #elif usage_modus == 'test':
     #      activities = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'CSI', 'CSO']
    
    all_segments = np.empty((0, 9))
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
        
    for act in activities:
        print(act)
        for sub in ids:
            print(sub)
            for recordings in range(1,act_record[act]+1):
                print(recordings)
            
                file_name_data = "{}/{}_{}_{}_annotated.csv".format(act, act, sub, recordings)
                print("\n{}".format(file_name_data))
                try:
                    # getting data
                    print(FOLDER_PATH + file_name_data)
                    data = reader_data(FOLDER_PATH + file_name_data)
                    print("\nFiles loaded in modus {}\n{}".format(usage_modus, file_name_data))
                    
                    IMU=np.array(data['IMU'])
                    all_segments = np.vstack((all_segments, IMU))
                    all_segments = norm_mobi(all_segments)
                    print("\nFiles loaded and normalised")
                    
                except:
                    print("\n1 In loading data,  in file {}".format(FOLDER_PATH + file_name_data))
                    continue
            
            frames = all_segments.shape[0]
            if frames != 0:
                train_no=round(0.70*frames)
                val_no=round(0.15*frames)
                tv= train_no+val_no
                
            print('train and val labels split')
            
            if usage_modus=='trainval':
                X_train = np.vstack((X_train, all_segments[0:train_no,:]))
                act_train = np.append(act_train, np.full((train_no), act))
                id_train = np.append(id_train, np.full((train_no), sub))
                print('done train')
                            
                X_val = np.vstack((X_val, all_segments[train_no:tv,:]))
                act_val = np.append(act_val, np.full((val_no), act))
                id_val = np.append(id_val, np.full((val_no), sub))
                print('done val')
            elif usage_modus=='test':
                X_test = np.vstack((X_test, all_segments[tv:frames,:]))
                act_test = np.append(act_test, np.full((frames-tv), act))
                id_test = np.append(id_test, np.full((frames-tv), sub))
                print('done test')
    
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
        
        print("training data save")
        if usage_modus=='train':
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
                    print("input values are")
                    print(seq.shape)
                    print(act_train[f])
                    print(labelid_train[f])
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
                    print("input values are")
                    print(seq.shape)
                    print(act_val[f])
                    print(labelid_val[f])
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
                    print("input values are")
                    print(seq.shape)
                    print(act_test[f])
                    print(labelid_test[f])
                    obj = {"data": seq, "act_label": act_test[f], "act_labels_all": act_all_test[f], "label": labelid_test[f]}
                
                    f = open(os.path.join(data_dir_test, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.close()

                    counter_seq += 1
                except:
                    raise('\nError adding the seq')
    except:
        print("error in saving") 
    
    '''
               try:
                  # Getting labels and attributes
                  labels = csv_reader.reader_labels(FOLDER_PATH + file_name_label)
                  class_labels = np.where(labels[:, 0] == 7)[0]

                  # Deleting rows containing the "none" class
                  data_x = np.delete(data_x, class_labels, 0)
                  labels = np.delete(labels, class_labels, 0)

                  #data_t, data_x, data_y = divide_x_y(data)
                  #del data_t
               except:
                  print("2 In generating data, Error getting the data {}".format(FOLDER_PATH
                                                                                       + file_name_data))
                  continue
               
               try:
                  data_x = norm_mbientlab(data_x)
               except:
                  print("\n3  In generating data, Plotting {}".format(FOLDER_PATH + file_name_data))
                  continue
              
               try:
                  # checking if annotations are consistent
                  if data_x.shape[0] == data_x.shape[0]:

                      # Sliding window approach
                      print("\nStarting sliding window")
                      X, y, y_all = opp_sliding_window(data_x, labels.astype(int), sliding_window_length,
                                                             sliding_window_step, label_pos_end=False)
                      print("\nWindows are extracted")

                            # Statistics

                      hist_classes = np.bincount(y[:, 0], minlength=NUM_CLASSES)
                      hist_classes_all += hist_classes
                      print("\nNumber of seq per class {}".format(hist_classes_all))
                      
                      for f in range(X.shape[0]):
                          try:

                              sys.stdout.write(
                                        '\r' +
                                        'Creating sequence file number {} with id {}'.format(f, counter_seq))
                              sys.stdout.flush()

                              # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                              seq = np.reshape(X[f], newshape=(1, X.shape[1], X.shape[2]))
                              seq = np.require(seq, dtype=np.float)
                              
                              
                              obj = {"data": seq, "act_label": y[f], "act_labels_all": y_all[f], "label": labels_persons[P]}
                                           
                              file_name = open(os.path.join(data_dir,
                                                                  'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                              pickle.dump(obj, file_name, protocol=pickle.HIGHEST_PROTOCOL)
                              file_name.close()

                              counter_seq += 1

                          except:
                              raise ('\nError adding the seq')

                      print("\nCorrect data extraction from {}".format(FOLDER_PATH + file_name_data))

                      del data
                      del data_x
                      del X
                      del labels
                      del class_labels

                  else:
                      print("\n4 Not consisting annotation in  {}".format(file_name_data))
                      continue
               except:
                   print("\n5 In generating data, No created file {}".format(FOLDER_PATH + file_name_data))
                   print("-----------------\n{}\n{}\n-----------------".format(file_name_data, file_name_label))
                   continue
               
           except KeyboardInterrupt:
               print('\nYou cancelled the operation.')

    return
'''

def generate_CSV(csv_dir, type_file, data_dir):
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

    np.savetxt(csv_dir + type_file, f, delimiter="\n", fmt='%s')
        
    return f


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

    return f


def create_dataset(identity_bool = False):
    '''
    create dataset
    - Segmentation
    - Storing sequences

    @param half: set for creating dataset with half the frequence.
    '''
    
    train_ids=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
            '11', '12', '16', '18', '19', '20', 
            '21', '22', '23', '24', '25', '26', '27', '29',
            '32', '33', '35', '36', '37', '38', '39', '40', 
            '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', 
            '51', '52', '53', '54', '55', '56', '58', '59', '60',
            '61', '62', '63', '64', '65', '66', '67']
    
    base_directory = '/data/nnair/mobiact/'
    
    
    print("Reading subject info...")
    start_time = time.time()
    subject_info = read_subject_info(SUBJECT_INFO_FILE)
    print(f"Subject info read in {time.time() - start_time:.2f} seconds.")
    #print(subject_info)
    generate_data(train_ids, sliding_window_length=200, sliding_window_step=50, base_directory=base_directory, usage_modus='trainval')
    
    
    return


    
'''   
    generate_data(train_ids, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_train, usage_modus='train')
    generate_data(val_ids, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_val, usage_modus='val')
    generate_data(test_ids, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_test, usage_modus='test')
    
    generate_CSV(base_directory, "train.csv", data_dir_train)
    generate_CSV(base_directory, "val.csv", data_dir_val)
    generate_CSV(base_directory, "test.csv", data_dir_test)
    generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)
'''
    

def norm_mobi(data):
    """
    Normalizes all sensor channels
    Zero mean and unit variance

    @param data: numpy integer matrix
    @return data_norm: Normalized sensor data
    """

    max_values= [ 19.53466397, 19.59746992, 19.27203208, 10.006583, 10.008111, 9.938472, 360.0, 179.99995, 89.948]
    min_values= [ -19.5243695, -19.60940892, -18.97174136, -10.007805, -10.008416, -9.990396, -89.79765, -179.9995, -89.852516]
    mean_values=np.array([ 0.265079537, 7.13106528, 0.387973281, -0.0225606363, -0.00302826137,  0.0131514254,  178.629895, -67.8435738, 2.02923485])
    std_values=np.array([ 3.49444826, 6.70119943, 3.31003981, 1.1238746, 1.12533643, 0.72129725, 105.81241608, 58.62837783, 17.58456297])
    
    mean_values = np.reshape(mean_values, [1, 9])
    
    std_values = np.reshape(std_values, [1, 9])

    mean_array = np.repeat(mean_values, data.shape[0], axis=0)
    std_array = np.repeat(std_values, data.shape[0], axis=0)

    max_values = mean_array + 2 * std_array
    min_values = mean_array - 2 * std_array

    data_norm = (data - min_values) / (max_values - min_values)

    data_norm[data_norm > 1] = 1
    data_norm[data_norm < 0] = 0


    return data_norm



if __name__ == '__main__':
    # Creating dataset for LARa Mbientlab
    # Set the path to where the segmented windows will be located
    # This path will be needed for the main.py

    # Dataset (extracted segmented windows) will be stored in a given folder by the user,
    # However, inside the folder, there shall be the subfolders (sequences_train, sequences_val, sequences_test)
    # These folders and subfolfders gotta be created manually by the user
    # This as a sort of organisation for the dataset
    # mbientlab/sequences_train
    # mbientlab/sequences_val
    # mbientlab/sequences_test

    create_dataset()
    # statistics_measurements()
    print("Done")
