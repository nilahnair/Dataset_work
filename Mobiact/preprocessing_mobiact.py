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
def generate_data(ids, sliding_window_length, sliding_window_step, data_dir=None,
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
    
    if usage_modus == 'train':
           activities = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'CSI', 'CSO']
           #activities = ['STD',]
    #elif usage_modus == 'val':
     #      activities = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'CSI', 'CSO']
    #elif usage_modus == 'test':
     #      activities = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'CSI', 'CSO']
    
    all_segments = np.empty((0, 9))
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
                    
                    print(len(data['IMU']))
                    IMU=np.array(data['IMU'])
                    print(IMU.shape)
                    all_segments = np.vstack((all_segments, IMU))
                    print('new size of all_segments')
                    print(all_segments.shape)
                  
                    print("\nFiles loaded")
                except:
                    print("\n1 In loading data,  in file {}".format(FOLDER_PATH + file_name_data))
                    continue
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
    
    base_directory = '/data/nnair/idimuall/'
    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'
    
    print("Reading subject info...")
    start_time = time.time()
    subject_info = read_subject_info(SUBJECT_INFO_FILE)
    print(f"Subject info read in {time.time() - start_time:.2f} seconds.")
    #print(subject_info)
    generate_data(train_ids, sliding_window_length=200, sliding_window_step=50, data_dir=data_dir_train, usage_modus='train')
    
    
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
    

def norm_mbientlab(data):
    """
    Normalizes all sensor channels
    Zero mean and unit variance

    @param data: numpy integer matrix
    @return data_norm: Normalized sensor data
    """

    mean_values = np.array([-0.56136913,  0.23381773,  0.3838226,   0.79076586,  0.45813304, -0.70334326,
                            0.03523825,  1.00726919, -0.1427787,   0.32435255,  0.55939433, -1.30199178,
                            -0.96324657, -0.09888434,  0.12263245, -0.22261515,  0.8984959,   0.49177392,
                            -0.59227687, -0.24910351,  0.43490187, -0.35732476,  0.8924354,   1.02112235,
                            -0.23097866, -0.85492054, -0.20215291,  0.0394256,   0.11252314,  0.5274977])
    mean_values = np.reshape(mean_values, [1, 30])

    std_values = np.array([0.42772949,  0.52758021,  0.46414677, 57.27246626, 72.281297,   59.67808402,
                           0.48215708,  0.23598994,  0.31527504, 28.65629199, 59.30216666, 58.69912234,
                           0.14558289,  0.21995655,  0.29484591, 39.2756242,  19.63945915, 18.32191831,
                           0.42880226,  0.51087836,  0.42606367, 57.17931987, 74.60050755, 62.19641315,
                           0.68380897,  0.42066544,  0.32898669, 35.61022222, 55.83724424, 59.23920043])
    std_values = np.reshape(std_values, [1, 30])

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
