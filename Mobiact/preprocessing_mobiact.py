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
import csv_reader
from sliding_window import sliding_window
import pickle

# folder path
FOLDER_PATH = "/vol/actrec/MobiAct_Dataset/"

#PERSONS = ['S14']

SENSORS = ['LA', 'LL', 'N', 'RA', 'RL', 'T']

SCENARIO = {'R01': 'L01', 'R02': 'L01', 'R03': 'L02', 'R04': 'L02', 'R05': 'L02', 'R06': 'L02', 'R07': 'L02',
            'R08': 'L02', 'R09': 'L02', 'R10': 'L02', 'R11': 'L02', 'R12': 'L02', 'R13': 'L02', 'R14': 'L02',
            'R15': 'L02', 'R16': 'L02', 'R17': 'L03', 'R18': 'L03', 'R19': 'L03', 'R20': 'L03', 'R21': 'L03',
            'R22': 'L03', 'R23': 'L03', 'R24': 'L03', 'R25': 'L03', 'R26': 'L03', 'R27': 'L03', 'R28': 'L03',
            'R29': 'L03', 'R30': 'L03'}


labels_persons = {"S07": 0, "S08": 1, "S09": 2, "S10": 3, "S11": 4, "S12": 5, "S13": 6, "S14": 7}

NUM_CLASSES = 8
NUM_ATTRIBUTES = 19



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

    print('Getting data from {}'.format(path))
    counter = 0
    IMU = []
    time = []
    data = []
    with open(path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            try:
                try:
                    if spamreader.line_num == 1:
                        # print('\n')
                        print(', '.join(row))
                    else:
                        if len(row) != 31:
                            idx_row = 0
                            IMU.append(row[idx_row])
                            idx_row += 1
                        else:
                            idx_row = 0
                        try:
                            time_d = datetime.datetime.strptime(row[idx_row], '%Y-%m-%d %H:%M:%S.%f')
                            idx_row += 1
                        except:
                            try:
                                time_d = datetime.datetime.strptime(row[idx_row.astype(int)], '%Y-%m-%d %H:%M:%S')
                                idx_row += 1
                            except:
                                print("strange time str {}".format(time_d))
                                continue
                        time.append(time_d)
                        data.append(list(map(float, row[idx_row:])))
                except:
                    print("Error in line {}".format(row))
            except KeyboardInterrupt:
                print('\nYou cancelled the operation.')

    if len(row) != 31:
        imu_data = {'IMU': IMU, 'time': time, 'data': data}
    else:
        imu_data = {'time': time, 'data': data}
        data_new=np.asarray(data)
        print(data_new.shape)
        
    return data_new



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
           persons = [ "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]
    elif usage_modus == 'val':
           persons = ["S07", "S08", "S09", "S10", "S11", "S12", "S13","S14"]
    elif usage_modus == 'test':
           persons = ["S07", "S08", "S09", "S10", "S11", "S12", "S13","S14"]
    #persons = ["S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]
    
    ID = {"S07": 0, "S08": 1, "S09": 2, "S10": 3, "S11": 4, "S12":5, "S13": 6, "S14": 7}
    train_ids = ["R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", 
                 "R13", "R14", "R16", "R17", "R18", "R19", "R20", "R21", "R22", "R23", 
                 "R24", "R25", "R26", "R27", "R28", "R29", "R30"]
    val_ids = ["R11","R12"]
    test_ids = ["R15"]

    counter_seq = 0
    hist_classes_all = np.zeros((NUM_CLASSES))

    for P in persons:
       if usage_modus == 'train':
           recordings = train_ids
       elif usage_modus == 'val':
           recordings = val_ids
       elif usage_modus == 'test':
           recordings =  test_ids 
            
       print("\nModus {} \n{}".format(usage_modus, recordings))
       for R in recordings:
           try:
               S = SCENARIO[R]
               file_name_data = "{}/{}_{}_{}.csv".format(P, S, P, R)
               file_name_label = "{}/{}_{}_{}_labels.csv".format(P, S, P, R)
               print("\n{}\n{}".format(file_name_data, file_name_label))
               try:
                  # getting data
                  data = reader_data(FOLDER_PATH + file_name_data)
                  print("\nFiles loaded in modus {}\n{}".format(usage_modus, file_name_data))
                  
                  data_x = data
                  
                  print("\nFiles loaded")
               except:
                  print("\n1 In loading data,  in file {}".format(FOLDER_PATH + file_name_data))
                  continue

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
    train_ids = ["R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", 
                 "R13", "R14", "R16", "R17", "R18", "R19", "R20", "R21", "R22", "R23", 
                 "R24", "R25", "R26", "R27", "R28", "R29", "R30"]
    val_ids = ["R11","R12"]
    test_ids = ["R15"]

    #all_data = ["S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]

    # general_statistics(train_ids)

    # base_directory = '/path_where_sequences_will_ve_stored/mbientlab_10_persons/'
    # base_directory = '/path_where_sequences_will_ve_stored/mbientlab_50_persons/'
    # base_directory = '/path_where_sequences_will_ve_stored/mbientlab_10_recordings/'
    #base_directory = '/path_where_sequences_will_ve_stored/mbientlab_50_recordings/'
    #base_directory = '/data/nnair/trial/imu_all/'
    base_directory = '/data/nnair/idimuall/'
    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'

    generate_data(train_ids, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_train, usage_modus='train')
    generate_data(val_ids, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_val, usage_modus='val')
    generate_data(test_ids, sliding_window_length=100, sliding_window_step=12, data_dir=data_dir_test, usage_modus='test')

    generate_CSV(base_directory, "train.csv", data_dir_train)
    generate_CSV(base_directory, "val.csv", data_dir_val)
    generate_CSV(base_directory, "test.csv", data_dir_test)
    generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)

    return


def statistics_measurements():
    '''
    Compute some statistics of the duration of the sequences data:

    print:
    Max and Min durations per class or attr
    Mean and Std durations per class or attr

    @param
    '''

    train_final_ids = ["P07", "P08", "P09", "P10", "P11", "P12"]

    persons = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11", "P12", "P13", "P14"]
    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]

    counter_seq = 0
    hist_classes_all = np.zeros((NUM_CLASSES))

    g, ax_x = plt.subplots(2, sharex=False)
    line3, = ax_x[0].plot([], [], '-b', label='blue')
    line4, = ax_x[1].plot([], [], '-b', label='blue')
    accumulator_measurements = np.empty((0,30))
    for P in persons:
        if P not in train_final_ids:
            print("\n6 No Person in expected IDS {}".format(P))
        else:
            for r, R in enumerate(recordings):
                S = SCENARIO[r]
                file_name_data = "{}/{}_{}_{}.csv".format(P, S, P, R)
                file_name_label = "{}/{}_{}_{}_labels.csv".format(P, S, P, R)
                print("------------------------------\n{}\n{}".format(file_name_data, file_name_label))
                try:
                    # getting data
                    data = reader_data(FOLDER_PATH + file_name_data)
                    data_x = data["data"]
                    accumulator_measurements = np.append(accumulator_measurements, data_x, axis = 0)
                    print("\nFiles loaded")
                except:
                    print("\n1 In loading data,  in file {}".format(FOLDER_PATH + file_name_data))
                    continue

    try:
        max_values = np.max(accumulator_measurements, axis=0)
        min_values = np.min(accumulator_measurements, axis=0)
        mean_values = np.mean(accumulator_measurements, axis=0)
        std_values = np.std(accumulator_measurements, axis=0)
    except:
        max_values = 0
        min_values = 0
        mean_values = 0
        std_values = 0
        print("Error computing statistics")
    return max_values, min_values, mean_values, std_values


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
