'''
Created on Oct 02, 2019

@author: fmoya


'''
import glob

import numpy as np
import csv
import os
import sys
import matplotlib.pyplot as plt
import datetime
import csv

from tqdm import tqdm


# from sliding_window import sliding_window

def sliding_window():
    raise NotImplementedError()


import pickle

# folder path
BASE_DIR = "/vol/actrec/PersonIDDataset/db_release/db_10min/"

ID_2_SUBJECT = {
    0: "5d00cb4666ffc90012433d37",
    1: "5d029a6b1866d90019d127b8",
    2: "5d00dd5e66ffc90012433d3f",
    3: "5d1366774d5e600012a54bf3",
    4: "5d1608a6c1e53d001261fb3b",
    5: "5d15e716c1e53d001261fac0",
    6: "5d15dcf6c1e53d001261f9ca",
    7: "5d1600c8c1e53d001261fb1e",
    8: "5d1b2efd53fefc0012b0a0e9",
    9: "5d1c8fbe53fefc0012b0b5f3",
    10: "5d1b41ad53fefc0012b0a2c6",
    11: "5d1b3f1653fefc0012b0a2b0",
    12: "5d1c797653fefc0012b0b068",
    13: "5d1b3b5b53fefc0012b0a28d",
    14: "5d1f702753fefc0012b0ef76",
    15: "5d1c841e53fefc0012b0b567",
    16: "5d19eaf2c1e53d0012621069",
    17: "5d1c886253fefc0012b0b58f",
    18: "5d19fd5e53fefc0012b095a5",
    19: "5d1cd4c753fefc0012b0bec9",
    20: "5d1ccbbd53fefc0012b0bc07",
    21: "5d19f321c1e53d00126210c6",
    22: "5d19f0b9c1e53d0012621095",
    23: "5d1a002f53fefc0012b095b5",
    24: "5da98051711fbe0013a4be13",
    25: "5dbac1391a65700013675e45",
    90: "testfolder"
}

NUM_CLASSES = 3
SAMPLING_RATE = 20 # Hz

def read_folder(record_file_path):
    sensor_data_out = np.empty((1, 6))
    labels_out = np.empty((1,))

    for file in tqdm(glob.glob(f"{record_file_path}{os.sep}*")):
        sensor_data = np.load(file, allow_pickle=True).item()

        identifier = sensor_data['id']
        acc_data = sensor_data['acc']
        gyr_data = sensor_data['gyr']
        sleep_indicator = sensor_data['sleep']
        step_indicator = sensor_data['step']

        if sleep_indicator == 1 and step_indicator == 0:  # SLEEPING
            label = 0
        elif sleep_indicator == 0 and step_indicator == 1:  # WALKING
            label = 1
        elif sleep_indicator == 0 and step_indicator == 0:  # RANDOM
            label = 2
        else:
            print(
                f"Skipping {file} -> unplausible indicators: Sleep Indicator {sleep_indicator}, Step Indicator {step_indicator}")
            continue

        try:
            # Quick fix for mismatching rows
            min_rows = min(acc_data.shape[0], gyr_data.shape[0]) 
            sensor_data_matrix = np.hstack((acc_data[:min_rows,:], gyr_data[:min_rows,:]))
        except:
            print(f"Skipping {file} -> sensor data dimension mismatch: ACC {acc_data.shape}, GYR {gyr_data.shape}")
            continue

        n_samples = sensor_data_matrix.shape[0]
        label_vector = np.ones((n_samples,)) * label

        sensor_data_out = np.vstack((sensor_data_out, sensor_data_matrix))
        labels_out = np.hstack((labels_out, label_vector))

    return sensor_data_out[1:, :], labels_out[1:]  # skip the first row due to initialization


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
        try:
            data_y_labels = []
            for sw in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1)):
                labels = np.zeros((20)).astype(int)
                count_l = np.bincount(sw[:, 0], minlength=NUM_CLASSES)
                idy = np.argmax(count_l)
                attrs = np.sum(sw[:, 1:], axis=0)
                attrs[attrs > 0] = 1
                labels[0] = idy
                labels[1:] = attrs
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

    subject_to_file_paths = {}

    counter_seq = 0
    hist_classes_all = np.zeros((NUM_CLASSES))

    for subject_id in ids:
        subject_folder = f"{BASE_DIR}{os.sep}db_release{os.sep}db_10min{os.sep}db_{ID_2_SUBJECT[subject_id]}{os.sep}"
        print('subject folder')
        print(subject_folder)
        data_x, labels = read_folder(subject_folder)
        # data_x is a [n x 6] matrix with the first 3 rows as acc and the next 3 rows as gyr
        # labels is a [n,] vector indicating the classes

        #data_x = norm_mbientlab(data_x)

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

                        obj = {"data": seq, "label": y[f], "labels": y_all[f],
                               "identity": labels_persons[P]}
                        file_name = open(os.path.join(data_dir,
                                                      'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                        pickle.dump(obj, file_name, protocol=pickle.HIGHEST_PROTOCOL)
                        file_name.close()

                        counter_seq += 1

                    except:
                        raise ('\nError adding the seq')
                print("\nCorrect data extraction from {}".format(FOLDER_PATH + file_name_data))
            else:
                print("\n4 Not consisting annotation in  {}".format(file_name_data))
                continue
        except:
            print("\n5 In generating data, No created file {}".format(FOLDER_PATH + file_name_data))
        print("-----------------\n{}\n{}\n-----------------".format(file_name_data, file_name_label))


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


def create_dataset(identity_bool=False):
    '''
    create dataset
    - Segmentation
    - Storing sequences

    @param half: set for creating dataset with half the frequence.
    '''
    train_ids = [0]
    val_ids = ["S11", "S12"]
    test_ids = ["S13", "S14"]

    all_data = ["S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]

   
    base_directory = '//data/nnair/datasetbias/personid/baseline/prepros/'

    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'

    generate_data(train_ids, sliding_window_length=200, sliding_window_step=12, data_dir=data_dir_train)
    generate_data(val_ids, sliding_window_length=200, sliding_window_step=12, data_dir=data_dir_val)
    generate_data(test_ids, sliding_window_length=200, sliding_window_step=12, data_dir=data_dir_test)

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
    accumulator_measurements = np.empty((0, 30))
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
                    accumulator_measurements = np.append(accumulator_measurements, data_x, axis=0)
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

    mean_values = np.array([-0.6018319, 0.234877, 0.2998928, 1.11102944, 0.17661719, -1.41729978,
                            0.03774093, 1.0202137, -0.1362719, 1.78369919, 2.4127946, -1.36437627,
                            -0.96302063, -0.0836716, 0.13035097, 0.08677377, 1.088766, 0.51141513,
                            -0.61147614, -0.22219321, 0.41094977, -1.45036893, 0.80677986, -0.1342488,
                            -0.02994514, -0.999678, -0.22073192, -0.1808128, -0.01197039, 0.82491874])
    mean_values = np.reshape(mean_values, [1, 30])

    std_values = np.array([1.17989719, 0.55680584, 0.65610454, 58.42857495, 74.36437559,
                           86.72291263, 1.01306, 0.62489802, 0.70924608, 86.47014857,
                           100.6318856, 61.02139095, 0.38256693, 0.21984504, 0.32184666,
                           42.84023413, 24.85339931, 18.02111335, 0.44021448, 0.51931148,
                           0.45731142, 78.58164965, 70.93038919, 76.34418105, 0.78003314,
                           0.32844988, 0.54919488, 26.68953896, 61.04472454, 62.9225945])
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
