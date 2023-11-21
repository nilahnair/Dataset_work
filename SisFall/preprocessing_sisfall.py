import re
import os
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from multiprocessing import Pool



# Configuration Parameters
DATA_DIR = '/vol/actrec/SisFall_dataset/' 
README_FILE_PATH = os.path.join(DATA_DIR, 'Readme.txt')
SUBJECT_IDS = ['SA01', 'SA02', 'SA03', 'SA04', 'SA05', 'SA06', 'SA07', 'SA08', 'SA09', 'SA10', 'SA11', 'SA12', 
               'SA13', 'SA14', 'SA15', 'SA16', 'SA17', 'SA18', 'SA19', 'SA20', 'SA21', 'SA22', 'SA23', 'SE01', 'SE02',
               'SE03', 'SE04', 'SE05', 'SE06', 'SE07', 'SE08', 'SE09', 'SE10', 'SE11', 'SE12', 'SE13', 'SE14', 'SE15']
WINDOW_SIZE = 200
STRIDE = 50
SOFT_BIOMETRICS = ['age', 'height', 'weight', 'gender']

def get_person_info(subject_id):
    # Open the readme file and read the lines
    file_path = DATA_DIR +'Readme.txt'
    print(file_path)
    with open(file_path, 'r', encoding='latin1') as file:
        strings = file.readlines()

    # Parse the person information for the given person ID
    person_list = []
    for s in strings:
        print('subject_id',s)
        if re.match('^\| {subject_id}', s):   
            temp = s.split('|')
            temp = [x.strip() for x in temp]
            if len(temp) == 7:
                person_list.append(temp[1:-1])
               # If person_list is empty, return an empty DataFrame
            if not person_list:
                return pd.DataFrame(columns=['subject', 'age', 'height', 'weight', 'gender'])

            # Create a DataFrame with the person information
            columns = ['subject', 'age', 'height', 'weight', 'gender']
            person_info = pd.DataFrame(person_list, columns=columns)

            # Convert the age, height, and weight columns to numeric values
            person_info[['age', 'height', 'weight']] = person_info[['age', 'height', 'weight']].apply(pd.to_numeric)

            # Encode the gender column as a categorical variable and then convert to numeric
            person_info['gender'] = pd.Categorical(person_info['gender'], categories=['M', 'F'])
            person_info['gender'].replace(['M', 'F'], [1, 0], inplace=True)
            print(person_info)

    return person_info  # Returning the DataFrame directly

def process_file(file_path, subject_id):
    segments = [] 
    try:
        # Read the data in chunks and process each chunk individually
        for chunk in pd.read_csv(file_path, header=0, chunksize=1000): 
# Apply sliding window segmentation
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


def main():
    for subject_id in SUBJECT_IDS:
        person_info = get_person_info(subject_id)
    '''
    sensor_cols = ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x', 'MMA8451Q_y', 'MMA8451Q_z']
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