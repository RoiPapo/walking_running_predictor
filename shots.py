import pandas as pd
import glob
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def evaluate_pred(pred, y_val):
    accuracy = metrics.accuracy_score(y_val, y_pred=pred)
    precision = metrics.precision_score(y_val, y_pred=pred)
    recall = metrics.recall_score(y_val, y_pred=pred)
    F1 = 2 * ((precision*recall)/(precision+recall))
    print("acuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("F1: ", F1)
    # cm = confusion_matrix(y_val, pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.show()




def process_csv(file_path):
    # Read the metadata from the CSV file
    with open(file_path, 'r') as file:
        metadata = [next(file).strip().split(",") for _ in range(1)]


    # Read the sensor data, skipping the initial metadata
    data = pd.read_csv(file_path)

    # Initialize the dictionary
    specimen = {
        "time": [np.float64(0)],  # The first time difference is 0
        "ACC X": [],
        "ACC Y": [],
        "ACC Z": [],
        "GYRO X": [],
        "GYRO Y": [],
        "GYRO Z": [],
        "FSR":[]
    }

    # Populate the dictionary
    for i, row in data.iterrows():
        try:
            if i > 0:  # Calculate time difference for subsequent rows
                time_diff = np.float64(data.at[i, 'Time [sec]']) - np.float64(data.at[i - 1, 'Time [sec]'])
                specimen["time"].append(time_diff)
        except:
            if i > 0:  # Calculate time difference for subsequent rows
                time_diff = np.float64(data.at[i, 'Time[sec]']) - np.float64(data.at[i - 1, 'Time[sec]'])
                specimen["time"].append(time_diff)
        specimen["ACC X"].append(row["ACC X"])
        specimen["ACC Y"].append(row["ACC Y"])
        specimen["ACC Z"].append(row["ACC Z"])
        specimen["GYRO X"].append(row["GYRO X"])
        specimen["GYRO Y"].append(row["GYRO Y"])
        specimen["GYRO Z"].append(row["GYRO Z"])
        specimen["FSR"].append(row["FSR"])

    return specimen


def main():
    # Example list of CSV file paths
    csv_files = glob.glob('shoots_ds/**')
    random.Random(4).shuffle(csv_files)

    # Initialize a list to hold the data from all files
    all_data = []
    labels = []
    all_actual_steps = []

    # Process each file and append the result to `all_data`
    for file_path in csv_files:
        specimen_data = process_csv(file_path)
        all_data.append(specimen_data)


    all_files_features = []
    for i in range(len(all_data)):
        features_of_one_file = []
        for key, value in all_data[i].items():
            if key != "time":

                data = np.array(value).astype(np.float64)
                peaks, _ = find_peaks(data)
                num_peaks = len(peaks)
                avg=np.mean(data)
                maximum = np.max(data)

                features_of_one_file.append(num_peaks)
                features_of_one_file.append(avg)
                features_of_one_file.append(maximum)

        sample_rate = np.mean(list(all_data[i].values())[0])
        num_samples = len(list(all_data[i].values())[-1])
        feature_vec_of_file_i = [sample_rate] + features_of_one_file + [num_samples]
        all_files_features.append(np.array(feature_vec_of_file_i).astype(np.float64))
    x_train, x_val, y_train, y_val = train_test_split(np.array(all_files_features), labels, test_size=0.3,
                                                      random_state=42)
    # preds =tree_fitter(x_train, y_train, x_val, y_val)
    # print(preds)

if __name__ == '__main__':
    main()
