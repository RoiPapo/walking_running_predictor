import pandas as pd
import glob
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

from sklearn.model_selection import train_test_split
from joblib import dump, load


def evaluate_pred_reg(pred, y_val):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    """
    Evaluate the performance of a regression model.

    Args:
    - pred: Predicted values (numpy array)
    - y_val: True target values (numpy array)

    Returns:
    - mae: Mean Absolute Error
    - mse: Mean Squared Error
    - rmse: Root Mean Squared Error
    - r2: R-squared
    """
    y_val = [float(n) for n in y_val]
    mae = mean_absolute_error(y_val, pred)
    mse = mean_squared_error(y_val, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, pred)
    print("mae: ", mae , "mse: ", mse, "rmse: ", rmse, "r2: ", r2)


    # return mae, mse, rmse, r2
def evaluate_pred(pred, y_val):
    accuracy = metrics.accuracy_score(y_val, y_pred=pred)
    precision = metrics.precision_score(y_val, y_pred=pred)
    recall = metrics.recall_score(y_val, y_pred=pred)
    F1 = 2 * ((precision * recall) / (precision + recall))
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
        "FSR": []
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


def random_forest_binary_classifier(x_train, y_train, x_val, y_val,ai_feature):
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_val)
    evaluate_pred(y_pred, y_val)
    dump(rf, f'{ai_feature}.joblib')

def create_dataset_from_csv_list(csv_files,train =False):
    all_data = []
    labels = []
    good_fratures = ["yashar", "strong", "stable", "accurate"]
    for file_path in csv_files:
        specimen_data = process_csv(file_path)
        all_data.append(specimen_data)
        label = file_path.split("\\")[-1].split("_")[0]
        if train:
            if label in good_fratures:
                labels.append(1)
            else:
                labels.append(0)

    all_files_features = []
    for i in range(len(all_data)):
        features_of_one_file = []
        for key, value in all_data[i].items():
            if key != "time":
                data = np.array(value).astype(np.float64)
                peaks, _ = find_peaks(data)
                num_peaks = len(peaks)
                avg = np.mean(data)
                maximum = np.max(data)

                features_of_one_file.append(num_peaks)
                features_of_one_file.append(avg)
                features_of_one_file.append(maximum)

        sample_rate = np.mean(list(all_data[i].values())[0])
        num_samples = len(list(all_data[i].values())[-1])
        feature_vec_of_file_i = [sample_rate] + features_of_one_file + [num_samples]
        all_files_features.append(np.array(feature_vec_of_file_i).astype(np.float64))
    return np.array(all_files_features),labels


def train():


    # Example list of CSV file paths
    for ai_feature in ["yashar_test", "strongness_test", "shaky_test", "accuracy_test"]:
        csv_files = glob.glob(f'shoots_ds/{ai_feature}/**')
        random.Random(4).shuffle(csv_files)
        all_files_features, labels =create_dataset_from_csv_list(csv_files,train=True)
        # Initialize a list to hold the data from all files



        # Process each file and append the result to `all_data`

        x_train, x_val, y_train, y_val = train_test_split(np.array(all_files_features), labels, test_size=0.2,
                                                          random_state=42)
        print(f"$$$$$$$$$$$---{ai_feature}---$$$$$$$$$$$")
        random_forest_binary_classifier(np.array(all_files_features), labels, x_val, y_val,ai_feature)

        print("moo")
    # preds =tree_fitter(x_train, y_train, x_val, y_val)
    # print(preds)

def predict_using_RandomForestRegressor(x_train, y_train, x_val, y_val):
    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_val)
    evaluate_pred_reg(y_pred, y_val)


def train2(ai_features):
    csv_files = glob.glob(f'shoots_ds//**.csv')
    all_files_features, labels = create_dataset_from_csv_list(csv_files, train=True)
    labels = [csv_name.split("_")[-1][:-4] for csv_name in csv_files]
    # Initialize a list to hold the data from all files
    for extra_features in ai_features:
        all_files_features = np.concatenate((all_files_features, extra_features[:, np.newaxis]), axis=1)

    # Process each file and append the result to `all_data`

    x_train, x_val, y_train, y_val = train_test_split(np.array(all_files_features), labels, test_size=0.2,
                                                      random_state=42)
    predict_using_RandomForestRegressor(np.array(all_files_features), labels, x_val, y_val)
    # print(f"$$$$$$$$$$$---{ai_feature}---$$$$$$$$$$$")
    # random_forest_binary_classifier(np.array(all_files_features), labels, x_val, y_val, ai_feature)


    print("moo")

def test(csv_src):

    for ai_feature in ["yashar_test", "strongness_test", "shaky_test", "accuracy_test"]:
        rf = load(f'{ai_feature}.joblib')
        all_data=[]
        csv_files = glob.glob(f'shoots_ds/**.csv')
        # csv_files = [csv_src]
        all_files_features, labels = create_dataset_from_csv_list(csv_files, train=False)
        y_pred = rf.predict(all_files_features)

        print(f"$$$$$$$$$$$---{ai_feature}---$$$$$$$$$$$")
        # print(y_pred)
        yield y_pred




if __name__ == '__main__':
    # train()
    csv_file = glob.glob(f'shoots_ds/**.csv')
    preds= list(test(csv_file))
    train2(preds)
