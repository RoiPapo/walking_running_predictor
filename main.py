import pandas as pd
import glob
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate_pred(pred, y_val):
    accuracy = metrics.accuracy_score(y_val, y_pred=pred)
    precision = metrics.precision_score(y_val, y_pred=pred)
    recall = metrics.recall_score(y_val, y_pred=pred)
    F1 = 2 * ((precision*recall)/(precision+recall))
    print("acuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("F1: ", F1)
    cm = confusion_matrix(y_val, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


def tree_fitter(x_train, y_train, x_val, y_val):
    print("Xgboost train...")
    params = {
        'min_child_weight': [0.5, 1, 5, 10],
        'gamma': [0.1, 0.5, 1, 1.5, 2, 5],
        'subsample': [0.4, 0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.2],
        'reg_lambda': [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 200],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 6]
    }
    # params = {
    #     'min_child_weight': [1,5],
    #     'gamma': [2],
    #     'subsample': [0.6,0.8],
    #     'reg_alpha': [0.1],
    #     'reg_lambda': [0, 0.1],
    #     'colsample_bytree': [0.6],
    #     'max_depth': [5,6]
    # }

    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                        nthread=1)
    folds = 3
    param_comb = 5

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=16, scoring='roc_auc', n_jobs=4,
                                       cv=skf.split(x_train, y_train), random_state=1001)

    random_search.fit(x_train, y_train)

    predicted_y = random_search.best_estimator_.predict(x_val)

    print("XGBOOST :")
    print("...................")
    evaluate_pred(predicted_y,y_val)

    return predicted_y


def process_csv(file_path):
    # Read the metadata from the CSV file
    with open(file_path, 'r') as file:
        metadata = [next(file).strip().split(",") for _ in range(5)]

    # Extract the activity type and assign a label
    activity_type = metadata[2][1].strip('"')
    actual_steps = int(metadata[3][1].strip('"'))
    label = 0 if activity_type == "Walking" else 1

    # Read the sensor data, skipping the initial metadata
    data = pd.read_csv(file_path, skiprows=5)

    # Initialize the dictionary
    specimen = {
        "time": [np.float64(0)],  # The first time difference is 0
        "ACC X": [],
        "ACC Y": [],
        "ACC Z": [],
        "GYRO X": [],
        "GYRO Y": [],
        "GYRO Z": []
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

    return specimen, label, actual_steps


def main():
    # Example list of CSV file paths
    csv_files = glob.glob('IoT_2024_dataset/**')
    random.Random(4).shuffle(csv_files)

    # Initialize a list to hold the data from all files
    all_data = []
    labels = []
    all_actual_steps = []

    # Process each file and append the result to `all_data`
    for file_path in csv_files:
        try:
            specimen_data, label, actual_steps = process_csv(file_path)
            all_data.append(specimen_data)
            labels.append(label)
            all_actual_steps.append(actual_steps)
        except:
            print(f"Error processing {file_path}")
    all_files_features = []
    for i in range(len(all_data)):
        features_of_one_file = []
        for key, value in all_data[i].items():
            if key != "time":

                data = np.array(value).astype(np.float64)
                peaks, _ = find_peaks(data)
                num_peaks = len(peaks)
                avg=np.mean(data)

                features_of_one_file.append(num_peaks)
                features_of_one_file.append(avg)

        sample_rate = np.mean(list(all_data[i].values())[0])
        num_samples = len(list(all_data[i].values())[-1])
        feature_vec_of_file_i = [sample_rate] + features_of_one_file + [all_actual_steps[i]] + [num_samples]
        all_files_features.append(np.array(feature_vec_of_file_i).astype(np.float64))
    x_train, x_val, y_train, y_val = train_test_split(np.array(all_files_features), labels, test_size=0.3,
                                                      random_state=42)
    preds =tree_fitter(x_train, y_train, x_val, y_val)
    print(preds)

if __name__ == '__main__':
    main()
