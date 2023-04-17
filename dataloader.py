from typing import Tuple
import os

import numpy as np
import pandas as pd

X_TEMPLATE = "subject_{}_{}__x.csv"
X_TIME_TEMPLATE = "subject_{}_{}__x_time.csv"
Y_TEMPLATE = "subject_{}_{}__y.csv"
Y_TIME_TEMPLATE = "subject_{}_{}__y_time.csv"

X_HEADER = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
Y_HEADER = ["label"]


class Dataloader:

    @staticmethod
    def read_and_join_subject_session_data(data_path, subject_id, session_number):

        x_path = os.path.join(data_path, X_TEMPLATE.format(subject_id, session_number))
        x_time_path = os.path.join(data_path, X_TIME_TEMPLATE.format(subject_id, session_number))
        y_path = os.path.join(data_path, Y_TEMPLATE.format(subject_id, session_number))
        y_time_path = os.path.join(data_path, Y_TIME_TEMPLATE.format(subject_id, session_number))

        x_data = pd.read_csv(x_path, names=X_HEADER)
        x_time = pd.read_csv(x_time_path, names=["time"])

        y_data = pd.read_csv(y_path, names=Y_HEADER)
        y_time = pd.read_csv(y_time_path, names=["time"])

        x = pd.concat([x_data, x_time], axis=1)
        y = pd.concat([y_data, y_time], axis=1)
        y = y.reset_index(drop=False).rename(columns={"index": "timestamp"})
        return x, y

    def read_and_join_data(self, data_path, subject_ids, session_numbers):
        combined_x = []
        combined_y = []
        for subject_id in subject_ids:
            for session_number in session_numbers:
                X, y = self.read_and_join_subject_session_data(data_path, subject_id, session_number)
                combined_x.append(X)
                combined_y.append(y)
        return pd.DataFrame(combined_x,
                            columns=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "time"]), pd.DataFrame(
            combined_y, columns=["labels", "time"])
