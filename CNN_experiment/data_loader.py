import os
import re
import numpy as np
from keras.utils import to_categorical


class DataLoader():
    def __init__(self, _workspace, _data_type):
        self.workspace = _workspace
        self.data_type = _data_type.lower()

    def set_folder(self):
        return_list = []
        for folder_name in os.listdir(self.workspace):
            # TODO: check dir or file
            if self.data_type in folder_name:
                return_list.append(folder_name)

        if not return_list:
            raise FileNotFoundError

        return return_list


    def get_data_in_folder(self, _folder_name):

        _data = []
        _labels = []
        data_root = os.path.join(self.workspace, _folder_name)
        label_idx = int(re.split('[_]+', _folder_name)[0])
        for file_name in os.listdir(data_root):
            data_path = os.path.join(data_root, file_name)
            _labels.append(to_categorical(int(re.split('[-.]+', file_name)[-2]) * label_idx, num_classes=4))
            _data.append(np.load(data_path))

        return _data, _labels
    """
    def get_data_in_folder(self, _folder_name):

        _data = []
        _labels = []
        data_root = os.path.join(self.workspace, _folder_name)
        for file_name in os.listdir(data_root):
            data_path = os.path.join(data_root, file_name)
            _labels.append(to_categorical(int(re.split('[-.]+', file_name)[-2]), num_classes=2))
            _data.append(np.load(data_path))

        return _data, _labels
    """
    def run(self):
        final_data = []
        final_label = []
        folder_list = self.set_folder()
        for folder_name in folder_list:
            data, label = self.get_data_in_folder(folder_name)
            final_data.extend(data)
            final_label.extend(label)

        return np.asarray(final_data), np.asarray(final_label)

