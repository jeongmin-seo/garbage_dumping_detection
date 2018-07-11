import keras
import data_loader
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np

Model_path = './model/6.hdf5'


def onehot_to_integer(_labels):
    return_label = []
    for label in _labels:
        return_label.append(int(label.argmax()))

    return return_label


if __name__ == '__main__':
    workspace = 'C:/Users/JM/Desktop/Data/ETRIrelated/additional'
    loader = data_loader.DataLoader(workspace, 'test')
    test_data, test_label = loader.run()
    print('Data Load Complete')

    model = load_model(Model_path)

    # result = model.evaluate(test_data, test_label, batch_size=32)
    # print(model.metrics_names)
    # print('Accuracy: %f, Precision: %f, Recall: %f' %result[])

    pred_label = model.predict(test_data, batch_size=32)
    test_label = onehot_to_integer(test_label)
    pred_label = onehot_to_integer(pred_label)

    cnf = confusion_matrix(test_label, pred_label)
    print(cnf)

