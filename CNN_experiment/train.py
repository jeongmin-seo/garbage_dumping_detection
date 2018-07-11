##########################
# import Library
##########################
import keras
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, Dense, Flatten, Merge, Dropout
import data_loader

#########################################################
#                   tensorboard setup                   #
#########################################################
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
mcCallBack = ModelCheckpoint('./model/{epoch:0}.hdf5', save_best_only=True)
rlCallBack = ReduceLROnPlateau()


class Model():
    # input_shape and num_classes are from datasets1
    # optimizer is Adam()
    def __init__(self, datasets1, nb_filters, nb_kernels, nb_strides):
        self.datasets1 = datasets1
        self.nb_filters = nb_filters
        self.nb_kernels =  nb_kernels
        self.nb_strides = nb_strides

    def make_model(self):
        # total layer # : 3
        _model = Sequential()
        _model.add(Conv1D(filters=self.nb_filters, kernel_size=self.nb_kernels, strides=self.nb_strides,
                          activation='relu', input_shape=self.datasets1.shape))
        _model.add(BatchNormalization())
        _model.add(Conv1D(filters=self.nb_filters * 2, kernel_size=self.nb_kernels,
                          strides=self.nb_strides, activation='relu', padding='same'))
        _model.add(BatchNormalization())
        _model.add(Conv1D(filters=self.nb_filters * 4, kernel_size=self.nb_kernels,
                          strides=self.nb_strides, activation='relu', padding='same'))
        _model.add(BatchNormalization())
        _model.add(Flatten())
        _model.add(Dense(units=500, activation='relu'))
        _model.add(Dense(units=50, activation='relu'))
        _model.add(Dense(units=4, activation='softmax'))
        _model.compile(optimizer='Adam', loss=keras.losses.binary_crossentropy,# keras.losses.categorical_crossentropy,
                       metrics=['accuracy'])

        return _model


if __name__ == '__main__':

    workspace = 'C:/Users/JM/Desktop/Data/ETRIrelated/additional'
    loader = data_loader.DataLoader(workspace, 'train')
    train_data, train_label = loader.run()
    print('Data Load Complete')

    model = Model(train_data[0], 32, 20, 2)
    train_model = model.make_model()
    train_model.summary()
    train_model.fit(train_data, train_label, batch_size=32, epochs=100,
                    callbacks=[tbCallBack, mcCallBack, rlCallBack], validation_split=0.2)


