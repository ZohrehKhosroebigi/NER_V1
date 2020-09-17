import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X_train, Y_train, batch_size,shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = Y_train
        self.list_IDs = X_train
        self.f_genarator = open("Raw_Data/generator.txt", "w")
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((self.list_IDs.shape[0]) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #self.f_genarator.write("indexes"+str(indexes)+"\n")

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #self.f_genarator.write("list_IDs_temp"+str(list_IDs_temp)+"\n")

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        #self.f_genarator.write(str(X) + "\n")
        #self.f_genarator.write("\n" + "+++++++++++++++++++++" + "\n")
        print("+" * 10)
        print(X)
        print(y)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.list_IDs.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batchsize samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        X = np.empty([self.batch_size, 30])
        y = np.empty([self.batch_size, 30,14])
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = (self.list_IDs[i])
            # Store class
            y[i,:,:] = self.labels[i]

        self.f_genarator.write(str(X)+"\n")
        self.f_genarator.write("\n" + "*************************" + "\n")

        return X, y


"""from keras.models import Sequential

from Data_generator import DataGenerator

history = model_.fit_generator(generator=DataGenerator(x_train, y_train, batch_size=batchsize, shuffle=True),
                               steps_per_epoch=x_train.shape[0] // batchsize, epochs=epoch)"""