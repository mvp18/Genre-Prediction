import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_dict, list_IDs, labels_dict, batch_size, shuffle):

        self.data_dict = data_dict
        self.batch_size = batch_size
        self.labels_dict = labels_dict
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        #'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def __data_generation(self, list_IDs_temp):
        #'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Generate data
        for ID in enumerate(list_IDs_temp):
            # Store sample
            X = self.data_dict[ID]
            # Store class
            y = self.labels_dict[ID]

        return X, y