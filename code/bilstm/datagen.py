import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self, mode, data_dict, list_IDs, labels_dict, num_classes, batch_size, shuffle):

        self.mode = mode
        self.data_dict = data_dict
        self.batch_size = batch_size
        self.labels_dict = labels_dict
        self.list_IDs = list_IDs
        self.num_classes = num_classes
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
        # Generate data
        if self.mode == 'train':

            y = np.empty((self.batch_size, self.num_classes), dtype=int)

            X=[]

            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                X.append(self.data_dict[ID])
                # Store class
                y[i] = self.labels_dict[ID]

            sent_lengths = [embedding.shape[0] for embedding in X]

            max_len = max(sent_lengths)

            padded_X = np.zeros([len(X), max_len, 4096], dtype='float32')

            for i, x_len in enumerate(sent_lengths):
                
                padded_X[i][:x_len] = X[i]

            return padded_X, y

        elif self.mode == 'val':

            assert len(list_IDs_temp)==1, "VAL_BATCH_SIZE is not 1."

            X = self.data_dict[list_IDs_temp[0]]

            y = self.labels_dict[list_IDs_temp[0]]

            return np.expand_dims(X, axis=0), y
