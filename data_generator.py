# Tensorflow
import numpy as np
import tensorflow.keras.utils
import scan_loader

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras.'
    
    def __init__(self, list_IDs, shuffle=True, batch_size=4, 
                 input_dim=(240,240,160), output_dim=None,
                 n_channels=4, n_classes=4, categorical_classes=True,
                 ground_truth=True, preprocessed=True):
        """
        Creates a new data generator for the given data.
        
        Parameters
        ----------
        list_IDs : list[tuple]
            List of MRI scan path. Each list element is a tuple containing the path of every modality and optionally 
            the segmentation, e.g.: (t1 path, t2 path, t1 contrasted path, flair path, segmentation path).
            If it contains a segmentation, it must be the last element of the tuple.
            
        shuffle : bool, default=True
            If True, the elements of the dataset are returned in random order. They are reshuffled after every epoch.
            
        batch_size : int, default=4
            The number of MRI images returned by the DataGenerator in every batch.
            
        input_dim : (int,int,int), default=(240,240,160)
            The dimensions of the input image in shape (x,y,z). The images in the BraTS dataset have a size of 240x240x155
            by default, but our data augmentation increases the size to 240x240x160 because 155 can't be divided by 2 and this
            leads to errors in the model construction.
            
        output_dim : (int,int,int), default=None
            The dimensions of the images returned by the data generator. If the output dimension is smaller than the input
            dimension, the image is cropped automatically. If the given output dimension is the same or None, the dimensions
            of the input image are kept.
            
        n_channels : int, default=4
            The number of channels in every MRT image. Each channel represents one modality.
            For our project there were four modalities: t1, t2, t1 contrasted and flair.
            
        n_classes : int, default=4
            The number of classes in the ground truth segmentation.
            For our project, there were four classes: No tumor, edema, enhancing tumor and necrotic tissue.
            
        categorical_classes : bool, default=True
            If this is True, each class in the ground truth data is represented by a separate binary channel.
                --> For our project: 4 channels with the value 1 at every voxel that contains that class.
            Otherwise, there is only one channel containing all labels.
                --> For our project: 1 channel with the values 0-3 as described in n_classes.
                
        ground_truth : bool, default=True
            If this is True, the data contains a ground truth segmentation for every image.
            Otherwise, there is no ground truth segmentation and the generator will return None instead of it.
            
        preprocessed : bool, default=True
            If this is True, the data has been preprocessed already and the generator only loads, crops and returns it.
            Otherwise, the data is normalized upon loading it.
        """
        
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Checks if input and output dimensions are the same.
        if self.input_dim == self.output_dim or self.output_dim is None:
            # Same or None: Don't crop
            self.crop_input = False
            self.output_dim = self.input_dim
        else:
            # Different: Crop
            self.crop_input = True
            #self.crop_offset = tuple(np.subtract(self.dim, self.output_dim) // 2)

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.categorical_classes = categorical_classes
        self.preprocessed = preprocessed
        self.ground_truth = ground_truth
        
        # Shuffles data once at the start.
        self.on_epoch_end()

        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))
    

    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generates indices for the batch.
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Gets list of image paths for each index.
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Loads the data.    
        X, y, IDs = self.__data_generation(list_IDs_temp)
        
        # Optional: Crops the data.
        if self.crop_input:
            X, y = scan_loader.make_size_batch(X, y, self.output_dim)
            # X, y = self.__data_cropping(X, y)
        
        # If end is reached: Resets generator.
        if index == self.__len__()-1:
            self.on_epoch_end()
        
        return X, y, IDs


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
  
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        
        # Initialization of return types.
        X = np.empty((self.batch_size, *self.input_dim, self.n_channels))
        y = np.empty((self.batch_size, *self.input_dim))
        patient_ids = [''] * len(list_IDs_temp)

        # Loads data. Only normalizes if data is not preprocessed.
        for i, IDs in enumerate(list_IDs_temp):
            X[i], y[i], patient_ids[i] = scan_loader.load_img(IDs, ground_truth = self.ground_truth, 
                                                              normalize=not self.preprocessed)            
        
        # Converts the ground truth to categorical data if necessary.
        if self.ground_truth and self.categorical_classes:
            y = tensorflow.keras.utils.to_categorical(y, self.n_classes)
        
        if self.ground_truth:
            return X.astype('float32'), y, patient_ids
        else:
            return X.astype('float32'), None, patient_ids
        

    def __data_cropping(self, X, Y):
        'Crops the data to the given dimension'
        
        X_crop = np.empty((X.shape[0], *self.output_dim, self.n_channels))
        Y_crop = np.empty((X.shape[0], *self.output_dim))
        
        for i in range(X.shape[0]):
            X_crop[i], Y_crop[i] = scan_loader.crop_img(X[i], Y[i], self.output_dim, self.crop_offset) 
            
        return X_crop, Y_crop
  