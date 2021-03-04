import tensorflow as tf
import numpy as np
import json
import sys, os
sys.path.append(os.path.abspath('..'))

from model import architecture
from helper_functions import scan_loader

class SegmentationModule:
    """
    This class is an interface to a trained Vox2Vox-model intended for simple segmentation of MRI brain scans.
    """
    
    def __init__(self, model_config_path: str = "config.json"):
        """
        Module constructor. Initializes the model.
        
        model_config_path:
            Path of a JSON file containing the model configuration.
        """
        
        # Loads model config from JSON file.
        with open(model_config_path, 'r') as model_config_file:
            config = json.load(model_config_file)
            
        # Saves needed parameters.
        self.model_dim = config["model_dim"]
        self.model_path = config["model_path"]
        self.class_weights = config["class_weights"]
        self.num_classes = config["num_classes"]
        self.num_modalities = config["num_modalities"]
        
        # Initializes Vox2Vox-model.
        self.__init_model()
    
    
    def __init_model(self):
        "Initializes the model architecture and loads trained weights."

        im_shape = (*self.model_dim, self.num_modalities)
        gt_shape = (*self.model_dim, self.num_classes)
        
        self.vox2vox = architecture.vox2vox(im_shape, gt_shape, self.class_weights)
        
        # Creates separate generator, as the discriminator isn't needed.
        #self.generator = vox2vox.Generator()
        self.vox2vox.generator.load_weights(self.model_path)
        
    
    def segment(self, mri_scan: np.array) -> np.array:
        """
        Calculates the brain tumor segmentation for a given MRI scan.
        
        Parameters
        ----------
        mri_scan : np.array
            MRI brain scan to segment. Should have shape (X, Y, Z, number of modalities).
            
        Returns
        -------
        np.array
            Brain tumor segmentation. Has shape (X, Y, Z) and stores the most likely class label for every voxel.
        """
        
        input_shape = mri_scan.shape[:3]
        
        # Creates a batch of size 1 for the single image
        mri_scan = mri_scan.reshape((1, *mri_scan.shape))
        
        # Resizes the image to the model size.
        # TODO: Right now the image is cropped for this - implement a better approach for this, e.g.:
        #   - Scaling the image
        #   - Separating the image into tiles, segmenting each tile and combining the results into one segmentation
        mri_scan, _ = scan_loader.make_size_batch(mri_scan, None, self.model_dim)
       
        # Predicts tumor segmentation for the batch.
        # gen_pred: 1 x X x Y x Z x num_classes
        gen_pred = self.vox2vox.generator.predict(mri_scan)
    
        # Transforms segmentation back to original dimensions.
        _, gen_pred = scan_loader.make_size_batch(None, gen_pred, input_shape)
        
        # Finds the most likely class for every voxel.
        seg = np.argmax(gen_pred[0], axis=-1).astype('float32')
        
        # Replaces label 3 with label 4.
        seg[seg == 3] = 4
        
        # Applies ET Threshold
        if np.sum(seg == 4) < 1500:
            seg[seg == 4] = 1
        
        # Applies NT Threshold
        if np.sum(seg == 1) < 500:
            seg[seg == 1] = 2
            
        return seg