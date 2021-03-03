import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, Dropout, ReLU, LeakyReLU, Concatenate
from tensorflow.keras.optimizers import Adam

import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

import matplotlib.image as mpim

import numpy as np
import os
import time
import csv

from print2file import *

class vox2vox():
    def __init__(self, img_shape, seg_shape, class_weights, Nfilter_start=64, depth=4, LAMBDA=5, batch_num = -1, output_path = None, save_images=False):
        """
        Initializes the Vox2Vox model.
        
        Parameters
        ----------
        img_shape : tuple of int
            Dimensions of the input image. Usually this is (x, y, z, number of modalities).
            
        seg_shape : tuple of int
            Dimensions of the ground truth segmentation. Usually this is (x, y, z, number of classes).
            
        class_weights : numpy array
            Initial class weights.
            
        NFilter_start : int, default=64
            Number for defining encoder/decoder layers.
            
        depth : int, default=4
            Number of encoder/decoder layers in the U-Net.
            
        LAMBDA : float, default=5
            Weight of the loss of the models to one another. (?)
            
        batch_num : int, default=-1
            Maximum number of batches to go through per training and validation epoch. Only recommended for debugging.
            If set to -1, all data will be used, no matter how many batches it takes.
            
        output_path : str, default=None
            Path at which model weights, epoch losses etc. are saved.
            
        save_images : bool, default=False
            If true, one image per epoch will be saved showing the ground truth and predicted segmentation of a single slice.   
        """
        
        
        self.img_shape = img_shape
        self.seg_shape = seg_shape
        self.class_weights = class_weights
        self.Nfilter_start = Nfilter_start
        self.depth = depth
        self.LAMBDA = LAMBDA
        self.batch_num = batch_num
        
        def diceLoss(y_true, y_pred, w=self.class_weights):
            y_true = tf.convert_to_tensor(y_true, 'float32')
            y_pred = tf.convert_to_tensor(y_pred, y_true.dtype)

            num = tf.math.reduce_sum(tf.math.multiply(w, tf.math.reduce_sum(tf.math.multiply(y_true, y_pred), axis=[0,1,2,3])))
            den = tf.math.reduce_sum(tf.math.multiply(w, tf.math.reduce_sum(tf.math.add(y_true, y_pred), axis=[0,1,2,3])))+1e-5

            return 1-2*num/den

        # Build and compile the discriminator
        self.discriminator = self.Discriminator()
        self.discriminator.compile(loss='mse', optimizer=Adam(2e-4, beta_1=0.5), metrics=['accuracy'])

        # Construct Computational Graph of Generator
        # Build the generator
        self.generator = self.Generator()

        # Input images and their conditioning images
        seg = Input(shape=self.seg_shape)
        img = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        seg_pred = self.generator(img)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([seg_pred, img])

        # Combined model.
        self.combined = Model(inputs=[seg, img], outputs=[valid, seg_pred])
        self.combined.compile(loss=['mse', diceLoss], loss_weights=[1, self.LAMBDA], optimizer=Adam(2e-4, beta_1=0.5))
        
        self.output_path = output_path
        self.save_images = save_images
        
    
    def Generator(self):
        '''
        Generator model
        '''

        inputs = Input(self.img_shape, name='input_image')     

        def encoder_step(layer, Nf, inorm=True):
            x = Conv3D(Nf, kernel_size=4, strides=2, kernel_initializer='he_normal', padding='same')(layer)
            if inorm:
                x = InstanceNormalization()(x)
            x = LeakyReLU()(x)
            
            return x
        
        def bottlenek(layer, Nf):
            x = Conv3D(Nf, kernel_size=4, strides=2, kernel_initializer='he_normal', padding='same')(layer)
            x = InstanceNormalization()(x)
            x = LeakyReLU()(x)
            for i in range(4):
                y = Conv3D(Nf, kernel_size=4, strides=1, kernel_initializer='he_normal', padding='same')(x)
                x = InstanceNormalization()(y)
                x = Dropout(0.2)(x)
                x = LeakyReLU()(x)
                x = Concatenate()([x, y])
                
            return x

        def decoder_step(layer, layer_to_concatenate, Nf):
            x = Conv3DTranspose(Nf, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(layer)
            x = InstanceNormalization()(x)
            x = ReLU()(x)
            x = Concatenate()([x, layer_to_concatenate])
            return x

        layers_to_concatenate = []
        x = inputs

        # encoder
        for d in range(self.depth-1):
            if d==0:
                x = encoder_step(x, self.Nfilter_start*np.power(2,d), False)
            else:
                x = encoder_step(x, self.Nfilter_start*np.power(2,d))
            layers_to_concatenate.append(x)
        
        # bottlenek
        x = bottlenek(x, self.Nfilter_start*np.power(2,self.depth-1))

        # decoder
        for d in range(self.depth-2, -1, -1): 
            x = decoder_step(x, layers_to_concatenate.pop(), self.Nfilter_start*np.power(2,d))

        # classifier
        last = Conv3DTranspose(4, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal', activation='softmax', name='output_generator')(x)

       # Create model
        return Model(inputs=inputs, outputs=last, name='Generator')

    def Discriminator(self):
        '''
        Discriminator model
        '''
        
        inputs = Input(self.img_shape, name='input_image')
        targets = Input(self.seg_shape, name='target_image')

        def encoder_step(layer, Nf, inorm=True):
            x = Conv3D(Nf, kernel_size=4, strides=2, kernel_initializer='he_normal', padding='same')(layer)
            if inorm:
                x = InstanceNormalization()(x)
            x = LeakyReLU()(x)
            return x

        x = Concatenate()([inputs, targets])

        for d in range(self.depth):
            if d==0:
                x = encoder_step(x, self.Nfilter_start*np.power(2,d), False)
            else:
                x = encoder_step(x, self.Nfilter_start*np.power(2,d))


        last = tf.keras.layers.Conv3D(1, 4, strides=1, padding='same', kernel_initializer='he_normal', name='output_discriminator')(x) 

        return Model(inputs=[targets, inputs], outputs=last, name='Discriminator')
    
    def train_step(self, Xbatch, Ybatch, mp=True, n_workers=16):
        """Performs one single training step on a batch of training data. Returns the overall model loss."""
        
        # Generator output
        gen_output = self.generator.predict(Xbatch, use_multiprocessing=mp, workers=n_workers)
        
        # Discriminator output shape    
        disc_output_shape = self.discriminator.output_shape
        disc_output_shape = (gen_output.shape[0], *disc_output_shape[1:])
        
        # Train Discriminator
        disc_loss_real = self.discriminator.fit([Ybatch, Xbatch], tf.ones(disc_output_shape), verbose=0, use_multiprocessing=mp, workers=n_workers)
        disc_loss_fake = self.discriminator.fit([gen_output, Xbatch], tf.zeros(disc_output_shape), verbose=0, use_multiprocessing=mp, workers=n_workers)

        # Train Generator
        gen_loss = self.combined.fit([Ybatch, Xbatch], [tf.ones(disc_output_shape), Ybatch], 
                                     verbose=0, use_multiprocessing=mp, workers=16)
        
        return gen_loss.history['loss'][0]
    
    def valid_step(self, Xbatch, Ybatch, mp=True, n_workers=16):
        """Evaluates the model using one batch of validation data."""
        
        # Generetor output
        gen_output = self.generator.predict(Xbatch, use_multiprocessing=mp, workers=n_workers)
        
        # Discriminator output shape    
        disc_output_shape = self.discriminator.output_shape
        disc_output_shape = (gen_output.shape[0], *disc_output_shape[1:])
        
        # Train Discriminator
        disc_loss_real = self.discriminator.evaluate([Ybatch, Xbatch], tf.ones(disc_output_shape), verbose=0, use_multiprocessing=mp, workers=n_workers)
        disc_loss_fake = self.discriminator.evaluate([gen_output, Xbatch], tf.zeros(disc_output_shape), verbose=0, use_multiprocessing=mp, workers=n_workers)

        # Train Generator
        gen_loss = self.combined.evaluate([Ybatch, Xbatch], [tf.ones(disc_output_shape), Ybatch], verbose=0, use_multiprocessing=mp, workers=n_workers)
        
        return gen_loss[0]

    
    def train(self, train_generator, valid_generator, nEpochs):
        """Trains the model for nEpochs."""
        
        print2file('Training process:')
        print2file('Training on {} and validating on {} batches.\n'.format(len(train_generator), len(valid_generator)))
        
        # Creates the output directory if it doesn't exist yet.
        if self.output_path is not None:
            path = self.output_path
            if os.path.exists(path)==False:
                os.mkdir(path)
   
        # Initializes history objects.
        train_losses = []
        valid_losses = []
        smallest_loss = np.inf
    
        # Opens CSV file for logging of losses during training.
        with open(path + "/losses.csv", "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")
            csv_writer.writerow(["epoch", "training", "validation"])
        
            # Runs nEpochs epochs.
            for e in range(nEpochs): 
                print2file('Epoch {}/{}'.format(e+1,nEpochs))
                
                start_time = time.time()           


                ### TRAINING
                batch_losses = []
                b = 0
              
                if self.batch_num > 0:
                    total_batch_num = self.batch_num
                else:
                    total_batch_num = len(train_generator)
                
                # Loads one batch of training data per iteration.
                for Xbatch, Ybatch, IDbatch in train_generator:
                    b+=1
                    
                    # Trains model on the batch.
                    gan_loss = self.train_step(Xbatch, Ybatch)
                    batch_losses.append(gan_loss)
                    print2file('Training Batch: {}/{} - v2v_loss: {:.4f}'.format(b, total_batch_num, gan_loss))

                    if b >= self.batch_num and self.batch_num > 0:
                        break

                del(Xbatch, Ybatch) 

                # Calculates average training loss for this epoch.
                train_avg = np.mean(batch_losses)
                print2file('Training Batch Average: v2v_loss: {:.4f}\n'.format(train_avg))


        
                ### VALIDATION
                batch_losses = []
                b = 0
 
                if self.batch_num > 0:
                    total_batch_num = self.batch_num
                else:
                    total_batch_num = len(valid_generator)
                
                # Loads one batch of validation data per iteration.
                for Xbatch, Ybatch, IDbatch in valid_generator:
                    b += 1

                    # Evalutes model on the batch.
                    gan_loss = self.valid_step(Xbatch, Ybatch)
                    batch_losses.append(gan_loss)             
                    print2file('Validation Batch: {}/{} - v2v_loss: {:.4f}'.format(b, total_batch_num, gan_loss))

                    if b >= self.batch_num and self.batch_num > 0:
                        break

                # Calculates average validation loss.
                valid_avg = np.mean(batch_losses)
                print2file('Validation Batch Average: v2v_loss_val: {:.4f}\n'.format(valid_avg))

                # Measures elapsed time for this epoch.
                elapsed_time = time.time() - start_time
                print2file('Elapsed time: {}:{} mm:ss\n'.format(int(elapsed_time//60), int(elapsed_time%60)))


                # Saves loss values to CSV and history files.
                csv_writer.writerow([e, train_avg, valid_avg])
                
                train_losses.append(train_avg)
                valid_losses.append(valid_avg)
                np.save(path + '/history_train', train_losses)
                np.save(path + '/history_valid', valid_losses)

                
                # Generates and saves image for epoch.
                if self.save_images and self.output_path is not None:

                    # Predicts segmentations for one batch.
                    y_pred = self.generator.predict(Xbatch)
                    
                    # Compiles most likely class for every voxel in ground truth and prediction.
                    Ybatch = np.argmax(Ybatch, axis=-1)
                    y_pred = np.argmax(y_pred, axis=-1)

                    # imsize: size of a single image
                    # r: number of rows of images
                    # c: number of columns of images
                    imsize, r, c = 128, 1, 3

                    canvas = np.zeros((r*imsize,c*imsize))
                    
                    # Every row shows one image from the batch.
                    for i in range(r):
                        s = Xbatch[i,:,:,imsize//2,2] 
                        
                        # Column 1: Image of t1ce modality without segmentation
                        canvas[i*imsize : (i+1)*imsize, 0 : imsize] = (s - np.min(s)) / (np.max(s)-np.min(s))
                        
                        # Column 2: Image of ground truth segmentation
                        canvas[i*imsize : (i+1)*imsize, imsize : 2*imsize] = Ybatch[i,:,:,imsize//2]/6
                        
                        # Column 3: Image of predicted segmentation
                        canvas[i*imsize : (i+1)*imsize, 2*imsize : 3*imsize] = y_pred[i,:,:,imsize//2]/6

                    del(Xbatch, Ybatch, IDbatch)
                    
                    # Saves image.
                    fname = (path + '/pred@epoch_{}.png').format(e+1)
                    mpim.imsave(fname, canvas, cmap='gray')

                    
                # Saves model if it has the smallest loss so far.
                if valid_avg < smallest_loss:
                    self.generator.save_weights(path + '/Generator.h5') 
                    self.discriminator.save_weights(path + '/Discriminator.h5') 
                    self.combined.save_weights(path + '/Vox2Vox.h5')

                    # Replaces the worst loss with the new loss value
                    smallest_loss = valid_avg

                    print2file(f"Epoch {e} was best model so far.")
        
        return train_losses, valid_losses