### HARDWARE
# Comma-separated list of GPUs to be used, or -1 to use CPU
gpu = '0'


### OUTPUT
output_path = '/mnt/Data/Vox2vox_output'
log_file = '/mnt/Data/Vox2vox_output/vox2vox.log'


### DATASET
dataset_mask_train = '/mnt/Data/Datasets/BRATS_2020/MICCAI_BraTS2020_TrainingData/*/'
dataset_mask_test = '/mnt/Data/Datasets/BRATS_2020/MICCAI_BraTS2020_ValidationData/*/'

num_classes = 4
num_channels = 4

dataset_dim = (240, 240, 155)


### AUGMENTATION
aug_export_path = '/mnt/Data/Datasets/BRATS_2020/TrainingData_Augmented'

num_augmentations = 1

augmented_dim = (240, 240, 160)


### TRAINING
num_epochs = 100
batch_size = 4

continue_training = True
make_new_splits = False
export_images = False

kfold_validation = True

#model_dim = (128, 128, 128)
#model_dim = (240, 240, 160)
model_dim = (160, 192, 128)


### EVALUATION
weight_folder = "/mnt/Data/Vox2vox_output/models/Vox2Vox_192_bs8_epoch100/output"
eval_export_path = "export"


### ENSEMBLE
use_ensemble = False
ensemble_folder = "/mnt/Data/Vox2vox_output/ensemble"