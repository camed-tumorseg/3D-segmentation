### HARDWARE
# Comma-separated list of GPUs to be used, or -1 to use CPU
gpu = '0'


### OUTPUT
output_path = '/mnt/Data/Vox2vox_output'
log_file = 'vox2vox.log'


### DATASET
dataset_mask_train = '/mnt/Data/Datasets/BRATS_2020/TrainingData_Augmented/*/'
dataset_mask_test = '/mnt/Data/Datasets/BRATS_2020/MICCAI_BraTS2020_ValidationData/*/'

num_classes = 4
num_channels = 4

dataset_dim = (240, 240, 155)


### AUGMENTATION
aug_export_path = 'TrainingData_Augmented_v4'

num_augmentations = 2

augmented_dim = (240, 240, 160)


### TRAINING
num_epochs = 100
batch_size = 4

continue_training = False
make_new_splits = True
export_images = False

model_dim = (128, 128, 128)


### EVALUATION
eval_export_path = "export"