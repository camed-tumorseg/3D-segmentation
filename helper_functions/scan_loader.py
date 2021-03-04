import os
import numpy as np
import nibabel as nib

# Loads one datapoint consisting of four modalities and one segmented ground truth
def load_img(img_files, ground_truth=True, normalize=True):

    N = len(img_files)

    if ground_truth:
        # Loads segmented ground truth (last image in the given list)
        Y = nib.load(img_files[N-1]).get_fdata(dtype='float32')

        # Ground truth has labels 0,1,2,4
        # 4 is replaces by 3 to remove the gap
        Y[Y==4]=3
    else:
        Y = None

    # Normalizes every MRT-image
    # X_norm = np.empty((Y.shape[0], Y.shape[1], Y.shape[2], 4))
    
    first_elem = True
    
    #X_norm = np.empty((240, 240, 155, 4))
    for channel in range(N-1):
        # Loads the MRT-image
        X = nib.load(img_files[channel]).get_fdata(dtype='float32')
        
        if first_elem:
            first_elem = False
            X_norm = np.empty((X.shape[0], X.shape[1], X.shape[2], 4))
        
        # Selects only brain voxels from array
        brain = X[X!=0] 
        
        # Creates an empty array to store the normalized result in
        brain_norm = np.zeros_like(X) # background at -100 <- ???
        
        if normalize:
            # Z-normalizes the brain array
            # norm = (brain - np.mean(brain))/np.std(brain)
            brain_norm = (X - np.mean(brain))/np.std(brain)
        
            # Writes normalized non-empty voxels to normalized array
            # brain_norm[X!=0] = norm
        else:
            brain_norm = X
        
        # Saves the normalized modality as a channel
        X_norm[:,:,:,channel] = brain_norm        
        
    # Also crops the MRT-image to 160x192x128
    # X_norm = X_norm[40:200,34:226,8:136,:] 
        
    t1_file = img_files[0]
    
    id_start = t1_file.rfind('/') + 1
    id_end = t1_file.rfind('_t1.nii.gz')
    
    patient_id = t1_file[id_start:id_end]
    
    
    del(X, brain, brain_norm)
    return X_norm, Y, patient_id

def crop_img(im, gt, dim, offset = (0,0,0)):
    x = offset[0]
    y = offset[1]
    z = offset[2]
        
    im = im[x:x+dim[0], y:y+dim[1], z:z+dim[2],:]
    
    if gt is not None:
        gt = gt[x:x+dim[0], y:y+dim[1], z:z+dim[2]]
        
    return im, gt

def extend_img(im, gt, dim):
    backgrounds = im.min(axis=(0,1,2))
    
    im_ext = np.ones((*dim, im.shape[3]))
    gt_ext = np.zeros(dim)
    
    for modality in range(im.shape[3]):
        im_ext[:,:,:,modality] *= backgrounds[modality]
    
    im_ext[:im.shape[0], :im.shape[1], :im.shape[2], :] = im
    gt_ext[:im.shape[0], :im.shape[1], :im.shape[2]] = gt
    return im_ext, gt_ext

def make_size(im, gt, dim):
    
    if im is not None:
        im_batch = im.reshape((1, *im.shape))
    else:
        im_batch = None
        
    if gt is not None:
        gt_batch = gt.reshape((1, *gt.shape))
    else:
        gt_batch = None
        
    im_batch, gt_batch = make_size_batch(im_batch, gt_batch, dim)
    
    if im_batch is not None:
        im_batch = im_batch[0]
    if gt_batch is not None:
        gt_batch = gt_batch[0]
        
    return im_batch, gt_batch
    

def make_size_batch(im, gt, dim):
    
    if im is not None:
        input_shape = im.shape[1:4]
    elif gt is not None:
        input_shape = gt.shape[1:4]
    else:
        return None, None
    
    # Calculates the dimension of the are where image and target dimension intersect.
    isect = np.min((input_shape, dim), axis=0)
        
    # Calculates the offset in the output or input image.
    out_offset = np.subtract(dim, isect)//2
    in_offset = np.subtract(input_shape, isect)//2
        
    # Defines the slices used for cropping/extending the image.
    out_slices = [slice(None)]
    in_slices = [slice(None)]
    for i in range(len(out_offset)):
        out_slices += [slice(out_offset[i], out_offset[i] + isect[i])]
        in_slices += [slice(in_offset[i], in_offset[i] + isect[i])]
    out_slices = tuple(out_slices)
    in_slices = tuple(in_slices)
 
    im_out = None
    gt_out = None

    # If a segmentation is given, it is resized.
    if gt is not None:
        
        if len(gt.shape) > 4:
            # Categorical ground truth: Contains one channel for each class 
            # (batch_size, X, Y, Z, num_classes)
            gt_out = np.zeros((gt.shape[0], *dim, gt.shape[4]))
        else:
            # Original ground truth: Contains one channel for all classes combined
            # (batch_size, X, Y, Z)
            gt_out = np.zeros((gt.shape[0], *dim))
            
        gt_out[out_slices] = gt[in_slices]
    
    # If an image is given, it is resized.   
    if im is not None:
        im_out = np.ones((im.shape[0], *dim, im.shape[4]))
        
        # Applies uniform background to new image to avoid empty regions.
        backgrounds = im.min(axis=(1,2,3))
        for b in range(im.shape[0]):
            im_out[b] *= backgrounds[b]

        # Applies cropping and extension.
        im_out[out_slices] = im[in_slices]
 
        
    return im_out, gt_out
        


def save_img(img, path):
    "Saves a single MRT image to a file."
    
    # Save axis for data (just identity)
    transform = np.array([
        [-1, 0, 0, 0], 
        [0, -1, 0, img.shape[1]-1], 
        [0, 0,  1, 0], 
        [0, 0,  0, 1]
    ])

    # img = nib.Nifti1Image(new_data, affine=np.eye(4)
    img = nib.Nifti1Image(img, affine=transform)

    # Save as NiBabel file
    img.header.get_xyzt_units()
    img.to_filename(path) 
    
    
def save_full_scan(im, gt, folder, file_base):
    "Saves a full MRT scan including all modalities and ground truth to a folder."
    
    # Checks if folder exists and creates it if not
    if not os.path.exists(folder):
            os.makedirs(folder)
            
    base_path = f"{folder}/{file_base}"
    
    # Saves every modality
    for class_id in range(im.shape[3]):
        path = f"{base_path}_{get_mod_name(class_id)}.nii.gz"
        save_img(im[:,:,:,class_id], path)
    
    # Saves ground truth
    path = f"{base_path}_seg.nii.gz"
    save_img(gt, path)
    

def get_mod_name(class_id):
    "Returns the name of a modality based on its ID."
    
    if class_id == 0:
        return "t1"
    elif class_id == 1:
        return "t2"
    elif class_id == 2:
        return "t1ce"
    else:
        return "flair"