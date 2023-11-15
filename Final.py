import os
import random
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import imageio
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage.transform import rotate, resize
from skimage.util import montage
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import (
    Input,
    Conv3D,
    MaxPooling3D,
    concatenate,
    Conv3DTranspose,
    BatchNormalization,
    Dropout,
    Lambda,
)
from keras.optimizers import Adam
from keras.metrics import MeanIoU
from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger,
)
from IPython.display import display, Image as IPImage
from IPython.display import Image, display as IPDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss, precision_score
from tifffile import imsave
from U_Net import U_Net
from Loading_Files import *
from Npy_Array import Save_Npy
import splitfolders
import gif_your_nifti.core as gif2nif
from array2gif import write_gif
from moviepy.editor import ImageSequenceClip
from keras.utils.vis_utils import plot_model
import visualkeras
from PIL import ImageFont
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

batch_size = 2
Classes = 4

SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC', 
    2 : 'EDEMA',
    3 : 'ENHANCING' 
}


#Preprocessing Data

TRAIN_DATASET_PATH = 'D:\Final Project\Brain Tumor\DataSet1\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
VALIDATION_DATASET_PATH = 'D:\Final Project\Brain Tumor\DataSet1\BraTS2020_ValidationData\MICCAI_BraTS2020_ValidationData'
input_folder = r'D:\Final Project\Brain Tumor\DataSet1\BraTS2020_TrainingData'
output_folder = r'D:\Final Project\Brain Tumor\DataSet1\BraTS2020_TrainingData\val'
t2_train_list = sorted(glob.glob(TRAIN_DATASET_PATH + '\*\*t2.nii'))
t1ce_train_list = sorted(glob.glob(TRAIN_DATASET_PATH +'\*\*t1ce.nii'))
flair_train_list = sorted(glob.glob(TRAIN_DATASET_PATH + '\*\*flair.nii'))
mask_train_list = sorted(glob.glob(TRAIN_DATASET_PATH + '\*\*seg.nii'))

t2_val_list = sorted(glob.glob(VALIDATION_DATASET_PATH + '\*\*t2.nii'))
t1ce_val_list = sorted(glob.glob(VALIDATION_DATASET_PATH + '\*\*t1ce.nii'))
flair_val_list = sorted(glob.glob(VALIDATION_DATASET_PATH + '\*\*flair.nii'))
    
################################################################################################
## This is for preproccessing and saving it . it only needs to be read one ##
#Save_Npy(t2_train_list,t1ce_train_list,flair_train_list,mask_train_list,t2_val_list,t1ce_val_list,flair_val_list,input_folder)
#splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None)

################################################################################################

train_img_dir = input_folder + r'\train\images\\'
train_mask_dir = input_folder + r'\train\masks\\'
val_img_dir = input_folder + r'\val\images\\'
val_mask_dir = input_folder + r'\val\masks\\'
train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)
val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)
steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size


train_img_datagen = ImgLoader(train_img_dir, train_img_list, 
                                  train_mask_dir, train_mask_list, batch_size)

val_img_datagen = ImgLoader(val_img_dir, val_img_list, 
                                  val_mask_dir, val_mask_list, batch_size)

#################Loss#####################

def categorical_crossentropy(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, (-1, 4))
    y_pred_flat = tf.reshape(y_pred, (-1, 4))
    loss = tf.keras.losses.categorical_crossentropy(y_true_flat, y_pred_flat, from_logits=False)
    return tf.reduce_mean(loss)

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-5):
    true_positives = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    false_negatives = tf.reduce_sum(y_true * (1 - y_pred), axis=(1, 2, 3))
    false_positives = tf.reduce_sum((1 - y_true) * y_pred, axis=(1, 2, 3))
    tversky_index = (true_positives + smooth) / (true_positives + alpha * false_negatives + beta * false_positives + smooth)
    tversky_loss = 1 - tversky_index
    return tversky_loss

def combined_loss(y_true, y_pred):
    ce_loss = categorical_crossentropy(y_true, y_pred)
    tv_loss = tversky_loss(y_true, y_pred)
    
    combined_loss = ce_loss + tv_loss
    
    return combined_loss

################Model####################


model = U_Net(Height=128, 
                          Width=128, 
                          Depth=128, 
                          Channels=3, 
                          Classes=4,
                          Drop = 0.25)


font = ImageFont.truetype("arial.ttf", 200) 
visualkeras.layered_view(model, legend=True, font=font) 

csv_logger = CSVLogger('Model.log')

opt = keras.optimizers.Adam( learning_rate = 0.001)

callbacks = [
    EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=1, mode='auto'),
    ReduceLROnPlateau(factor=0.9, patience=4, monitor='val_loss', mode='min' , min_lr= 1e-5, verbose=1),
    csv_logger,
]


model.compile(optimizer = opt, loss= combined_loss , metrics = ['accuracy'])

#######################Training ###########################3
# history=model.fit(train_img_datagen,
#             steps_per_epoch=steps_per_epoch,
#             epochs=10,
#             verbose=1,
#             validation_data=val_img_datagen,
#             validation_steps=val_steps_per_epoch,
#             shuffle=True,
#             )
# model.save('ModelAll.hdf5')

# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# plt.plot(epochs, acc, 'y', label='Training accuracy')
# plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

######### Example Plotting ##########
test_image_flair=nib.load(TRAIN_DATASET_PATH + '\BraTS20_Training_111\BraTS20_Training_111_flair.nii').get_fdata()
test_image_t1=nib.load(TRAIN_DATASET_PATH + '\BraTS20_Training_111\BraTS20_Training_111_t1.nii').get_fdata()
test_image_t1ce=nib.load(TRAIN_DATASET_PATH + '\BraTS20_Training_111\BraTS20_Training_111_t1ce.nii').get_fdata()
test_image_t2=nib.load(TRAIN_DATASET_PATH + '\BraTS20_Training_111\BraTS20_Training_111_t2.nii').get_fdata()
test_mask=nib.load(TRAIN_DATASET_PATH + '\BraTS20_Training_111\BraTS20_Training_111_seg.nii').get_fdata()


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))
slice_w = 25
ax1.imshow(test_image_flair[:,:,55], cmap = 'gray')
ax1.set_title('Image Flair')
ax2.imshow(test_image_t1[:,:,55], cmap = 'gray')
ax2.set_title('Image T1')
ax3.imshow(test_image_t1ce[:,:,55], cmap = 'gray')
ax3.set_title('Image T1ce')
ax4.imshow(test_image_t2[:,:,55], cmap = 'gray')
ax4.set_title('Image T2')
ax5.imshow(test_mask[:,:,55])
ax5.set_title('Mask')
plt.show()

# fig, ax2 = plt.subplots(1, 1, figsize = (15,15))
# ax2.imshow(rotate(montage(test_image_t1[:,:,:]), 90, resize=True), cmap ='gray')


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))
slice_w = 25
ax1.imshow(test_image_flair[56:184, 56:184 ,55], cmap = 'gray')
ax1.set_title('Image Flair')
ax2.imshow(test_image_t1[56:184, 56:184 ,55], cmap = 'gray')
ax2.set_title('Image T1')
ax3.imshow(test_image_t1ce[56:184, 56:184 ,55], cmap = 'gray')
ax3.set_title('Image T1ce')
ax4.imshow(test_image_t2[56:184, 56:184,55], cmap = 'gray')
ax4.set_title('Image T2')
ax5.imshow(test_mask[56:184, 56:184,55])
ax5.set_title('Mask')
plt.show()


#########Prediction###########
my_model = load_model(r'C:\Users\dorsa\Desktop\Codes\Loading\ModelAll.hdf5', 
                      compile=False)

# img_nums = [1, 8, 14, 15 , 52 , 56 , 57 , 63 , 70 ,95,96,111,112,113,116,132,134,136,137,143,146,153,154,166,173,174]
# ##1 , 52 , 63 , 70 , 95 , 96 , 112 , 111 , 
# for img_num in img_nums:

#     test_img = np.load(r"D:\Final Project\Brain Tumor\DataSet1\BraTS2020_TrainingData\val\images\image_"+str(img_num)+".npy")
    
#     test_mask = np.load(r"D:\Final Project\Brain Tumor\DataSet1\BraTS2020_TrainingData\val\masks\mask_"+str(img_num)+".npy")
    
    
#     test_mask_argmax=np.argmax(test_mask, axis=3)
#     test_img_input = np.expand_dims(test_img, axis=0)
#     test_prediction = my_model.predict(test_img_input)
#     test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]
    
#     n_slice = 55
    
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
#     axes[0].set_title('Testing Image')
#     axes[0].imshow(test_img[:, :, n_slice, 1], cmap='gray')
    
#     axes[1].set_title('Testing Label')
#     label_img = axes[1].imshow(test_mask_argmax[:, :, n_slice], cmap='viridis')  # You can change 'viridis' to the desired colormap
    
#     axes[2].set_title('Prediction on Test Image')
#     prediction_img = axes[2].imshow(test_prediction_argmax[:, :, n_slice], cmap='viridis')  # You can change 'viridis' to the desired colormap
    
#     plt.show()

img_num = 111

test_img = np.load(r"D:\Final Project\Brain Tumor\DataSet1\BraTS2020_TrainingData\val\images\image_"+str(img_num)+".npy")
    
test_mask = np.load(r"D:\Final Project\Brain Tumor\DataSet1\BraTS2020_TrainingData\val\masks\mask_"+str(img_num)+".npy")
    
    
test_mask_argmax=np.argmax(test_mask, axis=3)
test_img_input = np.expand_dims(test_img, axis=0)
test_prediction = my_model.predict(test_img_input)
test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]
    
n_slice = 55
    
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
axes[0].set_title('Testing Image')
axes[0].imshow(test_img[:, :, n_slice, 1], cmap='gray')
    
axes[1].set_title('Testing Label')
label_img = axes[1].imshow(test_mask_argmax[:, :, n_slice], cmap='viridis')  # You can change 'viridis' to the desired colormap
    
axes[2].set_title('Prediction on Test Image')
prediction_img = axes[2].imshow(test_prediction_argmax[:, :, n_slice], cmap='viridis')  # You can change 'viridis' to the desired colormap

plt.show()

def update_label(frame):
    plt.clf()
    ax = plt.gca()
    ax.set_title('Testing Label')
    ax.imshow(test_mask_argmax[:, :, frame], cmap='viridis')

def update_original(frame):
    plt.clf()
    ax = plt.gca()
    ax.set_title('Testing Image')
    ax.imshow(test_img[:, :, frame, 1], cmap='gray')

def update_prediction(frame):
    plt.clf()
    ax = plt.gca()
    ax.set_title('Prediction on Test Image')
    ax.imshow(test_prediction_argmax[:, :, frame], cmap='viridis')

# Create GIFs
label_gif = "label_img_u111.gif"
original_gif = "original_image_u111.gif"
prediction_gif = "prediction_img_u111.gif"

fig_label = plt.figure(figsize=(6, 6))
ani_label = FuncAnimation(fig_label, update_label, frames=test_mask_argmax.shape[2], repeat=False)
ani_label.save(label_gif, writer='pillow', fps=10)

fig_original = plt.figure(figsize=(6, 6))
ani_original = FuncAnimation(fig_original, update_original, frames=test_img.shape[2], repeat=False)
ani_original.save(original_gif, writer='pillow', fps=10)

fig_prediction = plt.figure(figsize=(6, 6))
ani_prediction = FuncAnimation(fig_prediction, update_prediction, frames=test_prediction_argmax.shape[2], repeat=False)
ani_prediction.save(prediction_gif, writer='pillow', fps=10)

plt.show()

#########Evaluation###########

test_img_datagen = ImgLoader(val_img_dir, val_img_list, 
                                  val_mask_dir, val_mask_list, batch_size)
test_image_batch, test_mask_batch = test_img_datagen.__next__()
test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)


IOU_keras = MeanIoU(num_classes=Classes)  
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)

print("Mean IoU =", IOU_keras.result().numpy())

##gt == ground truth
def Dice_Coef(seg_mask, gt_mask, num_classes = 4):
    dice_scores = []
    
    for class_id in range(num_classes):
        seg_class = seg_mask == class_id
        gt_class = gt_mask == class_id
        
        intersection = np.logical_and(seg_class, gt_class)
        dice = 2.0 * intersection.sum() / (seg_class.sum() + gt_class.sum())
        dice_scores.append(dice)
    
    mean_dice = np.mean(dice_scores)
    return mean_dice

dice = Dice_Coef(test_pred_batch_argmax, test_mask_batch_argmax , 4)

print(f"Dice Coefficient: {dice:.4f}")

def Sensitivity(gt_masks, predicted_masks, num_classes):
    sensitivities = []
    for class_id in range(num_classes):
        true_positives = np.sum((gt_masks == class_id) & (predicted_masks == class_id))
        false_negatives = np.sum((gt_masks == class_id) & (predicted_masks != class_id))
        sensitivity = true_positives / (true_positives + false_negatives)
        sensitivities.append(sensitivity)
    mean_sensitivity = np.mean(sensitivities)
    return mean_sensitivity

mean_sensitivity = Sensitivity(test_mask_batch_argmax, test_pred_batch_argmax, 4)
print("Mean Sensitivity:", mean_sensitivity)


def Precision(gt_masks, predicted_masks, num_classes):
    precisions = []
    for class_id in range(num_classes):
        true_positives = np.sum((gt_masks == class_id) & (predicted_masks == class_id))
        false_positives = np.sum((gt_masks != class_id) & (predicted_masks == class_id))
        precision = true_positives / (true_positives + false_positives)
        precisions.append(precision)
    mean_precision = np.mean(precisions)
    return mean_precision

mean_precision = Precision(test_mask_batch_argmax, test_pred_batch_argmax, 4)
print("Mean Precision:", mean_precision)

def Accuracy(gt_masks, predicted_masks):
    total_pixels = gt_masks.size
    correct_pixels = np.sum(gt_masks == predicted_masks)
    accuracy = correct_pixels / total_pixels
    return accuracy

mean_accuracy = Accuracy(test_mask_batch_argmax, test_pred_batch_argmax)
print("Mean Accuracy:", mean_accuracy)