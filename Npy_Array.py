import numpy as np
import nibabel as nib
from  tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


def Save_Npy(t2_train_list,t1ce_train_list,flair_train_list,mask_train_list,t2_val_list,t1ce_val_list,flair_val_list,path):
    
    for img in range(len(t2_train_list)):   
        
        temp_image_t2=nib.load(t2_train_list[img]).get_fdata()
        temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
       
        temp_image_t1ce=nib.load(t1ce_train_list[img]).get_fdata()
        temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
       
        temp_image_flair=nib.load(flair_train_list[img]).get_fdata()
        temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
            
        temp_mask=nib.load(mask_train_list[img]).get_fdata()
        temp_mask=temp_mask.astype(np.uint8)
        temp_mask[temp_mask==4] = 3  
    
        temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
        
        temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
        temp_mask = temp_mask[56:184, 56:184, 13:141]
        
        val, counts = np.unique(temp_mask, return_counts=True)
        
        if (1 - (counts[0]/counts.sum())) > 0.03: 
            temp_mask= to_categorical(temp_mask, num_classes=4)
            print(f'Train Image {img} is Saving')
            np.save(path + r'\input_data_3channels\images\image_'+str(img)+'.npy', temp_combined_images)
            np.save(path + r'\input_data_3channels\masks\mask_'+str(img)+'.npy', temp_mask)
            
            