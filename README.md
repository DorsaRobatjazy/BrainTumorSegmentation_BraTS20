# BrainTumorSegmentation_BraTS20
Brain Tumor Segmentation With UNET , VNET , UNET with Attention Gate

In this project, I employed the BraTS 2020 dataset. In order to preprocess the large images, I selected tumor images that were noticeable in size.
The dataset contained four protocol images, and I utilized three of them, stacking them to form a numpy array.
These MRI images were then cropped to a size of 128x128x128 and normalized using the MinMax scaler.
Subsequently, I split the dataset into a 75-25 ratio for training and testing.

For the modeling phase, I implemented three models: UNET, UNet with an attention gate, and VNET.
These models were trained using three different loss functions: Tversky loss, categorical crossentropy, and a combination of these losses.

Post-training, I conducted evaluations using various metrics, including Mean IOU, Dice coefficient, Mean Sensitivity, and Mean Precision.

In conclusion, the VNET model with the combined loss function emerged as the most effective, considering both time and epochs required for training, as well as evaluation metrics.
It achieved an IoU of 83%, Dice coefficient of 90%, Sensitivity of 93%, and Precision of 88%.
