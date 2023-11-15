import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Conv3D, MaxPooling3D, Conv3DTranspose,
    BatchNormalization, Dropout, concatenate, Lambda,
    ReLU, Add, AveragePooling3D, Flatten, Dense,
    UpSampling3D, Concatenate, add
)
import keras.backend as K
from keras import regularizers
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Maximum , PReLU

kernel_initializer = 'he_uniform'

def U_Net(Height, Width, Depth, Channels, Classes , Drop):

     inputs = Input((Height, Width, Depth, Channels))

     #مسیر فشرده سازی #     
     c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(inputs)
     c1 = Dropout(Drop)(c1)
     c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
     p1 = MaxPooling3D((2, 2, 2))(c1)
    
     c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
     c2 = Dropout(Drop)(c2)
     c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
     p2 = MaxPooling3D((2, 2, 2))(c2)
     
     c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
     c3 = Dropout(Drop)(c3)
     c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
     p3 = MaxPooling3D((2, 2, 2))(c3)
     
     c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
     c4 = Dropout(Drop)(c4)
     c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
     p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
     
     c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
     c5 = Dropout(Drop)(c5)
     c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
     #مسیر بسط دادن #     
     u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
     u6 = concatenate([u6, c4])
     c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
     c6 = Dropout(Drop)(c6)
     c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
     u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
     u7 = concatenate([u7, c3])
     c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
     c7 = Dropout(Drop)(c7)
     c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
     u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
     u8 = concatenate([u8, c2])
     c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
     c8 = Dropout(Drop)(c8)
     c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
     u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
     u9 = concatenate([u9, c1])
     c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
     c9 = Dropout(Drop)(c9)
     c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
     # لایه خروجی #     
     outputs = Conv3D(Classes, (1, 1, 1), activation='softmax')(c9) 
     model = Model(inputs=[inputs], outputs=[outputs])
    
     return model
 
    
def Attention(Map):
    MeanPool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=(1,2,3), keepdims=True))(Map)
    MaxPool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=(1,2,3), keepdims=True))(Map)
    Concat = tf.keras.layers.Concatenate(axis=-1)([MeanPool, MaxPool])
    Weights = tf.keras.layers.Conv3D(filters=1, kernel_size=(1,1,1), strides=(1,1,1), activation='softmax', use_bias=False)(Concat)
    AttMap = tf.keras.layers.Multiply()([Map, Weights])
    return AttMap


def Attention_Unet(Height,  Width, Depth, Channels, Classes):
    inputs = Input((Height,  Width, Depth, Channels))

    c1 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    p1 = Attention(p1)
    
    c2 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
    p2 = Attention(p2)

     
    c3 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)
    p3 = Attention(p3)

    c4 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
    p4 = Attention(p4)

    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(p4)
    u6 = concatenate([u6, c4])
    u6 = Attention(u6)
    c6 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Attention(u7)
    c7 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
    c7 = BatchNormalization()(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Attention(u8)
    c8 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
    c8 = BatchNormalization()(c8)

    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Attention(u9)
    c9 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
    c9 = BatchNormalization()(c9)

    outputs = Conv3D(Classes, (1, 1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def Residual_Block(input_block,filter_num,kernel_size):
  X = Conv3D(filter_num,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(input_block)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = Conv3D(filter_num,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X)
  X = add([input_block,X]);
  
  return X


def V_Net(Height, Width, Depth, Channels, Classes , Drop ): 
 
  input_shape = (Height, Width, Depth, Channels)
  input_img = Input(input_shape)

  c1 = Conv3D(8,kernel_size = (5,5,5) , strides = (1,1,1) , padding='same')(input_img)

  c2 = Conv3D(16,kernel_size = (2,2,2) , strides = (2,2,2) , padding = 'same' )(c1)
  
  c3 = Residual_Block(c2 , 16 , 5 )
  
  p3 = Conv3D(32,kernel_size = (2,2,2) , strides = (2,2,2), padding = 'same')(c3)
  p3 = Dropout(Drop)(p3)
  
  c4 = Residual_Block(p3,32,5)
  p4 = Conv3D(64,kernel_size = (2,2,2) , strides = (2,2,2) , padding='same')(c4)
  p4 = Dropout(Drop)(p4)
    
  c5 = Residual_Block(p4, 64 ,5)
  p6 = Conv3D(128,kernel_size = (2,2,2) , strides = (2,2,2) , padding='same')(c5)
  p6 = Dropout(Drop)(p6)


  p7 = Residual_Block(p6,128,5)
    
  u6 = Conv3DTranspose(64, (2,2,2), strides=(2, 2, 2), padding='same')(p7);
  u6 = concatenate([u6,c5]);
  c7 = Residual_Block(u6,128,5)
  c7 = Dropout(Drop)(c7)
  u7 = Conv3DTranspose(32,(2,2,2),strides = (2,2,2) , padding= 'same')(c7);

  
  u8 = concatenate([u7,c4]);
  c8 = Residual_Block(u8,64,5)
  c8 = Dropout(Drop)(c8)
  u9 = Conv3DTranspose(16,(2,2,2),strides = (2,2,2) , padding= 'same')(c8);
    
  u9 = concatenate([u9,c3]);
  c9 = Residual_Block(u9,32,5)
  c9 = Dropout(Drop)(c9)
  u10 = Conv3DTranspose(8,(2,2,2),strides = (2,2,2) , padding= 'same')(c9);
  
  
  u10 = concatenate([u10,c1]);
  c10 = Conv3D(16,kernel_size = (5,5,5),strides = (1,1,1) , padding = 'same')(u10);
  c10 = Dropout(Drop)(c10)
  c10 = add([c10,u10]);
  
  outputs = Conv3D(Classes, (1,1,1), activation='softmax')(c10)

  model = Model(inputs=input_img, outputs=outputs)

  return model

