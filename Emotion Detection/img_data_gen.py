'''The module generates augmented images data through 
flow of directory method of ImageDataGenerator.'''

#Image preprocessing
from keras.preprocessing.image import ImageDataGenerator

def img_data_gen(train_path,val_path,target_size,batch_size,color_mode,class_mode):
  ''' The function generates augmented data rescales the validation images given the paths.
  Parameters:
  train_path: path to the train directory of the images
  val_path: path to the validation directory of images
  target_size: image target size; example: (48*48)
  batch_size: The batches in which the data is supposed to be fed
  color_mode: example: 'grayscale' or 'rgb'
  class_mode: example: 'binary' or 'categorical'
   '''
  #Initialising the generators for the train and validation set
  #The rescale parameter ensures the input range in [0, 1] 
  train_datagen = ImageDataGenerator(rescale = 1./255,rotation_range = 10,horizontal_flip = True,width_shift_range=0.1,height_shift_range=0.1,
                                    fill_mode = 'nearest')
  val_datagen = ImageDataGenerator(rescale = 1./255) #validation data should not be augmented

  train_generator = train_datagen.flow_from_directory(
          train_path,
          target_size=target_size,
          batch_size=batch_size,
          color_mode=color_mode,
          seed = 42,
          shuffle= True,
          class_mode=class_mode)
    
  val_generator = val_datagen.flow_from_directory(
          val_path,
          target_size=target_size,
          batch_size=batch_size,
          color_mode=color_mode,
          seed = 42,
          shuffle = True,
          class_mode=class_mode)
  
  return train_generator, val_generator