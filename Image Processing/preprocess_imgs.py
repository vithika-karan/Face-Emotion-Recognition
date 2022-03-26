'''The module creates image directories for various 
    classes out of a dataframe for data augmentation purposes.'''

#importing libraries
import numpy as np
import pandas as pd
import os
from PIL import Image

def create_dir(path,class_list):
  ''' The function takes in the path and list of the classes to 
        create directories for different classes.
        Parameters:
        path: The path where train and validation directories needs to be created.
        class_list: The list of labels in the dataset.''' 
  #Create train and validation directories
  train_path = os.path.join(path,'train')
  val_path = os.path.join(path,'valid')
  os.mkdir(train_path)
  os.mkdir(val_path) 
  for data_path,cat in {train_path:'train-',val_path:'valid-'}.items():
    for label in class_list:
      label_dir = os.path.join(data_path,cat+str(label))
      os.mkdir(label_dir)

def save_imgs(df,df_path,pixel_col,class_col,class_list,prefix):
  '''This function takes in the dataframes and 
     creates images and saves images in directories.
     Parameters: 
     df: Dataframe that needs to be converted.
     df_path: Path to the directory (dtype-string)
              Example- If the training dataframe is fed, df_path should be the path
              to the train directory created.
     pixel_col: Name of the column containing pixels in string object
     class_col: Name of the column for data labels
     prefix: train- for training set, valid- for validation set  '''
  
  for i in range(len(df)):
      pixel_string = df[pixel_col][i]
      pixels = list(map(int, pixel_string.split()))
      
      matrix = np.array(pixels).reshape(48,48).astype(np.uint8)
      img = Image.fromarray(matrix)
      for label in class_list:
        if str(df[class_col][i]) in prefix + str(label):
            img.save(df_path +'/'+ prefix + str(label)+'/'+ prefix + str(label)+'-'+str(i)+'.png')
        else:
          continue








