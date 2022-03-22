'''The module helps in visualizing images in machine learning tasks.''' 
#import data analysis libraries
import numpy as np
import pandas as pd

def show_img(df_row,pixel_string_col,class_col):
    '''The function takes in pixels in the form of strings and 
    respective class of the picture; preprocessess it and returns an array of image matrix of 48X48 
    and the class which can be easily plotted to visualize the image.
    Parameters:
    df_row: A row of a dataframe which has two columns, one containing pixels in string datatype
    and other labelled class
    pixel_string_col: Name of the column containing the pixels (dtype:string)
    class_col: Name of the column containing class (dtype:string)'''
    #pass observation and gather pixel and class
    pixels = df_row[pixel_string_col]
    label = df_row[class_col]
    #split object and convert to array
    pic = np.array(pixels.split())
    pic = pic.reshape(48,48)
    image = np.zeros((48,48,3))
    #slice image and put the picture in three channels
    image[:,:,0] = pic
    image[:,:,1] = pic
    image[:,:,2] = pic
    #return image array and class
    return np.array([image.astype(np.uint8), label])