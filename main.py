import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from random import shuffle
from tqdm import tqdm
from PIL import Image
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
#print(os.listdir("../Project4"))


#establish the sigmoid function
def sigmoid(z):
    sigmoida = (1 + np.exp(-z))**(-1)
    return sigmoida

# Function to read MNIST image files
def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read the header information from the file
        buffer = f.read(16)
        # Extract the metadata
        _, num_images, rows, cols = np.frombuffer(buffer, dtype=np.dtype('>i4'), count=4)
        # Read the image data
        data = f.read()
        # Convert the raw byte data to numpy array
        images = np.frombuffer(data, dtype=np.uint8)
        # Reshape the array to get individual images
        images = images.reshape(num_images, rows, cols)
        return images
#main
#"Hey is this a worm or a number"
guy = input('Type 0 for MNIST or 1 for Worms\n')


if (guy == '1'):
    bias = np.load("wormbias.npy")
    weights = np.load("wormweights.npy")
    guy2 = input('What is the folder path for worm images?\n')
    labelarray = []
    patharray = []
    worm = 0
    noworm = 0
    
    for image in tqdm(os.listdir(guy2)):
        #print(image)
        path = os.path.join(guy2,image)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100,100))
        
        img = np.asarray(img)
        img = (img - np.min(img))/(np.max(img)-np.min(img))
        #flatten the arrays of the image
        img_flatten = img.reshape(1, img.shape[0]*img.shape[1])
        #sigma of w dot phi + b
        z = np.dot(weights.T, img_flatten.T) + bias
        decision = sigmoid(z)
       # print(decision)
        #decide if worm or not
        if (decision <= .5):
            label = 'no worm'
        else:
            label = 'worm'
        #add image path and label to an array
        labelarray.append(label)
        patharray.append(path)
        #add 1 to count of label usage
        if (label == 'no worm'):
            noworm = noworm + 1
        else:
            worm = worm + 1
    labelarray.append(worm)
    labelarray.append(noworm)
    patharray.append('Worms Identified:')
    patharray.append('No Worms Identified:')
    dictionary = {'path': patharray, 'label': labelarray}
        #excel file out which has 2XN matrix featuring column of image names, then number of worms/number of not worms in a 2X2 (1 row for labels) matrix
    df = pd.DataFrame(dictionary)
    df.to_excel('Magedov_Rasul_Wibert_worms.xlsx', index=False)
    print('Outfile path: Magedov_Rasul_Wibert_worms.xlsx')


else:
    #import the model
    bias = np.load("MNISTbias.npy")
    weights = np.load("MNISTweights.npy")
    guy2 = input('What is the path of the mnist file?\n')
    labelarray = []
    patharray = []
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    six = 0
    seven = 0
    eight = 0
    nine = 0
    zero = 0
    
    for image in tqdm(os.listdir(guy2)):
        #print(image)
        path = os.path.join(guy2, image)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28,28))
        img = np.asarray(img)
        #print(img.shape)

        img = (img - np.min(img))/(np.max(img)-np.min(img))
        #flatten the arrays of the image
        img_flatten = img.reshape(1, img.shape[0]*img.shape[1])
        #sigma of w dot phi + b
        z = np.dot(img_flatten, weights) + bias
        sigma = sigmoid(z)
        decision = np.argmax(sigma, axis=1)
        if (decision[0] == 0):
            label = 'zero'
            zero = zero +1
        elif (decision[0] == 1):
            label = 'one'
            one = one + 1
        elif (decision[0] == 2):
            label = 'two'
            two = two + 1
        elif (decision[0] == 3):
            label = 'three'
            three = three +1
        elif (decision[0] == 4):
            label = 'four'
            four = four + 1
        elif (decision[0] == 5):
            label = 'five'
            five = five + 1
        elif (decision[0] == 6):
            label = 'six'
            six = six + 1
        elif (decision[0] == 7):
            label = 'seven'
            seven = seven + 1
        elif (decision[0] == 8):
            label = 'eight'
            eight = eight + 1
        elif (decision[0] == 9):
            label = 'nine'
            nine = nine + 1
        #add image path and label to an array
        labelarray.append(label)
        patharray.append(path)
    labelarray.append(zero)
    labelarray.append(one)
    labelarray.append(two)
    labelarray.append(three)
    labelarray.append(four)
    labelarray.append(five)
    labelarray.append(six)
    labelarray.append(seven)
    labelarray.append(eight)
    labelarray.append(nine)
    patharray.append('Zeroes Identified:')
    patharray.append('Ones Identified:')
    patharray.append('Two Identified:')
    patharray.append('Threes Identified:')
    patharray.append('Fours Identified:')
    patharray.append('Fives Identified:')
    patharray.append('Sixes Identified:')
    patharray.append('Sevens Identified:')
    patharray.append('Eights Identified:')
    patharray.append('Nines Identified:')
    dictionary = {'path': patharray, 'label': labelarray}
    df = pd.DataFrame(dictionary)
    df.to_excel('Magedov_Rasul_Wibert_MNIST.xlsx', index=False)
    print('Outfile path: Magedov_Rasul_Wibert_MNIST.xlsx')
