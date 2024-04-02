# This script is used to generate the final crop type map. It loads a trained TempCNN model, reads satellite 
# images, processes their pixel values, reshapes them into tensors, iterates through a grid of pixels, 
# classifies each pixel using the model, and stores the results.

from TempCNNSelf import TempCNN
import torch
from glob import glob
import os
import rasterio
import numpy as np

model = TempCNN()

file_path = r'D:\Train_Data_Extra\Crop_classification_TempCNN_epoch_8_1.000_0.500'
model.load_state_dict(torch.load(file_path, map_location = 'cpu')) # change the path to your saved trained network file




# import breizhcrops as bzh
# model = bzh.models.pretrained("Transformer")


data_dir = r'D:\Final_Year_Project\Final\Data1\SatelliteImages_Tiles\6_Tile6'

#This is done to ensure that the folder gets read in sequence
file_list = []
for i in range(1,18):
    file = glob(os.path.join(data_dir, ('{}_*.tif').format(i)))
    file_list.append(file)
    
file_list = [item for sublist in file_list for item in sublist]


imageData = []
allImages = {}
index = 1


#Store the satellite image pixel value inside a imageData List
for images in file_list:
    with rasterio.open(images) as src:
        for i in range(1,12):
            band = src.read(i)
            imageData.append(band.astype(float))
        allImages[index] = imageData
        
        imageData = []
        index = index + 1

#Convert the list into Numpy Array and Tensor        
allImagesArray = []
for i in range (1,len(allImages)+1):
    allImagesArray.append(np.asarray(allImages[i]))
    #allImagesTensor.append(torch.from_numpy(allImagesArray[i])/10000)
allImagesNpArray = np.asarray(allImagesArray)

allImagesTensor = torch.as_tensor(allImagesNpArray/10000)
allImagesTensorPermuted = allImagesTensor.permute(2,3,0,1)

classified = torch.zeros(500,500)

for i in range(0,500):
    for j in range(0,500):
        x = allImagesTensorPermuted[:][i,j,:].unsqueeze(0)
        y_pred = model(x.float())
        
        y_prob = torch.exp(y_pred)
        
        npY_prob = y_prob[0].detach().numpy()
        
        maxProb = np.amax(npY_prob)
        
        label = np.where(npY_prob == maxProb)[0][0]
        #print(label)
        
        classified[i, j] = label
        
