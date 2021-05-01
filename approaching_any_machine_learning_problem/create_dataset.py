import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from tqdm import tqdm

def create_dataset(traininf_df,image_dir):

    images = []
    target = []

    for index,row in tqdm(
        traininf_df.iterrow(),
        total = len(traininf_df),
        desc = "processing images"
    ):
        image_id = row['ImageId']
        image = image.open(image_path+".png")
        image = image.resize((256,256),resample = Image.BILINEAR)
        image = np.array(image)
        images.append(image)
        targets.append(int(row["target"]))
    images = np.array(images)
    print(image.shape)
    return images, targets

