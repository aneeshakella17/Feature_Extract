from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from cv_utils.io import HDF5DatasetWriter
import os
import numpy as np
import random

bs = 32;
dataset_folder = "animals/images/";
final_output_size = 512 * 7 * 7;
output = "/artifacts/features.hdf5";
bufferSize = 1000;

print("[INFO] loading images ...")


classNames = list(os.listdir(dataset_folder));
imagePaths = [];
for _class in classNames:
    new_folder_name = dataset_folder + _class + '/';
    imagePaths.extend([new_folder_name + name for name in os.listdir(new_folder_name)]);
random.shuffle(imagePaths);

le = LabelEncoder();
labels = le.fit_transform(classNames);

model = VGG16(weights = "imagenet", include_top = False);
dataset = HDF5DatasetWriter((len(imagePaths), final_output_size), output, dataKey = "features", bufSize = bufferSize)
dataset.storeClassLabels(le.classes_);


for i in np.arange(0, len(imagePaths), bs):
    batchPaths = imagePaths[i: i + bs];
    batchLabels = labels[i: i + bs]
    batchImages = [];

    for (j, imagePath) in enumerate(batchPaths):

        image = load_img(imagePath, target_size=(224, 224));
        image = img_to_array(image);

        image = np.expand_dims(image, axis = 0);
        image = imagenet_utils.preprocess_input(image);

        batchImages.append(image);

    batchImages = np.vstack(batchImages);
    features = model.predict(batchImages, batch_size =bs);

    features = features.reshape((features.shape[0], final_output_size));

    dataset.add(features, batchLabels);

dataset.close();