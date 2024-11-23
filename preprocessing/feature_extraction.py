import os
import shutil
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from preprocessing.video_to_frames import video_to_frames

def model_cnn_load():
    model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    return Model(inputs=model.input, outputs=model.layers[-2].output)

def load_image(path):
    import cv2
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not read image {path}")
        return None
    img = cv2.resize(img, (224, 224))
    return img

def extract_features(video, model, input_path, output_path):
    video_id = video.split(".")[0]
    print(f'Processing video {video}')
    image_list = video_to_frames(video, input_path, output_path)

    if not image_list:
        return None

    images = np.zeros((len(image_list), 224, 224, 3))
    for i in range(len(image_list)):
        img = load_image(image_list[i])
        if img is not None:
            images[i] = img

    fc_feats = model.predict(images, batch_size=200)
    img_feats = np.array(fc_feats)

    shutil.rmtree(os.path.join(output_path, 'temporary_images'))
    return img_feats

