import cv2
import os
from tensorflow import keras
import numpy as np

CATEGORIES = ['Cat', 'Dog']

def image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Wrong path:', path)
    else:
        new_arr = cv2.resize(img, (60, 60))
        new_arr = np.array(new_arr)
        new_arr = new_arr.reshape(-1, 60, 60, 1)
        return new_arr


def main():
    model = keras.models.load_model('cat-vs-dog.model')
    imgs = os.listdir("catdog")
    print(len(imgs))
    for img in imgs:
        if img.endswith(('.jpg', '.png', 'jpeg')):
            prediction = model.predict([image(f"catdog/{img}")])
            print(f"Image {img} is the {CATEGORIES[prediction.argmax()]}")



if __name__ == '__main__':
    main()


