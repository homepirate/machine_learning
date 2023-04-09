import os

import cv2
from tensorflow import keras
import numpy as np

CATEGORIES = ['men', 'women']

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
    model = keras.models.load_model('men-vs-women.model')
    men = os.listdir('people/men')
    women = os.listdir('people/women')
    count_m = 0
    count_w = 0
    for m in men:
        prediction = model.predict([image(f'people/men/{m}')])
        if prediction.argmax() == 0:
            count_m += 1

    for w in women:
        prediction = model.predict([image(f'people/women/{w}')])
        if prediction.argmax() == 1:
            count_w += 1


    pr_1 = '%.2f' % (count_m/len(men)*100)
    pr_2 = '%.2f' % (count_w/len(women)*100)
    pr_3 = '%.2f' % ((count_m+count_w)/(len(men)+len(women)) * 100)
    print(f'men: {count_m} out of {len(men)} {pr_1}%\nwomen: {count_w} out of {len(women)} {pr_2}%')
    print(pr_3, '%')


if __name__ == "__main__":
    main()