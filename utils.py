import os
import numpy as np
from random import seed, randrange, shuffle

seed(0)

def load_dataset(src_path, zoom=False):
    X_train_orig, Y_train_orig, X_val_orig, Y_val_orig, X_test_orig, Y_test_orig = [], [], [], [], [], []

    for npy in os.listdir(src_path):
        if 'labels' in npy:
            array_name = npy.split("labels")[0] + 'images.npy'
            images = np.load(os.path.join(src_path, array_name))
            labels = np.load(os.path.join(src_path, npy))

            if 'train' in npy:
                print(os.path.join(src_path, array_name), os.path.join(src_path, npy))
                drones = 0

                for i in range(0, len(labels)):
                    if labels[i][0] == 1 and drones != 10000:
                        drones += 1
                        img = np.reshape(images[i], (40, 40, 1))
                        img = np.interp(img, (img.min(), img.max()), (0, 255))
                        if zoom:
                            zc = randrange(0, 3)
                        else:
                            zc = 0

                        X_train_orig.append(clipped_zoom(img, zc * 0.4, 'constant'))
                        Y_train_orig.append(labels[i])

                no_drones = 0
                i = 0

                # no drones random pick

                s = np.arange(images.shape[0])
                shuffle(s)
                images = images[s]
                labels = labels[s]

                while no_drones != int(drones):
                    if labels[i][0] == -1:
                        img = np.reshape(images[i], (40, 40, 1))
                        X_train_orig.append(np.interp(img, (img.min(), img.max()), (0, 255)))
                        Y_train_orig.append(labels[i])
                        no_drones += 1

                    i += 1

                print('Train samples:')
                print('- drones:', drones)
                print('- no drones:', no_drones)

            elif 'test' in npy and 'old' not in npy:
                print(os.path.join(src_path, array_name), os.path.join(src_path, npy))

                for i in range(0, len(labels)):
                    img = np.reshape(images[i], (40, 40, 1))
                    X_test_orig.append(np.interp(img, (img.min(), img.max()), (0, 255)))
                    Y_test_orig.append(labels[i])

            elif 'val' in npy:
                print(os.path.join(src_path, array_name), os.path.join(src_path, npy))

                for i in range(0, len(labels)):
                    img = np.reshape(images[i], (40, 40, 1))
                    X_val_orig.append(np.interp(img, (img.min(), img.max()), (0, 255)))
                    Y_val_orig.append(labels[i])

            print(len(images))

    X_train_orig = np.stack(X_train_orig)
    X_val_orig = np.stack(X_val_orig)
    X_test_orig = np.stack(X_test_orig)

    print("X_train_orig shape: " + str(X_train_orig.shape))
    print("Y_train_orig shape: " + str(len(Y_train_orig)))
    print("X_val_orig shape: " + str(X_val_orig.shape))
    print("Y_val_orig shape: " + str(len(Y_val_orig)))
    print("X_test_orig shape: " + str(X_test_orig.shape))
    print("Y_test_orig shape: " + str(len(Y_test_orig)))

    return X_train_orig, Y_train_orig, X_val_orig, Y_val_orig, X_test_orig, Y_test_orig


def convert_to_one_hot(labels, num_classes):
    one_hot_array = np.zeros((len(labels), num_classes))
    for i in range(0, len(labels)):
        if labels[i] == 1:
            one_hot_array[i] = 1
        else:
            one_hot_array[i] = 0

    return one_hot_array


def clipped_zoom(img, zoom_factor, mode):
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1 and zoom_factor != 0:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = randrange(0, h - zh)
        left = randrange(0, w - zw)

        # Zero-padding
        if mode == 'constant':
            out = np.zeros_like(img)
            out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, mode=mode)
        elif mode == 'nearest':
            # non funziona
            out = zoom(img, zoom_tuple, mode=mode)

    # If zoom_factor == 1, just return the input array
    elif zoom_factor == 0 or zoom_factor == 1:
        out = img
    return out