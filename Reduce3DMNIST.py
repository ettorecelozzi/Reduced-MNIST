import numpy as np
import torch
from torchvision.datasets import MNIST
from collections import defaultdict
import pickle as pkl
import h5py


def get_rmnist_idx(n, labels, test=False):
    """
    Generate indexes to get n elemnt for each digits
    :param n: number of element (int)
    :param labels: list of train labels
    :return: dictionary where the key is the label and the values are the indexes of the samples
    """
    indices = defaultdict(list)
    for i, label in enumerate(labels):
        if len(indices[label]) < n:
            indices[label].append(i)
    filename = "./3DRMNIST/indexes/rmnist" + str(n) + "_train" if test is False else "./3DRMNIST/indexes/rmnist" + str(
        n) + "_test"
    with open(filename, "wb") as f:
        pkl.dump(indices, f)
    return indices


def get_rmnist(n, indices, imgs, labels, test=False):
    """
    Retrieve the samples given the dictionary of the samples indexes
    :param n: number of samples for each digit
    :param indices: dictionary (see get_rmnist_idx)
    :param imgs: full dataset of images
    :param labels: full dataset of labels
    """
    imgs_r = []
    labels_r = []
    for key in indices.keys():
        for idx in indices[key]:
            imgs_r.append(imgs[idx])
            labels_r.append(labels[idx])
    filename = './3DRMNIST/dataset/trainImg_' + str(
        n) + '.npy' if test is False else './3DRMNIST/dataset/testImg_' + str(n) + '.npy'
    filename_label = './3DRMNIST/dataset/trainlabel_' + str(
        n) + '.npy' if test is False else './3DRMNIST/dataset/testlabel_' + str(n) + '.npy'
    np.save(filename, np.asarray(imgs_r))
    np.save(filename_label, np.asarray(labels_r))


# Load 3D MNIST train
with h5py.File("./3DMNIST/full_dataset_vectors.h5", 'r') as h5:
    X_train, ds_trainY = h5["X_train"][:], h5["y_train"][:]
    X_test, ds_testY = h5["X_test"][:], h5["y_test"][:]
ds_trainX = []
for img in X_train:
    ds_trainX.append(img.reshape(16, 16, 16))
ds_testX = []
for img in X_test:
    ds_testX.append(img.reshape(16, 16, 16))
print(f'Images shape: {ds_trainX[0].shape}')

n = 40
indices = get_rmnist_idx(n, ds_trainY, test=False)
with open("./3DRMNIST/indexes/rmnist" + str(n) + "_train", "rb") as f:
    data = pkl.load(f)
get_rmnist(n, data, ds_trainX, ds_trainY, test=False)

n = 25
indices_test = get_rmnist_idx(n, ds_testY, test=True)
with open("./3DRMNIST/indexes/rmnist" + str(n) + "_test", "rb") as f:
    data_test = pkl.load(f)
get_rmnist(n, data_test, ds_testX, ds_testY, test=True)
