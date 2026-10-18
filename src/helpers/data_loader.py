import pickle
from tqdm import tqdm
import torch
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

#########################################################################
#                    Load Real Trained Weights from Pickle
#########################################################################
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        return data

def load_lenetwghts_from_pickl(rel_traindata_path, class_labels, num_classes):
    print(f"\tLoading real trained weights")
    data_lst = load_pickle(rel_traindata_path)

    real_weights_list = []
    real_labels_list = []

    label_to_index = {label: idx for idx, label in class_labels.items()}

    for item in tqdm(data_lst, desc="Processing real weights"):
        layer_tensors = item['parameters_flat']
        real_weights_list.append(layer_tensors.view(1, -1))
        
        if 'model' in item:
            label_split = item['model'].split('_')[:-1]
            label_str = '_'.join(label_split)
            label = label_to_index.get(label_str)
            real_labels_list.append(label)
        else:
            print(f"\nWarning: No label found, using RANDOM")
            real_labels_list.append(np.random.randint(0, num_classes))

    real_weights_tensor = torch.stack(real_weights_list, dim=0).squeeze(1)
    real_labels_tensor = torch.tensor(real_labels_list, dtype=torch.long)

    print(f"\tLoaded {len(real_weights_tensor)} real weight samples")
    print(f"\tWeight tensor shape: {real_weights_tensor.shape}")

    weight_vector_len = real_weights_tensor.shape[1]

    return real_weights_tensor, real_labels_tensor, weight_vector_len, data_lst

#########################################################################
#                    Load Imagenette Data
#########################################################################

#def load_imagenette():
def load_imagenette(data_dir):
    print("Loading Imagenette dataset")

    #data_dir = r'C:\Users\dk412\Desktop\David\Python Projects\HyperGenAI\tensorflow_datasets'
    os.makedirs(data_dir, exist_ok=True)

    imagenette_dataset = tfds.load(
        'imagenette/320px-v2', 
        as_supervised=True, 
        data_dir=data_dir
    )

    def preprocess_and_collect(dataset):
        images, labels = [], []
        for img, label in tfds.as_numpy(dataset):
            img = tf.image.resize(img, [32, 32]).numpy() / 255.0
            images.append(img)
            labels.append(label)
        return np.array(images), np.array(labels)

    x_train, y_train = preprocess_and_collect(imagenette_dataset['train'])
    x_test, y_test = preprocess_and_collect(imagenette_dataset['validation'])

    print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")
    print(f"Image shape: {x_train[0].shape}")

    return x_train, y_train, x_test, y_test

