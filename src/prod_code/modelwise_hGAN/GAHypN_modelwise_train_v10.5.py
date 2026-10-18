# -*- coding: utf-8 -*-
##########################################################################################
#                  MULTI CLASS Training Class Specific Discriminator
#                        Generator replaced with HMLP Hypernetwork
#         **Z Dynamic Latent Vector internally maintained as per class embedding**
#                           Added Task Data Augmentation
#                              weight decay regularization
#                    Task Loss averaged over batch size vs samp size = 4
#                   Single Gen Weight Z_test sample size vs averaged z_test
#     Evaluation performed on BALANCED test samples over BINARY LABELS
#                    *** Target LeNets trained with balanced 50/50 data ***
#                    *** Weight ordering matches PyTorch state_dict ***
#                  Added framework to handle full training dataset 100K samples
#                          Training Data Sample Selection: Top N
#                  V10.5 adds latent_rep of trained weights to generator input
##########################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
from datetime import datetime
import torch.nn.functional as F
import gc
import torchvision.transforms as T
import pandas as pd
from torch.func import vmap
import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys

tf.config.set_visible_devices([], 'GPU')
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
tf.config.optimizer.set_jit(False)
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device.type}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

#########################################################################
#                    Load Imagenette Data
#########################################################################
 
def load_imagenette(data_dir):
    print("Loading Imagenette dataset")
 
    os.makedirs(data_dir, exist_ok=True)
 
    imagenette_dataset = tfds.load(
        'imagenette/320px-v2', 
        as_supervised=True, 
        data_dir=data_dir
    )

    def preprocess_and_collect(dataset):
        images, labels = [], []
        for img, label in tfds.as_numpy(dataset):
            #if label == 0:
            img = tf.image.resize(img, [32, 32]).numpy() / 255.0
            images.append(img)
            labels.append(label)
        images = np.array(images)
        images = images.transpose(0, 3, 1, 2).astype(np.float32)
        return images, np.array(labels)
 
    x_train, y_train = preprocess_and_collect(imagenette_dataset['train'])
    x_test, y_test = preprocess_and_collect(imagenette_dataset['validation'])
 
    print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")
    print(f"Image shape: {x_train[0].shape}")  # Now (3, 32, 32)
 
    return x_train, y_train, x_test, y_test

#########################################################################
#                    Load Real Trained Weights from Pickle
#########################################################################
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        return data

def load_lenetwghts_from_pickl(rel_traindata_path, class_labels, num_classes, traindata_samp_size):
    ###### FILTER THE TRAINING DATA TO TOP N SAMPLES PER CLASS BASED ON PERFORMANCE METRICS CSV 
    #train_data_performance_df = pd.read_csv('/homes/dkurtenb/projects/HyperGENAI/hnet_trainingdata_build_pytorch_10000/output/run_20260314_090634/model_performance_metrics.csv')
    train_data_performance_df = pd.read_csv('/homes/dkurtenb/projects/HyperGENAI/hnet_trainingdata_build_pytorch_10000/output/run_20260314_090634/model_performance_metrics.csv')
    df = train_data_performance_df.sort_values(by=['Class', 'Accuracy'], ascending=[True, False])
    top_x_sample_lst = []

    for class_nm in class_labels.values():
        test_df = df[df['Class']==class_nm]
        top_x = test_df.head(traindata_samp_size)

        for index, row in top_x.iterrows():
            class_nm = row['Class']
            count = row['Count']
            index_nm = class_nm+'_'+str(count)
            top_x_sample_lst.append(index_nm)
    ################################################################################################
    
    print(f"\tLoading real trained weights")
    data_lst = load_pickle(rel_traindata_path)
 
    real_weights_list = []
    real_labels_list = []
 
    label_to_index = {label: idx for idx, label in class_labels.items()} #maps class name to an inex {'tench': 0, 'english_springer': 1, '... 'gas_pump': 7, 'golf_ball': 8, 'parachute': 9}}
 
    print(f"\tPickle contains {len(data_lst)} entries")
    print(f"\tExpected class labels: {label_to_index}")
 
    none_count = 0
    none_examples = []
 
    for item in tqdm(data_lst, desc="Processing real weights"):
        layer_tensors = item['parameters_flat']

        if 'model' in item:
            model_name = item['model']
            #print(f"\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ MODEL NAME {model_name}")

            if model_name in top_x_sample_lst:  ##### FILTER TO top_x
                #print(f"%%%%%%%%%%%%%%%%%%%% GOT ONE")
                label_split = model_name.split('_')[:-1]
                label_str = '_'.join(label_split)          
            
                label = label_to_index.get(label_str)  # converts the class name to an index label using the label_to_index mapping
                if label is not None:
                    real_weights_list.append(layer_tensors.view(1, -1))
                    real_labels_list.append(label)
                else:
                    none_count += 1
                    if len(none_examples) < 5:
                        none_examples.append(label_str)
 
    if none_count > 0:
        print(f"\n\tWARNING: {none_count} entries had unrecognized model names and were SKIPPED.")
        print(f"\tExamples of unmatched names: {none_examples}")
        print(f"\tExpected format: '{{class_name}}_{{count}}' where class_name is one of: "
              f"{list(label_to_index.keys())}")

    model_name_examples = [item.get('model', '?') for item in data_lst[:10]]
    del data_lst
    gc.collect()

    if len(real_weights_list) == 0:
        raise ValueError(
            f"No weights could be loaded! None of the model names matched the class labels.\n"
            f"Model name examples from pickle: {model_name_examples}\n"
            f"Expected class names: {list(label_to_index.keys())}\n"
            f"The 'model' field should be formatted as '{{class_name}}_{{count}}' "
            f"e.g. 'tench_0', 'english_springer_42'"
        )

    real_weights_tensor = torch.stack(real_weights_list, dim=0).squeeze(1)
    real_labels_tensor = torch.tensor(real_labels_list, dtype=torch.long)

    del real_weights_list, real_labels_list
    gc.collect()

    print(f"\tLoaded {len(real_weights_tensor)} real weight samples")
    print(f"\tWeight tensor shape: {real_weights_tensor.shape}")

    weight_vector_len = real_weights_tensor.shape[1]

    return real_weights_tensor, real_labels_tensor, weight_vector_len



#########################################################################
#                 PyTorch Functional LeNet5 (DIFFERENTIABLE)
#########################################################################
 
class FunctionalLeNet5:
 
    @staticmethod
    def get_weight_shapes():
        shapes = [
            ('conv1.weight', (6, 3, 5, 5), 6*3*5*5), # 450
            ('conv1.bias', (6,), 6), # 6
            ('conv2.weight', (16, 6, 5, 5), 16*6*5*5), # 2400
            ('conv2.bias', (16,), 16), # 16
            ('conv3.weight', (120, 16, 5, 5), 120*16*5*5), # 48000
            ('conv3.bias', (120,), 120), # 120
            ('fc1.weight', (84, 480), 84*480), # 40320
            ('fc1.bias', (84,), 84), # 84
            ('fc2.weight', (1, 84), 1*84), # 84
            ('fc2.bias', (1,), 1), # 1
        ]
        return shapes
    
    @staticmethod
    def get_total_params():
        return sum(s[2] for s in FunctionalLeNet5.get_weight_shapes())
    
    @staticmethod
    def forward(x, weights_flat):

        if x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        
        shapes = FunctionalLeNet5.get_weight_shapes()
        
        idx = 0
        params = {}
        for name, shape, size in shapes:
            params[name] = weights_flat[idx:idx+size].reshape(shape)
            idx += size
        
        x = F.conv2d(x, params['conv1.weight'], params['conv1.bias'], padding=2)
        x = torch.tanh(x)
        x = F.avg_pool2d(x, 2, stride=2)
        
        x = F.conv2d(x, params['conv2.weight'], params['conv2.bias'], padding=0)
        x = torch.tanh(x)
        x = F.avg_pool2d(x, 2, stride=2)
        
        x = F.conv2d(x, params['conv3.weight'], params['conv3.bias'], padding=0)
        x = torch.tanh(x)
        
        x = x.reshape(x.size(0), -1)
        
        x = F.linear(x, params['fc1.weight'], params['fc1.bias'])
        x = torch.tanh(x)
        
        x = F.linear(x, params['fc2.weight'], params['fc2.bias'])
        x = torch.sigmoid(x)
        
        return x
    
#########################################################################
#            PyTorch LeNet5 for Evaluation ONLY (replaces Keras)
#########################################################################
 
class LeNet5Eval(nn.Module):
    def __init__(self):
        super(LeNet5Eval, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc1   = nn.Linear(120 * 2 * 2, 84)
        self.fc2   = nn.Linear(84, 1)
 
    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = torch.tanh(self.conv3(x))
        #x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
def load_weights_into_lenet5_pytorch(model, weights_array):
    if isinstance(weights_array, torch.Tensor):
        weights_array = weights_array.detach().cpu()
    else:
        weights_array = torch.from_numpy(weights_array).float()
    
    shapes = FunctionalLeNet5.get_weight_shapes()
    
    state_dict = {}
    idx = 0
    for name, shape, size in shapes:
        state_dict[name] = weights_array[idx:idx+size].reshape(shape)
        idx += size
    
    model.load_state_dict(state_dict)
    return model

#########################################################################
#                    Task Loss Computation (DIFFERENTIABLE)
#########################################################################
 
def compute_task_loss_differentiable(weights_flat, x_batch, y_batch, target_class_id, device):
    pos_idx = np.where(y_batch == target_class_id)[0]
    neg_idx = np.where(y_batch != target_class_id)[0]

    if len(pos_idx) == 0:
        return None, None  

    if len(neg_idx) == 0:
        x_bal = x_batch[pos_idx]
        y_bal = np.ones(len(pos_idx), dtype=np.float32)
    else:
        n = min(len(pos_idx), len(neg_idx))
        pos_chosen = np.random.choice(pos_idx, n, replace=False)
        neg_chosen = np.random.choice(neg_idx, n, replace=False)
        bal_idx = np.concatenate([pos_chosen, neg_chosen])
        np.random.shuffle(bal_idx)
        x_bal = x_batch[bal_idx]
        y_bal = (y_batch[bal_idx] == target_class_id).astype(np.float32)

    x_tensor = torch.from_numpy(x_bal).float().to(device)
    y_tensor = torch.from_numpy(y_bal).float().to(device).unsqueeze(1)
    predictions = FunctionalLeNet5.forward(x_tensor, weights_flat)
    loss = F.binary_cross_entropy(predictions, y_tensor)

    with torch.no_grad():
        acc = ((predictions > 0.5).float() == y_tensor).float().mean().item()

    return loss, acc

def compute_task_loss_batch(gen_weights_flat, x_batch_aug, y_batch, target_class_id, device, w_abs_max):
    pos_idx = np.where(y_batch == target_class_id)[0]
    neg_idx = np.where(y_batch != target_class_id)[0]

    if len(pos_idx) == 0:
        return None, None

    if len(neg_idx) == 0:
        x_bal = x_batch_aug[pos_idx].to(device)
        y_bal = torch.ones(len(pos_idx), 1, dtype=torch.float32, device=device)
    else:
        n = min(len(pos_idx), len(neg_idx))
        pos_chosen = np.random.choice(pos_idx, n, replace=False)
        neg_chosen = np.random.choice(neg_idx, n, replace=False)
        bal_idx = np.concatenate([pos_chosen, neg_chosen])
        np.random.shuffle(bal_idx)
        x_bal = x_batch_aug[bal_idx].to(device)
        y_bal = torch.from_numpy(
            (y_batch[bal_idx] == target_class_id).astype(np.float32)
        ).unsqueeze(1).to(device)

    # vmap maps over dim-0 of gen_weights_flat; x_bal/y_bal are fixed constants.
    def per_weight_forward(w_flat):
        preds = FunctionalLeNet5.forward(x_bal, w_flat * w_abs_max)
        loss = F.binary_cross_entropy(preds, y_bal)
        return loss, preds

    losses, all_preds = vmap(per_weight_forward)(gen_weights_flat)
    accs = ((all_preds.detach() > 0.5).float() == y_bal.unsqueeze(0)).float().mean(dim=[1, 2])
    return losses.mean(), accs.mean().item()


def compute_task_loss_eval(weights_flat, x_batch, y_batch, target_class_id, device, model=None):
    if isinstance(weights_flat, torch.Tensor):
        weights_np = weights_flat.detach().cpu()
    else:
        weights_np = torch.from_numpy(weights_flat).float()
    
    if model is None:
        model = LeNet5Eval()
    
    load_weights_into_lenet5_pytorch(model, weights_np)
    model.eval()
    model.to(device)
    
    x_tensor = torch.from_numpy(x_batch).float().to(device)
    binary_labels = (y_batch == target_class_id).astype(np.float32)
    
    with torch.no_grad():
        predictions = model(x_tensor).cpu().numpy().flatten()
    
    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.mean(binary_labels * np.log(predictions) + (1 - binary_labels) * np.log(1 - predictions))
    accuracy = np.mean((predictions > 0.5) == binary_labels)
    
    return float(loss), float(accuracy)

#########################################################################
#          Balanced Test Set Builder 
#   Creates a 50/50 balanced subset for each class, matching how
#########################################################################
 
def build_balanced_test_set(x_test, y_test, class_idx, rng=None):
    pos_idx = np.where(y_test == class_idx)[0]
    neg_idx = np.where(y_test != class_idx)[0]

    if len(pos_idx) == 0:
        return None, None  

    if len(neg_idx) == 0:
        return x_test[pos_idx], np.ones(len(pos_idx), dtype=np.float32)

    n_pos = len(pos_idx)
    replace = n_pos > len(neg_idx)
    neg_sample_idx = rng.choice(neg_idx, size=n_pos, replace=replace)
    balanced_idx = np.concatenate([pos_idx, neg_sample_idx])
    rng.shuffle(balanced_idx)
    x_bal = x_test[balanced_idx]
    y_bal_binary = (y_test[balanced_idx] == class_idx).astype(np.float32)
    return x_bal, y_bal_binary

#########################################################################
#   Generator (Hypernetwork) z is an internally maintained per-class embedding, looked up by cond_id 
#########################################################################
class ConditionalWeightGenerator(nn.Module):
    def __init__(self, latent_dim, noise_dim,weight_vector_len, num_classes, hidden_dim,
                 gen_leakyrelu_slope, gen_dropout_rate=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.weight_vector_len = weight_vector_len
 
        self.z_embeddings = nn.Embedding(num_classes, latent_dim)
        self.noise_dim = noise_dim
 
        self.model = nn.Sequential(nn.Linear(latent_dim + noise_dim, hidden_dim),
            nn.LeakyReLU(gen_leakyrelu_slope, inplace=True),
            nn.Dropout(gen_dropout_rate),
 
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(gen_leakyrelu_slope, inplace=True),
            nn.Dropout(gen_dropout_rate),
 
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.LeakyReLU(gen_leakyrelu_slope, inplace=True),
            nn.Dropout(gen_dropout_rate),
 
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(gen_leakyrelu_slope, inplace=True),
            nn.Dropout(gen_dropout_rate),
 
            nn.Linear(hidden_dim * 2, weight_vector_len),
            nn.Tanh()
        )
 
    def forward(self, cond_id, batch_size=1):
        c = self.z_embeddings(torch.tensor([cond_id], device=self.z_embeddings.weight.device)).expand(batch_size, -1)
        noise = torch.randn(batch_size, self.noise_dim, device=c.device)   # fresh every call
        return self.model(torch.cat([c, noise], dim=1))
    
#########################################################################
#                    Discriminator  
#########################################################################
class ConditionalDiscriminator(nn.Module):
 
    def __init__(self, weight_vector_len, num_classes, embedding_dim, hidden_dim, dropout_rate, dis_leakyrelu_slope):
        super(ConditionalDiscriminator, self).__init__()
 
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
 
        self.model = nn.Sequential(
            nn.Linear(weight_vector_len + embedding_dim, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(dis_leakyrelu_slope, inplace=True),
            nn.Dropout(dropout_rate),
 
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(dis_leakyrelu_slope, inplace=True),
            nn.Dropout(dropout_rate),
 
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(dis_leakyrelu_slope, inplace=True),
            nn.Dropout(dropout_rate),
 
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
 
    def forward(self, weights, labels):
        class_embed = self.class_embedding(labels)
        weights_flat = weights.view(weights.size(0), -1) if len(weights.shape) > 2 else weights
        disc_input = torch.cat([weights_flat, class_embed], dim=1)
        return self.model(disc_input)
 
#########################################################################
#                        Early Stopping
#########################################################################
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_test_acc = 0
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, test_acc):
        if test_acc > self.best_test_acc + self.delta:
            self.best_test_acc = test_acc
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")

#########################################################################
#                    Training Function
#########################################################################
def train_gan(generator, discriminator, real_weights_by_class, w_abs_max_by_class,
              x_train, y_train, x_test, y_test, model_dir, plot_dir, class_labels, traindata_samp_size,
              config, current_datetime, version, early_stopping):

    num_classes = config['num_classes']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    task_loss_weight = config['task_loss_weight']
    g_adversarial_loss_weight = config['g_adversarial_loss_weight']
    lr_g = config['lr_g']
    lr_d = config['lr_d']
    beta1 = config['beta1']
    device = config['device']

    w_abs_max = max(w_abs_max_by_class.values())  
    """
    weights need to be normalized to [-1, 1] to match the Thanh output range.
    When this was trained without w_abs_max normalization, weights exploded.
    w_abs is sensitve to outliers, this can cause issues with modelwise approach but seems to be resolved with layerwise approach
    """ 
    #print(f'Per-class weight scale factors:')
    #for _cls, _wmax in w_abs_max_by_class.items():
    #    print(f'  class {_cls} ({class_labels[_cls]}): {_wmax:.6f}')

    print(f"Training with {num_classes} classes: {list(class_labels.values())}")

    optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, 0.999), weight_decay=1e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, 0.999), weight_decay=1e-4)

    adversarial_loss = nn.BCELoss()

    g_losses = []
    d_losses = []
    task_losses = []
    task_accuracies = []
    g_adversarial_losses = []
    test_task_losses = []
    test_task_accuracies = []

    per_class_test_accuracies = {i: [] for i in range(num_classes)}

    num_batches = len(x_train) // batch_size  # x_train is always the smaller dataset
    print(f"Num batches per epoch: {num_batches}")

    train_augment = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    eval_rng = np.random.default_rng(42)
    balanced_test_sets = {}
    for class_idx in range(num_classes):
        x_bal, y_bal = build_balanced_test_set(x_test, y_test, class_idx, rng=eval_rng)
        balanced_test_sets[class_idx] = (x_bal, y_bal)
        print(f"  Balanced test for class {class_labels[class_idx]}: "
              f"{int(y_bal.sum())} pos / {int(len(y_bal) - y_bal.sum())} neg")

    balanced_test_tensors = {
        ci: torch.from_numpy(x_bal).float().to(device)
        for ci, (x_bal, _) in balanced_test_sets.items()
    }

    eval_model = LeNet5Eval().to(device)

    print(f"\nStarting training for {num_batches} batches per epoch...")
    print("=" * 70)

    best_test_acc = 0

    for epoch in range(1, num_epochs + 1):

        if epoch % 200 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\nEpoch {epoch}/{num_epochs}:")

        epoch_g_loss = []
        epoch_d_loss = []
        epoch_task_loss = []
        epoch_task_acc = []
        epoch_g_adversarial_loss = []

        eval_predictions_lst = []
        eval_binary_labels_lst = []

        shuffle_idx = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[shuffle_idx]
        y_train_shuffled = y_train[shuffle_idx]

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{num_epochs}")

        generator.train()
        discriminator.train()

        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            x_batch = x_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            batch_real_label = torch.ones((batch_size, 1), dtype=torch.float32).to(device)
            batch_fake_label = torch.zeros((batch_size, 1), dtype=torch.float32).to(device)

            #########################
            # Train Discriminator
            #########################
            optimizer_D.zero_grad()
            d_loss_total = 0.0

            for class_idx in range(num_classes):
                gen_weights_flat_d = generator(cond_id=class_idx, batch_size=batch_size)
                gen_labels_d = torch.full((batch_size,), class_idx, dtype=torch.long).to(device)

                class_pool = real_weights_by_class[class_idx]  # CPU tensor
                real_indices = torch.randint(0, len(class_pool), (batch_size,))
                real_weights_batch = class_pool[real_indices].to(device)  # move only the batch to GPU
                real_labels_batch = torch.full((batch_size,), class_idx, dtype=torch.long).to(device)

                fake_output = discriminator(gen_weights_flat_d.detach(), gen_labels_d)
                d_fake_loss = adversarial_loss(fake_output, batch_fake_label)

                real_output = discriminator(real_weights_batch, real_labels_batch)
                d_real_loss = adversarial_loss(real_output, batch_real_label)

                d_loss_total += (d_real_loss + d_fake_loss)

            d_loss = d_loss_total/num_classes
            d_loss.backward()
            optimizer_D.step()

            #########################
            # Train Generator
            #########################
            optimizer_G.zero_grad()

            g_adversarial_loss_total = 0
            task_loss_total = 0
            task_acc_total = 0

            x_batch_aug = torch.stack([train_augment(img) for img in torch.from_numpy(x_batch).float()])

            for class_idx in range(num_classes):
                gen_weights_flat = generator(cond_id=class_idx, batch_size=batch_size)
                gen_labels = torch.full((batch_size,), class_idx, dtype=torch.long).to(device)

                fake_output = discriminator(gen_weights_flat, gen_labels)
                g_adv_loss = adversarial_loss(fake_output, batch_real_label)
                g_adversarial_loss_total += g_adv_loss

                t_loss, t_acc = compute_task_loss_batch(
                    gen_weights_flat,
                    x_batch_aug,
                    y_batch,
                    class_idx,
                    device,
                    w_abs_max_by_class[class_idx]
                )
                if t_loss is not None:
                    task_loss_total += t_loss
                    task_acc_total  += t_acc

            g_adversarial_loss = g_adversarial_loss_total / num_classes
            task_loss = task_loss_total / num_classes
            task_acc = task_acc_total  / num_classes

            g_loss = g_adversarial_loss_weight * g_adversarial_loss + task_loss_weight * task_loss

            g_loss.backward()
            optimizer_G.step()

            epoch_g_loss.append(g_loss.item())
            epoch_d_loss.append(d_loss.item())
            epoch_task_loss.append(task_loss.item() if hasattr(task_loss, 'item') else float(task_loss))
            epoch_task_acc.append(task_acc.item() if hasattr(task_acc, 'item') else float(task_acc))
            epoch_g_adversarial_loss.append(g_adversarial_loss.item())

            pbar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}',
                'Task_loss': f'{task_loss:.4f}',
                'Task_acc': f'{task_acc:.4f}'
            })

        g_losses.append(np.mean(epoch_g_loss))
        d_losses.append(np.mean(epoch_d_loss))
        task_losses.append(np.mean(epoch_task_loss))
        task_accuracies.append(np.mean(epoch_task_acc))
        g_adversarial_losses.append(np.mean(epoch_g_adversarial_loss))

        #########################################################
        # Evaluation
        #########################################################
        generator.eval()

        with torch.no_grad():
            all_class_test_losses = []

            for class_idx in range(num_classes):
                gen_weights = generator(cond_id=class_idx, batch_size=2)
                weights_flat = gen_weights[0] * w_abs_max_by_class[class_idx]  # scale to real weight range

                load_weights_into_lenet5_pytorch(eval_model, weights_flat)
                eval_model.eval()

                if class_idx not in balanced_test_sets:
                    continue

                _, y_bal_binary = balanced_test_sets[class_idx]
                x_bal_tensor = balanced_test_tensors[class_idx]  # already on GPU

                with torch.no_grad():
                    preds = eval_model(x_bal_tensor).cpu().numpy().flatten()
                preds = np.clip(preds, 1e-7, 1 - 1e-7)

                #Per-class binary loss
                loss = -np.mean(y_bal_binary * np.log(preds) + (1 - y_bal_binary) * np.log(1 - preds))
                all_class_test_losses.append(float(loss))

                #Per-class accuracy on balanced set
                class_acc = float(np.mean((preds > 0.5) == y_bal_binary))
                per_class_test_accuracies[class_idx].append(class_acc)

                print(f"  Class {class_labels[class_idx]}: "
                      f"loss={loss:.4f}, acc={class_acc:.4f} "
                      f"(balanced {len(y_bal_binary)} samples)")

                eval_predictions_lst.append(preds)
                eval_binary_labels_lst.append(y_bal_binary)

            #Overall accuracy across all balanced class evaluations
            if eval_predictions_lst:
                flat_preds = np.concatenate(eval_predictions_lst).tolist()
                flat_blabels = np.concatenate(eval_binary_labels_lst).tolist()

                predicted_classes = (np.array(flat_preds) >= 0.5).astype(int)
                test_acc = float(np.mean(predicted_classes == flat_blabels))
                test_loss = np.mean(all_class_test_losses)
            else:
                flat_preds, flat_blabels, predicted_classes = [], [], []
                test_acc, test_loss = 0.0, 0.0

        test_task_losses.append(test_loss)
        test_task_accuracies.append(test_acc)

        print(f"  Combined Test Loss: {test_loss:.4f}, Combined Test Acc: {test_acc:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            checkpoint_path = f'{model_dir}/best_checkpoint.pt'
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'z_embeddings': generator.z_embeddings.weight.detach().clone(),
                'w_abs_max': w_abs_max,
                'w_abs_max_by_class': w_abs_max_by_class,
                'g_losses': g_losses,
                'd_losses': d_losses,
                'task_losses': task_losses,
                'task_accuracies': task_accuracies,
                'test_task_accuracies': test_task_accuracies,
                'per_class_test_accuracies': per_class_test_accuracies,
            }, checkpoint_path)
            print(f"  New best! Saved checkpoint: {checkpoint_path}")

        if early_stopping:
            early_stopping.check_early_stop(test_acc)
            if early_stopping.stop_training:
                print(f"Early stopping at epoch {epoch}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"\nEpoch {epoch} Summary:")
            print(f"  G Loss: {g_losses[-1]:.4f}")
            print(f"  D Loss: {d_losses[-1]:.4f}")
            print(f"  Task Loss: {task_losses[-1]:.4f}")
            print(f"  Task Accuracy: {task_accuracies[-1]:.4f}")

    #Final plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(g_losses, label='G Loss')
    axes[0, 0].plot(d_losses, label='D Loss')
    axes[0, 0].plot(task_losses, label='Train Task Loss')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(); axes[0, 0].grid(True)
    axes[0, 0].set_title('Training Losses')

    axes[0, 1].plot(task_accuracies, label='Train Accuracy')
    axes[0, 1].plot(test_task_accuracies, label='Test Accuracy')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend(); axes[0, 1].grid(True)
    axes[0, 1].set_title('Combined Accuracy (All Classes)')

    for class_idx in range(num_classes):
        axes[1, 0].plot(per_class_test_accuracies[class_idx],
                        label=f'{class_labels[class_idx]}', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Test Accuracy')
    axes[1, 0].legend(fontsize=8, loc='lower right')
    axes[1, 0].grid(True); axes[1, 0].set_title('Per-Class Test Accuracy (Balanced)')

    final_accs = [per_class_test_accuracies[i][-1] if per_class_test_accuracies[i] else 0
                  for i in range(num_classes)]
    axes[1, 1].bar(range(num_classes), final_accs)
    axes[1, 1].set_xticks(range(num_classes))
    axes[1, 1].set_xticklabels([class_labels[i][:8] for i in range(num_classes)],
                                rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_ylabel('Final Test Accuracy')
    axes[1, 1].set_title('Final Per-Class Test Accuracy (Balanced)')
    axes[1, 1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/training_plots_v{version}_epochs{num_epochs}_mpc{traindata_samp_size}_{current_datetime}.png")
    plt.close()

    return {
        'g_losses': g_losses,
        'd_losses': d_losses,
        'task_losses': task_losses,
        'task_accuracies': task_accuracies,
        'g_adversarial_losses': g_adversarial_losses,
        'test_task_losses': test_task_losses,
        'test_task_accuracies': test_task_accuracies,
        'per_class_test_accuracies': per_class_test_accuracies,
        'z_embeddings': generator.z_embeddings.weight.detach().clone(),
        'w_abs_max': w_abs_max,
        'w_abs_max_by_class': w_abs_max_by_class,
    }, eval_predictions_lst, eval_binary_labels_lst, predicted_classes, test_acc

########################################################################
#                    Main Execution
#########################################################################
if __name__ == "__main__":
 
    version = '10.3.3'
    traindata_samp_size = 10000
    rel_traindata_path = '/homes/dkurtenb/projects/HyperGENAI/hnet_trainingdata_build_pytorch_10000/output/lenet5_10000_onevall_wghts_bymodel_20260326_080442.pkl'    
    output_pth = '/homes/dkurtenb/projects/HyperGENAI/hyperGAN/outputs/'
    data_dir = '/homes/dkurtenb/projects/HyperGENAI/hyperGAN/data/tensorflow_datasets'
 
    config = {
        'num_classes': 10,
        'batch_size': 16,
        'num_epochs': 1500,
        'latent_dim': 64,
        'noise_dim': 32,
        'task_loss_weight': 0.69,
        'g_adversarial_loss_weight': 0.34,
        'lr_g': 0.0001,
        'lr_d': 0.00001,
        'beta1': 0.22,
        'device': device,
        'hidden_dim': 128,
        'embedding_dim': 64,
        'dropout_rate': 0.4,
        'gen_dropout_rate': 0.4,
        'dis_leakyrelu_slope': 0.3,
        'gen_leakyrelu_slope': 0.3,
        'patience': 5,
    }
 
    print("\n" + "=" * 70)
    print("GAN-based Hypernetwork for 10 Class Imagenette Binary Classification")
    print("=" * 70)
 
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_pth, f"GAHypN_v{version}_numperclass{traindata_samp_size}_{current_datetime}_epochs_{config['num_epochs']}")
 
    model_dir = os.path.join(save_dir, "models")
    plot_dir  = os.path.join(save_dir, "plots")
 
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
 
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
 
    class_labels = {
        0: 'tench',
        1: 'english_springer',
        2: 'cassette_player',
        3: 'chain_saw',
        4: 'church',
        5: 'french_horn',
        6: 'garbage_truck',
        7: 'gas_pump',
        8: 'golf_ball',
        9: 'parachute'
    }    

    real_weights_tensor, real_labels_tensor, weight_vector_len = load_lenetwghts_from_pickl( rel_traindata_path, class_labels, config['num_classes'],traindata_samp_size)

    expected_params = FunctionalLeNet5.get_total_params()
    print(f"Expected LeNet params: {expected_params}")
    print(f"Actual weight vector length: {weight_vector_len}")

    if weight_vector_len != expected_params:
        print(f"WARNING: Weight vector length mismatch!")

    print("\nClass distribution in real weights:")
    for class_idx in range(config['num_classes']):
        count = (real_labels_tensor == class_idx).sum().item()
        print(f"  Class {class_idx} ({class_labels[class_idx]}): {count} samples")

    real_weights_by_class = {}
    w_abs_max_by_class = {}
    for _cls in range(config['num_classes']):
        _mask = real_labels_tensor == _cls # T/F tensor for each class
        _cls_w = real_weights_tensor[_mask] # tensor of wghts for that class based on T/F _mask        
        _w_max = _cls_w.abs().max().item() #max value in wghts
        w_abs_max_by_class[_cls] = _w_max
        real_weights_by_class[_cls] = _cls_w / _w_max #normalized wghts for each class. This normalizes by abs max over all wghts in each class
    del real_weights_tensor, real_labels_tensor     
    gc.collect()
 
    num_classes = config['num_classes']
    latent_dim = config['latent_dim']
    noise_dim = config['noise_dim']
    embedding_dim = config['embedding_dim']
    hidden_dim = config['hidden_dim']
    gen_leakyrelu_slope = config['gen_leakyrelu_slope']
    dropout_rate = config['dropout_rate']
    gen_dropout_rate = config['gen_dropout_rate']
    dis_leakyrelu_slope = config['dis_leakyrelu_slope']
 
    print("\nInitializing Generator")
    generator = ConditionalWeightGenerator(
        latent_dim = latent_dim,
        noise_dim = noise_dim,
        weight_vector_len = weight_vector_len,
        num_classes = num_classes,
        hidden_dim = hidden_dim,
        gen_leakyrelu_slope = gen_leakyrelu_slope,
        gen_dropout_rate = gen_dropout_rate
    ).to(device)
 
    weight_shapes = FunctionalLeNet5.get_weight_shapes()

    #apply_hyperfan_pytorch(generator, weight_shapes, nonlinearity='tanh')

    print(f"Generator created with {sum(p.numel() for p in generator.parameters())} parameters")
    print(f"Per-class z embeddings (conditional_params): {generator.z_embeddings.weight.shape}")
 
    print("\nInitializing Discriminator")
    discriminator = ConditionalDiscriminator(
        weight_vector_len = weight_vector_len,
        num_classes = num_classes,
        embedding_dim = embedding_dim,
        hidden_dim = hidden_dim,
        dropout_rate = dropout_rate,
        dis_leakyrelu_slope = dis_leakyrelu_slope
    ).to(device)
 
    print(f"Discriminator created with {sum(p.numel() for p in discriminator.parameters())} parameters")
 
    print("\nConfiguration:")
    print(f"  Num classes: {num_classes}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Latent dim (z size per class): {latent_dim}")
    print(f"  Task loss weight: {config['task_loss_weight']}")
    print(f"  Weight vector length: {weight_vector_len}")
 
    x_train, y_train, x_test, y_test = load_imagenette(data_dir)
 
    early_stopping = EarlyStopping(patience=config['patience'], delta=0.01, verbose=True)
 
    x_train_tensor = torch.from_numpy(x_train).float()
    print(f"\nx_train information: \n\tType: {type(x_train_tensor)} \n\tShape: {x_train_tensor.shape}")
 
    results, predictions_lst, binary_labels_lst, predicted_classes, test_acc = train_gan(
        generator = generator,
        discriminator = discriminator,
        real_weights_by_class = real_weights_by_class,
        w_abs_max_by_class = w_abs_max_by_class,
        x_train = x_train,
        y_train = y_train,
        x_test = x_test,
        y_test = y_test,
        model_dir = model_dir,
        plot_dir = plot_dir,
        class_labels = class_labels,
        traindata_samp_size = traindata_samp_size,
        config = config,
        current_datetime = current_datetime,
        version = version,
        early_stopping = None
    )
 
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best test accuracy: {max(results['test_task_accuracies']):.4f}")
    print("\nFinal per-class test accuracies (balanced):")
    for class_idx in range(num_classes):
        if results['per_class_test_accuracies'][class_idx]:
            final_acc = results['per_class_test_accuracies'][class_idx][-1]
            print(f"  {class_labels[class_idx]}: {final_acc:.4f}")
    print("=" * 70)
 
    test_result = {
        "Best G Loss": min(results['g_losses']),
        "Average G Loss": np.mean(results['g_losses']),
        "Best D Loss": min(results['d_losses']),
        "Average D Loss": np.mean(results['d_losses']),
        "Best Task Loss": min(results['task_losses']),
        "Average Task Loss": np.mean(results['task_losses']),
        "Best Task Accuracy": max(results['task_accuracies']),
        "Average Task Accuracy": np.mean(results['task_accuracies']),
        "Best Test Accuracy": max(results['test_task_accuracies']),
        "Average Test Accuracy": np.mean(results['test_task_accuracies']),
    }
 
    with open(os.path.join(save_dir, f"config_file_{version}_epochs{config['num_epochs']}_mpc{traindata_samp_size}.txt"), "w") as file:
        file.write(str(config))
        file.write(str(test_result))