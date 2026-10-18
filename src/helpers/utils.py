import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pickle
import os
import torch.nn.functional as F
import gc
import pandas as pd
from torch.func import vmap
import tensorflow_datasets as tfds
import tensorflow as tf

import sys
sys.path.append(r'C:\Users\dk412\Desktop\David\Python Projects\HyperGenAI\hyperGAN\src')

from models.hyperforge import FunctionalLeNet5, LeNet5Eval, load_weights_into_lenet5_pytorch, assemble_full_weights

#########################################################################
# Load Imagenette Data
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
# Load Real Trained Weights from Pickle
#########################################################################
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        return data

def load_lenetwghts_from_pickl(rel_traindata_path, class_labels, num_classes, traindata_samp_size, train_data_performance_csv_path):
    ###### FILTER THE TRAINING DATA TO TOP N SAMPLES PER CLASS BASED ON PERFORMANCE METRICS CSV 
    #train_data_performance_df = pd.read_csv('/homes/dkurtenb/projects/HyperGENAI/hnet_trainingdata_build_pytorch_10000/output/run_20260314_090634/model_performance_metrics.csv')
    train_data_performance_df = pd.read_csv(train_data_performance_csv_path)
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
# Early Stopping
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
                    print("Stopping early")

#########################################################################
# Balanced Test Set Builder 
#########################################################################
 
def build_balanced_test_set(x_test, y_test, class_idx, rng=None):
    pos_idx = np.where(y_test == class_idx)[0]
    neg_idx = np.where(y_test != class_idx)[0]

    if len(pos_idx) == 0:
        return None, None  # no data for this class, skip it

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
# Task Loss Computation (DIFFERENTIABLE)
#########################################################################
 
def compute_task_loss_differentiable(weights_flat, x_batch, y_batch, target_class_id, device):
    pos_idx = np.where(y_batch == target_class_id)[0]
    neg_idx = np.where(y_batch != target_class_id)[0]

    if len(pos_idx) == 0:
        return None, None  # no data for this class at all

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

def compute_task_loss_batch(gen_weights_flat, x_batch_aug, y_batch, target_class_id, device, w_abs_max,
                             reference_tail, layer_start):
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

    # Denormalize the generated layer segment, then splice it into the frozen reference weights for every other layer to get full model params.
    #print(f"$$$$$$$$$$$$$$$$$$$$$ gen_weights_flt length: {gen_weights_flat.shape}")
    full_weights_flat = assemble_full_weights(gen_weights_flat * w_abs_max, reference_tail, layer_start)

    def per_weight_forward(w_flat):
        preds = FunctionalLeNet5.forward(x_bal, w_flat)
        loss = F.binary_cross_entropy(preds, y_bal)
        return loss, preds

    losses, all_preds = vmap(per_weight_forward)(full_weights_flat)
    # all_preds: (batch_size, n_images, 1) Ã¢â‚¬â€ y_bal broadcasts as (1, n_images, 1)
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
#                        Generator Hyper Fan-In
#########################################################################
 
def apply_hyperfan_pytorch(generator, weight_shapes, nonlinearity='tanh'):
    """
    HyperFan initialization for the generator's output layer.
    
    Updated for PyTorch weight ordering:
        Conv shapes: (Cout, Cin, H, W) → fan_in = Cin * H * W, fan_out = Cout
        FC shapes:   (out, in)          → fan_in = in, fan_out = out
    """
    final_layer = None
    for m in reversed(list(generator.model.modules())):
        if isinstance(m, nn.Linear):
            final_layer = m
            break
    if final_layer is None:
        raise ValueError("No nn.Linear found in generator.model")
 
    actual_hidden_dim = final_layer.in_features
    print(f"HyperFan: using final Linear({final_layer.in_features} → {final_layer.out_features})")
 
    with torch.no_grad():
        nn.init.zeros_(final_layer.bias)
 
        idx = 0
        for name, shape, size in weight_shapes:
            weight_slice = final_layer.weight[idx : idx + size]
            
            if 'bias' in name and len(shape) == 1:
                nn.init.zeros_(weight_slice)
            elif len(shape) == 4:
                # PyTorch conv shape is (Cout, Cin, H, W)
                fan_in  = shape[1] * shape[2] * shape[3]  # Cin * H * W
                fan_out = shape[0] * shape[2] * shape[3]  # Cout * H * W
                target_var = 2.0 / (fan_in + fan_out) if nonlinearity == 'tanh' else 2.0 / fan_in
                hyperfan_std = math.sqrt(target_var / actual_hidden_dim)
                nn.init.normal_(weight_slice, std=hyperfan_std)
            elif len(shape) == 2:
                # ---- CHANGED: PyTorch FC shape is (out_features, in_features) ----
                fan_out, fan_in = shape
                target_var = 2.0 / (fan_in + fan_out) if nonlinearity == 'tanh' else 2.0 / fan_in
                hyperfan_std = math.sqrt(target_var / actual_hidden_dim)
                nn.init.normal_(weight_slice, std=hyperfan_std)
            
            idx += size
    
    print("HyperFan initialization applied to generator output layer")
    
    print("\nExpected output std per weight group:")
    idx = 0
    for name, shape, size in weight_shapes:
        if 'bias' in name and len(shape) == 1:
            print(f"  {name:15s}: 0.0 (bias → zeros)")
        elif len(shape) == 4:
            fan_in  = shape[1] * shape[2] * shape[3]   # Cin * H * W
            fan_out = shape[0] * shape[2] * shape[3]   # Cout * H * W
            target_var = 2.0 / (fan_in + fan_out)
            print(f"  {name:15s}: std={math.sqrt(target_var):.4f} (fan_in={fan_in}, fan_out={fan_out})")
        elif len(shape) == 2:
            fan_out, fan_in = shape
            target_var = 2.0 / (fan_in + fan_out)
            print(f"  {name:15s}: std={math.sqrt(target_var):.4f} (fan_in={fan_in}, fan_out={fan_out})")
        idx += size
