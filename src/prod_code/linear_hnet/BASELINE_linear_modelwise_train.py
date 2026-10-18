import torch
import numpy as np
import os
from datetime import datetime
import tensorflow as tf
from tqdm import tqdm
import pandas as pd

import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import gc

import sys
sys.path.append('/homes/dkurtenb/projects/HyperGENAI/hyperGAN/src/helpers/')
from data_loader import *
from hnet_linear import *
from mnet_lenet5_binary import *


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
        #if self.best_test_acc is None or test_acc < self.best_test_acc - self.delta: #stops on min use it for val_loss not acc
            self.best_test_acc = test_acc
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")

#########################################################################
#                    Training Function (Linear Hypernetwork Baseline)
#########################################################################
def train_linear_hnet(hnet, real_weights_tensor, real_labels_tensor,
                      x_train, y_train, x_test, y_test, model_dir, plot_dir, class_labels,
                      config, current_datetime, version, early_stopping):

    num_classes = config['num_classes']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    task_loss_weight = config['task_loss_weight']
    lr = config['lr']
    device = config['device']

    print(f"Training with {num_classes} classes: {list(class_labels.values())}")

    optimizer = optim.Adam(hnet.parameters(), lr=lr, weight_decay=1e-4)

    task_losses = []
    task_accuracies = []
    test_task_losses = []
    test_task_accuracies = []
    per_class_test_accuracies = {i: [] for i in range(num_classes)}

    num_batches = len(x_train) // batch_size
    print(f"Num batches per epoch: {num_batches}")

    #--------------------------------- Task Data Augmentation
    train_augment = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5
        ),
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])
    #---------------------------------------------------------

    best_test_acc = 0
    eval_models = []
    for _ in range(num_classes):
        model = build_lenet5_keras()
        model.compile(optimizer='adam', loss='binary_crossentropy')
        eval_models.append(model)

    print(f"Initial MNET Model Setup\n{eval_models}")

    for epoch in range(1, num_epochs + 1):

        if epoch % 200 == 0:
            tf.keras.backend.clear_session()
            eval_models = []
            for _ in range(num_classes):
                model = build_lenet5_keras()
                model.compile(optimizer='adam', loss='binary_crossentropy')
                eval_models.append(model)
            print(f"Cache Clear MNET Model Setup\n{eval_models}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\nEpoch {epoch}/{num_epochs}:")

        epoch_task_loss = []
        epoch_task_acc = []

        shuffle_idx = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[shuffle_idx]
        y_train_shuffled = y_train[shuffle_idx]

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{num_epochs}")

        hnet.train()

        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            x_batch = x_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            #------------------- Task Data Augmentation
            x_batch_tensor = torch.from_numpy(x_batch).float().permute(0, 3, 1, 2)
            x_batch_aug = train_augment(x_batch_tensor).permute(0, 2, 3, 1).numpy()
            #-------------------------------------------

            optimizer.zero_grad()

            task_loss_total = 0
            task_acc_total = 0

            for class_idx in range(num_classes):
                label = torch.tensor([class_idx], dtype=torch.long).to(device)
                gen_weights = hnet(label)  # shape: (1, weight_vector_len)

                t_loss, t_acc = compute_task_loss_differentiable(
                    gen_weights[0],
                    x_batch_aug,
                    y_batch,
                    class_idx,
                    device
                )

                task_loss_total += t_loss
                task_acc_total += t_acc

            task_loss = task_loss_total / num_classes
            task_acc = task_acc_total / num_classes

            loss = task_loss_weight * task_loss
            loss.backward()
            optimizer.step()

            epoch_task_loss.append(task_loss.item())
            epoch_task_acc.append(task_acc)

            pbar.set_postfix({
                'Task_loss': f'{task_loss.item():.4f}',
                'Task_acc': f'{task_acc:.4f}'
            })

        task_losses.append(np.mean(epoch_task_loss))
        task_accuracies.append(np.mean(epoch_task_acc))

        # Evaluation
        hnet.eval()

        with torch.no_grad():
            all_class_test_accs = []
            all_class_test_losses = []

            for class_idx in range(num_classes):
                label = torch.tensor([class_idx], dtype=torch.long).to(device)
                gen_weights = hnet(label)

                t_loss, t_acc = compute_task_loss_eval(
                    gen_weights[0],
                    x_test,
                    y_test,
                    class_idx,
                    device,
                    model=eval_models[class_idx]
                )
                all_class_test_losses.append(t_loss)
                all_class_test_accs.append(t_acc)
                per_class_test_accuracies[class_idx].append(t_acc)

            test_loss = np.mean(all_class_test_losses)
            test_acc = np.mean(all_class_test_accs)

        test_task_losses.append(test_loss)
        test_task_accuracies.append(test_acc)

        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print(f"  Per-class Test Acc: {[f'{acc:.3f}' for acc in all_class_test_accs]}")

        # Save best checkpoint
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            checkpoint_path = f'{model_dir}/best_checkpoint.pt'
            torch.save({
                'epoch': epoch,
                'hnet_state_dict': hnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
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
            print(f"  Task Loss: {task_losses[-1]:.4f}")
            print(f"  Task Accuracy: {task_accuracies[-1]:.4f}")

    # Final plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(task_losses, label='Train Task Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title('Training Loss')

    axes[1].plot(task_accuracies, label='Train Accuracy')
    axes[1].plot(test_task_accuracies, label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title('Combined Accuracy (All Classes)')

    for class_idx in range(num_classes):
        axes[2].plot(per_class_test_accuracies[class_idx],
                     label=f'{class_labels[class_idx]}', alpha=0.7)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Test Accuracy')
    axes[2].legend(fontsize=8, loc='lower right')
    axes[2].grid(True)
    axes[2].set_title('Per-Class Test Accuracy')

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/training_plots_v{version}_{current_datetime}.png")
    plt.close()

    return {
        'task_losses': task_losses,
        'task_accuracies': task_accuracies,
        'test_task_accuracies': test_task_accuracies,
        'per_class_test_accuracies': per_class_test_accuracies,
    }


#########################################################################
#                    Main Execution
#########################################################################
if __name__ == "__main__":

    version = 'BASELINE_LINEAR'
    rel_traindata_path = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/lenet5_onevall_10000_models/pickle_files/lenet5_onevall_wghts_bymodel_20250502_102356.pkl'
    output_pth = '/homes/dkurtenb/projects/HyperGENAI/hyperGAN/outputs/'
    imagenet_data_dir = '/homes/dkurtenb/projects/HyperGENAI/hyperGAN/data/tensorflow_datasets'


    config = {
        'num_classes': 10,
        'batch_size': 16,
        'num_epochs': 500,
        'task_loss_weight': 1.0,
        'lr': 1e-3,
        'device': device,
        'embedding_dim': 50,
        'patience': 5,
    }

    print("\n" + "=" * 70)
    print("Linear Hypernetwork Baseline for 10-Class Imagenette Binary Classification")
    print("=" * 70)

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_pth, f"LinearHypN_v{version}_{current_datetime}_epochs_{config['num_epochs']}")

    model_dir = os.path.join(save_dir, "models")
    plot_dir = os.path.join(save_dir, "plots")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Seed
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

    real_weights_tensor, real_labels_tensor, weight_vector_len, data_lst = load_lenetwghts_from_pickl(
        rel_traindata_path, class_labels, config['num_classes']
    )

    expected_params = FunctionalLeNet5.get_total_params()
    print(f"Expected LeNet params: {expected_params}")
    print(f"Weight Tensor Shape: {real_weights_tensor.shape}")

    if weight_vector_len != expected_params:
        print(f"WARNING: Weight vector length mismatch! Expected {expected_params}, got {weight_vector_len}")

    print("\nClass distribution in real weights:")
    for class_idx in range(config['num_classes']):
        count = (real_labels_tensor == class_idx).sum().item()
        print(f"  Class {class_idx} ({class_labels[class_idx]}): {count} samples")

    num_classes = config['num_classes']
    embedding_dim = config['embedding_dim']

    print("\nInitializing Linear Hypernetwork")
    hnet = LinearHypernetwork(
        num_classes=num_classes,
        weight_vector_len=weight_vector_len,
        embedding_dim=embedding_dim
    ).to(device)

    hyperfan_in_init(hnet)

    print(f"LinearHypernetwork created with {sum(p.numel() for p in hnet.parameters())} parameters")

    print("\nArchitecture:")
    print(hnet)

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    print(f"  weight_vector_len: {weight_vector_len}")

    x_train, y_train, x_test, y_test = load_imagenette(imagenet_data_dir)

    early_stopping = EarlyStopping(patience=config['patience'], delta=0.01, verbose=True)

    results = train_linear_hnet(
        hnet=hnet,
        real_weights_tensor=real_weights_tensor,
        real_labels_tensor=real_labels_tensor,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        model_dir=model_dir,
        plot_dir=plot_dir,
        class_labels=class_labels,
        config=config,
        current_datetime=current_datetime,
        version=version,
        early_stopping=False
    )

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best test accuracy: {max(results['test_task_accuracies']):.4f}")
    print("\nFinal per-class test accuracies:")
    for class_idx in range(num_classes):
        final_acc = results['per_class_test_accuracies'][class_idx][-1]
        print(f"  {class_labels[class_idx]}: {final_acc:.4f}")
    print("=" * 70)

    with open(os.path.join(save_dir, "config_file.txt"), "w") as file:
        file.write(str(config))