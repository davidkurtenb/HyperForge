import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import gc

import sys
sys.path.append(r'C:\Users\dk412\Desktop\David\Python Projects\HyperGenAI\hyperGAN\src\helpers')
from data_loader import *
from hnet_cond_GAHypN import *
from mnet_lenet5_binary import *

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
#                    Training Function
#########################################################################
def train_gan(generator, discriminator, real_weights_tensor, real_labels_tensor, 
              x_train, y_train, x_test, y_test, model_dir, plot_dir, class_labels, 
              config, current_datetime, version, early_stopping):

    num_classes = config['num_classes']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    latent_dim = config['latent_dim']
    task_loss_weight = config['task_loss_weight']
    g_adversarial_loss_weight = config['g_adversarial_loss_weight']
    lr_g = config['lr_g']
    lr_d = config['lr_d']
    beta1 = config['beta1']
    device = config['device']
    z_test_batch_size = config['z_test_batch_size']
    
    #binary_class_idx = 0
    #target_imagenette_class = test_class[0]  # Original class ID for binary label computation
    
    #print(f"Target Imagenette class: {target_imagenette_class} ({test_class[1]})")
    #print(f"Binary class index for embedding: {binary_class_idx}")
    print(f"Training with {num_classes} classes: {list(class_labels.values())}")

    optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, 0.999), weight_decay = 1e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, 0.999), weight_decay = 1e-4)
    
    adversarial_loss = nn.BCELoss()
    
    g_losses = []
    d_losses = []
    task_losses = []
    task_accuracies = []
    g_adversarial_losses = []
    test_task_losses = []
    test_task_accuracies = []

    per_class_test_accuracies = {i: [] for i in range(num_classes)}

    num_batches = min(len(real_weights_tensor), len(x_train)) // batch_size
    print(f"Num batches per epoch: {num_batches}")
    
    #z = torch.randn(1, latent_dim).to(device)  
    z_eval = torch.randn(1, latent_dim).to(device) # adding dyanmic latent vector
    
    print(f"Starting training for {num_batches} batches per epoch...")
    print("=" * 70)
    
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

    for epoch in range(1, num_epochs + 1):

        if epoch % 200 == 0:
            tf.keras.backend.clear_session()
            eval_models = []
            for _ in range(num_classes):
                model = build_lenet5_keras()
                model.compile(optimizer='adam', loss='binary_crossentropy')
                eval_models.append(model)            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        print(f"\nEpoch {epoch}/{num_epochs}:")
        
        epoch_g_loss = []
        epoch_d_loss = []
        epoch_task_loss = []
        epoch_task_acc = []
        epoch_g_adversarial_loss = []
        
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
            #batch_real_label = torch.full((batch_size,1), 0.9, dtype=torch.float32).to(device)  # Added Label Smoothing
            #batch_fake_label = torch.full((batch_size,1), 0.1, dtype=torch.float32).to(device)  # Added Label Smoothing
            
            #gen_labels = torch.full((batch_size,), binary_class_idx, dtype=torch.long).to(device)
            
            #z_batch = z.expand(batch_size, -1)
            #z_batch = torch.randn(batch_size,latent_dim).to(device) # adding dyanmic latent vector

            #########################
            # Train Discriminator
            #########################
            optimizer_D.zero_grad()
            
            d_loss_total = 0

            for class_idx in range(num_classes):
                gen_labels = torch.full((batch_size,), class_idx, dtype = torch.long).to(device)
                z_batch = torch.randn(batch_size, latent_dim).to(device)
                gen_weights_flat = generator(z_batch, gen_labels)

                class_mask = real_labels_tensor == class_idx
                class_real_weights = real_weights_tensor[class_mask]
                real_indices = torch.randint(0, len(class_real_weights), (batch_size,))
                real_weights_batch = real_weights_tensor[real_indices].to(device)
                real_labels_batch = torch.full((batch_size,), class_idx, dtype=torch.long).to(device)
            
                fake_output = discriminator(gen_weights_flat.detach(), gen_labels)
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

            #------------------- Task Data Augmentation
            x_batch_tensor = torch.from_numpy(x_batch).float().permute(0, 3, 1, 2)   
            x_batch_aug = train_augment(x_batch_tensor).permute(0, 2, 3, 1).numpy() 
            #------------------- Task Data Augmentation

            for class_idx in range(num_classes):
                gen_labels = torch.full((batch_size,), class_idx, dtype=torch.long).to(device)
                z_batch = torch.randn(batch_size, latent_dim).to(device)
                gen_weights_flat = generator(z_batch, gen_labels)
            
                fake_output = discriminator(gen_weights_flat, gen_labels)
                g_adv_loss = adversarial_loss(fake_output, batch_real_label)
                g_adversarial_loss_total += g_adv_loss
                #g_adversarial_loss = adversarial_loss(fake_output, batch_real_label)

                target_imagenette_class = class_idx
                
                task_losses_batch = []
                task_accs_batch = []

                #for i in range (min(4, batch_size)):
                for i in range (min(8, batch_size)):
                    t_loss, t_acc = compute_task_loss_differentiable(
                        gen_weights_flat[i],  
                        #x_batch,
                        x_batch_aug,  # Task Data Agumentation
                        y_batch,
                        target_imagenette_class,
                        device
                    )

                    task_losses_batch.append(t_loss)
                    task_accs_batch.append(t_acc)
                
                task_loss_class = torch.stack(task_losses_batch).mean()
                task_acc_class = np.mean(task_accs_batch)

                task_loss_total += task_loss_class
                task_acc_total += task_acc_class
                
            g_adversarial_loss = g_adversarial_loss_total/num_classes
            task_loss = task_loss_total/num_classes
            task_acc = task_acc_total/num_classes

            g_loss = g_adversarial_loss_weight * g_adversarial_loss + task_loss_weight * task_loss
            
            g_loss.backward()
            optimizer_G.step()
            
            epoch_g_loss.append(g_loss.item())
            epoch_d_loss.append(d_loss.item())
            epoch_task_loss.append(task_loss.item())
            epoch_task_acc.append(task_acc)
            epoch_g_adversarial_loss.append(g_adversarial_loss.item())
            
            pbar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}',
                'Task_loss': f'{task_loss.item():.4f}',
                'Task_acc': f'{task_acc:.4f}'
            })
        
        g_losses.append(np.mean(epoch_g_loss))
        d_losses.append(np.mean(epoch_d_loss))
        task_losses.append(np.mean(epoch_task_loss))
        task_accuracies.append(np.mean(epoch_task_acc))
        g_adversarial_losses.append(np.mean(epoch_g_adversarial_loss))
        
        # Evaluation
        generator.eval()
        
        with torch.no_grad():
            all_class_test_accs = []
            all_class_test_losses = []

            for class_idx in range(num_classes):
                target_imagenette_class = class_idx
                test_label = torch.tensor([target_imagenette_class], dtype=torch.long).to(device)

                #gen_weights = generator(z, test_label)

                class_test_accs =[]
                class_test_losses = []

                for _ in range(z_test_batch_size):
                    z_test = torch.randn(1, latent_dim).to(device)
                    gen_weights = generator(z_test, test_label)
                    t_loss, t_acc = compute_task_loss_eval(
                        gen_weights[0], 
                        x_test, 
                        y_test, 
                        target_imagenette_class, 
                        device,
                        model = eval_models[class_idx]
                        )
                    class_test_losses.append(t_loss)
                    class_test_accs.append(t_acc)
                                    
                class_avg_loss = np.mean(class_test_losses)
                class_avg_acc = np.mean(class_test_accs)

                all_class_test_losses.append(class_avg_loss)
                all_class_test_accs.append(class_avg_acc)
                per_class_test_accuracies[class_idx].append(class_avg_acc)

            test_loss = np.mean(all_class_test_losses)
            test_acc = np.mean(all_class_test_accs)
            
        test_task_losses.append(test_loss)
        test_task_accuracies.append(test_acc)
        
        print(f"  Combined Test Loss: {test_loss:.4f}, Combined Test Acc: {test_acc:.4f}")
        print(f"  Per-class Test Acc: {[f'{acc:.3f}' for acc in all_class_test_accs]}")

        # Save best checkpoint
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            checkpoint_path = f'{model_dir}/best_checkpoint.pt'
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                #'z_embedding': z,
                'z_embedding': z_eval, # adding dyanmic latent vector
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
    
    # Final plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss plot
    axes[0, 0].plot(g_losses, label='G Loss')
    axes[0, 0].plot(d_losses, label='D Loss')
    axes[0, 0].plot(task_losses, label='Train Task Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_title('Training Losses')
    
    # Combined accuracy plot
    axes[0, 1].plot(task_accuracies, label='Train Accuracy')
    axes[0, 1].plot(test_task_accuracies, label='Test Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_title('Combined Accuracy (All Classes)')
    
    # Per-class test accuracy plot
    for class_idx in range(num_classes):
        axes[1, 0].plot(per_class_test_accuracies[class_idx], 
                        label=f'{class_labels[class_idx]}', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Test Accuracy')
    axes[1, 0].legend(fontsize=8, loc='lower right')
    axes[1, 0].grid(True)
    axes[1, 0].set_title('Per-Class Test Accuracy')
    
    # Final per-class accuracy bar chart
    final_accs = [per_class_test_accuracies[i][-1] if per_class_test_accuracies[i] else 0 
                  for i in range(num_classes)]
    axes[1, 1].bar(range(num_classes), final_accs)
    axes[1, 1].set_xticks(range(num_classes))
    axes[1, 1].set_xticklabels([class_labels[i][:8] for i in range(num_classes)], 
                                rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_ylabel('Final Test Accuracy')
    axes[1, 1].set_title('Final Per-Class Test Accuracy')
    axes[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/training_plots_v{version}_{current_datetime}.png")
    plt.close()

    return {
        'g_losses': g_losses,
        'd_losses': d_losses,
        'task_losses': task_losses,
        'task_accuracies': task_accuracies,
        'g_adversarial_losses': g_adversarial_losses,
        'test_task_accuracies': test_task_accuracies,
        'per_class_test_accuracies': per_class_test_accuracies,
        'z_embedding': z_eval
    }
