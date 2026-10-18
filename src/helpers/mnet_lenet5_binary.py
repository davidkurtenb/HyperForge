import torch
import torch.nn.functional as F
import numpy as np

#########################################################################
#                    PyTorch Functional LeNet5 (DIFFERENTIABLE)
#########################################################################

class FunctionalLeNet5:

    @staticmethod
    def get_weight_shapes():
        shapes = [
            ('conv1_b', (6,), 6), # 6
            ('conv1_w', (5, 5, 3, 6), 5*5*3*6), # 450
            ('conv2_b', (16,), 16), # 16
            ('conv2_w', (5, 5, 6, 16), 5*5*6*16), # 2400
            ('conv3_b', (120,), 120), # 120
            ('conv3_w', (5, 5, 16, 120), 5*5*16*120), # 48000
            ('fc1_b', (84,), 84), # 84
            ('fc1_w', (480, 84), 480*84), # 40320
            ('fc2_b', (1,), 1), # 1
            ('fc2_w', (84, 1), 84*1), # 84
        ]
        return shapes
    
    @staticmethod
    def get_total_params():
        return sum(s[2] for s in FunctionalLeNet5.get_weight_shapes())
    
    @staticmethod
    def forward(x, weights_flat):

        if x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        
        shapes = FunctionalLeNet5.get_weight_shapes()
        
        idx = 0
        params = {}
        for name, shape, size in shapes:
            params[name] = weights_flat[idx:idx+size].reshape(shape)
            idx += size
        
        conv1_w = params['conv1_w'].permute(3, 2, 0, 1)
        x = F.conv2d(x, conv1_w, params['conv1_b'], padding=2)
        x = torch.tanh(x)
        x = F.avg_pool2d(x, 2, stride=2)
        
        conv2_w = params['conv2_w'].permute(3, 2, 0, 1)
        x = F.conv2d(x, conv2_w, params['conv2_b'], padding=0)
        x = torch.tanh(x)
        x = F.avg_pool2d(x, 2, stride=2)
        
        conv3_w = params['conv3_w'].permute(3, 2, 0, 1)
        x = F.conv2d(x, conv3_w, params['conv3_b'], padding=0)
        x = torch.tanh(x)
        
        x = x.reshape(x.size(0), -1)  
        
        fc1_w = params['fc1_w'].t()  
        x = F.linear(x, fc1_w, params['fc1_b'])
        x = torch.tanh(x)
        
        fc2_w = params['fc2_w'].t()  
        x = F.linear(x, fc2_w, params['fc2_b'])
        x = torch.sigmoid(x)
        
        return x
    
#########################################################################
#                    Keras LeNet5 for Evaluation ONLY
#########################################################################

def build_lenet5_keras():
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(32, 32, 3), padding='same'),
        layers.AveragePooling2D((2, 2), strides=(2, 2)),
        layers.Conv2D(16, (5, 5), activation='tanh'),
        layers.AveragePooling2D((2, 2), strides=(2, 2)),
        layers.Conv2D(120, (5, 5), activation='tanh'),
        layers.Flatten(),
        layers.Dense(84, activation='tanh'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def load_weights_into_lenet5_keras(model, weights_array):

    if isinstance(weights_array, torch.Tensor):
        weights_array = weights_array.detach().cpu().numpy()
    
    pickle_order = [
        ('conv1_b', (6,)),
        ('conv1_w', (5, 5, 3, 6)),
        ('conv2_b', (16,)),
        ('conv2_w', (5, 5, 6, 16)),
        ('conv3_b', (120,)),
        ('conv3_w', (5, 5, 16, 120)),
        ('fc1_b', (84,)),
        ('fc1_w', (480, 84)),
        ('fc2_b', (1,)),
        ('fc2_w', (84, 1)),
    ]
    
    idx = 0
    extracted = {}
    for name, shape in pickle_order:
        size = int(np.prod(shape))
        extracted[name] = weights_array[idx:idx+size].reshape(shape)
        idx += size
    
    keras_weights = [
        extracted['conv1_w'], # conv1 kernel
        extracted['conv1_b'], # conv1 bias
        extracted['conv2_w'], # conv2 kernel
        extracted['conv2_b'], # conv2 bias
        extracted['conv3_w'], # conv3 kernel
        extracted['conv3_b'], # conv3 bias
        extracted['fc1_w'], # dense1 kernel
        extracted['fc1_b'], # dense1 bias
        extracted['fc2_w'], # dense2 kernel
        extracted['fc2_b'], # dense2 bias
    ]
    
    model.set_weights(keras_weights)
    return model

#########################################################################
#                    Task Loss Computation (DIFFERENTIABLE)
#########################################################################

def compute_task_loss_differentiable(weights_flat, x_batch, y_batch, target_class_id, device):
    x_tensor = torch.from_numpy(x_batch).float().to(device) 
    binary_labels = (y_batch == target_class_id).astype(np.float32)
    y_tensor = torch.from_numpy(binary_labels).float().to(device).unsqueeze(1)
    predictions = FunctionalLeNet5.forward(x_tensor, weights_flat)
    loss = F.binary_cross_entropy(predictions, y_tensor)
    
    with torch.no_grad():
        acc = ((predictions > 0.5).float() == y_tensor).float().mean().item()
    
    return loss, acc

def compute_task_loss_differentiable_weighted(weights_flat, x_batch, y_batch, target_class_id, device):
    x_tensor = torch.from_numpy(x_batch).float().to(device)
    binary_labels = (y_batch == target_class_id).astype(np.float32)
    y_tensor = torch.from_numpy(binary_labels).float().to(device).unsqueeze(1)

    # Count positives/negatives to set pos_weight dynamically
    n_pos = binary_labels.sum()
    n_neg = len(binary_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)  # ~9.0 typically

    predictions = FunctionalLeNet5.forward(x_tensor, weights_flat)
    # sigmoid already applied in forward, so use binary_cross_entropy not _with_logits
    loss = F.binary_cross_entropy(predictions, y_tensor,
                                   weight=pos_weight.expand_as(y_tensor))

    with torch.no_grad():
        acc = ((predictions > 0.5).float() == y_tensor).float().mean().item()

    return loss, acc

def compute_task_loss_differentiable_debug(weights_flat, x_batch, y_batch, target_class_id, device):
    x_tensor = torch.from_numpy(x_batch).float().to(device)
    binary_labels = (y_batch == target_class_id).astype(np.float32)
    y_tensor = torch.from_numpy(binary_labels).float().to(device).unsqueeze(1)

    # Manually step through LeNet5 to inspect activations
    shapes = FunctionalLeNet5.get_weight_shapes()
    idx = 0
    params = {}
    for name, shape, size in shapes:
        params[name] = weights_flat[idx:idx+size].reshape(shape)
        idx += size

    if x_tensor.shape[-1] == 3:
        x = x_tensor.permute(0, 3, 1, 2)

    import torch.nn.functional as F_
    conv1_w = params['conv1_w'].permute(3, 2, 0, 1)
    pre_act1 = F_.conv2d(x, conv1_w, params['conv1_b'], padding=2)
    act1 = torch.tanh(pre_act1)
    
    print(f"  conv1 pre-activation: mean={pre_act1.mean().item():.3f}, std={pre_act1.std().item():.3f}")
    print(f"  conv1 post-tanh:      mean={act1.mean().item():.3f}, std={act1.std().item():.3f}")
    print(f"  tanh saturation:      {((act1.abs() > 0.99).float().mean().item()*100):.1f}% of neurons at ±1")
    
    predictions = FunctionalLeNet5.forward(x_tensor, weights_flat)
    print(f"  Final predictions:    mean={predictions.mean().item():.4f}, std={predictions.std().item():.4f}")
    
    loss = F_.binary_cross_entropy(predictions, y_tensor)
    return loss, 0.0

def compute_task_loss_eval(weights_flat, x_batch, y_batch, target_class_id, device, model = None):
    if isinstance(weights_flat, torch.Tensor):
        weights_flat = weights_flat.detach().cpu().numpy()
    
    if model is None:
        model = build_lenet5_keras()
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
    load_weights_into_lenet5_keras(model, weights_flat)
    
    binary_labels = (y_batch == target_class_id).astype(np.float32)
    #predictions = model.predict(x_batch, verbose=0).flatten()
    predictions = model(x_batch, training=False).numpy().flatten() 

    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.mean(binary_labels * np.log(predictions) + (1 - binary_labels) * np.log(1 - predictions))
    accuracy = np.mean((predictions > 0.5) == binary_labels)
    
    return float(loss), float(accuracy)