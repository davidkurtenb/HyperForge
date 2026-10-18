import torch 
import torch.nn as nn
import torch.nn.functional as F

#########################################################################
# Generator
#########################################################################
class ConditionalWeightGenerator(nn.Module):
    def __init__(self, latent_dim, weight_vector_len, num_classes, hidden_dim,
                 gen_leakyrelu_slope, gen_dropout_rate=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.weight_vector_len = weight_vector_len
 
        self.z_embeddings = nn.Embedding(num_classes, latent_dim)
 
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
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
        z = self.z_embeddings(
            torch.tensor([cond_id], device=self.z_embeddings.weight.device)
        )
        z = z.expand(batch_size, -1)
        return self.model(z)
    
#########################################################################
#  Discriminator  
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
# PyTorch Functional LeNet5 (DIFFERENTIABLE)
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
    def get_layer_slice(layer_prefix):
        shapes = FunctionalLeNet5.get_weight_shapes()
        idx = 0
        start = None
        size = 0
        for name, shape, s in shapes:
            if name.split('.')[0] == layer_prefix:
                if start is None:
                    start = idx
                size += s
            idx += s
        if start is None:
            raise ValueError(f"No params found for layer prefix '{layer_prefix}'")
        return start, size

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
# PyTorch LeNet5 for Evaluation ONLY (replaces Keras)
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

def assemble_full_weights(gen_layer_batch, reference_tail, layer_start):
    batch_size = gen_layer_batch.shape[0]
    prefix_len = layer_start
    tail_expanded = reference_tail.unsqueeze(0).expand(batch_size, -1)
    prefix = tail_expanded[:, :prefix_len]
    suffix = tail_expanded[:, prefix_len:]
    return torch.cat([prefix, gen_layer_batch, suffix], dim=1)