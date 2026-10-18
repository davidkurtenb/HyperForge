from helpers.mnet_lenet5_binary import FunctionalLeNet5
import torch
import torch.nn as nn


#########################################################################
#    Linear Hypernetwork V9 Baseline
#    Mirrors V9 ConditionalWeightGenerator architecture but trained
#    with pure task loss only (no discriminator / adversarial signal).
#
#    Drop-in replacement for ConditionalWeightGenerator:
#        hnet(cond_id=0, batch_size=B)  -->  (B, weight_vector_len)
#########################################################################

class LinearHypernetwork(nn.Module):
    """
    Per-class class_embeddings serve as the learned z input (replacing
    z_embeddings + GAN noise in V9).  Architecture is identical to the
    V9 generator MLP (hidden_dim -> *2 -> *4 -> *2 -> weight_vector_len)
    so parameter counts are comparable for a fair baseline.
    """
    def __init__(self, num_classes, weight_vector_len, hidden_dim,
                 gen_leakyrelu_slope, gen_dropout_rate=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.weight_vector_len = weight_vector_len

        # Per-class learned embeddings — mirrors z_embeddings in V9 generator.
        self.class_embeddings = nn.Embedding(num_classes, hidden_dim)

        self.model = nn.Sequential(
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
        z = self.class_embeddings(
            torch.tensor([cond_id], device=self.class_embeddings.weight.device)
        )                              # (1, hidden_dim)
        z = z.expand(batch_size, -1)  # (batch_size, hidden_dim)
        return self.model(z)           # (batch_size, weight_vector_len)

def hyperfan_in_init(hnet):
    """
    Hyperfan-in weight initialization for hypernetworks.
    Reference: Chang et al. (2020) "Principled Weight Initialization for Hypernetworks"

    Hidden hnet layers: He fan-in  ->  var = 2 / fan_in_hnet
    Output layer:       per-chunk  ->  var = 1 / (fan_in_hnet * fan_in_mnet)
    Bias chunks in output:         ->  zero init
    """
    def get_fan_in(shape):
        """fan-in of a mnet parameter tensor"""
        if len(shape) == 1:       return 1              # bias
        elif len(shape) == 2:     return shape[0]       # fc: (in, out)
        elif len(shape) == 4:     return shape[0] * shape[1] * shape[2]  # conv: (kH,kW,Cin,Cout)
        return 1

    linear_layers = [m for m in hnet.model if isinstance(m, nn.Linear)]

    # --- Hidden layers: He fan-in ---
    for layer in linear_layers[:-1]:
        fan_in = layer.weight.shape[1]
        nn.init.normal_(layer.weight, 0.0, (2.0 / fan_in) ** 0.5)
        nn.init.zeros_(layer.bias)

    # --- Output layer: hyperfan-in, one chunk per mnet layer ---
    out_layer = linear_layers[-1]
    fan_in_hnet = out_layer.weight.shape[1]  # 100 in your case

    mnet_shapes = FunctionalLeNet5.get_weight_shapes()   # (name, shape, size)

    with torch.no_grad():
        row = 0
        for name, shape, size in mnet_shapes:
            fan_in_mnet = get_fan_in(shape)
            if len(shape) == 1:
                # bias params -> generated biases should start near zero
                out_layer.weight[row:row + size, :].zero_()
            else:
                std = (1.0 / (fan_in_hnet * fan_in_mnet)) ** 0.5
                out_layer.weight[row:row + size, :].normal_(0.0, std)
            row += size
        nn.init.zeros_(out_layer.bias)

    # Embedding: small normal so class embeddings don't dominate at start
    nn.init.normal_(hnet.class_embedding.weight, 0.0, 0.01)



