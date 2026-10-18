# HyperForge

HyperForge is a GAN-based hypernetwork that generates the full weight vector of a small convolutional classifier. A class-conditional generator learns to produce LeNet-5 weights, and those weights are trained against two signals: a discriminator that compares them with a library of conventionally trained LeNet-5 models, and a task loss computed by running the generated network on real images.

The target task is one-vs-all binary classification on the 10 classes of [Imagenette](https://github.com/fastai/imagenette) (320px v2, resized to 32x32). For each class, the generator produces a LeNet-5 that decides whether an image belongs to that class.

## How it works

**Target network (main network).** A LeNet-5 with tanh activations, average pooling, and a single sigmoid output. It has 91,481 parameters, flattened in the same order as a PyTorch `state_dict`:

| Layer | Shape | Params |
|---|---|---|
| conv1 | 6 x 3 x 5 x 5 (+6 bias) | 456 |
| conv2 | 16 x 6 x 5 x 5 (+16 bias) | 2,416 |
| conv3 | 120 x 16 x 5 x 5 (+120 bias) | 48,120 |
| fc1 | 84 x 480 (+84 bias) | 40,404 |
| fc2 | 1 x 84 (+1 bias) | 85 |

**Generator (hypernetwork).** `ConditionalWeightGenerator` holds a learned latent embedding for each class. Given a class id, it looks up that embedding and passes it through an MLP (hidden_dim -> 2x -> 4x -> 2x -> weight vector) with LayerNorm, LeakyReLU, and dropout. A final tanh keeps the output in [-1, 1], and it is rescaled by the per-class maximum absolute weight of the real models.

**Discriminator.** `ConditionalDiscriminator` takes a flattened weight vector concatenated with a class embedding and predicts whether the weights came from a really trained LeNet-5 or from the generator.

**Training loop.** For each batch and each class:

1. The discriminator is updated on real weights sampled from that class's pool and on generated weights.
2. The generator is updated on a weighted sum of the adversarial loss and the task loss. The task loss runs the generated LeNet-5 on a class-balanced (50/50 positive/negative), augmented image batch using a functional forward pass, so gradients flow back into the generator. Task loss is computed for the whole generated batch at once with `torch.func.vmap`.

After every epoch, one generated network per class is evaluated on a fixed, balanced test set built from the Imagenette validation split. The best checkpoint by combined test accuracy is saved.

## Repository layout

```
src/
  models/
    hyperforge.py                 Generator, discriminator, functional and eval LeNet-5,
                                  weight loading helpers
  helpers/
    utils.py                      Data loading, top-N training sample selection, balanced
                                  test sets, task loss functions, hyperfan init, early stopping
    model_train.py                Shared GAN training loop
    data_loader.py                Imagenette and weight pickle loaders
    mnet_lenet5_binary.py         Functional LeNet-5 and Keras LeNet-5 used by the baseline
    hnet_linear.py                Linear hypernetwork used for the baseline
    best_model_acc_parser.py      Parses training logs for the best test accuracy
  prod_code/
    modelwise_hGAN/
      GAHypN_modelwise_train_v10.3.3.py
      GAHypN_modelwise_train_v10.4.py
      GAHypN_modelwise_train_v10.5.py
    linear_hnet/
      BASELINE_linear_modelwise_train.py
      plot_linear_hnet.py
      linear_hnet_architecture.png
```

### Training script versions

All three scripts are self-contained and share the same training and evaluation pipeline.

| Script | Change |
|---|---|
| v10.3.3 | Generator input is the per-class latent embedding only. Task loss is batched with `vmap` instead of looping over each generated sample. |
| v10.4 | Adds a random noise vector (`noise_dim = 16`), drawn fresh on every forward pass, to the generator input alongside the class embedding. |
| v10.5 | Same architecture as v10.4 with `noise_dim = 32`. |

### Baseline

`BASELINE_linear_modelwise_train.py` trains `LinearHypernetwork`, which uses the same MLP as the generator but is trained on task loss alone, with no discriminator and no real weights. It gives a fair comparison for how much the adversarial signal adds. `plot_linear_hnet.py` draws the architecture diagram in `linear_hnet_architecture.png`.

## Requirements

- Python 3.11 or later
- PyTorch with `torch.func` (2.0 or later) and torchvision
- TensorFlow and tensorflow-datasets (used only to download and preprocess Imagenette; TensorFlow is pinned to CPU)
- NumPy, pandas, scikit-learn, matplotlib, tqdm

A CUDA GPU is strongly recommended.

## Data

Training needs two inputs that are not included in this repository:

1. **Imagenette.** Downloaded automatically through `tensorflow_datasets` into `data_dir` on the first run.
2. **A library of trained LeNet-5 weights.** A pickle containing a list of dicts, each with:
   - `model`: a name of the form `{class_name}_{count}`, for example `tench_0` or `english_springer_42`
   - `parameters_flat`: a flat tensor of the 91,481 LeNet-5 parameters in `state_dict` order

   Plus a `model_performance_metrics.csv` with `Class`, `Count`, and `Accuracy` columns. The training scripts use it to keep only the top `traindata_samp_size` models per class by accuracy.

## Running

Set the paths at the bottom of the training script (`rel_traindata_path`, `output_pth`, `data_dir`) and the path to `model_performance_metrics.csv` inside `load_lenetwghts_from_pickl`, then run:

```
python src/prod_code/modelwise_hGAN/GAHypN_modelwise_train_v10.5.py
```

Hyperparameters are set in the `config` dictionary in the same section, including batch size, epochs, latent and noise dimensions, the task and adversarial loss weights, and the learning rates.

Each run creates a time-stamped folder under `output_pth` containing:

- `models/best_checkpoint.pt`: generator and discriminator state, optimizer state, per-class latent embeddings, weight scale factors, and loss and accuracy history
- `plots/training_plots_*.png`: losses, combined train and test accuracy, and per-class test accuracy
- `config_file_*.txt`: the run configuration and summary metrics

## License

See [LICENSE](LICENSE).
