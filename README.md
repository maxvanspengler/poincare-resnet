# Poincaré ResNet
Repository containing the code for the [Poincaré ResNet paper](https://arxiv.org/abs/2303.14027).

# Installation
This repository requires `python >= 3.10`. To install the required packages, run
```
pip install -r requirements.txt
```
The repository uses CIFAR-10 and CIFAR-100 for training and Places365, SVHN and Textures for out-of-distrbution (OOD) detection. Your root directory (where this README is located) must contain a `config.ini` file containing the paths to these datasets. An example ini file can be found in `example_config.ini`. Instructions for downloading the required datasets can be found at:

- [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Places365](http://places2.csail.mit.edu/download.html)
- [SVHN](http://ufldl.stanford.edu/housenumbers/)
- [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

Note that the last three of these datasets are only required for OOD detection. If you are only interested in other components of this repository, feel free to ignore the paths to these datasets in the ini file. If you are interested in the OOD detection but do not have access to one or more of these datasets, you can remove the corresponding sections from the `ood_detection.py` file. 

# Training
To train a model, use the CLI tool in `train.py`. The naming convention for the models is as follows:
```
<hyperbolic|euclidean|euclideanwhypclass>-<channel_size1>-<channel_size2>-<channel_size3>-resnet-<depth>
```
where depth = 3 * 2 * block_depth + 2. As an example, 
```
hyperbolic-8-16-32-resnet-32
```
leads to a hyperbolic ResNet with channel sizes (8, 16, 32) and with block sizes (5, 5, 5). 

For simplicity, the `train.sh` script contains an example of a call to the train tool with some sensible arguments.

# Robustness experiments
Each of the robustness experiments has its own CLI tool: `ood_detection.py`, `adversarial_attacks.py`, `gradcam.py`

These experiments require models to have already been trained and stored by using the train tool mentioned above with the `-s` flag (stores the weights in ./weights directory). 

Examples of how to run these experiments with sensible arguments are shown in the corresponding shell scripts.
