# Reconstructing rock art chronology with transfer learning: A case study from Arnhem Land, Australia

*by Jarrad Kowlessar, James Keal, Daryl Wesley, Ian Moffat, Dudley Lawrence, Abraham Weson, Alfred Nayinggul & Mimal Land Management Aboriginal Corporation*

This repository contains the official implementation of the article [Reconstructing rock art chronology with transfer learning: A case study from Arnhem Land, Australia](https://doi.org/10.1080/03122417.2021.1895481). The images of rock art used in the publication were derived from cultural material that remains the property of the Marrku Traditional Owners and cannot be openly distributed. For this reason, the rock art data has not been included in this repository.

![Figure 2](https://jameskeal.com.au/img/rockart_figure2.png)


## Requirements

 - Python 3, [PyTorch](https://pytorch.org/docs/stable/index.html) and [torchvision](https://pytorch.org/vision/stable/index.html)
 
 The following example installs these with CUDA support using Conda:

```sh
conda install python=3.8
conda install pytorch torchvision cudatoolkit -c pytorch
```

Take care to match the version of Python to the requirements of the latest version of PyTorch. More details, including CPU only installation can be found [here](https://pytorch.org/get-started/locally/).

 - [scikit-learn](https://scikit-learn.org/stable/index.html), [seaborn](https://seaborn.pydata.org/) and [tqdm](https://tqdm.github.io/)

```sh
conda install scikit-learn seaborn tqdm
```

#### To use the "Quick, Draw!" dataset:

- [gsutil](https://cloud.google.com/storage/docs/gsutil) and [cairocffi](https://pypi.org/project/cairocffi/)

These are only required to download and rasterize [Google's "Quick, Draw!" dataset](https://quickdraw.withgoogle.com/data).

```sh
sudo snap install google-cloud-sdk
conda install cairocffi
```


## Usage

#### Download training data

Data must separated into training (background) and evaluation datasets. Each must be arranged in accordance with the PyTorch [ImageFolder class](https://pytorch.org/vision/stable/datasets.html#imagefolder). For example:

```
data/
├── my_dataset_background/
│   ├── class_A/
│   │   ├── xxx.png
│   │   ├── xxy.png
│   │   └── ...
│   ├── class_B/
│   │   ├── 123.png
│   │   ├── nsdf3.png
│   │   └── ...
│   ...
└── my_dataset_evaluation/
    ├── class_A/
    │   ├── xxz.png
    │   ├── xyx.png
    │   └── ...
    ├── class_B/
    │   ├── 456.png
    │   ├── asd932_.png
    │   └── ...
    ...
```

Images must be 224×224 pixels. Example bash scripts are provided that download, resize and correctly organize [MNIST](http://yann.lecun.com/exdb/mnist/), [Omniglot](https://github.com/brendenlake/omniglot) and [Quick, Draw!](https://quickdraw.withgoogle.com/data) datsets.

```sh
cd data
./get_mnist.sh
./get_omniglot.sh
./get_quickdraw.sh
```

#### Training

```sh
python train.py --model {kochnet, alexnet, vgg, resnet} --train-data my_dataset
```

Optional arguments:

 - --model {kochnet, alexnet, vgg, resnet} — The architecture to train.
 - --train-data DATASET_NAME — The prefix of the background data directory.
 - --eval-data DATASET_NAME — The prefix of the evaluation data directory, if different from training.
 - --epochs N — The number of epochs to train for.
 - --learn-rate LR — The initial learning rate.
 - --output-dir DIR — The folder in which to store trained parameters.
 - --seed X — Sets the random seed.

 The expected output is a directory containing files `0.params` ... `N.params` where `N` is the number of training epochs.

#### Evaluating

```sh
python evaluate.py --model {kochnet, alexnet, vgg, resnet} --eval-data my_dataset --params best.params
```

Optional arguments:

 - --model {kochnet, alexnet, vgg, resnet} — The architecture to evaluate.
 - --eval-data DATASET_NAME — The prefix of the evaluation data directory.
 - --params PARAMS — The trained parameters to load. Use "imagenet" to load parameters trained on the [ImageNet dataset](http://www.image-net.org/).
 - --train-classes N — The number of classes with which to initialize the architecture, if different from the number of classes in the eval data.
 - --distance {euclidean, cosine} — The distance metric to use.
 - --pca — Use PCA after neural net embedding.
 - --output-dir DIR — The folder in which to store results and plots.
 - --seed X — Sets the random seed.

The expected output is a directory containing `accuracy.txt` which stores the network accuracy in a one-shot setting. The output directory should also contain plots with and without a class legend, and one flattened in the Y axis. Finally, `ordered_paths.txt` should list the paths to the original image for each data point in order from left to right. This can be used to construct and interpret the inferred stylistic chronology.

## Licence

All source code is made available under a BSD 3-clause license. You may freely use and modify this code, without warranty, so long as you provide attribution to the authors. The manuscript text is not open source. The article's content has been published in the journal Australian Archeology.

## Citation

 > Jarrad Kowlessar, James Keal, Daryl Wesley, Ian Moffat, Dudley Lawrence, Abraham Weson, Alfred Nayinggul & Mimal Land Management Aboriginal Corporation. "Reconstructing rock art chronology with transfer learning: A case study from Arnhem Land, Australia", Australian Archaeology, 2021, DOI: 10.1080/03122417.2021.1895481
