
import os, time, random
from random import getrandbits, choice, sample
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import pairwise_distance, cosine_similarity
from tqdm import tqdm, trange

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import seaborn as sns
from matplotlib import pyplot as plt

from utils.model import KochNet, AlexNet, VGG, ResNet
from utils.data import load_siamese, normalise


# return a fitted pca transform function
def fit_pca(vectors, n_components=None):
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.cpu().numpy()

    # normalize
    scaler = StandardScaler()
    scaler.fit(vectors)
    vectors = scaler.transform(vectors)

    # fit
    pca = PCA(n_components)
    pca.fit(vectors)

    def f(x):
        x = x.cpu().numpy()
        x = scaler.transform(x)
        x = pca.transform(x)
        x = torch.tensor(x)
        return x

    return f


def one_shot_eval(embed, data, device, distance, pca, tests=1000, classes=6):
    results = []

    for _ in trange(tests):
        with torch.no_grad():

            # get an example from each class
            K = list(data.dataset.targets)
            if 0 < classes: K = sample(K, classes)
            S = [choice(data.dataset.targets[k]) for k in K]
            S = [data.dataset.imgs[s][0] for s in S]

            # choose a test image
            k = choice(K)
            x = choice(data.dataset.targets[k])
            x = data.dataset.imgs[x][0]

            # stack images
            x_i = torch.stack(len(S) * [x]).to(device)
            x_j = torch.stack(S).to(device)

            # embed images
            x_i, x_j = embed(x_i), embed(x_j)

            # transform images
            if pca:
                x_i, x_j = pca(x_i), pca(x_j)

            # predict
            d = distance(x_i, x_j)
            results.append(k == K[torch.argmin(d)])

    return sum(results) / len(results)


def plot_tsne(data, vectors, distance, pca):
    if pca:
        vectors = pca(vectors)

    # get class names and paths
    paths, targets = zip(*data.dataset.imgs.samples)
    target_to_class = {v:k for k, v in data.dataset.imgs.class_to_idx.items()}
    class_names = [target_to_class[t] for t in targets]
    n = len(vectors)

    # compute distances between vectors
    distances = np.zeros((n, n))
    with torch.no_grad():
        for i, j in tqdm(list(combinations(range(n), 2))):
            d = distance(vectors[i].unsqueeze(0), vectors[j].unsqueeze(0))
            distances[i, j] = distances[j, i] = d

    # t-distributed stochastic neighbour embedding
    tsne = TSNE(n_components=1, perplexity=4, learning_rate=25,
                metric='precomputed', square_distances=True)
    tsne_vectors = tsne.fit_transform(distances)
    _, paths = zip(*sorted([(v, p) for v, p in zip(tsne_vectors, paths)]))

    # set visual range to a (0, 1)
    tsne_vectors = tsne_vectors - np.min(tsne_vectors)
    tsne_vectors = tsne_vectors / np.max(tsne_vectors)
    tsne_vectors = .8 * tsne_vectors + .1

    # prepare dataframe
    df = pd.DataFrame({'value': tsne_vectors[:, 0],
        'method': len(class_names) * [''], 'label': class_names})

    # create figure without legend
    fig, ax = plt.subplots()

    # plot tsne vectors
    sns.stripplot(x='value', y='method', hue='label', ax=ax,
        data=df, dodge=True, jitter=False, alpha=.75, zorder=1)

    # show the conditional means
    sns.pointplot(x="value", y="method", hue="label", ax=ax,
        data=df, dodge=.66, join=False, palette="dark",
        markers="d", scale=1, ci=None)

    ax.set_xlim(0, 1)
    ax.get_legend().set_visible(False)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis('off')

    # create figure with legend
    fig_legend, ax = plt.subplots()

    # plot tsne vectors
    sns.stripplot(x='value', y='method', hue='label', ax=ax,
        data=df, dodge=True, jitter=False, alpha=.75, zorder=1)

    # show the conditional means
    sns.pointplot(x="value", y="method", hue="label", ax=ax,
        data=df, dodge=.66, join=False, palette="dark",
        markers="d", scale=1, ci=None)

    ax.set_xlim(0, 1)
    ax.get_legend().set_visible(True)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis('off')

    # create figure without dodge
    fig_1d, ax = plt.subplots()

    # plot tsne vectors
    sns.stripplot(x='value', y='method', hue='label', ax=ax,
        data=df, dodge=False, jitter=False, alpha=.75, zorder=1)

    # show the conditional means
    sns.pointplot(x="value", y="method", hue="label", ax=ax,
        data=df, dodge=False, join=False, palette="dark",
        markers="d", scale=1, ci=None)

    ax.set_xlim(0, 1)
    ax.get_legend().set_visible(False)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis('off')

    return fig, fig_legend, fig_1d, paths


def main(model, eval_data, params, train_classes, distance, pca, output_dir, seed):

    # set random seed
    seed = getrandbits(32) if seed is None else seed
    print('Seed:', seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # parse arguments
    model = {
        'kochnet': KochNet,
        'alexnet': AlexNet,
        'vgg':     VGG,
        'resnet':  ResNet,
        'none':    None
    }[model]

    eval_data = Path('data') / f'{eval_data}_evaluation'

    # load data
    print('\nImporting data...')
    eval_data = load_siamese(eval_data, normalise, batch_size=1, batches=100)
    print('Evaluation', eval_data.dataset)

    # detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # build model
    if model:
        print('\nBuilding model...')
        model = model(
            classes=train_classes if train_classes else len(eval_data.dataset.imgs.classes),
            pretrained=str(params)=='imagenet'
        ).to(device)

        # load pre-trained parameters
        if str(params) == 'imagenet':
            print('Using ImageNet parameters')
        elif params:
            print(f'Loading {params}')
            params = torch.load(params)
            try:
                model.load_state_dict(params)
            except RuntimeError:
                model = torch.nn.DataParallel(model)
                model.load_state_dict(params)
                model = model.module

    # get embed function
    if model:
        model.eval()
        embed = model.embed
    else:
        embed = lambda x: x.flatten(1)

    # get distance function
    if distance == 'euclidean':
        distance = pairwise_distance
    elif distance == 'cosine':
        distance = lambda a, b: 1 - cosine_similarity(a, b)
    else:
        raise ValueError(f'Unknown distance: {distance}')

    # get vectors
    with torch.no_grad():
        vectors = torch.cat([embed(x.unsqueeze(0).to(device))
            for x, y in eval_data.dataset.imgs])

    # get pca function
    if pca:
        pca = fit_pca(vectors, min(*vectors.shape, 4096))

    # check output directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # # evaluate
    # print('\nEvaluating...')
    # accuracy = one_shot_eval(embed, eval_data, device, distance, pca)
    # line = f'{100 * accuracy :.4f}%'
    # print(line)

    # with open(output_dir / 'one_shot_accuracy.txt', 'w+') as f:
    #     f.write(line)

    # plot
    print('\nPlotting...')
    fig, fig_legend, fig_1d, paths = plot_tsne(eval_data, vectors, distance, pca)
    
    fig.tight_layout()
    fig.savefig(output_dir / 'figure.png')

    fig_legend.tight_layout()
    fig_legend.savefig(output_dir / 'figure_with_legend.png')

    fig_1d.tight_layout()
    fig_1d.savefig(output_dir / 'figure_flat.png')

    with open(output_dir / 'ordered_paths.txt', 'w+') as f:
        f.writelines(path + '\n' for path in paths)


if __name__ == '__main__':
    networks = 'kochnet', 'alexnet', 'vgg', 'resnet', 'none'
    distances = 'euclidean', 'cosine'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=networks, default='none')
    parser.add_argument('-e', '--eval-data', default='mnist')
    parser.add_argument('-p', '--params', type=Path)
    parser.add_argument('-c', '--train-classes', type=int, default=0)
    parser.add_argument('-d', '--distance', choices=distances, default='euclidean')
    parser.add_argument('-pca', '--pca', action='store_true')
    parser.add_argument('-o', '--output-dir', type=Path, default='results')
    parser.add_argument('--seed', type=int)
    main(**vars(parser.parse_args()))
