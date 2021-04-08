
import os, time, random
from random import getrandbits
from datetime import timedelta
from pathlib import Path

import torch
from tqdm import tqdm

from utils.model import KochNet, AlexNet, VGG, ResNet
from utils.data import load, normalise, augment_and_normalise


# evaluate accuracy
def evaluate(model, data, device, verbose=False):
    _data = tqdm(data) if verbose else data

    # evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in _data:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += len(x)

    return correct / total


# train one epoch
def train(model, data, lossfn, optimr, schdlr, device, epoch=0, verbose=False):

    # train
    model.train()
    start = time.time()
    for x, y in data:
        optimr.zero_grad()
        x, y = x.to(device), y.to(device)
        loss = lossfn(model(x), y)
        loss.backward()
        optimr.step()
        del x, y, loss

    train_time = timedelta(seconds=time.time() - start)
    lr = [group['lr'] for group in optimr.param_groups]
    if len(lr) == 1: lr = lr[0]

    # get training loss
    losses = []
    model.eval()
    with torch.no_grad():
        for x, y in data:
            x, y = x.to(device), y.to(device)
            loss = lossfn(model(x), y)
            losses.append(loss.item())
            del x, y, loss

    loss = sum(losses) / len(losses)
    schdlr.step(loss)

    # report
    if verbose:
        print(' | '.join((
            'Epoch {}'.format(epoch),
            'LR: {:.4e}'.format(lr),
            'Loss: {:.4e}'.format(loss),
            'Time: {}'.format(train_time),
        )))


def main(model, train_data, eval_data, epochs, learn_rate, output_dir, seed=None):

    # set random seed
    seed = getrandbits(32) if seed is None else seed
    print('Seed:', seed)
    torch.manual_seed(seed)
    random.seed(seed)

    model = {
        'kochnet': KochNet,
        'alexnet': AlexNet,
        'vgg':     VGG,
        'resnet':  ResNet,
    }[model]

    if not eval_data:
        eval_data = train_data
    
    train_data = Path('data') / f'{train_data}_background'
    eval_data = Path('data') / f'{eval_data}_evaluation'

    # load data
    print('\nImporting data...')
    train_data = load(train_data, augment_and_normalise, batch_size=64, batches=100)
    eval_data = load(eval_data, normalise, batch_size=64, batches=100)
    print('Training', train_data.dataset)
    print('Evaluation', eval_data.dataset)

    # detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model
    print('\nBuilding model...')
    model = model(
        classes=len(train_data.dataset.classes),
        pretrained=False
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # loss function, optimizer, scheduler
    lossfn = torch.nn.CrossEntropyLoss()
    optimr = torch.optim.Adam(model.parameters(), lr=learn_rate)
    schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimr)

    # check output directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # save initial parameters
    torch.save(model.state_dict(), output_dir / '0.params')
    best_score = 0

    # begin training
    print('\nTraining...')
    for epoch in range(epochs):
        train(model, train_data, lossfn, optimr, schdlr, device, epoch, verbose=True)

        # evaluate
        if not (epoch + 1) % 10:

            print('\nEvaluating...')
            score = evaluate(model, eval_data, device, verbose=True)
            print(f'accuracy: {100 * score :.2f}%')

            # save model parameters
            torch.save(model.state_dict(), output_dir / f'{epoch + 1}.params')

            if score > best_score:
                print('New best!')
                torch.save(model.state_dict(), output_dir / 'best.params')
                best_score = score

            print()


if __name__ == '__main__':
    networks = 'kochnet', 'alexnet', 'vgg', 'resnet'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=networks, default='kochnet')
    parser.add_argument('-t', '--train-data', default='mnist')
    parser.add_argument('-e', '--eval-data', default=None)
    parser.add_argument('-n', '--epochs', type=int, default=200)
    parser.add_argument('-lr', '--learn-rate', type=float, default=1e-4)
    parser.add_argument('-o', '--output-dir', type=Path, default='trained')
    parser.add_argument('--seed', type=int)
    main(**vars(parser.parse_args()))
