from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

from datasets import Slides
from network import SemanticInstanceSegmentation


def nearest(nearest_neighbors, query):
    return nearest_neighbors.kneighbors(query, n_neighbors=1, return_distance=False)[0][0]


NearestNeighbors.nearest = nearest


def mean(iterable):
    total = 0
    terms = 1
    for term in iterable:
        total += term
        terms += 1
    return total / terms


class Clustering:
    def __init__(self, embedding, cluster):
        self.embedding = embedding
        self.cluster = cluster
        self.indices = [int(i) for i in np.unique(cluster.cpu().numpy())]

    def __getitem__(self, index):
        return self.embedding[:, :, self.cluster == index].mean(dim=2, keepdim=True)

    def __iter__(self):
        for index in self.indices:
            yield self[index]


class SemanticLabels:
    def __init__(self, embeddings, labels, instances):
        self.embeddings = embeddings
        self.labels = labels
        self.instances = instances
        self.dimensions = embeddings.shape[1]

    def __getitem__(self, index):
        mask = self.labels.view(-1) == index
        embeddings = self.embeddings.view(1, self.dimensions, -1)[..., mask]
        target_instances = self.instances.view(-1)[mask]
        return embeddings, Clustering(embeddings, target_instances)

    def __iter__(self):
        for index in np.unique(self.labels.cpu().numpy()):
            if index > 0:
                yield self[int(index)]


class DiscriminativeLoss(nn.Module):
    def __init__(self, alpha=1, beta=2, gamma=0.001, delta_v=0.5, delta_distance=1.5):
        super(DiscriminativeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.delta_distance = 2 * delta_distance
        self.delta_v = delta_v

    def variance_loss(self, x, clusters):
        return mean((F.relu(((x - cluster)**2).sum(dim=1) - self.delta_v)**2).mean()
                    for cluster in clusters)

    def distance_loss(self, clusters):
        return mean(F.relu(self.delta_distance - ((cluster_A - cluster_B)**2).sum())**2
                    for cluster_A, cluster_B in combinations(clusters, 2))

    def regularization_loss(self, clusters):
        return mean((cluster**2).sum() for cluster in clusters)

    def forward(self, embedding, clusters):
        return (self.alpha * self.variance_loss(embedding, clusters) +
                self.beta * self.distance_loss(clusters) +
                self.gamma * self.regularization_loss(clusters))


def instances_from_colors(image):
    image_colors = image.reshape(-1, 3)
    colors, indices = np.unique(image_colors, axis=0, return_inverse=True)
    return indices.reshape(image.shape[:2])


def mean_shift(label_embedding):
    neigh = NearestNeighbors(radius=0.5, metric='euclidean').fit(label_embedding.copy().T)

    predicted_instances = -np.ones(label_embedding.shape[1])
    unlabeled = np.where(predicted_instances < 0)[0]

    dimensions = label_embedding.shape[0]

    while unlabeled.size > 0:
        index = np.random.choice(unlabeled)
        indices = set([index])
        centre = label_embedding.T[index]

        for i in range(100):
            neighbors = neigh.radius_neighbors(centre.reshape(1, dimensions), return_distance=False)[0]
            new_centre = label_embedding.T[neighbors].mean(axis=0)
            if np.allclose(centre, new_centre):
                break
            indices.update(neighbors)
            centre = new_centre

        neighbors = neigh.radius_neighbors(centre.reshape(1, dimensions), return_distance=False)[0]
        indices.update(neighbors)

        centre_index = neigh.nearest(centre.reshape(1, dimensions))
        predicted_instances[list(indices)] = centre_index
        unlabeled = np.where(predicted_instances < 0)[0]

    return predicted_instances


def visualise_embeddings(label_embedding, predicted_instances, target_instances=None):
    if target_instances is not None:
        plt.subplot(1, 2, 1)

    ax = plt.gca()
    for index in np.unique(predicted_instances):
        cluster = label_embedding[:, predicted_instances == index]
        plt.plot(*cluster, '+', label=index)
        ax.add_patch(plt.Circle(cluster.mean(axis=1), radius=1.5, fill=False, linestyle='--'))
        ax.add_patch(plt.Circle(cluster.mean(axis=1), radius=0.5, fill=False, linestyle='--'))

    if target_instances is not None:
        plt.subplot(1, 2, 2)
        for index in np.unique(target_instances):
            cluster = label_embedding[:, target_instances == index]
            plt.plot(*cluster, '+', label=index)

    plt.legend()
    plt.show()


def visualise_instances(predicted_instances, labels, class_index=2):
    predicted_image = np.zeros(labels.shape).flatten()

    _, predicted_indices = np.unique(predicted_instances, return_inverse=True)
    predicted_image[labels.view(-1).cpu().numpy() == class_index] = predicted_indices + 1

    plt.imshow(predicted_image.reshape(labels.shape))
    plt.show()


if __name__ == '__main__':
    data = Slides(download=True, train=True, root='data')

    model = SemanticInstanceSegmentation().cuda()
    instance_clustering = DiscriminativeLoss().cuda()

    imgs = [torch.Tensor(np.asarray(Image.open(filename)).transpose(2, 0, 1)[np.newaxis])
            for filename in ['image.png', 'image2.png']]
    img_instances = [torch.Tensor(instances_from_colors(np.asarray(Image.open(filename)).astype(np.uint8))).long()
                     for filename in ['image_instances.png', 'image_instances2.png']]
    img_labels = [torch.Tensor(instances_from_colors(np.asarray(Image.open(filename)).astype(np.uint8))).long()
                  for filename in ['image_labels.png', 'image_labels2.png']]

    train = False

    if train:
        # model.pretrained.load_state_dict(torch.load(Path('models') / 'epoch_3200'))

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 150, gamma=0.1)
        model.train()
        losses = np.zeros(50)

        for iteration in range(50):
            scheduler.step()
            optimizer.zero_grad()

            for img, instances, labels in zip(imgs, img_instances, img_labels):
                img = img.cuda()
                instances = instances.cuda()
                labels = labels.cuda()

                _, instance_embeddings = model(Variable(img))

                loss = sum(instance_clustering(embeddings, target_clusters)
                           for embeddings, target_clusters in SemanticLabels(instance_embeddings, labels, instances))

                print(iteration, loss.data[0])
                losses[iteration] = loss.data[0]
                loss.backward()

            optimizer.step()

        torch.save(model.state_dict(), Path('models') / 'instance_model')
    else:
        model.load_state_dict(torch.load(Path('models') / 'instance_model'))

    img = imgs[0].cuda()
    instances = img_instances[0].cuda()
    labels = img_labels[0].cuda()

    _, instance_embeddings = model(Variable(img))
    mask = labels.view(-1) > 0
    label_embedding = instance_embeddings.view(1, instance_embeddings.shape[1], -1)[..., mask]
    label_instances = instances.view(-1)[mask]

    label_embedding = label_embedding.data.cpu().numpy()[0]
    label_instances = label_instances.cpu().numpy()

    predicted_instances = mean_shift(label_embedding)

    # visualise_embeddings(label_embedding, predicted_instances)
    visualise_instances(predicted_instances, labels)
