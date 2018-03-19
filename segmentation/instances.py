from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from torch import nn
import torch.nn.functional as F


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
    def __init__(self, alpha=1, beta=1, gamma=0.001, delta_v=0.5, delta_distance=1.5):
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


def visualise_class_instances(predicted_instances, labels, class_index=2):
    predicted_image = np.zeros(labels.shape).flatten()

    _, predicted_indices = np.unique(predicted_instances, return_inverse=True)
    predicted_image[labels.view(-1).cpu().numpy() == class_index] = predicted_indices + 1

    return predicted_image.reshape(labels.shape)


def visualise_instances(predicted_instances, labels, num_classes):
    instance_image = np.zeros(labels.shape)
    total = -1
    for class_index in range(1, num_classes):
        class_instances = visualise_class_instances(predicted_instances[class_index], labels, class_index=class_index)
        class_instances[class_instances > 0] += total + 1
        total = class_instances.max()
        instance_image += class_instances
    return instance_image
