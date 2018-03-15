from itertools import islice
import logging
from pathlib import Path

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms

from datasets import Slides
from instances import DiscriminativeLoss, SemanticLabels, mean_shift, visualise_embeddings, visualise_instances
from network import SemanticInstanceSegmentation


logging.basicConfig(format='[%(asctime)s] %(message)s', filename='training.log', filemode='w', level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def visualise_segmentation(predicted_class, colours):
    class_image = np.zeros((predicted_class.shape[1], predicted_class.shape[2], 3))
    prediction = predicted_class[0].cpu().numpy()
    for j in range(len(colours)):
        class_image[prediction == j] = colours[j]
    return class_image / 255


def visualise_results(output, image, predicted_class, colours, n=6, dpi=250):
    gs = gridspec.GridSpec(2, n, width_ratios=[1]*n, wspace=0.1, hspace=0, top=0.95, left=0.17, right=0.845)
    plt.figure(figsize=(n, 2))

    for i in range(n):
        plt.subplot(gs[0, i])
        plt.imshow(image.data[i].cpu().numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.subplot(gs[1, i])
        plt.imshow(visualise_segmentation(predicted_class[i], colours))
        plt.axis('off')
    plt.savefig(str(output), dpi=dpi, bbox_inches='tight')
    plt.close('all')


def torch_zip(*args):
    for items in zip(*args):
        yield tuple(item.unsqueeze(0) for item in items)


def train(model, instance_clustering, train_loader, test_loader):
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, 2500, gamma=0.1)

    losses = {'train': {'semantic': [], 'instance': [], 'total': []},
              'test':  {'semantic': [], 'instance': [], 'total': []}}
    accuracies = {'train': [], 'test': []}

    for epoch in range(12000):
        scheduler.step()

        if epoch % scheduler.step_size == 0:
            logging.debug(f'Learning rate set to {scheduler.get_lr()[0]}')

        model.train()

        for i, (image, labels, instances) in enumerate(train_loader):
            image, labels, instances = Variable(image).cuda(), Variable(labels).cuda(), Variable(instances).cuda()
            optimizer.zero_grad()

            logits, instance_embeddings = model(image)
            logits_per_pixel = logits.view(image.shape[0], 5, -1).transpose(1, 2).contiguous()
            semantic_loss = cross_entropy(logits_per_pixel.view(-1, 5), labels.view(-1))

            instance_loss = sum(sum(instance_clustering(embeddings, target_clusters)
                                    for embeddings, target_clusters
                                    in SemanticLabels(image_instance_embeddings, image_labels, image_instances))
                                for image_instance_embeddings, image_labels, image_instances
                                in torch_zip(instance_embeddings, labels, instances))

            loss = semantic_loss * 10 + instance_loss

            loss.backward()
            optimizer.step()

            predicted_class = logits.data.max(1, keepdim=True)[1]
            correct_prediction = predicted_class.eq(labels.data.view_as(predicted_class))
            accuracy = correct_prediction.int().sum().item() / np.prod(predicted_class.shape)

            losses['train']['semantic'].append(semantic_loss.item())
            losses['train']['instance'].append(instance_loss.item())
            losses['train']['total'].append(loss.item())
            accuracies['train'].append(accuracy)
            logging.debug(f'Epoch: {epoch + 1:{3}}, Batch: {i:{3}}, Cross-entropy loss: {loss.item()}, Accuracy: {(accuracy * 100)}%')

        if (epoch + 1) % 5 == 0:
            model.eval()

            total_loss = 0
            total_accuracy = 0

            num_test_batches = 1

            with torch.no_grad():
                for image, labels, instances in islice(test_loader, num_test_batches):
                    image, labels, instances = (Variable(tensor).cuda() for tensor in (image, labels, instances))

                    logits, instance_embeddings = model(image)
                    logits_per_pixel = logits.view(image.shape[0], 5, -1).transpose(1, 2).contiguous()
                    semantic_loss = cross_entropy(logits_per_pixel.view(-1, 5), labels.view(-1))

                    instance_loss = sum(sum(instance_clustering(embeddings, target_clusters)
                                            for embeddings, target_clusters
                                            in SemanticLabels(image_instance_embeddings, image_labels, image_instances))
                                        for image_instance_embeddings, image_labels, image_instances
                                        in torch_zip(instance_embeddings, labels, instances))

                    loss = semantic_loss * 10 + instance_loss

                    total_loss += loss.item()

                    predicted_class = logits.data.max(1, keepdim=True)[1]
                    correct_prediction = predicted_class.eq(labels.data.view_as(predicted_class))
                    accuracy = correct_prediction.int().sum().item() / np.prod(predicted_class.shape)
                    total_accuracy += accuracy

            average_loss = total_loss / num_test_batches
            average_accuracy = total_accuracy / num_test_batches
            losses['test']['total'].append(average_loss)
            accuracies['test'].append(average_accuracy)
            logging.info(f'Epoch: {epoch + 1:{3}}, Test Set, Cross-entropy loss: {average_loss}, Accuracy: {(average_accuracy * 100)}%')

        if (epoch + 1) % 10 == 0:
            visualise_results(Path('results') / f'epoch_{epoch + 1}.png', image, predicted_class,
                              colours=train_loader.dataset.colours)
            np.save('losses.npy', [{'train': losses['train'], 'test': losses['test']}])
            np.save('accuracies.npy', [{'train': accuracies['train'], 'test': accuracies['test']}])

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), Path('models') / f'epoch_{epoch + 1}')


if __name__ == '__main__':
    model = SemanticInstanceSegmentation().cuda()
    instance_clustering = DiscriminativeLoss().cuda()

    transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomCrop((256, 768)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()])

    target_transform = transforms.Compose([transform, transforms.Lambda(lambda x: (x * 255).long())])

    # WARNING: Don't use multiple workers for loading! Doesn't work with setting random seed
    train_data = Slides(download=True, train=True, root='data', transform=transform, target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=6, drop_last=True, shuffle=True)
    test_data = Slides(download=True, train=False, root='data', transform=transform, target_transform=target_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=6, drop_last=True, shuffle=True)

    # train(model, instance_clustering, train_loader, test_loader)
    model.load_state_dict(torch.load('models/epoch_12000'))

    model.eval()

    test_data = Slides(download=True, train=False, root='data', transform=transform, target_transform=target_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=6, shuffle=True)

    image, labels, instances = next(iter(test_loader))

    image = Variable(image)
    instances = Variable(instances + 1)
    logits, instance_embeddings = model(image.cuda())

    for i in range(1):
        current_logits = logits[i]
        current_labels = labels[i, 0].cuda()
        current_instances = instances[i].cuda()

        mask = current_labels.view(-1) == 2
        label_embedding = instance_embeddings[i].view(1, instance_embeddings.shape[1], -1)[..., mask]
        label_instances = current_instances.view(-1)[mask]

        label_embedding = label_embedding.data.cpu().numpy()[0]
        label_instances = label_instances.cpu().numpy()

        predicted_instances = mean_shift(label_embedding)

        visualise_embeddings(label_embedding, predicted_instances, target_instances=label_instances)
        plt.subplot(2, 3, 1)
        plt.imshow(image[i].data.numpy().transpose(1, 2, 0))
        plt.subplot(2, 3, 2)
        plt.imshow(current_labels.cpu().numpy().squeeze())
        plt.subplot(2, 3, 3)
        plt.imshow(current_instances.cpu().numpy().squeeze())
        plt.subplot(2, 3, 5)
        plt.imshow(current_logits.data.max(0)[1].cpu().numpy().squeeze())
        plt.subplot(2, 3, 6)
        instance_image = visualise_instances(predicted_instances, current_labels, class_index=2)
        plt.imshow(instance_image)
        plt.show()

    # image_original = torch.Tensor((plt.imread('test/010650967_816387_1428652.JPG') / 255).transpose(2, 0, 1)[np.newaxis])
    # image = F.pad(image_original, (112, 112, 91, 91))
    # logits, instance_embeddings = model(image.cuda())
    # current_logits = logits[0]
    # predicted_class = F.pad(current_logits, (-112, -112, -91, -91)).data.max(0)[1]
    # instance_embeddings = F.pad(instance_embeddings, (-112, -112, -91, -91))[0]
    #
    # predicted_instances = [None] * 5
    # for class_index in range(5):
    #     mask = predicted_class.view(-1) == class_index
    #     if mask.max() > 0:
    #         label_embedding = instance_embeddings.view(1, instance_embeddings.shape[0], -1)[..., mask]
    #         label_embedding = label_embedding.data.cpu().numpy()[0]
    #
    #         predicted_instances[class_index] = mean_shift(label_embedding)
    #
    # plt.subplot(3, 1, 1)
    # plt.imshow(image_original[0].data.numpy().transpose(1, 2, 0))
    # plt.subplot(3, 1, 2)
    # plt.imshow(predicted_class.cpu().numpy())
    # plt.subplot(3, 1, 3)
    #
    # instance_image = np.zeros((image_original.shape[2], image_original.shape[3]))
    # total = -1
    # for class_index in range(1, 5):
    #     class_instances = visualise_instances(predicted_instances[class_index], predicted_class, class_index=class_index)
    #     class_instances[class_instances > 0] += total + 1
    #     total = class_instances.max()
    #     instance_image += class_instances
    #
    # plt.imshow(instance_image)
    # plt.show()
