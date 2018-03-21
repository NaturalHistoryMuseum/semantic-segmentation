from itertools import chain, cycle, islice
import logging
from pathlib import Path

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

from segmentation.instances import SemanticLabels


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


def visualise_results(output, original_image, reconstructed_image, predicted_class, colours, n=5, dpi=500):
    gs = gridspec.GridSpec(3, n, width_ratios=[2.42]*n, wspace=0.05, hspace=0)
    plt.figure(figsize=(n * 2.42, 3))

    for i in range(n):
        plt.subplot(gs[0, i])
        plt.imshow(original_image.data[i].cpu().numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.subplot(gs[1, i])
        plt.imshow(reconstructed_image.data[i].cpu().numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.subplot(gs[2, i])
        plt.imshow(visualise_segmentation(predicted_class[i], colours))
        plt.axis('off')
    plt.savefig(str(output), dpi=dpi, bbox_inches='tight')
    plt.close('all')


def torch_zip(*args):
    for items in zip(*args):
        yield tuple(item.unsqueeze(0) for item in items)


def train(model, instance_clustering, train_loader_labelled, train_loader_unlabelled, test_loader_labelled, test_loader_unlabelled):
    cross_entropy = nn.CrossEntropyLoss()
    l2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, 25, gamma=0.1)

    losses = {'train': {'semantic': [], 'instance': [], 'total': []},
              'test':  {'semantic': [], 'instance': [], 'total': []}}
    accuracies = {'train': [], 'test': []}

    for epoch in range(100):
        scheduler.step()

        if epoch % scheduler.step_size == 0:
            logging.debug(f'Learning rate set to {scheduler.get_lr()[0]}')

        model.train()

        for i, training_data in enumerate(chain(*zip(train_loader_unlabelled, cycle(train_loader_labelled)))):
            labelled = isinstance(training_data, tuple) or isinstance(training_data, list)

            if labelled:
                image, labels, instances = training_data
                image, labels, instances = Variable(image).cuda(), Variable(labels).cuda(), Variable(instances).cuda()
            else:
                image = training_data
                image = Variable(image).cuda()

            optimizer.zero_grad()

            z_hat1, x_hat, logits, instance_embeddings = model(image)
            z1 = model.forward_clean(image)[0]
            reconstruction_loss = l2(z_hat1, Variable(z1.data, requires_grad=False)) + l2(x_hat, image)
            loss = 20 * reconstruction_loss

            if labelled:
                logits_per_pixel = logits.view(image.shape[0], 5, -1).transpose(1, 2).contiguous()
                semantic_loss = cross_entropy(logits_per_pixel.view(-1, 5), labels.view(-1))

                instance_loss = sum(sum(instance_clustering(embeddings, target_clusters)
                                        for embeddings, target_clusters
                                        in SemanticLabels(image_instance_embeddings, image_labels, image_instances))
                                    for image_instance_embeddings, image_labels, image_instances
                                    in torch_zip(instance_embeddings, labels, instances))

                loss += semantic_loss * 10 + instance_loss

                predicted_class = logits.data.max(1, keepdim=True)[1]
                correct_prediction = predicted_class.eq(labels.data.view_as(predicted_class))
                accuracy = correct_prediction.int().sum().item() / np.prod(predicted_class.shape)

            loss.backward()
            optimizer.step()

            # losses['train']['semantic'].append(semantic_loss.item())
            # losses['train']['instance'].append(instance_loss.item())
            losses['train']['total'].append(loss.item())
            # accuracies['train'].append(accuracy)
            info = f'Epoch: {epoch + 1:{3}}, Batch: {i:{3}}, Loss: {loss.item()}'
            if labelled:
                info += f', Accuracy: {(accuracy * 100)}%'
            logging.info(info)

        # if (epoch + 1) % 5 == 0:
        #     model.eval()
        #
        #     total_loss = 0
        #     total_accuracy = 0
        #
        #     num_test_batches = 1
        #
        #     with torch.no_grad():
        #         for image, labels, instances in islice(test_loader, num_test_batches):
        #             image, labels, instances = (Variable(tensor).cuda() for tensor in (image, labels, instances))
        #
        #             z_hat1, x_hat, logits, instance_embeddings = model(image)
        #             z1 = model.forward_clean(image)[0]
        #             # logits_per_pixel = logits.view(image.shape[0], 5, -1).transpose(1, 2).contiguous()
        #             # semantic_loss = cross_entropy(logits_per_pixel.view(-1, 5), labels.view(-1))
        #             #
        #             # instance_loss = sum(sum(instance_clustering(embeddings, target_clusters)
        #             #                         for embeddings, target_clusters
        #             #                         in SemanticLabels(image_instance_embeddings, image_labels, image_instances))
        #             #                     for image_instance_embeddings, image_labels, image_instances
        #             #                     in torch_zip(instance_embeddings, labels, instances))
        #             #
        #             # loss = semantic_loss * 10 + instance_loss
        #             loss = l2(z_hat1, z1) + l2(x_hat, image)
        #
        #             total_loss += loss.item()
        #
        #             predicted_class = logits.data.max(1, keepdim=True)[1]
        #             correct_prediction = predicted_class.eq(labels.data.view_as(predicted_class))
        #             accuracy = correct_prediction.int().sum().item() / np.prod(predicted_class.shape)
        #             total_accuracy += accuracy
        #
        #     average_loss = total_loss / num_test_batches
        #     average_accuracy = total_accuracy / num_test_batches
        #     losses['test']['total'].append(average_loss)
        #     accuracies['test'].append(average_accuracy)
        #     logging.info(f'Epoch: {epoch + 1:{3}}, Test Set, Cross-entropy loss: {average_loss}, Accuracy: {(average_accuracy * 100)}%')

        if (epoch + 1) % 1 == 0:
            visualise_results(Path('results') / f'epoch_{epoch + 1}.png', image, x_hat, predicted_class,
                              colours=train_loader_labelled.dataset.colours)
            np.save('losses.npy', [{'train': losses['train'], 'test': losses['test']}])
            np.save('accuracies.npy', [{'train': accuracies['train'], 'test': accuracies['test']}])

        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), Path('models') / f'epoch_{epoch + 1}')
