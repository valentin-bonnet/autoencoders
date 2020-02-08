import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2


def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('..\imgs\image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


def generate_and_save_images_compare(model, epoch, test_input, file_name_head='image'):
    x_logits = model.reconstruct(test_input)
    test_input = test_input[:2, :, :, :]
    x_logit = x_logits[:2, :, :, :]
    # predictions = model.sample(test_input)
    fig = plt.figure(figsize=(2, 2))

    for i in range(x_logit.shape[0]):
        plt.subplot(2, 2, (2 * i) + 1)
        plt.imshow(cv2.cvtColor(np.float32(test_input[i]), cv2.COLOR_Lab2RGB))
        plt.axis('off')
        plt.subplot(2, 2, 2 * (i + 1))
        plt.imshow(cv2.cvtColor(np.float32(x_logit[i]), cv2.COLOR_Lab2RGB))
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(file_name_head+'_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()

def img_loss_accuracy(train_loss_results, test_loss_results, train_accuracy_results, test_accuracy_results, filename="loss_accuracy"):
    e = np.linspace(1, len(train_loss_results) + 1, len(train_loss_results))
    fig = plt.figure()
    plt.subplot(211)
    train_plot, = plt.plot(e, train_loss_results, 'r:', label="train")
    test_plot, = plt.plot(e, test_loss_results, 'r', label="test")
    plt.legend(handles=[train_plot, test_plot])

    plt.subplot(212)
    train_plot_acc, = plt.plot(e, train_accuracy_results, 'r:', label="train")
    test_plot_acc, = plt.plot(e, test_accuracy_results, 'r', label="test")
    plt.legend(handles=[train_plot_acc, test_plot_acc])
    plt.show()
    plt.savefig(filename+'.png')