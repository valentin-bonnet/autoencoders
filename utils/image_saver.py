import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
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


def generate_and_save_images_compare(model, test_input, file_name_head='image', path='./'):
    x_logits = model.reconstruct(test_input)
    test_input = test_input[:2, :, :, :]
    x_logit = x_logits[:2, :, :, :]
    # predictions = model.sample(test_input)
    fig = plt.figure(figsize=(2, 2))
    print(x_logit.dtype)
    print(test_input.dtype)

    for i in range(x_logit.shape[0]):
        plt.subplot(2, 2, (2 * i) + 1)
        test_input_color = cv2.cvtColor(np.float32(test_input[i]), cv2.COLOR_Lab2RGB)
        test_logit_color = cv2.cvtColor(np.float32(x_logit[i]), cv2.COLOR_Lab2RGB)
        print(test_input_color)
        print(test_logit_color)
        plt.imshow(test_input_color)
        plt.axis('off')
        plt.subplot(2, 2, 2 * (i + 1))
        plt.imshow(test_logit_color)
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = os.path.join(path, file_name_head)
    plt.savefig(file_path + '.png')
    # plt.show()


def extract_single_dim_from_LAB_convert_to_RGB(image, idim):
    '''
    image is a single lab image of shape (None,None,3)
    '''
    z = np.zeros(image.shape)
    if idim != 0:
        z[:, :, 0] = 50  ## I need brightness to plot the image along 1st or 2nd axis
    z[:, :, idim] = image[:, :, idim]
    z = cv2.cvtColor(np.float32(z), cv2.COLOR_Lab2RGB)
    return (z)


def generate_and_save_images_compare_lab(model, test_input, file_name_head='image', path='./'):
    x_logits = model.reconstruct(test_input)
    test_input = test_input * [100, 255.0, 255.0]
    test_input = test_input - [0, 128, 128]
    x_logits = x_logits * [100, 255.0, 255.0]
    x_logits = x_logits - [0, 128, 128]

    test_input = test_input[:2, :, :, :]
    x_logit = x_logits[:2, :, :, :]
    # predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4, 4))

    for i in range(x_logit.shape[0]):
        input_l = extract_single_dim_from_LAB_convert_to_RGB(test_input[i], 0)
        input_a = extract_single_dim_from_LAB_convert_to_RGB(test_input[i], 1)
        input_b = extract_single_dim_from_LAB_convert_to_RGB(test_input[i], 2)
        logit_l = extract_single_dim_from_LAB_convert_to_RGB(x_logit[i], 0)
        logit_a = extract_single_dim_from_LAB_convert_to_RGB(x_logit[i], 1)
        logit_b = extract_single_dim_from_LAB_convert_to_RGB(x_logit[i], 2)
        test_input_color = cv2.cvtColor(np.float32(test_input[i]), cv2.COLOR_Lab2RGB)
        test_logit_color = cv2.cvtColor(np.float32(x_logit[i]), cv2.COLOR_Lab2RGB)
        plt.subplot(4, 4, (8 * i) + 1)
        plt.imshow(test_input_color)
        plt.axis('off')
        plt.subplot(4, 4, (8 * i) + 2)
        plt.imshow(input_l)
        plt.axis('off')
        plt.subplot(4, 4, (8 * i) + 3)
        plt.imshow(input_a)
        plt.axis('off')
        plt.subplot(4, 4, (8 * i) + 4)
        plt.imshow(input_b)
        plt.axis('off')
        plt.subplot(4, 4, (8 * i) + 5)
        plt.imshow(test_logit_color)
        plt.axis('off')
        plt.subplot(4, 4, (8 * i) + 6)
        plt.imshow(logit_l)
        plt.axis('off')
        plt.subplot(4, 4, (8 * i) + 7)
        plt.imshow(logit_a)
        plt.axis('off')
        plt.subplot(4, 4, (8 * i) + 8)
        plt.imshow(logit_b)
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = os.path.join(path, file_name_head)
    plt.savefig(file_path + '.png')

def compare_images(imgs_ground_truth, imgs_reconstruct, filename, path):
    if len(imgs_ground_truth) != len(imgs_reconstruct):
        print("Number of ground truth image and reconstructed image are not equals")

    else:
        nb_imgs = len(imgs_ground_truth)
        plt.figure(figsize=(nb_imgs, 2))
        for i in range(nb_imgs):
            plt.subplot(nb_imgs, 2, (2*i))
            plt.imshow(imgs_ground_truth[i])
            plt.axis('off')
            plt.subplot(nb_imgs, 2, (2*i))
            plt.imshow(imgs_reconstruct[i])
            plt.axis('off')

        if not os.path.isdir(path):
            os.makedirs(path)
        file_path = os.path.join(path, filename)
        plt.savefig(file_path+'.png')

def compare_multiple_images_Lab(images, legends, filename, path):
    nb_images = len(images)
    nb_models = len(images[0])
    print("nb_images ", nb_images)
    print("nb_models ", nb_models)

    images = images * [100, 255.0, 255.0]
    images = images - [0, 128, 128]

    plt.figure(figsize=(nb_images, nb_models))
    for i in range(nb_models):
        for j in range(nb_images):
            ax = plt.subplot(nb_images, nb_models, i*nb_models+j+1)
            plt.imshow(images[i][j])
            plt.axis('off')
            if i == nb_models-1:
                ax.set_title(legends[j])

    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = os.path.join(path, filename)
    plt.savefig(file_path + '.png')

def compare_multiple_images(images, legends, filename, path):
    nb_images = len(images)
    nb_models = len(images[0])

    plt.figure(figsize=(nb_images, nb_models))
    for i in range(nb_images):
        for j in range(nb_models):
            plt.subplot(nb_images, nb_models, i*nb_models+j+1)
            plt.imshow(images[i][j])
            plt.axis('off')

    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = os.path.join(path, filename)
    plt.savefig(file_path + '.png')

def curves(curves, legendes, file_name, path, x_axis_label='', y_axis_label=''):
    handles = []
    fig = plt.figure()
    for i, curve in enumerate(curves):
        e = np.linspace(1, len(curve), len(curve))
        handle, = plt.plot(e, curve, label=legendes[i])
        handles.append(handle)
    plt.legend(handles=handles)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = os.path.join(path, file_name)
    plt.savefig(file_path + '.png')
    plt.show()

def points(points, legendes, file_name, path):
    handles = []
    fig = plt.figure()
    for i, point in enumerate(points):
        e = np.linspace(1, len(point), len(point))
        handle, = plt.plot(e, point, 'ro', label=legendes[i])
        handles.append(handle)
    plt.legend(handles=handles)
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = os.path.join(path, file_name)
    plt.savefig(file_path + '.png')
    plt.show()

def img_loss_accuracy(train_loss_results, test_loss_results, train_accuracy_results, test_accuracy_results,
                      filename="loss_accuracy", path='./'):
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
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = os.path.join(path, filename)
    plt.savefig(file_path + '.png')
    plt.show()


