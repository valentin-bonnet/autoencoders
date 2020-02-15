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
    plt.savefig(file_name_head+'_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()



def extract_single_dim_from_LAB_convert_to_RGB(image,idim):
    '''
    image is a single lab image of shape (None,None,3)
    '''
    z = np.zeros(image.shape)
    if idim != 0 :
        z[:,:,0]=50 ## I need brightness to plot the image along 1st or 2nd axis
    z[:,:,idim] = image[:,:,idim]
    z = cv2.cvtColor(np.float32(z), cv2.COLOR_Lab2RGB)
    return(z)

def generate_and_save_images_compare_lab(model, epoch, test_input, file_name_head='image'):
    x_logits = model.reconstruct(test_input)
    test_input = test_input*[100, 255.0, 255.0]
    test_input = test_input-[0, 128, 128]
    x_logits = x_logits*[100, 255.0, 255.0]
    x_logits = x_logits-[0, 128, 128]

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
    plt.savefig(file_name_head+'_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()

def curves(curves, legendes, file_name):

    handles = []
    for i, curve in enumerate(curves):
        e = np.linspace(1, len(curve), len(curve))
        handle, = plt.plot(e, curve, label=legendes[i])
        handles.append(handle)
    plt.legend(handles=handles)
    plt.show()
    print(file_name)



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