import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image

palette_DAVIS = tf.constant([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0]])




def db_eval_boundary(foreground_mask,gt_mask,bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.

    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask);
    gt_boundary = seg2bmap(gt_mask);

    from skimage.morphology import binary_dilation,disk
    disk_fg = disk(bound_pix)
    fg_dil = binary_dilation(fg_boundary,disk_fg)
    gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg     = np.sum(fg_boundary)
    n_gt     = np.sum(gt_boundary)

    #% Compute precision and recall
    if n_fg == 0 and  n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0  and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match)/float(n_fg)
        recall    = np.sum(gt_match)/float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2*precision*recall/(precision+recall);

    return F

def seg2bmap(seg,width=None,height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]

    Returns:
        bmap (ndarray):	Binary boundary map.

     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
 """

    seg = seg.astype(np.bool)
    seg[seg>0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width  = seg.shape[1] if width  is None else width
    height = seg.shape[0] if height is None else height

    h,w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
            'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

    e  = np.zeros_like(seg)
    s  = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:,:-1]    = seg[:,1:]
    s[:-1,:]    = seg[1:,:]
    se[:-1,:-1] = seg[1:,1:]

    b        = seg^e | seg^s | seg^se
    b[-1,:]  = seg[-1,:]^e[-1,:]
    b[:,-1]  = seg[:,-1]^s[:,-1]
    b[-1,-1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height,width))
        for x in range(w):
            for y in range(h):
                if b[y,x]:
                    j = 1+floor((y-1)+height / h)
                    i = 1+floor((x-1)+width  / h)
                    bmap[j,i] = 1;

    return bmap

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

def generate_gif(images, file_name_head='image', path='./'):
    im = []
    for image in images:
        im.append(Image.fromarray(image.numpy()*255.0))

    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = os.path.join(path, file_name_head)
    im[0].save(file_path+'.gif', save_all=True, append_images=im[1:], duration=150)
    plt.savefig(file_path + '.png')

def generate_gif_concat(model, input, file_name_head='image', path='./'):
    output = model.reconstruct(input)
    input = tf.squeeze(input[0, ...]) #(seq_size, im_shape, im_shape)
    output = tf.squeeze(output[0, ...]) #(seq_size, im_shape, im_shape)
    concated_images = tf.concat([input, output], axis=2) ##(seq_size, im_shape, im_shape*2) # The image are concated horizontally

    generate_gif(concated_images, file_name_head, path)

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


def KAST_test_ResNet(kast, davis, file_name_head='image', path='./'):
    #output_v, v_j, i_drop = kast.call(davis, training=False)
    output_v, v_j, v_0 = kast.call_ResNet_Local(davis, training=False)
    #raw = tf.image.resize(i_drop[0][1:], [64, 64]).numpy()
    output_v = output_v[0].numpy()
    v_j = v_j[0].numpy()
    v_0 = v_0[0].numpy()
    all_white = np.ones([64, 64, 3]) * [1.0, 0., 0.]


    #LAB to RGB
    output_v = cv2.cvtColor(np.float32((output_v + 1.0) * [50.0, 127.5, 127.5] - [0., 128., 128.]), cv2.COLOR_Lab2RGB)
    v_j = cv2.cvtColor(np.float32((v_j + 1.0) * [50.0, 127.5, 127.5] - [0., 128., 128.]), cv2.COLOR_Lab2RGB)
    v_0 = cv2.cvtColor(np.float32((v_0 + 1.0) * [50.0, 127.5, 127.5] - [0., 128., 128.]), cv2.COLOR_Lab2RGB)
    all_white = cv2.cvtColor(np.float32((all_white + 1.0) * [50.0, 127.5, 127.5] - [0., 128., 128.]), cv2.COLOR_Lab2RGB)


    if not os.path.isdir(path):
        os.makedirs(path)

    #IMAGES
    fig = plt.figure(figsize=(2, 2))
    plt.subplot(2, 2, 1)
    plt.imshow(v_0)
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(v_j)
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(all_white)
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(output_v)
    plt.axis('off')

    file_path = os.path.join(path, file_name_head)
    plt.savefig(file_path + '_DAVIS.png')
    plt.close(fig)

    # GIF
    #images = tf.concat([output_v, v_j], axis=2).numpy()
    #im = []
    #for image in images:
    #    im.append(Image.fromarray(np.uint8(image*[255.0, 255.0, 255.0])))

    #file_path = os.path.join(path, file_name_head)
    #im[0].save(file_path + '_DAVIS.gif', save_all=True, append_images=im[1:], duration=150)

def KAST_JF(kast, davis):
    i_raw, v = tf.nest.flatten(davis)
    output_v, v_j, i_drop = kast.call(davis, training=False)
    output_v = output_v[0]
    seq_size = output_v.shape[0]
    max_value_output = tf.argmax(output_v, -1)
    v_j = v_j[0]
    max_value = tf.argmax(v_j, -1)

    number_size = tf.reduce_max(max_value)

    jss = []
    fss = []
    for i in range(seq_size):
        js = []
        fs = []
        for class_id in range(1, number_size + 1):
            gt = (max_value[i] == class_id)
            segment = (max_value_output[i] == class_id)
            j_and = gt & segment
            # print(j_and)
            j_and_float = np.float32(j_and)
            # print(j_and_float)
            j_and_sum = np.sum(j_and_float)
            # print(j_and_sum)
            j_or = gt | segment
            # print(j_or)
            j_or_float = np.float32(j_or)
            # print(j_or_float)
            j_or_sum = np.sum(j_or_float)
            j = j_and_sum / j_or_sum
            # j = tf.reduce_sum(gt & segment) / tf.reduce_sum(gt | segment)
            js.append(j)
            f = db_eval_boundary(np.float32(gt), np.float32(segment))
            fs.append(f)

        j_mean = np.mean(js)
        f_mean = np.mean(fs)
        jss.append(j_mean)
        fss.append(f_mean)



    return np.mean(jss), np.mean(fss)


def KAST_test(kast, davis, file_name_head='image', path='./'):
    #output_v, v_j, i_drop = kast.call(davis, training=False)
    output_v, v_j, i_drop = kast.call(davis, training=False)
    raw = tf.image.resize(i_drop[0][1:], [64, 64]).numpy()
    #output_v = output_v[0].numpy()
    output_v = output_v[0]
    max_value_output = tf.argmax(output_v, -1)
    output_v = tf.gather(palette_DAVIS, max_value_output).numpy()
    #v_j = v_j[0].numpy()
    v_j = v_j[0]
    max_value = tf.argmax(v_j, -1)
    v_j = tf.gather(palette_DAVIS, max_value).numpy()
    seq_size = output_v.shape[0]

    #LAB to RGB
    for i in range(seq_size):
        #output_v[i] = cv2.cvtColor(np.float32((output_v[i] + 1.0) * [50.0, 127.5, 127.5] - [0., 128., 128.]), cv2.COLOR_Lab2RGB)
        #v_j[i] = cv2.cvtColor(np.float32((v_j[i] + 1.0) * [50.0, 127.5, 127.5] - [0., 128., 128.]), cv2.COLOR_Lab2RGB)
        raw[i] = cv2.cvtColor(np.float32((raw[i] + 1.0) * [50.0, 127.5, 127.5] - [0., 128., 128.]), cv2.COLOR_Lab2RGB)


    #RGB to RGB
    #output_v = np.uint8((output_v+1.)*127.5)
    #v_j = np.uint8((v_j+1.)*127.5)
    #raw = np.uint8((raw+1.)*127.5)

    if not os.path.isdir(path):
        os.makedirs(path)

    #IMAGES
    fig = plt.figure(figsize=(seq_size, 3))
    for i in range(seq_size):
        plt.subplot(3, seq_size, 1+i)
        plt.imshow(output_v[i])
        plt.axis('off')
        plt.subplot(3, seq_size, 1+i+seq_size)
        plt.imshow(v_j[i])
        plt.axis('off')
        plt.subplot(3, seq_size, i+1+(seq_size*2))
        plt.imshow(raw[i])
        plt.axis('off')

    number_size = tf.reduce_max(max_value)

    jss = []
    fss = []
    for i in range(seq_size):
        js = []
        fs = []
        for class_id in range(1, number_size + 1):
            gt = (max_value[i] == class_id)
            segment = (max_value_output[i] == class_id)
            j_and = gt & segment
            # print(j_and)
            j_and_float = np.float32(j_and)
            # print(j_and_float)
            j_and_sum = np.sum(j_and_float)
            # print(j_and_sum)
            j_or = gt | segment
            # print(j_or)
            j_or_float = np.float32(j_or)
            # print(j_or_float)
            j_or_sum = np.sum(j_or_float)
            j = j_and_sum / j_or_sum
            # j = tf.reduce_sum(gt & segment) / tf.reduce_sum(gt | segment)
            js.append(j)
            f = db_eval_boundary(np.float32(gt), np.float32(segment))
            fs.append(f)

        j_mean = np.mean(js)
        f_mean = np.mean(fs)
        jss.append(j_mean)
        fss.append(f_mean)


    file_path = os.path.join(path, file_name_head)
    plt.savefig(file_path + '_DAVIS.png')
    plt.close(fig)

    # GIF
    #images = tf.concat([output_v, v_j], axis=2).numpy()
    #im = []
    #for image in images:
    #    im.append(Image.fromarray(np.uint8(image*[255.0, 255.0, 255.0])))

    #file_path = os.path.join(path, file_name_head)
    #im[0].save(file_path + '_DAVIS.gif', save_all=True, append_images=im[1:], duration=150)

def KAST_View_Resnet(kast, input_data, training=True, file_name_head='image', path='./'):
    output_v, v_j, v_0 = kast.reconstruct_ResNet(input_data, training=training)

    output_v = output_v[0].numpy()
    v_j = v_j[0].numpy()
    v_0 = v_0[0].numpy()
    all_white = np.ones([64, 64, 3]) * [1.0, 0., 0.]


    # LAB to RGB
    output_v = cv2.cvtColor(np.float32((output_v + 1.0) * [50.0, 127.5, 127.5] - [0., 128., 128.]), cv2.COLOR_Lab2RGB)
    v_j = cv2.cvtColor(np.float32((v_j + 1.0) * [50.0, 127.5, 127.5] - [0., 128., 128.]), cv2.COLOR_Lab2RGB)
    v_0 = cv2.cvtColor(np.float32((v_0 + 1.0) * [50.0, 127.5, 127.5] - [0., 128., 128.]), cv2.COLOR_Lab2RGB)
    all_white = cv2.cvtColor(np.float32((all_white + 1.0) * [50.0, 127.5, 127.5] - [0., 128., 128.]), cv2.COLOR_Lab2RGB)

    if not os.path.isdir(path):
        os.makedirs(path)

    # IMAGES
    fig = plt.figure(figsize=(2, 2))
    plt.subplot(2, 2, 1)
    plt.imshow(v_0)
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(v_j)
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(all_white)
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(output_v)
    plt.axis('off')

    file_path = os.path.join(path, file_name_head)
    plt.savefig(file_path + '_Resnet.png')
    plt.close(fig)

def KAST_View(kast, input_data, training=True, file_name_head='image', path='./'):
    output, ground_truth, image_drop_out = kast.reconstruct(input_data, training)
    ground_truth = ground_truth[0].numpy()
    seq_size = ground_truth.shape[0]
    output = output[0].numpy()
    image_drop_out = image_drop_out[0].numpy()


    # Input / output / drop_out to LAB
    all_white = np.ones([1, 64, 64, 3]) * [1.0, 0., 0.]
    output = np.concatenate([all_white, output], axis=0)
    for i in range(seq_size):
        ground_truth[i] = cv2.cvtColor(np.float32((ground_truth[i] + 1.0) * [50.0, 127.5, 127.5] - [0., 128., 128.]), cv2.COLOR_Lab2RGB)
        output[i] = cv2.cvtColor(np.float32((output[i] + 1.0) * [50.0, 127.5, 127.5] - [0., 128., 128.]), cv2.COLOR_Lab2RGB)
        image_drop_out[i] = cv2.cvtColor(np.float32((image_drop_out[i] + 1.0) * [50.0, 127.5, 127.5] - [0., 128., 128.]), cv2.COLOR_Lab2RGB)

    # RGB to RGB
    #ground_truth = np.uint8((ground_truth + 1.) * 127.5)
    #output = np.uint8((output + 1.) * 127.5)
    #image_drop_out = np.uint8((image_drop_out + 1.) * 127.5)


    # Input with input / drop_out / output
    if not os.path.isdir(path):
        os.makedirs(path)

    fig = plt.figure(figsize=(seq_size, 3))

    for i in range(seq_size):
        plt.subplot(3, seq_size, i + 1)
        plt.imshow(ground_truth[i])
        plt.axis('off')
        plt.subplot(3, seq_size, seq_size + i + 1)
        plt.imshow(image_drop_out[i])
        plt.axis('off')
        plt.subplot(3, seq_size, seq_size * 2 + i + 1)
        plt.imshow(output[i])
        plt.axis('off')

    file_path = os.path.join(path, file_name_head)
    plt.savefig(file_path + '.png')
    plt.close(fig)

    # Gif with input / drop_out / output
    #images = tf.concat([ground_truth, image_drop_out, output], axis=2).numpy()
    #im = []
    #for image in images:
    #    im.append(Image.fromarray(np.uint8(image)))

    #file_path = os.path.join(path, file_name_head)
    #im[0].save(file_path + '.gif', save_all=True, append_images=im[1:], duration=150)





def generate_and_save_images_compare_seq_lab(model, test_inputs, file_name_head='image', path='./', seq_size=8):
    seq_size=seq_size-1
    x_logits, test_input = model.reconstruct(test_inputs)
    #print("test_inputs: ", test_inputs[0, 0])
    #print("test_input: ", test_input[0, 0])
    #print("x_logits: ", x_logits[0, 0])
    #x_logits_vae = model.reconstruct_vae(test_input)
    #test_input = np.squeeze(test_input)
    #x_logits = np.squeeze(x_logits)
    test_input = test_input + 1.0
    test_input = test_input * [50.0, 127.5, 127.5]
    test_input = tf.cast(test_input - [0, 128, 128], tf.int32)
    x_logits = x_logits + 1.0
    x_logits = x_logits * [50.0, 127.5, 127.5]
    x_logits = tf.cast(x_logits - [0, 128, 128], tf.int32)
    #x_logits_vae = np.squeeze(x_logits_vae)
    #print(x_logits)
    #print("\n\n#######\n input:\n", test_input[0, 5, :, :])
    #print("max: ", tf.reduce_max(test_input[0, 5, :, :]))
    #print("\n\n#######\n output:\n", x_logits[0, 5, :, :])
    #print("max: ", tf.reduce_max(x_logits[0, 5, :, :]))
    nb_imgs = 5
    fig = plt.figure(figsize=(nb_imgs, 2))
    for i in range(2):
        test_input_0 = cv2.cvtColor(np.float32(test_input[i, 0]), cv2.COLOR_Lab2RGB)
        test_input_5 = cv2.cvtColor(np.float32(test_input[i, seq_size//4]), cv2.COLOR_Lab2RGB)
        test_input_10 = cv2.cvtColor(np.float32(test_input[i, 2*seq_size//4]), cv2.COLOR_Lab2RGB)
        test_input_15 = cv2.cvtColor(np.float32(test_input[i, 3*seq_size//4]), cv2.COLOR_Lab2RGB)
        test_input_19 = cv2.cvtColor(np.float32(test_input[i, seq_size-1]), cv2.COLOR_Lab2RGB)
        test_inputs_seq = [test_input_0, test_input_5, test_input_10, test_input_15, test_input_19]
        x_logit_0 = cv2.cvtColor(np.float32(x_logits[i, 0]), cv2.COLOR_Lab2RGB)
        x_logit_5 = cv2.cvtColor(np.float32(x_logits[i, seq_size//4]), cv2.COLOR_Lab2RGB)
        x_logit_10 = cv2.cvtColor(np.float32(x_logits[i, 2*seq_size//4]), cv2.COLOR_Lab2RGB)
        x_logit_15 = cv2.cvtColor(np.float32(x_logits[i, 3*seq_size//4]), cv2.COLOR_Lab2RGB)
        x_logit_19 = cv2.cvtColor(np.float32(x_logits[i, seq_size-1]), cv2.COLOR_Lab2RGB)
        x_logits_seq = [x_logit_0, x_logit_5, x_logit_10, x_logit_15, x_logit_19]


        """
        x_logit_vae_0 = x_logits_vae[:2, 0, :, :]
        x_logit_vae_5 = x_logits_vae[:2, 5, :, :]
        x_logit_vae_10 = x_logits_vae[:2, 10, :, :]
        x_logit_vae_15 = x_logits_vae[:2, 15, :, :]
        x_logit_vae_19 = x_logits_vae[:2, 19, :, :]
        x_logits_vae = [x_logit_vae_0, x_logit_vae_5, x_logit_vae_10, x_logit_vae_15, x_logit_vae_19]
        """
        # predictions = model.sample(test_input)

        #fig = plt.figure(figsize=(nb_imgs, 6))





        for j in range(nb_imgs):
            #plt.subplot(6, nb_imgs, nb_imgs*3*i + j + 1)
            plt.subplot(4, nb_imgs, nb_imgs*2*i + j + 1)
            plt.imshow(test_inputs_seq[j])
            plt.axis('off')
        for j in range(nb_imgs):
            #plt.subplot(6, nb_imgs, nb_imgs*(3*i+1) + j + 1)
            plt.subplot(4, nb_imgs, nb_imgs*(2*i+1) + j + 1)
            plt.imshow(x_logits_seq[j])
            plt.axis('off')
        """
        for j in range(nb_imgs):
            plt.subplot(6, nb_imgs, nb_imgs*(3*i+2) + j + 1)
            plt.imshow(x_logits_vae[j][i])
            plt.axis('off')"""

    # tight_layout minimizes the overlap between 2 sub-plots
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = os.path.join(path, file_name_head)
    plt.savefig(file_path + '.png')

def generate_and_save_images_compare_seq(model, test_input, file_name_head='image', path='./', seq_size=8):
    x_logits = model.reconstruct(test_input)
    #x_logits_vae = model.reconstruct_vae(test_input)
    test_input = np.squeeze(test_input)
    x_logits = np.squeeze(x_logits)
    #x_logits_vae = np.squeeze(x_logits_vae)
    #print(x_logits)
    #print("\n\n#######\n input:\n", test_input[0, 5, :, :])
    #print("max: ", tf.reduce_max(test_input[0, 5, :, :]))
    #print("\n\n#######\n output:\n", x_logits[0, 5, :, :])
    #print("max: ", tf.reduce_max(x_logits[0, 5, :, :]))


    test_input_0 = test_input[:2, 0, :, :, :]
    test_input_5 = test_input[:2, seq_size//4, :, :]
    test_input_10 = test_input[:2, 2*seq_size//4, :, :]
    test_input_15 = test_input[:2, 3*seq_size//4, :, :]
    test_input_19 = test_input[:2, seq_size-1, :, :]
    test_inputs = [test_input_0, test_input_5, test_input_10, test_input_15, test_input_19]
    x_logit_0 = x_logits[:2, 0, :, :]
    x_logit_5 = x_logits[:2, seq_size//4, :, :]
    x_logit_10 = x_logits[:2, 2*seq_size//4, :, :]
    x_logit_15 = x_logits[:2, 3*seq_size//4, :, :]
    x_logit_19 = x_logits[:2, seq_size-1, :, :]
    x_logits = [x_logit_0, x_logit_5, x_logit_10, x_logit_15, x_logit_19]
    """
    x_logit_vae_0 = x_logits_vae[:2, 0, :, :]
    x_logit_vae_5 = x_logits_vae[:2, 5, :, :]
    x_logit_vae_10 = x_logits_vae[:2, 10, :, :]
    x_logit_vae_15 = x_logits_vae[:2, 15, :, :]
    x_logit_vae_19 = x_logits_vae[:2, 19, :, :]
    x_logits_vae = [x_logit_vae_0, x_logit_vae_5, x_logit_vae_10, x_logit_vae_15, x_logit_vae_19]
    """
    # predictions = model.sample(test_input)
    nb_imgs = len(test_inputs)
    #fig = plt.figure(figsize=(nb_imgs, 6))
    fig = plt.figure(figsize=(nb_imgs, 4))



    for i in range(2):
        for j in range(nb_imgs):
            #plt.subplot(6, nb_imgs, nb_imgs*3*i + j + 1)
            plt.subplot(4, nb_imgs, nb_imgs*2*i + j + 1)
            plt.imshow(test_inputs[j][i])
            plt.axis('off')
        for j in range(nb_imgs):
            #plt.subplot(6, nb_imgs, nb_imgs*(3*i+1) + j + 1)
            plt.subplot(4, nb_imgs, nb_imgs*(2*i+1) + j + 1)
            plt.imshow(x_logits[j][i])
            plt.axis('off')
        """
        for j in range(nb_imgs):
            plt.subplot(6, nb_imgs, nb_imgs*(3*i+2) + j + 1)
            plt.imshow(x_logits_vae[j][i])
            plt.axis('off')"""

    # tight_layout minimizes the overlap between 2 sub-plots
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = os.path.join(path, file_name_head)
    plt.savefig(file_path + '.png')

def generate_and_save_images_compare_lab(model, test_input, file_name_head='image', path='./'):
    print(test_input.shape)
    x_logits = model.reconstruct(test_input)
    print(x_logits.shape)
    test_input = test_input * [100, 255.0, 255.0]
    test_input = test_input - [0, 128, 128]
    x_logits = x_logits * [100, 255.0, 255.0]
    x_logits = x_logits - [0, 128, 128]

    test_input = test_input[0, :, :, :]
    x_logit = x_logits[0, :, :, :]
    # predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4, 2))

    input_l = extract_single_dim_from_LAB_convert_to_RGB(test_input, 0)
    input_a = extract_single_dim_from_LAB_convert_to_RGB(test_input, 1)
    input_b = extract_single_dim_from_LAB_convert_to_RGB(test_input, 2)
    logit_l = extract_single_dim_from_LAB_convert_to_RGB(x_logit, 0)
    logit_a = extract_single_dim_from_LAB_convert_to_RGB(x_logit, 1)
    logit_b = extract_single_dim_from_LAB_convert_to_RGB(x_logit, 2)
    test_input_color = cv2.cvtColor(np.float32(test_input), cv2.COLOR_Lab2RGB)
    test_logit_color = cv2.cvtColor(np.float32(x_logit), cv2.COLOR_Lab2RGB)
    plt.subplot(2, 4, 1)
    plt.imshow(test_input_color)
    plt.axis('off')
    plt.subplot(2, 4, 2)
    plt.imshow(input_l)
    plt.axis('off')
    plt.subplot(2, 4, 3)
    plt.imshow(input_a)
    plt.axis('off')
    plt.subplot(2, 4, 4)
    plt.imshow(input_b)
    plt.axis('off')
    plt.subplot(2, 4, 5)
    plt.imshow(test_logit_color)
    plt.axis('off')
    plt.subplot(2, 4, 6)
    plt.imshow(logit_l)
    plt.axis('off')
    plt.subplot(2, 4, 7)
    plt.imshow(logit_a)
    plt.axis('off')
    plt.subplot(2, 4, 8)
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
        plt.figure(figsize=(1, 2))
        plt.subplot(2, 1, 1)
        plt.imshow(imgs_ground_truth)
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.imshow(imgs_reconstruct)
        plt.axis('off')

        if not os.path.isdir(path):
            os.makedirs(path)
        file_path = os.path.join(path, filename)
        plt.savefig(file_path+'.png')

def compare_multiple_images_Lab(images, legends, filename, path):
    nb_images = len(images[0])
    nb_models = len(images)
    print("nb_images ", nb_images)
    print("nb_models ", nb_models)

    images = np.asarray(images, dtype=np.float32) * [100, 255.0, 255.0]
    images = images - [0, 128, 128]

    legends = ['GT'] + legends

    fontdic = {'size': 6,
               'verticalalignment': 'top'}

    plt.figure(figsize=(nb_images, nb_models))
    for i in range(nb_models):
        for j in range(nb_images):
            ax = plt.subplot(nb_images, nb_models, j*nb_models+i+1)

            plt.imshow(cv2.cvtColor(np.float32(images[i][j]), cv2.COLOR_Lab2RGB))
            plt.axis('off')
            if j == 0:
                ax.set_title(legends[i], fontdict=fontdic)

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

def curves(curves, legendes, file_name, path, x_axis_label='', y_axis_label='', x_axis=None):
    handles = []
    fig = plt.figure()
    len_max = max(list(map(len, curves)))
    if x_axis is None:
        e = np.linspace(1, len_max, len_max)
    else:
        e = x_axis
    for i, curve in enumerate(curves):

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


