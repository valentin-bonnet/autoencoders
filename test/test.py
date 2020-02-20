import os


def multitest(ds_test, model, my_drive_path, directory_name, subfolders=None):

    directory_path = os.path.join(my_drive_path, directory_name)
    if not os.path.isdir(directory_path):
        print("No directory found at : "+directory_path)
        return

    if subfolders is None:
        subfolders = [x[0] for x in os.walk(directory_path)]

    for sub in subfolders:
        sub_path = os.path.join(directory_path, sub)
        if not os.path.isdir(sub_path):
            print("No directory found at : " + directory_path)
            return

"""def multitraining(datasets, models_type, models_arch, models_latent_space, models_use_bn, lrs, epochs, batch_size, ckpt_epochs, directory_name, path):

    model_args = [datasets, models_type, models_arch, models_latent_space, models_use_bn, lrs, epochs, batch_size, ckpt_epochs]
    max_len = max(map(len, model_args))
    print(max_len)

    path_directory = os.path.join(path, directory_name)
    if not os.path.isdir(path_directory):
        os.makedirs(path_directory)

    #Adapt to get all parameters to same size
    for i, arg in enumerate(model_args):
        temp_arg = arg.copy()
        for j in range((max_len-1)//len(arg)):
            arg.extend(temp_arg)
        model_args[i] = arg[:max_len]

    #Use this to get unique dataset used multiple times
    ds = {}
    for dataset in datasets:
        if dataset not in ds:
            train_test_ds, shape = datasetLoader.get_dataset(dataset)
            ds.update({dataset: [train_test_ds, shape]})

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    legendes = []
    # Construct the model,
    for i in range(max_len):
        #Get name
        str_ds = model_args[0][i]
        str_model = model_args[1][i]
        str_arch = '_'.join(str(x) for x in model_args[2][i])
        str_lat = 'lat' + str(model_args[3][i])
        str_use_bn = 'BN' if model_args[4][i] else ''
        str_all = '_'.join(filter(None, [str_ds, str_model, str_arch, str_lat, str_use_bn]))

        #Construct the model
        dataset = ds[model_args[0][i]][0]
        shape = ds[model_args[0][i]][1]
        model_type = model_args[1][i]
        model_arch = model_args[2][i].copy()
        model_lat = model_args[3][i]
        model_use_bn = model_args[4][i]

        lr = model_args[5][i]
        epoch = model_args[6][i]
        bs = model_args[7][i]
        ckpt_epoch = model_args[8][i]

        print(model_lat)

        model = construct_model.get_model(model_type, model_arch, model_lat, shape, model_use_bn)

        #Train
        print(dataset)
        print(model)
        print(lr)
        print(epoch)
        print(bs)
        print(ckpt_epoch)


        train_l, test_l, train_acc, test_acc = train(dataset, model, lr, epoch, bs, ckpt_epoch, path_directory,  str_all, True)

        #Get curves and output them
        train_losses.append(train_l)
        train_accs.append(train_acc)
        test_losses.append(test_l)
        test_accs.append(test_acc)



        legendes.append(str_all)

    image_saver.curves(test_accs, legendes, directory_name, path_directory)"""



