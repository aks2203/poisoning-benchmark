# The Benchmark Challenge

### The problem setup for transfer learning tests:
#### CIFAR-10
- 25 poisons, 2,500 images in the finetuning dataset (random subset of CIFAR10 standard training set).
- Models pre-trained on CIFAR100 transfered to CIFAR10.
- Perturbations bound in the l-infinity sense with epsilon = 8/255.

We provide a ResNet18 pretrained on CIFAR100 on which the white-box tests will be done. The black-box tests are done on VGG11 and MobileNetV2 models.

#### TinyImageNet
- 250 poisons, 50,000 images in the finetuning dataset (classes 100-199 in TinyImageNet standard training set).
- Models pre-trained on classes 0-99 of TinyImageNet standard training set.
- Perturbations bound in the l-infinity sense with epsilon = 8/255.

We provide a VGG16 pretrained on classes 0-99 of TinyImageNet standard training set on which the white-box tests will be done. The black-box tests are done on ResNet34 and MobileNetV2 models.

---

### Problem set up for training from scratch:
#### CIFAR-10
- 500 poisons, with entire CIFAR10 standard training set.
- Perturbations bound in the l-infinity sense with epsilon = 8/255.
- We train ResNet18, VGG11, and MobileNetV2 models for this test (i.e. no white-box / grey-box / black-box version).
#### TinyImageNet
- 250 poisons, with entire TinyImageNet standard training set.
- Perturbations bound in the l-infinity sense with epsilon = 8/255.
- We train VGG16, ResNet34, and MobileNetV2 models for this test (i.e. no white-box / grey-box / black-box version).
___
Python dictionaries with set-up information are stored in the [poison_setups](poison_setups) directory.  The files there each contain a list of dictionaries. Each dictionary has four entries:

   (i) The value corresponding to the key "target class" is an integer denoting the label of the target image in the dataset.
   
   (ii) The value corresponding to the key "target index" is an integer denoting the index of the target image in the CIFAR10 testing data.
  
  (iii) The value corresponding to the key "base class" is an integer denoting the label of the base images in the dataset.
  
   (iv) The value corresponding to the key "base indices" is an numpy array denoting the indices of the base images in the CIFAR10 training data.

The output from the attacker should be three different pickle files: poisons.pickle, base_indices.pickle, target.pickle.  The file poisons.pickle should contain a list of tuples, where each tuple has two entries. Each tuple is one clean-label poison example where the first entry is a PIL image object and the second is the label (this should match the poisoned_label variable). In base_indices.pickle the indices of the clean base images within the training set should be saved so that we can removed the clean images from the training set when evaluating the attack. The file target.pickle should have a single tuple with a PIL image object and an integer class label. In the triggered backdoor setting, this target tuple should also conain the 5x5 patch and and (x,y) coordinate locating the patch in the target image. Even triggerless attacks must submit this, since the evaluation will look to this file in order to load the target.

The setups file can be opened and loaded, and the output files can be saved with the following example python code.

With PyTorch:
```
    import pickle
    from torchvision import datasets, transforms

    with open("poison_setups/cifar10_transfer_learning.pickle", "rb") as handle:
        setup_dicts = pickle.load(handle)

    # Which set up to do in this run?
    setup = setup_dicts[0]    # this can be changed to choose a different setup

    # get set up for this trial
    target_class = setup["target class"]
    target_img_idx = setup["target index"]
    poisoned_label = setup["base class"]
    base_indices = setup["base indices"]
    num_poisons = len(base_indices)

    # load the CIFAR10 datasets
    trainset = datasets.CIFAR10(root="./data", train=True, download=True,
                                        transform=transforms.ToTensor())
    testset = datasets.CIFAR10(root="./data", train=False, download=True,
                                       transform=transforms.ToTensor())
    # get single target
    target_img, target_label = testset[target_img_idx]

    # get multiple bases
    base_imgs = torch.stack([trainset[i][0] for i in base_indices])
    base_labels = torch.LongTensor([trainset[i][1] for i in base_indices])

    ###
    # craft poisons here with the above inputs
    ###

    # save poisons, labels, and target
    
    # poison_tuples should be a list of tuples with two entries each (img, label), example:
    # [(poison_0, label_0), (poison_1, label_1), ...]
    # where poison_0, poison_1 etc are PIL images (so they can be loaded like the CIFAR10 data in pytorch)
    with open("poisons.pickle", "wb") as handle:
        pickle.dump(poison_tuples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # base_indices should be a list of indices witin the CIFAR10 data of the bases, this is used for testing for clean-lable
    # i.e. that the poisons are within the l-inf ball of radius 8/255 from their respective bases
    with open("base_indices.pickle", "wb") as handle:
        pickle.dump(base_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # For triggerless attacks use this
    with open("target.pickle", "wb") as handle:
        pickle.dump((transforms.ToPILImage()(target_img), target_label), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # For triggered backdoor attacks use this where patch is a 3x5x5 tensor conataing the patch 
    # and [startx, starty] is the location of the top left pixel of patch in the pathed target 
    with open("target.pickle", "wb") as handle:
        pickle.dump((transforms.ToPILImage()(target_img), target_label, patch, [startx, starty]), handle, 
                    protocol=pickle.HIGHEST_PROTOCOL)
```

With TensorFlow:
```
    import numpy as np
    import tensorflow as tf
    import pickle
    from PIL import Image

    with open("poison_setups/cifar10_transfer_learning.pickle", "rb") as handle:
        setup_dicts = pickle.load(handle)

    # Which set up to do in this run?
    setup = setup_dicts[0]    # this can be changed to choose a different setup

    # get set up for this trial
    target_class = setup["target class"]
    target_img_idx = np.array(setup["target index"])
    poisoned_label = setup["base class"]
    base_indices = np.array(setup["base indices"])
    num_poisons = len(base_indices)


    # load the CIFAR10 datasets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # get single target
    target_img = x_test[target_img_idx]
    target_label = y_test[target_img_idx]
    assert target_label == target_class

    # get multiple bases
    base_imgs = x_train[base_indices]
    base_labels = y_train[base_indices]

    ###
    # craft poisons here with the above inputs
    # if the output is a numpy array poison_array which has shape (num_poisons, 32, 32, 3) and values of integers from 0 to 255
    poison_array = craft_poisons(args)  # crafting function here is a stand-in and will not work
    ###

    # save poisons, labels, and target
    poison_tuples = []
    for i in range(num_poisons):
        poison_tuples.append(Image.fromarray(poison_array[0]), poisoned_label)
    
    # poison_tuples should be a list of tuples with two entries each (img, label), example:
    # [(poison_0, label_0), (poison_1, label_1), ...]
    # where poison_0, poison_1 etc are PIL images (so they can be loaded like the CIFAR10 data in pytorch)
    with open("poisons.pickle", "wb") as handle:
        pickle.dump(poison_tuples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # base_indices should be a list of indices within the CIFAR10 data of the bases, this is used for testing for clean-lable
    # i.e. that the poisons are within the l-inf ball of radius 8/255 from their respective bases
    with open("base_indices.pickle", "wb") as handle:
        pickle.dump(base_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # For triggerless attacks use this
    with open("target.pickle", "wb") as handle:
        pickle.dump((Image.fromarray(target_img), target_label), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # For triggered backdoor attacks use this where patch is a 3x5x5 numpy array conataing the patch 
    # and [startx, starty] is the location of the top left pixel of patch in the pathed target 
    with open("target.pickle", "wb") as handle:
        pickle.dump((Image.fromarray(target_img), target_label, patch, [startx, starty]), handle, 
                    protocol=pickle.HIGHEST_PROTOCOL)
```