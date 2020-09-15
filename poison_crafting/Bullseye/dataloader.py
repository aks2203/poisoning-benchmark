import torch
import torch.utils.data as data


class PoisonedDataset(data.Dataset):
    def __init__(self, path, subset='clean_train', transform=None, num_per_label=-1, poison_tuple_list=[], poison_indices=[],
                 class_labels=[i for i in range(10)], subset_group=0):
        """
        Made to be compatible with specifying class labels with class_labels
        """
        self.img_label_list = torch.load(path)[subset]
        self.transform = transform
        self.poison_indices = poison_indices
        self.poison_tuple_list = poison_tuple_list
        self.get_valid_indices(num_per_label, poison_indices, class_labels, subset_group)

    def get_valid_indices(self, num_per_label, poison_indices, class_labels, subset_group):
        # remove poisoned ones
        num_per_label_dict = {} # recording number of samples for each label
        idx_cursors = {l: 0 for l in class_labels}
        # put the poisons into the dataset
        for pidx in poison_indices:
            # count the poisoned class
            img, label = self.img_label_list[pidx]
            if label not in class_labels:
                continue
            # reduce the number of poison classes
            if label not in num_per_label_dict:
                num_per_label_dict[label] = 0
            num_per_label_dict[label] += 1

        # put the rest into the dataset
        if num_per_label > 0:
            self.valid_indices = []
            start_idx = subset_group * num_per_label
            end_idx = (subset_group + 1) * num_per_label
            for idx, (img, label) in enumerate(self.img_label_list):
                if label not in class_labels:
                    continue
                idx_cursors[label] += 1
                if idx in poison_indices:
                    continue
                if label not in num_per_label_dict:
                    num_per_label_dict[label] = 0
                if num_per_label_dict[label] < num_per_label and idx_cursors[label] > start_idx and idx_cursors[label] <= end_idx:
                    self.valid_indices.append(idx)
                    num_per_label_dict[label] += 1

        else:
            # Otherwise, use the whole clean set by default
            self.valid_indices = [i for i in range(len(self.img_label_list))]

    def __len__(self):
        return len(self.valid_indices) + len(self.poison_tuple_list)

    def __getitem__(self, index):
        if index < len(self.poison_tuple_list):
            img, label = self.poison_tuple_list[index]
        else:
            idx = self.valid_indices[index - len(self.poison_tuple_list)]
            img, label = self.img_label_list[idx]
            if self.transform is not None:
                img = self.transform(img)

        return img, label


class FeatureSet(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(self, train_loader, net, device):
        # Extract the features
        feat_list, label_list = [], []
        with torch.no_grad():
            for ite, (input, target) in enumerate(train_loader):
                if device == 'cuda':
                    input, target = input.to('cuda'), target.to('cuda')

                feat = net.module.penultimate(input).detach()

                feat_list.append(feat)
                label_list.append(target)

        self.feature_tensor = torch.cat(feat_list, 0)
        self.label_tensor = torch.cat(label_list, 0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        feature, target = self.feature_tensor[index], self.label_tensor[index]

        return feature, target

    def __len__(self):
        return self.feature_tensor.size(0)


class SubsetOfList(data.Dataset):
    def __init__(self, img_label_list, transform=None, start_idx=0, end_idx=1e10,
                    poison_tuple_list=[],
                    class_labels=[i for i in range(10)]):
        # logistics work for mixing data from CINIC10 with CIFAR10
        # suppose the poison samples are disjoint with the CINIC dataset
        self.img_label_list = img_label_list #torch.load(path)[subset]
        self.transform = transform
        self.poison_tuple_list = poison_tuple_list
        self.get_valid_list(start_idx, end_idx, class_labels)

    def get_valid_list(self, start_idx, end_idx, class_labels):
        # remove poisoned ones
        num_per_label_dict = {}
        selected_img_label_list = [] #[pt for pt in self.poison_tuple_list]
        if len(self.poison_tuple_list) > 0:
            poison_label = self.poison_tuple_list[0][1]
            print("Poison label: {}".format(poison_label))
        else:
            poison_label = -1

        for idx, (img, label) in enumerate(self.img_label_list):
            if label not in class_labels:
                continue
            if label not in num_per_label_dict:
                num_per_label_dict[label] = 0
            if num_per_label_dict[label] >= start_idx and num_per_label_dict[label] < end_idx:
                if label == poison_label and num_per_label_dict[label] - start_idx < len(self.poison_tuple_list):
                    pass
                else:
                    selected_img_label_list.append([img, label])
            num_per_label_dict[label] += 1

        self.img_label_list = selected_img_label_list


    def __len__(self):
        return len(self.img_label_list) + len(self.poison_tuple_list)

    def __getitem__(self, index):
        if index < len(self.poison_tuple_list):
            img, label = self.poison_tuple_list[index]
        else:
            img, label = self.img_label_list[index-len(self.poison_tuple_list)]
            if self.transform is not None:
                img = self.transform(img)

        return img, label
