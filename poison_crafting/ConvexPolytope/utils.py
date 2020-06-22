from models import *


def load_pretrained_net(net_name, chk_name, model_chk_path, test_dp=0):
    """
    Load the pre-trained models. CUDA only :)
    """
    net = eval(net_name)(test_dp=test_dp)
    net = torch.nn.DataParallel(net).cuda()
    net.eval()
    print("==> Resuming from checkpoint for %s.." % net_name)
    checkpoint = torch.load("./{}/{}".format(model_chk_path, chk_name) % net_name)
    if "module" not in list(checkpoint["net"].keys())[0]:
        # to be compatible with DataParallel
        net.module.load_state_dict(checkpoint["net"])
    else:
        net.load_state_dict(checkpoint["net"])

    return net


def fetch_target(target_label, target_index, start_idx, path, subset, transforms):
    """
    Fetch the "target_index"-th target, counting starts from start_idx
    """
    img_label_list = torch.load(path)[subset]
    counter = 0
    for idx, (img, label) in enumerate(img_label_list):
        if label == target_label:
            counter += 1
            if counter == (target_index + start_idx + 1):
                if transforms is not None:
                    return transforms(img)[None, :, :, :]
                else:
                    return np.array(img)[None, :, :, :]
    raise Exception(
        "Target with index {} exceeds number of total samples (should be less than {})".format(
            target_index, len(img_label_list) / 10 - start_idx
        )
    )


def fetch_all_target_cls(
    target_label, num_per_class_transfer, subset, path, transforms
):
    img_label_list = torch.load(path)[subset]
    counter = 0
    targetcls_img_list = []
    idx_list = []
    for idx, (img, label) in enumerate(img_label_list):
        if label == target_label:
            counter += 1
            if counter <= num_per_class_transfer:
                targetcls_img_list.append(transforms(img))
                idx_list.append(idx)
    return torch.stack(targetcls_img_list), idx_list


def get_target_nearest_neighbor(
    subs_net_list, target_cls_imgs, target_img, num_imgs, idx_list, device="cuda"
):
    target_img = target_img.to(device)
    target_cls_imgs = target_cls_imgs.to(device)
    total_dists = 0
    with torch.no_grad():
        for n_net, net in enumerate(subs_net_list):
            target_img_feat = net.module.penultimate(target_img)
            target_cls_imgs_feat = net.module.penultimate(target_cls_imgs)
            dists = (
                torch.sum((target_img_feat - target_cls_imgs_feat) ** 2, 1)
                .cpu()
                .detach()
                .numpy()
            )
            total_dists += dists
    min_dist_idxes = np.argsort(total_dists)
    # print("Selected feature dist squares: {}".format(total_dists[min_dist_idxes[:num_imgs]]))
    return (
        target_cls_imgs[min_dist_idxes[:num_imgs]],
        [idx_list[midx] for midx in min_dist_idxes[:num_imgs]],
    )


def fetch_nearest_poison_bases(
    sub_net_list,
    target_img,
    num_poison,
    poison_label,
    num_per_class,
    subset,
    train_data_path,
    transforms,
):
    imgs, idxes = fetch_all_target_cls(
        poison_label, num_per_class, subset, train_data_path, transforms
    )

    nn_imgs_batch, nn_idx_list = get_target_nearest_neighbor(
        sub_net_list, imgs, target_img, num_poison, idxes, device="cuda"
    )
    base_tensor_list = [nn_imgs_batch[n] for n in range(nn_imgs_batch.size(0))]
    print("Selected nearest neighbors: {}".format(nn_idx_list))
    return base_tensor_list, nn_idx_list


def fetch_poison_bases(poison_label, num_poison, subset, path, transforms):
    """
    Only going to fetch the first num_poison image as the base class from the poison_label class
    """
    img_label_list = torch.load(path)[subset]
    base_tensor_list, base_idx_list = [], []
    for idx, (img, label) in enumerate(img_label_list):
        if label == poison_label:
            base_tensor_list.append(transforms(img))
            base_idx_list.append(idx)
        if len(base_tensor_list) == num_poison:
            break
    return base_tensor_list, base_idx_list


def get_poison_tuples(poison_batch, poison_label):
    """
    Includes the labels
    """
    poison_tuple = [
        (poison_batch.poison.data[num_p].detach().cpu(), poison_label)
        for num_p in range(poison_batch.poison.size(0))
    ]

    return poison_tuple


def get_poison_list(poison_batch):
    """
    Doesn't have the labels
    """
    poison_tuple = [
        (poison_batch.poison.data[num_p].detach().clone())
        for num_p in range(poison_batch.poison.size(0))
    ]

    return poison_tuple
