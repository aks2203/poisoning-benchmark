from models import *
from PIL import Image, ExifTags
import os


def load_pretrained_net(net_name, chk_name, model_chk_path, test_dp=0, bdp=0, device='cuda'):
    """
    Load the pre-trained models. CUDA only :)
    """
    net = eval(net_name)(test_dp=test_dp, bdp=bdp)
    if device == 'cuda':
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = torch.nn.DataParallel(net)
    net.eval()
    print('==> Resuming from checkpoint for %s..' % net_name)
    if device == 'cuda':
        checkpoint = torch.load('./{}/{}'.format(model_chk_path, chk_name) % net_name)
    else:
        checkpoint = torch.load('./{}/{}'.format(model_chk_path, chk_name) % net_name,
                                map_location=lambda storage, loc: storage)
    if 'module' not in list(checkpoint['net'].keys())[0]:
        # to be compatible with DataParallel
        net.module.load_state_dict(checkpoint['net'])
    else:
        net.load_state_dict(checkpoint['net'])

    return net


def crop_resize_opencv(path):
    try:
        img = Image.open(path)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(img._getexif().items())

        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass

    w, h = img.size

    img = img.crop((0, 0, min([w, h]), min([w, h])))
    w, h = img.size

    if w > h:
        l = (w - h) / 2
        img = img.crop((l, 0, w - l, h))
    w, h = img.size
    assert w == h, "size error, {}, {}!".format(w, h)

    imgnp = np.asarray(img)
    img.close()
    imgcv2 = cv2.resize(imgnp, (32, 32), interpolation=cv2.INTER_AREA)

    return imgcv2


def fetch_camera_target(target_index, target_print_size, transforms):
    import cv2
    path = "targets/camera/printed-size-{}/target{}.png".format(target_print_size, "%.3d" % target_index)
    if os.path.exists(path):
        img = crop_resize_opencv(path)
        img = img[:, :, :3]  # remove the alpha/transparency channel
        if transforms is not None:
            img = transforms(img)[None, :, :, :]
        else:
            img = np.array(img)[None, :, :, :]
        return img
    else:
        print("camera image for target {} does not exist".format(target_index))
        print(path)
        # assert False
        return None


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
    raise Exception("Target with index {} exceeds number of total samples (should be less than {})".format(
        target_index, len(img_label_list) / 10 - start_idx))


def fetch_all_target_cls(target_label, num_per_class_transfer, subset, path, transforms):
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


def get_target_nearest_neighbor(subs_net_list, target_cls_imgs, target_img, num_imgs, idx_list, device='cuda'):
    target_img = target_img.to(device)
    target_cls_imgs = target_cls_imgs.to(device)
    total_dists = 0
    with torch.no_grad():
        for n_net, net in enumerate(subs_net_list):
            target_img_feat = net.module.penultimate(target_img)
            target_cls_imgs_feat = net.module.penultimate(target_cls_imgs)
            dists = torch.sum((target_img_feat - target_cls_imgs_feat) ** 2, 1).cpu().detach().numpy()
            total_dists += dists
    min_dist_idxes = np.argsort(total_dists)
    # print("Selected feature dist squares: {}".format(total_dists[min_dist_idxes[:num_imgs]]))
    return target_cls_imgs[min_dist_idxes[:num_imgs]], [idx_list[midx] for midx in min_dist_idxes[:num_imgs]]


def fetch_nearest_poison_bases(sub_net_list, target_img, num_poison, poison_label, num_per_class, subset,
                               train_data_path, transforms):
    imgs, idxes = fetch_all_target_cls(poison_label, num_per_class, subset, train_data_path, transforms)

    nn_imgs_batch, nn_idx_list = get_target_nearest_neighbor(sub_net_list, imgs, target_img, num_poison,
                                                             idxes, device='cuda')
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
    poison_tuple = [(poison_batch.poison.data[num_p].detach().cpu(), poison_label) for num_p in
                    range(poison_batch.poison.size(0))]

    return poison_tuple


def get_poison_list(poison_batch):
    """
    Doesn't have the labels
    """
    poison_tuple = [(poison_batch.poison.data[num_p].detach().clone()) for num_p in range(poison_batch.poison.size(0))]

    return poison_tuple
