import torch

def collate_keep_images_as_list(batch):
    # batch items are (img,label) or (img,label,idx)
    if len(batch[0]) == 3:
        imgs, labels, idxs = zip(*batch)
        return list(imgs), torch.as_tensor(labels), torch.as_tensor(idxs)
    imgs, labels = zip(*batch)
    return list(imgs), torch.as_tensor(labels)