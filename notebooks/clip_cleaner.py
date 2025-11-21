# use_clipcleaner.py
# Usage: adapt the ImageFolderDataset example to your data, then run the script.
# This script is modeled on the combined_selection logic in clipcleaner.py,
# but works with any dataset object that provides the small CLIPCleaner-like interface.

import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from types import SimpleNamespace
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import clip
import shutil

# ---------- helper utilities (same logic used in clipcleaner.py) ----------

def get_score(prediction, labels, mode='celoss'):
    num_classes = len(np.unique(labels))
    if mode == 'celoss':
        loss = prediction.log()
        score = torch.gather(loss, 1, labels.view(-1, 1)).squeeze()
    elif mode == 'perclass_celoss':
        loss = prediction.log()
        score = torch.gather(loss, 1, labels.view(-1, 1)).squeeze()
        id_by_label = [np.where(labels == i)[0] for i in range(num_classes)]
        for id in id_by_label:
            # normalize within-class
            sc = score[id]
            denom = (sc.max() - sc.min())
            if denom == 0:
                score[id] = 0.0
            else:
                score[id] = (sc - sc.min()) / denom
    else:  # 'consistency'
        vote_y = torch.gather(prediction, 1, labels.view(-1, 1)).squeeze()
        vote_max = prediction.max(dim=1)[0]
        score = vote_y / (vote_max + 1e-12)
    return score

def logistic_regression(features, labels):
    # CPU scikit-learn logistic regression
    classifier = LogisticRegression(random_state=0, max_iter=10000, class_weight='balanced').fit(features.cpu(), labels)
    prediction = torch.tensor(classifier.predict_proba(features.cpu()))
    return prediction

# ---------- core cleaning function (works with a dataset object) ----------

def run_clip_cleaner(dataset,
                     model_name='ViT-L/14@336px',
                     theta_gmm=0.5,
                     theta_cons=0.7,
                     batch_size=1024,
                     device='cuda'):
    """
    dataset: PyTorch Dataset with attributes described above.
    model_name: CLIP model identifier (same strings used by clip.load)
    theta_gmm: min posterior probability threshold from the GMM component
    theta_cons: consistency threshold (used for 'consistency' scores)
    """
    # load CLIP
    clip_model, preprocess = clip.load(model_name, device=device)
    clip_model = clip_model.eval()

    num_classes = dataset.num_classes
    class_names = dataset.class_names
    suffix = dataset.suffix
    all_labels = torch.tensor(dataset.label, dtype=torch.long)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # compute image features
    all_features_list = []
    with torch.no_grad():
        for (imgs, labels, indices) in loader:
            imgs = imgs.to(device)
            feats = clip_model.encode_image(imgs).float()
            all_features_list.append(feats.cpu())
    all_features = torch.cat(all_features_list, dim=0)
    all_features = F.normalize(all_features, dim=1)
    # all_features on CPU

    # build zero-shot text features for each class using dataset.detailed_features
    # expected: dataset.detailed_features is list-of-list: for class i, a list of extra text tokens (could be ["", "close up"])
    text_features_by_class = []
    with torch.no_grad():
        for i in range(num_classes):
            # class name can be str or list of synonyms
            cls_name = class_names[i]
            prompts = []
            # detailed_features could be empty list or a list of prompt fragments
            # If detailed_features[i] is [] treat as single empty fragment
            frags = dataset.detailed_features[i] if len(dataset.detailed_features) > 0 else [""]
            if isinstance(cls_name, list):
                base_names = cls_name
            else:
                base_names = [cls_name]
            for base in base_names:
                for frag in frags:
                    prompt = f"A photo of a surgical instrument with the damage class: {base.lower()} that is described as the following: {frag}" + (suffix or "")
                    prompts.append(prompt)
            # tokenize and encode all prompts for this class
            tokens = clip.tokenize(prompts).to(device)
            tfeats = clip_model.encode_text(tokens).float().cpu()
            tfeats = F.normalize(tfeats, dim=1)
            text_features_by_class.append(tfeats)  # (num_prompts_for_class x d)

    # compute similarity: for each sample, get per-class aggregated score (sum of exp(sim / T) across prompts)
    T = 0.07
    sim = torch.zeros(len(all_features), num_classes)
    all_features_device = all_features.to(device)
    with torch.no_grad():
        for i in range(num_classes):
            tfeats = text_features_by_class[i].to(device)  # P x d
            # similarity: N x P = all_features (Nxd) @ tfeats.T (d x P)
            sim_np = (all_features_device @ tfeats.t())  # N x P
            sim[:, i] = torch.exp(sim_np / T).sum(dim=1).cpu()
    # convert to probabilities across classes
    prediction_zero = (sim / sim.sum(dim=1, keepdim=True)).cpu()

    # visual logistic regression predictions
    prediction_lr = logistic_regression(all_features, all_labels)

    # compute selection scores
    CLIP_similarityprob_loss = get_score(prediction_zero, all_labels, 'perclass_celoss')
    CLIP_similarityprob_consistency = get_score(prediction_zero, all_labels, 'consistency')
    CLIP_visuallr_loss = get_score(prediction_lr, all_labels, 'perclass_celoss')
    CLIP_visuallr_consistency = get_score(prediction_lr, all_labels, 'consistency')

    method_score = [
        CLIP_similarityprob_loss.numpy(), CLIP_similarityprob_consistency.numpy(),
        CLIP_visuallr_loss.numpy(), CLIP_visuallr_consistency.numpy()
    ]
    method_name = [
        'zeroshot_perclassgmm', 'zeroshot_consistency',
        'visuallr_perclassgmm', 'visuallr_consistency'
    ]
    types = ['loss', 'consistency', 'loss', 'consistency']

    labels_np = all_labels.numpy()
    id_by_label = [np.where(labels_np == i)[0] for i in range(num_classes)]

    select_per_method = []
    for i, score in enumerate(method_score):
        clean_id_all = []
        for k in range(num_classes):
            ids = id_by_label[k]
            if len(ids) == 0:
                continue
            if types[i] == 'loss':
                gmm = GaussianMixture(n_components=2, random_state=0)
                gmm.fit(score[ids].reshape(-1, 1))
                # choose the component with larger mean as "clean" (higher prob)
                comp_idx = gmm.means_.argmax()
                prob = gmm.predict_proba(score[ids].reshape(-1, 1))[:, comp_idx]
                sel = ids[np.where(prob >= theta_gmm)[0]]
            else:
                sel = ids[np.where(score[ids] >= theta_cons)[0]]
            clean_id_all.append(sel)
        if len(clean_id_all) > 0:
            clean_id_all = np.concatenate(clean_id_all)
        else:
            clean_id_all = np.array([], dtype=int)
        per_class_counts = np.array([np.sum(labels_np[clean_id_all] == c) for c in range(num_classes)])
        print(f"Method {method_name[i]} per-class counts: {per_class_counts}")
        select_per_method.append(clean_id_all)

    # take intersection across methods
    if len(select_per_method) == 0:
        selected = np.array([], dtype=int)
    else:
        selected = select_per_method[0]
        for s in select_per_method[1:]:
            selected = np.intersect1d(selected, s)

    # ensure no class is empty: fill from zero-shot perclass list if necessary
    per_class_selected = np.array([np.sum(labels_np[selected] == i) for i in range(num_classes)])
    print("Selected per-class before fill:", per_class_selected)
    num_smallest = int(len(labels_np) / num_classes / num_classes)
    zero_class = np.where(per_class_selected == 0)[0]
    if len(zero_class) != 0:
        non_zero = np.where(per_class_selected != 0)[0]
        if len(non_zero) > 0:
            num_smallest = max(int(len(labels_np) / num_classes / num_classes), per_class_selected[non_zero].min())
        all_by_class = [np.where(labels_np == i)[0] for i in range(num_classes)]
        # use the zeroshot perclass loss scoring (method_score[0]) to pick best samples for zero-count classes
        for clx in zero_class:
            clx_scores = method_score[0][all_by_class[clx]]
            # want largest scores (higher probability means cleaner after normalization)
            rank = clx_scores.argsort()[::-1]  # descending
            to_take = num_smallest
            selected = np.concatenate([selected, all_by_class[clx][rank[:to_take]]])
    selected = np.unique(selected)
    per_class_selected_after = np.array([np.sum(labels_np[selected] == i) for i in range(num_classes)])
    print("Selected per-class final:", per_class_selected_after)

    return selected, labels_np[selected]

# ---------- Example minimal Dataset (adapt to your files) ----------

class ImageFolderDataset(Dataset):
    """
    Minimal dataset wrapper: given a list of (path, label). Images are preprocessed using CLIP preprocess.
    It sets attributes required by the cleaner:
      - num_classes
      - detailed_features: list of lists (one list per class) of additional prompt fragments (can be [""])
      - class_names: list of class names (strings)
      - suffix: extra string appended to each prompt (e.g. "")
      - label: numpy array of labels aligned with file list order
    """
    def __init__(self, items, class_names, detailed_features=None, suffix="", preprocess=None):
        # items: list of (path, label)
        self.items = items
        self.class_names = class_names
        self.num_classes = len(class_names)
        if detailed_features is None:
            self.detailed_features = [[""] for _ in range(self.num_classes)]
        else:
            self.detailed_features = detailed_features
        self.suffix = suffix
        self.label = np.array([lbl for _, lbl in items], dtype=int)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, lbl = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.preprocess is not None:
            img = self.preprocess(img)
        return img, lbl, idx

# ---------- Example usage ----------

if __name__ == "__main__":
    data_path = "../data/surface/train"
    
    base_dataset = ImageFolder(data_path)
    
    # Example: adapt these lists to your dataset (image paths and noisy labels)
    # items = [("data/img1.jpg", 0), ("data/img2.jpg", 1), ...]
    # items = []  # fill with your image paths and (possibly noisy) labels
    # optional per-class prompt fragments (can be extra phrases, empty string is fine)
    items = base_dataset.imgs
    class_names = base_dataset.classes
    
    detailed_features = [
        ["A deformation where part of the surgical instrument is visibly curved, twisted, or out of its intended straight alignment."], 
        ["Dark or black discoloration on the instrument's surface, typically caused by burnt residue, oxidation, or contamination."], 
        ["A visible fracture or split in the material of the instrument, ranging from small hairline cracks to clearly open breaks."],
        ["A demonstration or example of a gap defect, where two parts that should be tightly joined have an unintended visible space between them."],
        ["A defect where the tip or end portion of the instrument is broken off, incomplete, or entirely missing."],
        ["The surgical instrument shows no visible defects; surfaces are clean, intact, and in proper condition."],
        ["Brown or reddish surface corrosion, often appearing as spots, patches, or roughened areas caused by oxidation."],
        ["Fine or deep linear abrasions on the instrument's surface, caused by wear, handling, or mechanical damage."],
        ["White, cloudy, or chalk-like deposits on the metal surface, commonly originating from water residue or cleaning agents containing silicates."],
        ["Stains or discoloration present on the instrument that cannot be clearly associated with a known source or contamination type."],
        ["An undefined or ambiguous defect category used when the type of issue cannot be reliably classified."],
        ["Light-colored, often irregular or spot-like residue patterns left by dried water, typically after inadequate drying or mineral-rich water exposure."]
    ]
    args = SimpleNamespace(model='ViT-L/14@336px', theta_gmm=0.5, theta_cons=0.7)
    
    # load clip preprocess to pass into dataset
    print("Loading model....")
    clip_model, preprocess = clip.load(args.model, device="cpu")
    dataset = ImageFolderDataset(items, class_names, detailed_features=detailed_features, suffix="", preprocess=preprocess)
    selected_idx, selected_labels = run_clip_cleaner(dataset,
                                                      model_name=args.model,
                                                      theta_gmm=args.theta_gmm,
                                                      theta_cons=args.theta_cons,
                                                      batch_size=64,
                                                      device='cuda' if torch.cuda.is_available() else 'cpu')

    print("Number selected:", len(selected_idx))
    print("Selected labels:", selected_labels)
    
    
    clean_base = os.path.join(os.path.dirname(data_path), "clean")
    os.makedirs(clean_base, exist_ok=True)

    copied = 0
    for idx in selected_idx:
        idx = int(idx)
        src, lbl = items[idx]
        cls = class_names[lbl]
        dest_dir = os.path.join(clean_base, cls)
        os.makedirs(dest_dir, exist_ok=True)
        fname = os.path.basename(src)
        dest_path = os.path.join(dest_dir, f"{idx}_{fname}")
        try:
            shutil.copy2(src, dest_path)
            copied += 1
        except Exception as e:
            print(f"Failed to copy {src} -> {dest_path}: {e}")

    print(f"Copied {copied} files to {clean_base}")