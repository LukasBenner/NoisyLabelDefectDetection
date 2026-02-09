from torchvision.datasets import ImageFolder
from typing import Dict, List, Optional, Sequence


def filter_classes(
    dataset: ImageFolder,
    classes_to_keep: Optional[Sequence[str]],
    allow_missing: bool = False,
) -> ImageFolder:
    """
    Filter the given ImageFolder dataset to only include samples from the specified classes.

    Args:
        dataset (ImageFolder): The original ImageFolder dataset.
        classes_to_keep (Optional[Sequence[str]]): A list of class names to keep in the dataset.

    Returns:
        ImageFolder: A new ImageFolder dataset containing only the specified classes.
    """
    if not classes_to_keep:
        return dataset

    missing = [name for name in classes_to_keep if name not in dataset.class_to_idx]
    if missing and not allow_missing:
        raise ValueError(f"Classes not found in dataset: {missing}")

    allowed = {dataset.class_to_idx[name] for name in classes_to_keep if name in dataset.class_to_idx}
    new_classes = list(classes_to_keep)
    new_class_to_idx = {name: idx for idx, name in enumerate(new_classes)}

    new_samples = []
    new_targets = []
    for path, target in dataset.samples:
        if target in allowed:
            class_name = dataset.classes[target]
            new_target = new_class_to_idx[class_name]
            new_samples.append((path, new_target))
            new_targets.append(new_target)

    if len(new_samples) == 0:
        raise ValueError("No samples left after class filtering.")

    dataset.samples = new_samples
    dataset.targets = new_targets
    dataset.imgs = new_samples
    dataset.classes = new_classes
    dataset.class_to_idx = new_class_to_idx
    return dataset


def merge_classes(
    dataset: ImageFolder,
    merge_map: Optional[Dict[str, Sequence[str]]],
    allow_missing: bool = False,
) -> ImageFolder:
    """
    Merge classes in the given ImageFolder dataset according to the provided mapping.

    Args:
        dataset (ImageFolder): The original ImageFolder dataset.
        merge_map (Optional[Dict[str, Sequence[str]]]): A dictionary where keys are target class names and values are lists of source class names to merge.

    Returns:
        ImageFolder: A new ImageFolder dataset with merged classes.
    """
    if not merge_map:
        return dataset

    reverse_map: Dict[str, str] = {}
    for target_name, source_names in merge_map.items():
        for source_name in source_names:
            if (
                source_name in reverse_map
                and reverse_map[source_name] != target_name
            ):
                raise ValueError(
                    f"Class '{source_name}' assigned to multiple merge targets."
                )
            reverse_map[source_name] = target_name

    missing = [name for name in reverse_map if name not in dataset.class_to_idx]
    if missing and not allow_missing:
        raise ValueError(f"Classes not found in dataset: {missing}")

    if allow_missing and missing:
        reverse_map = {k: v for k, v in reverse_map.items() if k in dataset.class_to_idx}

    final_classes = []
    for name in dataset.classes:
        mapped = reverse_map.get(name, name)
        if mapped not in final_classes:
            final_classes.append(mapped)

    new_class_to_idx = {name: idx for idx, name in enumerate(final_classes)}

    new_samples = []
    new_targets = []
    for path, target in dataset.samples:
        class_name = dataset.classes[target]
        mapped = reverse_map.get(class_name, class_name)
        new_target = new_class_to_idx[mapped]
        new_samples.append((path, new_target))
        new_targets.append(new_target)

    dataset.samples = new_samples
    dataset.targets = new_targets
    dataset.imgs = new_samples
    dataset.classes = final_classes
    dataset.class_to_idx = new_class_to_idx
    return dataset