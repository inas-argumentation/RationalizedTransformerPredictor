import json
import numpy as np
import os
from settings import Config
from torch.utils.data import Dataset
from data import process_Bio_annotations, process_Movies_annotations

def get_data_folder(dataset_type=None):
    if dataset_type is None:
        return os.path.join(os.path.dirname(__file__), Config.dataset_type)
    else:
        return os.path.join(os.path.dirname(__file__), dataset_type)

def load_Bio_annotations():
    with open(os.path.join(get_data_folder("Bio"), "annotation/Second Annotation Project.json"), "r") as f:
        annotations = json.loads(f.read())
    return annotations

def load_bio_dataset(folder=os.path.join(get_data_folder("Bio"), "abstracts_new")):
    result = {}
    files = os.listdir(folder)
    for f in files:
        if f == ".gitkeep":
            continue
        with open(os.path.join(folder, f), "r", encoding="latin-1") as file:
            title, abstract, sub_labels = file.read().split("\n")
            sub_labels = sub_labels.split(",")
            labels = set([int(x[0]) for x in sub_labels])
            index = int(f[:-4])
            result[index] = {"title": title, "abstract": abstract, "labels": labels, "index": index, "sub_labels": sub_labels}
    return result

def load_bio_train_val_test_split_indices():
    folder = get_data_folder("Bio")
    test_indices = [int(x) for x in open(os.path.join(folder, "test_set_indices.txt"), "r").read().split(",")]
    val_indices = [int(x) for x in open(os.path.join(folder, "val_set_indices.txt"), "r").read().split(",")]
    train_indices = [int(x) for x in open(os.path.join(folder, "train_set_indices.txt"), "r").read().split(",")]
    return train_indices, val_indices, test_indices

def one_hot_encode(labels):
    vector = np.zeros(Config.num_class_labels)
    for l in labels:
        vector[l] = 1
    return vector

class BioDataset(Dataset):
    def __init__(self):
        train_indices, val_indices, test_indices = load_bio_train_val_test_split_indices()

        self.indices = {"train": train_indices,
                        "val": val_indices,
                        "test": test_indices}

        self.samples = load_bio_dataset()
        self.split = "train"

        for k in self.samples.keys():
            self.samples[k]["prediction_text"] = f"{self.samples[k]['title']}. {self.samples[k]['abstract']}"
            self.samples[k]["one_hot_label"] = one_hot_encode(self.samples[k]["labels"])

        self.annotated_indices = [int(x[-7:-4]) for x in os.listdir(os.path.join(get_data_folder("Bio"), "annotation/Second Annotation Project"))]

    def __len__(self):
        return len(self.indices[self.split])

    def __getitem__(self, idx):
        sample = self.samples[self.indices[self.split][idx]]
        return sample["text"], sample["label"]

    def get_full_sample(self, idx):
        return self.samples[idx]

    def set_split(self, split):
        self.split = split

def load_movie_data_for_JSONL_file(file):
    folder = get_data_folder("Movies")
    with open(os.path.join(folder, f"{file}"), 'r') as f:
        data = list(f)

    data_dict = {}
    i = 0
    for d in data:
        parsed = json.loads(d)
        with open(os.path.join(folder, f"docs/{parsed['annotation_id']}"), 'r') as f:
            parsed["lines"] = f.read().split("\n")
            parsed["text"] = " ".join(parsed["lines"])
        parsed["label"] = 1 if parsed["classification"] == 'POS' else 0
        data_dict[i] = parsed
        i += 1
    return data_dict

class MovieDataset(Dataset):
    def __init__(self):

        train_data = load_movie_data_for_JSONL_file("train.jsonl")
        val_data = load_movie_data_for_JSONL_file("val.jsonl")
        test_data = load_movie_data_for_JSONL_file("test.jsonl")

        self.samples = {}
        for x in list(train_data.values()) + list(val_data.values()) + list(test_data.values()):
            idx = int(x["annotation_id"][5:8]) * 2 + (1 if x["annotation_id"][0] == "n" else 0)
            self.samples[idx] = {
                "labels": {x["label"]},
                "index": idx,
                "prediction_text": x["text"],
                "one_hot_label": one_hot_encode([x["label"]]),
                "evidences": x["evidences"]}

        train_indices = list(range(1600))
        val_indices = list(range(1600, 1800))
        test_indices = [x for x in range(1800, 2000) if x in self.samples]

        self.indices = {"train": train_indices,
            "val": val_indices,
            "test": test_indices}

        self.split = None
        self.set_split("train")

    def set_split(self, split):
        self.split = split

    def __len__(self):
        return len(self.indices[self.split])

    def __getitem__(self, idx):
        sample = self.samples[self.indices[self.split][idx]]
        return sample["text"], sample["label"]

    def get_full_sample(self, idx):
        return self.samples[idx]

def load_dataset():
    if Config.dataset_type == "Bio":
        return BioDataset()
    elif Config.dataset_type == "Movies":
        return MovieDataset()

def create_gt_array_from_annotations(gt_annotations):
    gt_arrays = {}
    for idx, data in gt_annotations.items():
        if len(data) == 0:
            continue
        num_tokens = sum([x["n_tokens"] for x in list(data.values())[0]])
        arrays = []
        for label in range(Config.num_class_labels):
            if label in data:
                label_arrays = []
                for word in data[label]:
                    if len(word["annotation"]) > 0:
                        for i in range(word["n_tokens"]):
                            label_arrays.append([0, 1])
                    else:
                        for i in range(word["n_tokens"]):
                            label_arrays.append([1, 0])
                arrays.append(np.array(label_arrays))
            else:
                arrays.append(np.array([1, 0]).reshape(1, -1).repeat(num_tokens, 0))
        gt_arrays[idx] = np.stack(arrays, axis=1)
    return gt_arrays

def load_annotations_and_ground_truth(tokenizer, dataset):
    if Config.dataset_type == "Bio":
        # Load raw annotations from file
        annotations = load_Bio_annotations()
        # Match annotations to tokens created by tokenizer
        annotations = process_Bio_annotations.create_gt_annotations(annotations, tokenizer, dataset)
    elif Config.dataset_type == "Movies":
        annotations = process_Movies_annotations.create_gt_annotations(tokenizer, dataset)
    # Create ground-truth array
    ground_truth_array = create_gt_array_from_annotations(annotations)
    return annotations, ground_truth_array
