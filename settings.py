post_hoc_methods = ["MaRC", "Occlusion", "Saliency_L2", "Saliency_Sum", "InputXGrad_L2", "InputXGrad_Sum",
                          "LIME", "Integrated_Gradients_L2", "Integrated_Gradients_Sum", "Shapley"]
weakly_supervised_methods = ["Brinner_2024", "Lei_2016", "Yu_2019"]

model_checkpoints = {"Bio": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                    "Movies": "bert-base-uncased"}

output_fns = {"Bio": "sigmoid", "Movies": "softmax"}

num_labels = {"Bio": 10, "Movies": 2}

class Config:
    save_name = None
    dataset_type = None
    num_class_labels = None
    model_checkpoint = None
    output_fn = None

def set_save_name(save_name):
    Config.save_name = save_name

def set_dataset_type(dataset_type):
    Config.dataset_type = dataset_type
    Config.num_class_labels = num_labels[dataset_type]
    Config.model_checkpoint = model_checkpoints[dataset_type]
    Config.output_fn = output_fns[dataset_type]