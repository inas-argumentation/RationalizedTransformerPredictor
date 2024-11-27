from settings import set_save_name, set_dataset_type, post_hoc_methods
from post_hoc_methods.train_base_classifier import train_classifier, evaluate_base_classifier
from supervised_span_model.train_supervised_span_model import train_supervised_span_prediction_model
from supervised_span_model.evaluate_supervised_span_model import evaluate_supervised_span_model
from post_hoc_methods.create_rationales import create_rationales_for_all_methods
from post_hoc_methods.evaluate_rationales import evaluate_post_hoc_rationales
from L2E.train_L2E_model import train_L2E_span_prediction_model, evaluate_L2E_span_prediction_model
from weakly_supervised_models.train_weakly_supervised_model import train_weakly_supervised_model, evaluate_weakly_supervised_model
from A2R.evaluate_A2R_predictions import evaluate_A2R_rationales

training_configurations = {
    "Bio": [
        {"model_type": "Brinner_2024", "weight_decay": 1e-4, "max_num_epochs": 120},
        {"model_type": "Lei_2016", "weight_decay": 0, "max_num_epochs": 25, "weight_penalty": 0.4, "total_variation_penalty": 0.2},
        {"model_type": "Yu_2019", "weight_decay": 0, "max_num_epochs": 25, "weight_penalty": 2, "total_variation_penalty": 0.3},
        {"model_type": "Chang_2019", "weight_decay": 0, "max_num_epochs": 30, "weight_penalty": 0.6, "total_variation_penalty": 0.3}
    ],
    "Movies": [
        {"model_type": "Brinner_2024", "weight_decay": 1e-4, "max_num_epochs": 50},
        {"model_type": "Lei_2016", "weight_decay": 0, "max_num_epochs": 25, "weight_penalty": 0.4, "total_variation_penalty": 0.2},
        {"model_type": "Yu_2019", "weight_decay": 0, "max_num_epochs": 25, "weight_penalty": 1.5, "total_variation_penalty": 0.3},
        {"model_type": "Chang_2019", "weight_decay": 0, "max_num_epochs": 30, "weight_penalty": 1.1, "total_variation_penalty": 0.2}
    ]}

def train_all_models():

    # We train models for both the movie review dataset and the INAS dataset of biological abstracts.
    for dataset_type in ["Bio", "Movies"]:
        set_dataset_type(dataset_type) # Sets the dataset type globally, so that the correct dataset is loaded and models are saved correctly.

        # We train four weakly supervised models, one of them is our own proposed RTP model.
        for training_config in training_configurations[dataset_type]:
            train_weakly_supervised_model(False, **training_config)

        # We need to train a normal classifier as basis for the post-hoc explainers. We then predict and store rationales for all post-hoc methods.
        train_classifier()
        create_rationales_for_all_methods(split="test", methods=post_hoc_methods)

        # The L2E method is trained as supervised method on rationales created by a post-hoc explainer. We chose "MaRC" as base explainer, since it produces good rationales especially w.r.t. spans.
        # Rationales need to be created and stored before calling this method.
        create_rationales_for_all_methods(split="train", methods=["MaRC"])
        create_rationales_for_all_methods(split="val", methods=["MaRC"])
        train_L2E_span_prediction_model(base_explainer="MaRC")

        # We also train a supervised model on the ground-truth annotations for comparison.
        train_supervised_span_prediction_model()

def evaluate_all_methods():

    # We evaluate models for both the movie review dataset and the INAS dataset of biological abstracts.
    for dataset_type in ["Bio", "Movies"]:
        print(f"\nResults for {dataset_type} dataset:\n")
        set_dataset_type(dataset_type) # Sets the dataset type globally, so that the correct dataset is loaded and models are saved correctly.


        for training_config in training_configurations[dataset_type]:
            print(f"\nMethod: {training_config['model_type']}")
            evaluate_weakly_supervised_model(model_type=training_config["model_type"], split="test")

        # Evaluate classification performance of base classifier
        print(f"\nMethod: Standard Classifier")
        evaluate_base_classifier(split="test")

        # Evaluate post-hoc rationales
        evaluate_post_hoc_rationales(split="test")

        # Evaluate L2E method
        print(f"\nMethod: L2E")
        evaluate_L2E_span_prediction_model(base_explainer="MaRC", split="test")

        # Evaluate supervised span predictor
        print(f"\nMethod: Supervised span predictor")
        evaluate_supervised_span_model(split="test")

        # Evaluate A2R and A2R-Noise predictions.
        evaluate_A2R_rationales(split="test")



if __name__ == '__main__':
    # The global save name lets you save different models/predictions for different runs without overwriting the data from previous runs.
    set_save_name("paper_run")

    train_all_models()
    evaluate_all_methods()
