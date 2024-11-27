This is the code repository for the paper [Rationalizing Transformer Predictions via End-To-End Differentiable Self-Training](https://aclanthology.org/2024.emnlp-main.664/) by Marc Brinner and Sina Zarrieß, published at EMNLP 2024.

## Repository Structure

The repository contains code for training and evaluating all methods tested in our paper, except A2R and A2R-Noise, for which we used
the [original paper's code](https://github.com/adamstorek/noise_injection).

* `A2R`: This folder contains code for loading and evaluating the predicted rationales we exported from the A2R code repo. These rationales are included.
* `L2E`: This folder contains code for training and evaluating the Learning to Explain method by [Situ et al.](https://aclanthology.org/2021.acl-long.415/)
* `auxiliary`: This folder contains code that is commonly used by many other components, like loss functions and text visualization methods.
* `data`: This folder contains code for loading the datasets, as well as the data itself.
* `evaluation`: This folder contains code for evaluating class and rationale predictions as well as faithfulness.
* `output_data`: This folder contains outputs like saved model checkpoints and rationale predictions from post-hoc explainers.
* `post_hoc_methods`: This folder contains code from training a standard classifier, as well as for creating, storing and evaluating rationales for the following post-hoc methods:
    * MaRC [(Brinner et al.)](https://aclanthology.org/2023.findings-acl.867/)
    * Saliency [(Simonyan et al.)](https://arxiv.org/abs/1312.6034)
    * Input times gradient [(Shrikumar et al.)](https://arxiv.org/abs/1605.01713)
    * Occlusion [(Zeiler et al.)](https://arxiv.org/abs/1311.2901)
    * LIME [(Ribeiro et al.)](https://arxiv.org/abs/1602.04938)
    * Integrated Gradients [(Sundararajan et al.)](https://arxiv.org/abs/1703.01365)
    * Shapley value sampling [(Castro et al.)](https://www.sciencedirect.com/science/article/pii/S0305054808000804)
* `supervised_span_model`: This folder contains code for training a supervised model on a dataset annotated with spans.
* `weakly_supervised_models`: This folder contains code for training and evaluating the weakly supervised rationalized classifiers (except A2R and A2R-Noise):
    * 2-Player [(Lei et al.)](https://aclanthology.org/D16-1011/)
    * 3-Player [(Yu et al.)](https://aclanthology.org/D19-1420/)
    * CAR [(Chang et al.)](https://papers.nips.cc/paper_files/paper/2019/hash/5ad742cd15633b26fdce1b80f7b39f7c-Abstract.html)
    * RTP (Our model)

## Setup

The datasets we used for our evaluation are not included in this repository.
* For the movie review dataset, please visit [https://www.eraserbenchmark.com/](https://www.eraserbenchmark.com/) and download the dataset listed under "Movies".
Then, place the data located in the `movies` folder inside of `data/Movies`.
* For the INAS dataset (biological abstracts), we are unfortunately not allowed to publicly distribute the dataset due to copyright issues.
On request, we might be able to provide a private copy for research purposes. For this, please reach out to marc.brinner@uni-bielefeld.de

Afterwards, create an environment using the `requirements.txt` file provided.

## Training/Evaluation

The `main.py` file contains two methods, one for training all models and creating all rationales for the post-hoc methods (`train_all_models`), and the other for
evaluating all models and predicted rationales (`evaluate_all_methods`). The method names for training the individual models should be self-explanatory, so you may comment out specific methods in case you only want to train specific ones.

The `settings.py` file contains some global settings like model checkpoints that are loaded for training. It also contains two global parameters that control the dataset that is used for training or evaluation (which can be set by calling `set_dataset_type()` with "Bio" or "Movies" as argument) as well as
a global `save_name` that is used to name saved model checkpoints or stored rationales. In this way, you can perform runs with different settings without overwriting previously saved data.
Note, that for the `save_name` "paper_run", rationale predictions for the post-hoc methods already exist, so they will be skipped if you try to predict new ones.
Trained model checkpoints are not included in the repo due to their size. If you need access to the trained models from our experiments, please reach out to marc.brinner@uni-bielefeld.de

## Citation

If you use our code or want to reference our paper, please cite:
```bibtex
@inproceedings{brinner-zarriess-2024-rationalizing,
    title = "Rationalizing Transformer Predictions via End-To-End Differentiable Self-Training",
    author = "Brinner, Marc Felix  and
      Zarrie{\ss}, Sina",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.664",
    pages = "11894--11907",
    abstract = "We propose an end-to-end differentiable training paradigm for stable training of a rationalized transformer classifier. Our approach results in a single model that simultaneously classifies a sample and scores input tokens based on their relevance to the classification. To this end, we build on the widely-used three-player-game for training rationalized models, which typically relies on training a rationale selector, a classifier and a complement classifier. We simplify this approach by making a single model fulfill all three roles, leading to a more efficient training paradigm that is not susceptible to the common training instabilities that plague existing approaches. Further, we extend this paradigm to produce class-wise rationales while incorporating recent advances in parameterizing and regularizing the resulting rationales, thus leading to substantially improved and state-of-the-art alignment with human annotations without any explicit supervision.",
}