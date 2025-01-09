This is the code repository for the paper [Rationalizing Transformer Predictions via End-To-End Differentiable Self-Training](https://aclanthology.org/2024.emnlp-main.664/) by Marc Brinner and Sina Zarrieß, published at EMNLP 2024.

## Use the RTP!

We created an easy-to-use wrapper that turns transformer encoders like BERT, RoBERTa, ELECTRA or DeBERTa into rationalized transformer predictors. The code and a conceptual usage example are available [here](use_it_yourself).

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
* `use_it_yourself`: This folder contains a python file that can be included in your project to quickly turn any base model into an rationalized transformer predictor.
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

Afterwards, create an environment using the `requirements.txt` file provided. The project was created using Python 3.12.

## Training/Evaluation

The `main.py` file contains two methods, one for training all models and creating all rationales for the post-hoc methods (`train_all_models`), and the other for
evaluating all models and predicted rationales (`evaluate_all_methods`). The method names for training the individual models should be self-explanatory, so you may comment out specific methods in case you only want to train specific ones.

The `settings.py` file contains some global settings like model checkpoints that are loaded for training. It also contains two global parameters that control the dataset that is used for training or evaluation (which can be set by calling `set_dataset_type()` with "Bio" or "Movies" as argument) as well as
a global `save_name` that is used to name saved model checkpoints or stored rationales. In this way, you can perform runs with different settings without overwriting previously saved data.
Note, that for the `save_name` "paper_run", rationale predictions for the post-hoc methods already exist, so they will be skipped if you try to predict new ones.
Trained model checkpoints are not included in the repo due to their size. If you need access to the trained models from our experiments, please reach out to marc.brinner@uni-bielefeld.de

## Results

### INAS Dataset

| Method     | Clf-F1    | AUC-PR    | Token-F1   | D-Token-F1 | IoU-F1    | D-IoU-F1  | Suff. $\downarrow$ | Comp.$\uparrow$ | Perf.     |
|------------|-----------|-----------|------------|-----------|-----------|-----------|-------------|-----------|-----------|
| Random     | -         | 0.220     | 0.255      | 0.222     | 0.067     | 0.003     | 0.194       | 0.191     | 0.289     |
| Supervised | 0.730     | 0.557     | 0.406      | 0.509     | 0.231     | 0.257     | 0.005       | 0.396     | 1.028     |
| MaRC       | 0.776     | 0.366     | 0.336      | 0.351     | 0.219     | 0.178     | 0.040       | 0.459     | 0.974     |
| Occlusion  | 0.776     | 0.307     | 0.277      | 0.294     | 0.145     | 0.071     | 0.078       | 0.352     | 0.696     |
| Int. Grads | 0.776     | 0.315     | 0.302      | 0.318     | 0.087     | 0.013     | 0.030       | 0.538     | 0.897     |
| LIME       | 0.776     | 0.272     | 0.280      | 0.273     | 0.082     | 0.007     | 0.097       | 0.406     | 0.671     |
| Shapley    | 0.776     | 0.309     | 0.301      | 0.320     | 0.084     | 0.009     | -0.012      | 0.587     | 0.984     |
| L2E-MaRC   | 0.776     | 0.431     | 0.359      | 0.402     | 0.174     | 0.131     | 0.044       | 0.503     | 0.992     |
| 2-Player   | 0.753     | 0.272     | 0.286      | 0.270     | 0.085     | 0.007     | **-0.052**  | 0.367     | 0.790     |
| 3-Player   | 0.703     | 0.287     | 0.296      | 0.286     | 0.080     | 0.004     | 0.017       | 0.472     | 0.831     |
| CAR        | -         | 0.314     | 0.281      | 0.280     | 0.184     | 0.133     | -           | -         | -         |
| A2R        | 0.654     | 0.268     | 0.287      | 0.264     | 0.084     | 0.008     | 0.128       | 0.338     | 0.581     |
| A2R-Noise  | 0.618     | 0.258     | 0.275      | 0.249     | 0.081     | 0.005     | 0.211       | 0.408     | 0.553     |
| RTP (Ours) | **0.777** | **0.445** | **0.362**  | **0.416** | **0.226** | **0.194** | 0.088       | **0.697** | **1.197** |

![Exemplary output for the INAS dataset](/output_data/images/example_5.png)

### Movie Review Dataset

| Method     | Clf-F1    | AUC-PR    | Token-F1  | D-Token-F1 | IoU-F1    | D-IoU-F1  | Suff. (lower is better) | Comp.     | Perf.     |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|------------|-----------|-----------|
| Random     | -         | 0.316     | 0.326     | 0.312     | 0.061     | 0.002     | 0.227      | 0.238     | 0.398     |
| Supervised | 0.980     | 0.670     | 0.514     | 0.626     | 0.144     | 0.169     | 0.001      | 0.638     | 1.295     |
| MaRC       | 0.965     | 0.428     | 0.404     | 0.423     | 0.181     | 0.118     | 0.036      | 0.478     | 1.027     |
| Occlusion  | 0.965     | 0.409     | 0.367     | 0.377     | 0.151     | 0.079     | -0.021     | 0.569     | 1.108     |
| Int. Grads | 0.965     | 0.376     | 0.358     | 0.371     | 0.067     | 0.009     | 0.049      | 0.484     | 0.860     |
| LIME       | 0.965     | 0.379     | 0.361     | 0.369     | 0.076     | 0.014     | 0.005      | 0.603     | 1.035     |
| Shapley    | 0.965     | 0.442     | 0.390     | 0.426     | 0.082     | 0.020     | **-0.029** | 0.827     | 1.328     |
| L2E-MaRC   | 0.965     | 0.565     | 0.460     | 0.534     | 0.126     | 0.104     | -0.016     | 0.652     | 1.254     |
| 2-Player   | 0.930     | 0.516     | 0.449     | 0.508     | 0.113     | 0.066     | -0.024     | 0.210     | 0.796     |
| 3-Player   | 0.955     | 0.458     | 0.422     | 0.465     | 0.089     | 0.023     | 0.003      | 0.354     | 0.862     |
| CAR        | -         | 0.384     | 0.364     | 0.376     | 0.078     | 0.013     | -          | -         | -         |
| A2R        | 0.955     | 0.474     | 0.433     | 0.486     | 0.111     | 0.046     | 0.109      | 0.320     | 0.755     |
| A2R-Noise  | 0.950     | 0.483     | 0.440     | 0.492     | 0.107     | 0.044     | 0.005      | 0.338     | 0.880     |
| RTP (Ours) | **0.975** | **0.567** | **0.466** | **0.544** | **0.203** | **0.195** | **-0.029** | **0.851** | **1.549** |

![Exemplary output for the movie review dataset](/output_data/images/example_2.png)

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
