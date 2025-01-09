# An easy and quick way to use the Rationalized Transformer Predictor yourself!

We created a model wrapper that turns many common transformer encoder models into rationalized transformer predictors.

Tested with:
* BERT (base and large)
* RoBERTa
* Electra
* DeBERTa
* DeBERTa v3


### How to use it:
Just place the ```RTP.py``` file in your project. Then, you can wrap an existing model in the RTP class.
Note that you need to provide the base model to the RTP class (i.e., a ```BertModel``` or a ```DebertaModel```, not a ```BertForSequenceClassification```).
In case you want to visualize the prediction, you need to have the ```sty``` package installed. The remaining code works with pytorch (tested with version 2.3.0).


Usage example for BERT (works without changes for "google-bert/bert-base-uncased", google-bert/bert-large-uncased, "FacebookAI/roberta-base", google/electra-base-discriminator", "microsoft/deberta-base" and "microsoft/deberta-v3-base" and will most likely work with many other models):

```
from easy_RTP import RTP

# Load your dataset...
dataset = load_dataset()

# Loading the base model and tokenizer
model = AutoModel.from_pretrained("google-bert/bert-base-uncased").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = RTP(model, tokenizer, embedding_fn=lambda m: m.embeddings, n_class_labels=10, loss_fn=torch.nn.CrossEntropyLoss(), output_fn="softmax")

optimizer = torch.optim.AdamW(list(model.parameters()), lr=4e-5, weight_decay=1e-4)

for input_texts, labels in dataset:
    input_batch = tokenizer(input_texts, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
    
    loss = model.train_step(batch, labels)
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    
    if you_want_to_visualize_the_word_importance:
        model_input = tokenizer([input_text], return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
        model.visualize_sample(model_input)
        
    if you_want_to_get_the_classification_logits_or_importance_masks:
        model_input = tokenizer([input_text], return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
        model_output = model(model_input)
        
        class_logits = model_output["logits"]
        class_importance_masks = model_output["importance_scores"]
```

Arguments to the ```RTP``` class are:
* ```model```: The model instance that you want to turn into an RTP.
* ```tokenizer```: The tokenizer for the model.
* ```embedding_fn```: A lambda function that gets the model's embedding function that turns input_ids into embeddings.
* ```n_class_labels```: The number of classes of your classification problem.
* ```loss_fn```: The loss function for your classification problem. The input to this function will be the model output (logits of shape (n_samles, n_classes)), as well as the label you provide for the sample.
* ```output_fn```: Either "softmax" or "sigmoid", depending on the type of classification problem (single label or multi-label classification).

Hyperparameters can be adjusted in the dict at the top of ```RTP.py```.
Note that the forward pass for a single sample requires (during training) multiple forward passes through the base model.
You may need to choose a lower batch size compared to normal training and use gradient accumulation instead.

Please make sure to cite our work if you use the RTP in your research.
