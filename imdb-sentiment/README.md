# IMDB Sentiment Example

This example fine-tunes DistilBERT on the IMDb movie review dataset using the Hugging Face sequence-classification workflow, saves the model locally, and serves predictions through the Chanterelle handler.

## Files

- `train_model.py`: loads IMDb, tokenizes text, fine-tunes DistilBERT, evaluates, and saves model artifacts.
- `handler_io.py`: loads the saved tokenizer and model locally for inference.
- `model_meta.json`: Chanterelle model metadata.
- `IMDB_reviews-explore.ipynb`: lightweight notebook for inspecting the dataset and trying model inference.
- `requirements.txt`: Python dependencies for the example.

## Training Steps

The training flow in `train_model.py` follows the standard Hugging Face tutorial structure:

1. Load the IMDb dataset using `datasets.load_dataset("imdb")`.
2. Load the DistilBERT tokenizer for `distilbert/distilbert-base-uncased`.
3. Tokenize each movie review with truncation so raw text becomes model inputs.
4. Load `AutoModelForSequenceClassification` with two labels: negative and positive.
5. Define training arguments such as learning rate, batch size, epochs, evaluation strategy, and checkpoint behavior.
6. Build a `Trainer` with the model, tokenized datasets, tokenizer, dynamic padding collator, and accuracy metric.
7. Fine-tune the pretrained DistilBERT model on the IMDb training split.
8. Evaluate the fine-tuned model on the test split.
9. Save the trained model and tokenizer into the local `imdb_distilbert_model` directory.
10. Run a few sample predictions as a quick sanity check.
11. Save metadata, evaluation results, and example predictions to `model_info.json`.

In short, the pipeline is:

`raw text -> tokenizer -> DistilBERT classifier -> fine-tuning -> evaluation -> save model and tokenizer -> local inference`

## How Inference Works

The Chanterelle handler in `handler_io.py` does not download models from the Hugging Face Hub during normal prediction.

It:

1. Loads the tokenizer and model from the local `imdb_distilbert_model` directory.
2. Tokenizes incoming `review_text`.
3. Runs the model with PyTorch under `torch.no_grad()`.
4. Converts logits into probabilities and returns a positive or negative sentiment result.

## Run Locally

From this directory:

```bash
/Users/alieladi/Dev/Chanterelles/chanterelle-examples/venv/bin/python train_model.py
```

This will create:

- `imdb_distilbert_model/`
- `model_info.json`

## Notes

- A Hugging Face login is not required for this example because it uses public models and datasets and saves artifacts locally.
- The first run may download the public base model and dataset into your local Hugging Face cache.
- The handler expects training to have already been run so `imdb_distilbert_model/` exists.

## Sources

The updated workflow is based on these references:

- Hugging Face Transformers text classification tutorial: https://huggingface.co/docs/transformers/tasks/sequence_classification
- DistilBERT base model: https://huggingface.co/distilbert/distilbert-base-uncased
- DistilBERT SST-2 fine-tuned example model card: https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
- BERT base model card for comparison: https://huggingface.co/google-bert/bert-base-uncased

This example follows the Hugging Face training pattern closely, but saves artifacts locally for Chanterelle instead of pushing them to the Hugging Face Hub.