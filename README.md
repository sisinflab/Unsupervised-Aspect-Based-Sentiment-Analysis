# From Overall Sentiment to Aspect-Level Insights: A Pretraining Strategy for Unsupervised ABSA

Code and datasets of our paper "From Overall Sentiment to Aspect-Level Insights: A Pretraining Strategy for Unsupervised ABSA" 

## Requirements
- torch~=2.5.1
- transformers~=4.47.0
- pandas~=2.2.2
- tqdm~=4.66.5
- scikit-learn~=1.5.1
- numpy~=1.26.4
- evaluate~=0.4.3
- datasets~=2.19.1

## Preparation

- This project uses the `bert-base-uncased` model from Hugging Face as the embedding layer for text.
- You can download train data from: [https://www.dropbox.com/scl/fo/6faq3ormtc0p7qfi9yon1/AAbMHrP4VUcA6wF4og9HziQ?rlkey=c5fbsuksv3kah6ne2vxy316v3&st=9hojuloz&dl=0](https://www.dropbox.com/scl/fo/6faq3ormtc0p7qfi9yon1/AAbMHrP4VUcA6wF4og9HziQ?rlkey=c5fbsuksv3kah6ne2vxy316v3&st=9hojuloz&dl=0)
- By default, the data is loaded from the folder: `data/train/`.
- If you'd like to use a different dataset, you can preprocess it using the script `script/train_process.py` before training.

## ABSA trained model
To use the trained model for aspect-based sentiment analysis, you can download the model weights from the following link:
[https://www.dropbox.com/scl/fo/ekp4o1avws6tgycg8kpyc/AAFWG_ViaC-iNLbP9Y2JAWc?rlkey=ttsrb84rll23vat6txcwa4p5t&st=lkgtgfgh&dl=0](https://www.dropbox.com/scl/fo/ekp4o1avws6tgycg8kpyc/AAFWG_ViaC-iNLbP9Y2JAWc?rlkey=ttsrb84rll23vat6txcwa4p5t&st=lkgtgfgh&dl=0)

Once downloaded, save the weights to the specified folder. The default folder path is:
[model/trained_model](model/trainde_model)
Ensure that the model weights are placed correctly in the specified folder to allow the system to load them during analysis.

## Run
To run the model, run:

```bash
sh run.sh
