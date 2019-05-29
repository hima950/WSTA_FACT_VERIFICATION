# Automatic Fact Verification Project
Fact Verification System for Unimeln WSTA course by Team: White&Brown.

# INSTRUCTIONS FOR RUNNING:


## BERT Preparation and Running:
1. Download the pretrained weights to be tuned. [BERT-Base Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)

2. Unzip the bert.zip it is modified version of BERT repository with our parsers added to `run_classifier.py`.

*Sentence Selection part:*

3. This notebook [Sentence Selection Notebook](https://github.com/hima950/WSTA_FACT_VERIFICATION/blob/master/sentence_selection.ipynb) formats the claim and top-k results sentences pairs into the format of MRPC (GLUE DATASET) for the BERT SENTENCE SELECTION TUNING. It creates the train.tsv, dev.tsv and test.tsv.

4. Run the `bert-sentence-train.sh` script to tune the BERT Weights. Then there should be a folder /bert_output created in bert directory. Then to make predictions on the tuned weights change the name of the tuned weights according checkpoint created in line2 of `bert-sentence-train.sh` and run this script.

*Label Prediction and tuning Bert:*

5. This notebook [Label Prediction Notebook](https://github.com/hima950/WSTA_FACT_VERIFICATION/blob/master/label_prediction.ipynb) formats the claim and combined evidence pairs into the format of RTE (GLUE DATASET) for the BERT Label Prediction TUNING. It creates the train.tsv, dev.tsv and test.tsv.

6. Please make sure the datasets are created and use `bert_train_label.sh` to tune the weights just as before. `bert_test_label.sh` is used to make predictions.
Note: a file `eval_results.txt` created while tuning which contains the evaluation accuracy.

