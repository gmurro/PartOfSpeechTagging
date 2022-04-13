# ✅ Part-of-speech tagging ⁉️

This repository contains a project realized as part of the *Natural Language Processing* course of the [Master's degree in Artificial Intelligence](https://corsi.unibo.it/2cycle/artificial-intelligence), University of Bologna.

## Description

*Part-of-speech (POS) tagging* is a popular NLP task which refers to categorizing words in a text (corpus) in correspondence with a particular part of speech, depending on the definition of the word and its context.

To tackle this problem, different types of Recurrent Neural Networks and parameters are been used, namely BiLSTM, Bi-GRU, double BiLSTM with different combinations of fully connected layers. The input was embedded using GloVE embeddings, and the Out-Of-Vocabulary words were randomly instantiated. After sufficient hyperparameter tuning, the best models were BiLSTM and BiGRU, and they achieved Macro-F1 scores of 0.6183 and 0.6117 on the validation set.

## Dataset
For this experiment, the [Dependency Parsed Treebank](https://www.nltk.org/nltk_data/) dataset is used.
It contains 199 documents annotated, but in oder to achieve better results each document is been splitted into sentences.
The train-val-test split proprortin is 50%-25%-25%.

## Request and solution proposed
The task to comply with is described in the [assignment description](./Assignment_1.ipynb).
In order to have a better understanging of our proposed solution, take a look to the [notebook](./POS_tagging.ipynb) and the [report](./report.pdf).

## Results
The metric which has been used to compare models is the macro F1 score with the validation set, results are the following:
|               |        |        |                   |                     |
|:-------------:|:------:|:------:|:-----------------:|:-------------------:|
|    Metric     | BiLSTM | BiGRU  |  2 BiLSTM + Dense |   BiLSTM + 2 Dense  |
| Accuracy val  | 0.8457 | 0.8362 |       0.7564      |       0.7907        |
| F1 score val  | 0.6183 | 0.6117 |       0.5094      |       0.4840        |
| F1 score test | 0.6246 | 0.6062 |   not tested      |   not tested        |


This is an example of the prediction made by the BiGRU model on a test sentence:
```
Original sentence:  ['but' 'courts' 'quickly' 'tumbled' 'down' 'a' 'slippery' 'slope' '.']
Original POS tagging:  ['CC' 'NNS' 'RB' 'VBD' 'IN' 'DT' 'JJ' 'NN' '.']
Predicted POS tagging:  ['CC' 'NNS' 'RB' 'VBD' 'RP' 'DT' 'JJ' 'NN' '.']
```

## Resources & Libraries

* scikit-learn
* Tensorflow + Keras



## Versioning

We use Git for versioning.



## Group members

| Reg No. |   Name    |  Surname  |                 Email                  |                       Username                        |
| :-----: | :-------: | :-------: | :------------------------------------: | :---------------------------------------------------: |
| 1005271 | Giuseppe  |   Boezio  | `giuseppe.boezio@studio.unibo.it`      | [_giuseppeboezio_](https://github.com/giuseppeboezio) |
|  983806 | Simone    |  Montali  |    `simone.montali@studio.unibo.it`    |         [_montali_](https://github.com/montali)         |
|  997317 | Giuseppe  |    Murro  |    `giuseppe.murro@studio.unibo.it`    |         [_gmurro_](https://github.com/gmurro)         |



## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details