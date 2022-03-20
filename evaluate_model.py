from text_vectorizer import TextVectorizer, TargetVectorizer
from data_input import DataInput
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test, labels):
    """
    given a trained model and a test set returns the f-score and the confusion matrix
    taking into account only classes in labels
    """
    raw_y_true = np.array(y_test)
    raw_y_pred = model.predict(X_test)
    # shape of the output is doc x len_sen x classes
    # argmax for label predictions
    len_sentence = raw_y_pred.shape[1]
    num_sentences = raw_y_pred.shape[0]
    y_pred = np.empty((num_sentences, len_sentence))
    y_true = np.empty((num_sentences, len_sentence))
    # assign label with the highest probability
    for i in range(num_sentences):
        for j in range(len_sentence):
            y_pred[i,j] = np.argmax(raw_y_pred[i,j,:])
            y_true[i,j] = np.argmax(raw_y_true[i,j,:])
    # flatten the numpy array to have a 1D array
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    # show confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(conf_matrix)
    disp.plot()
    plt.show()
    # f1 score
    print("F score:\n-------------------------------\n")
    f1_score(y_true, y_pred, labels=labels, average='macro')


if __name__ == "__main__":
    
    # import of the corpus
    di = DataInput("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/dependency_treebank.zip", 0.5, 0.25)
    tv = TextVectorizer(max_tokens=20000)
    tv.adapt(di.datasets[0][0])
    X_train = tv.transform(di.datasets[0][0])
    tv.adapt(di.datasets[1][0])
    X_dev = tv.transform(di.datasets[1][0])
    tv.adapt(di.datasets[2][0])
    X_test = tv.transform(di.datasets[2][0])
    target_vec = TargetVectorizer()
    target_vec.adapt(di.datasets[0][1])
    y_train = target_vec.transform(di.datasets[0][1])
    y_dev = target_vec.transform(di.datasets[1][1])
    y_test = target_vec.transform(di.datasets[2][1])
    # train and validation steps
    
    # ...
    
    # test the best two models

    # it is possible to obtain this information using classes_ attribute of LabelBinarizer
    # the classes are: "$", "''", ",", "-LRB-", "-RRB-" (parenthesis), ".", ":", "LS" (list item marker),
    # "SYM" (symbol), "``"
    punctuation_indexes = [0, 1, 2, 3, 4, 5, 6, 16, 30, 43]
    classes = target_vec.get_classes()
    valid_labels = list(set(range(len(classes))) - set(punctuation_indexes))
    evaluate_model(best_model, X_test, y_test, valid_labels)
