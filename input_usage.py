from data_input import DataInput
from text_vectorizer import TextVectorizer, TargetVectorizer
import os

if __name__ == "__main__":
    dataset = DataInput(
        data_url="https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/dependency_treebank.zip",
        train_size=0.5,
        dev_size=0.25,
        dataset_folder=os.path.join(os.getcwd(), "dataset"),
        split_into_sentences=False
    )

    tv = TextVectorizer(max_tokens=20000)
    tv.adapt(dataset.train[0])
    X_train = tv.transform(dataset.train[0])
    tv.adapt(dataset.dev[0])
    X_dev = tv.transform(dataset.dev[0])
    tv.adapt(dataset.test[0])
    X_test = tv.transform(dataset.test[0])

    target_vec = TargetVectorizer()
    target_vec.adapt(dataset.train[1])
    y_train = target_vec.transform(dataset.train[1])
    y_dev = target_vec.transform(dataset.dev[1])
    y_test = target_vec.transform(dataset.test[1])
    print(X_train, y_train)
