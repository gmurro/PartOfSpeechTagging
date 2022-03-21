from data_input import DataInput
from text_vectorizer import TextVectorizer, TargetVectorizer
import os

if __name__ == "__main__":
    dataset = DataInput(
        data_url="https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/dependency_treebank.zip",
        train_size=0.50,
        dev_size=0.25,
        dataset_folder=os.path.join(os.getcwd(), "dataset"),
        split_into_sentences=True
    )

    # do preprocessing for train, validation and test sets
    dataset.preprocessing("train")
    dataset.preprocessing("dev")
    dataset.preprocessing("test")
    
    # separate inputs and targets
    X_train, y_train = dataset.train
    X_dev, y_dev = dataset.dev
    X_test, y_test = dataset.test

    # convert inputs to vector representation
    text_vectorizer = TextVectorizer(
        max_tokens=200000,
        embedding_dim=50,
        embedding_folder=os.path.join(os.getcwd(), "glove")
    )
    text_vectorizer.adapt(X_train)
    X_train = text_vectorizer.transform(X_train)
    text_vectorizer.adapt(X_dev)
    X_dev = text_vectorizer.transform(X_dev)
    text_vectorizer.adapt(X_test)
    X_test = text_vectorizer.transform(X_test)

    # convert targets to one-hot representation
    target_vectorizer = TargetVectorizer()
    # adapt the target vectorizer with only the training set: we do not consider possible targets that are not seen in training set but they are in the dev/test set
    target_vectorizer.adapt(y_train)  
    y_train = target_vectorizer.transform(y_train)
    y_dev = target_vectorizer.transform(y_dev)
    y_test = target_vectorizer.transform(y_test)
    
    print(f"First data input shape: {X_train[0].shape}")
    print(f"First data target shape: {y_train[0].shape}")
    print(f"Second data input shape: {X_train[1].shape}")
    print(f"Second data target shape: {y_train[1].shape}")