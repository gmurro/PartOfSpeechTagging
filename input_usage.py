from data_input import DataInput
from text_vectorizer import TextVectorizer, TargetVectorizer

if __name__ == "__main__":
    di = DataInput(
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/dependency_treebank.zip",
        0.5,
        0.25,
    )
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
    print(X_train, y_train)
