import numpy as np
import os
import glob
from urllib import request
import zipfile
from sklearn.preprocessing import LabelBinarizer


class NotAdaptedError(Exception):
    pass


class TextVectorizer:
    def __init__(
        self,
        glove_url="http://nlp.stanford.edu/data/glove.6B.zip",
        max_tokens=20000,
        embedding_dim=100,
        to_lower=True,
    ):
        """
        This class parses the GloVe embeddings, the input documents are expected
        to be in the form of a list of lists.
        [["word1", "word2", ...], ["word1", "word2", ...], ...]

        Args:
            glove_url: The url of the GloVe embeddings.
            max_tokens: The maximum number of words in the vocabulary.
            embedding_dim: The dimension of the embeddings(pick one of 50, 100, 200, 300).
            to_lower: Whether to convert the words to lowercase.
        """
        self.max_tokens = max_tokens
        self.embedding_dim = embedding_dim
        self.to_lower = to_lower
        self.download_glove_if_needed(glove_url=glove_url)
        self.parse_glove()

    def download_glove_if_needed(self, glove_url):
        """
        Downloads the glove embeddings from the internet

        Args:
            glove_url: The url of the GloVe embeddings.
        """
        if not glob.glob("glove*.txt"):
            request.urlretrieve(glove_url, "glove.6B.zip")
            with zipfile.ZipFile("glove.6B.zip", "r") as zip_ref:
                zip_ref.extractall("glove_files")
            os.remove("glove.6B.zip")

    def parse_glove(self):
        """
        Parses the GloVe embeddings from their files, filling the vocabulary.
        """
        self.vocabulary = {}
        with open("glove.6B." + str(self.embedding_dim) + "d.txt") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                self.vocabulary[word.lower()] = coefs

    def adapt(self, documents: np.array):
        """
        Computes the OOV words for a single data split, and adds them to the dictionary.

        Args:
            documents: The data split (might be training set, validation set, or test set).
        """
        words = {
            word.lower() if self.to_lower else word for doc in documents for word in doc
        }  # Use a set containing words
        oov_words = words - self.vocabulary.keys()
        for word in oov_words:
            self.vocabulary[word] = np.random.uniform(-1, 1, size=self.embedding_dim)
        print(f"Generated embeddings for {len(oov_words)} OOV words.")

    def transform(self, documents):
        """
        Transform the data into the input structure.
        """
        return np.array([self._transform_document(document) for document in documents])

    def _transform_document(self, document):
        """
        Transforms a single document to the GloVe embedding
        """
        try:
            return np.array([self.vocabulary[word.lower()] for word in document])
        except KeyError:
            raise NotAdaptedError(
                f"The whole document is not in the vocabulary. Please adapt the vocabulary first."
            )


class TargetVectorizer:
    """
    One-hot encodes the target documents, containing the POS tags.
    """

    def __init__(self):
        self.vectorizer = LabelBinarizer()

    def adapt(self, targets):
        """
        Fits the vectorizer for the classes.
        """
        self.vectorizer.fit(
            [target for doc_targets in targets for target in doc_targets]
        )

    def transform(self, targets):
        """
        Performs the one-hot encoding for the dataset Ys, returning a list of encoded document tags.
        """
        if self.vectorizer.classes_.shape[0] == 0:
            raise NotAdaptedError(
                "The target vectorizer has not been adapted yet. Please adapt it first."
            )
        return [self.vectorizer.transform(document) for document in targets]


if __name__ == "__main__":
    data_dir = "data/" + os.listdir("data")[0] + "/"
    docs = os.listdir(data_dir)
    X = []
    y = []
    for doc in docs:
        np_doc = np.loadtxt(data_dir + doc, str, delimiter="\t")
        X.append(np_doc[:, 0])  # " ".join(np_doc[:,0]))
        y.append(np_doc[:, 1])  # " ".join(np_doc[:,1]))
    X, y = np.array(X), np.array(y)
    tv = TextVectorizer(max_tokens=20000)
    tv.adapt(X)
    print(X)
    print(tv.transform(X)[0])
    targbvec = TargetVectorizer()
    targbvec.adapt(y)
    print(targbvec.transform(y))
