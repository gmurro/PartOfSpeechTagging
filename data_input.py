import os, random
import numpy as np
import glob
from urllib import request
import zipfile


class DataInput:
    def __init__(self, data_url, train_size, dev_size, sequence_length=200):
        docs = self.import_data(data_url)
        self.sequence_length = sequence_length
        self.datasets = self.train_dev_test_split(docs, train_size, dev_size)

    def import_data(self, data_url):
        """
        Import POS dataset from URL.
        """
        if not glob.glob("data/*.dp"):
            request.urlretrieve(data_url, "data.zip")
            with zipfile.ZipFile("data.zip", "r") as zip_ref:
                zip_ref.extractall("data")
            os.remove("data.zip")
        self.data_dir = "data/" + os.listdir("data")[0] + "/"
        docs = os.listdir(self.data_dir)
        return docs

    def parse_dataset(self, docs):
        """
        Parse the dependency treebank dataset.
        """
        X = []
        y = []
        for doc in docs:
            np_doc = np.loadtxt(self.data_dir + doc, str, delimiter="\t")
            X.append(np_doc[:, 0])
            y.append(np_doc[:, 1])
        return np.array(X), np.array(y)

    def train_dev_test_split(self, docs, train_size, dev_size):
        """
        Split dataset into train and test.

        Args:
            docs: list of documents
            train_size: float, percentage of train data
            dev_size: float, percentage of dev data (note that test size is 1-train_size-dev_size)

        Returns:
            train_docs: list of train documents
        """
        random.shuffle(docs)
        print(int(train_size * len(docs)))
        train_docs = self.parse_dataset(docs[: int(train_size * len(docs))])
        np.savetxt(
            "train_X.txt", train_docs[0], fmt="%s"
        )  # Save txt for alacarte embedding
        dev_docs = self.parse_dataset(
            docs[int(train_size * len(docs)) : int((train_size + dev_size) * len(docs))]
        )
        np.savetxt("dev_X.txt", dev_docs[0], fmt="%s")
        test_docs = self.parse_dataset(docs[int((train_size + dev_size) * len(docs)) :])
        np.savetxt("test_X.txt", test_docs[0], fmt="%s")
        return [train_docs, dev_docs, test_docs]
