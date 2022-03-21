import os, random
import numpy as np
import glob
from urllib import request
import zipfile


class DataInput:
    """
    This class is used to load the data from the data_url and create the train, dev and test datasets.
    """
    def __init__(self, data_url, train_size, dev_size, dataset_folder="data", split_into_sentences=False, shuffle=False):
        """
        Constructor for DataInput class that loads the data from the data_url and creates the train, dev and test datasets.
        The dataset splitted is available through the instance variables self.train, self.dev, self.test
        
        Parameters
        ----------    
        data_url : URL of the dataset
        train_size : percentage of the dataset that will be used for training
        dev_size : percentage of the dataset that will be used for development
        dataset_folder : folder where the dataset will be downloaded
        split_into_sentences : boolean indicating if each document in the dataset should be splitted into sentences or not
        shuffle : boolean indicating if the dataset should be shuffled or not before splitting
        """
        docs = self.import_data(data_url, dataset_folder)           
        X, y = self.parse_dataset(docs, split_into_sentences)
        self.train, self.dev, self.test = self.train_dev_test_split(X, y, train_size, dev_size, path_store=os.path.join(dataset_folder, "split"), shuffle=shuffle)

    def import_data(self, data_url, dataset_folder):
        """
        Import dataset from URL.

        Parameters
        ----------  
        data_url : URL of the dataset
        dataset_folder : folder where the dataset will be downloaded
        files_extension : extension of the files inside the dataset (default: dp)

        Returns
        -------
        list of document paths
        """
        # create dataset folder if it does not exist
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

        # extract the dataset if it is not extracted
        if not glob.glob(os.path.join(dataset_folder, "**/*.dp"), recursive=True):

            # download the dataset if it does not exist 
            dataset_zip = os.path.join(dataset_folder, data_url.split("/")[-1])
            if not os.path.exists(dataset_zip):
                print("Downloading the dataset...")
                request.urlretrieve(data_url, dataset_zip)
                print("Successful download!")

            # extract the dataset
            print("Extracting the dataset...")
            with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
                zip_ref.extractall(dataset_folder)
                print("Successfully extracted the dataset!")
            os.remove(dataset_zip)
        
        dataset_extracted_dir = os.path.join(dataset_folder, os.listdir(dataset_folder)[0])
        docs = [os.path.join(dataset_extracted_dir, doc) for doc in os.listdir(dataset_extracted_dir)]
        return docs

    def parse_dataset(self, docs, split_into_sentences):
        """
        Parse the dependency treebank dataset. This takes into account if the dataset should be splitted into sentences or not (according to self.split_into_sentences).
        
        Parameters
        ----------  
        docs : list of document paths
        split_into_sentences : boolean indicating if each document in the dataset should be splitted into sentences or not
        
        Returns
        -------
        pair where the first element is a list of lists representing tokens in each document/sentence, the second is a list of lists representing POS tag of tokens in each document/sentence
        """
        X = []
        y = []
        for doc in docs:
            if split_into_sentences:
                # if split_into_sentences is True, then we split the document into sentences considering as new sentence all the tokens after an empty line in the document
                with open(doc, mode='r', encoding='utf-8') as text_file:
                    sentence_text = []
                    sentence_tags = []
                    
                    # read the doc and extract informations
                    for line in text_file.readlines():
                        if line.strip() == "":
                            # if the line is empty, then we add the sentence to the list of sentences
                            X.append(np.array(sentence_text))
                            y.append(np.array(sentence_tags))
                            sentence_text = []
                            sentence_tags = []
                        else:
                            # if the line is not empty, then we add the token to the sentence
                            token, tag, number = line.split("\t")
                            sentence_text.append(token)
                            sentence_tags.append(tag)
                    X.append(np.array(sentence_text))
                    y.append(np.array(sentence_tags))
            else:
                # otherwise we consider the whole document as a single data point
                np_doc = np.loadtxt(doc, str, delimiter="\t", usecols = (0, 1))
                X.append(np_doc[:, 0])
                y.append(np_doc[:, 1])

        # add a padding appending dots to the end of each doc to make sure that all docs have the same length
        max_tokens = max([len(doc) for doc in X])
        for i, (doc, target) in enumerate(zip(X, y)):
            X[i] = np.pad(doc, (0, max_tokens-len(doc)), constant_values=".")
            y[i] = np.pad(target, (0, max_tokens-len(target)), constant_values=".")  
        return np.array(X), np.array(y)

    def train_dev_test_split(self, X, y, train_size, dev_size, path_store=None, shuffle=False):
        """
        Split dataset into train, validation and test.

        Parameters
        ----------
        X : list of lists of tokens in each document/sentence
        y : list of lists of POS tags in each document/sentence
        train_size : percentage of the dataset used for training
        dev_size : percentage of the dataset used for validation (note that test size is 1-train_size-dev_size)
        shuffle : boolean indicating if the dataset should be shuffled before splitting
        path_store : path where the split datasets will be stored. If None, then the split datasets will not be stored.
        
        Returns
        -------
        triple where the first element is the train-set, the second element is the dev-set and the third element is the test-set
        """
        # shuffle the dataset
        if shuffle:
            dataset = list(zip(X, y))
            random.shuffle(dataset)
            X, y = zip(*dataset)
            X, y = np.array(X), np.array(y)

        # create folder where the split datasets will be stored if it does not exist
        if path_store is not None and not os.path.exists(path_store):
            os.makedirs(os.path.join(path_store, "train"))
            os.makedirs(os.path.join(path_store, "dev"))
            os.makedirs(os.path.join(path_store, "test"))

        # build the train set
        train_size = int(np.ceil(train_size*len(X)))
        train_set = (X[:train_size], y[:train_size])
        if path_store is not None:
            np.savetxt(os.path.join(path_store, "train", "X_train.txt"), train_set[0], fmt="%s")
            np.savetxt(os.path.join(path_store, "train", "y_train.txt"), train_set[1], fmt="%s")
        print("Train set size:", len(train_set[0]))

        # build the dev set
        dev_size = int(np.ceil(dev_size*len(X)))
        dev_set = (X[train_size:train_size+dev_size], y[train_size:train_size+dev_size])
        if path_store is not None:
            np.savetxt(os.path.join(path_store, "dev", "X_dev.txt"),  dev_set[0], fmt="%s")
            np.savetxt(os.path.join(path_store, "dev", "y_dev.txt"),  dev_set[1], fmt="%s")
        print("Dev set size:", len(dev_set[0]))

        # build the test set
        test_set = (X[train_size+dev_size:], y[train_size+dev_size:])
        if path_store is not None:
            np.savetxt(os.path.join(path_store, "test", "X_test.txt"), test_set[0], fmt="%s")
            np.savetxt(os.path.join(path_store, "test", "y_test.txt"), test_set[1], fmt="%s")
        print("Test set size:", len(test_set[0]))
        return train_set, dev_set, test_set

    def preprocessing(self, set="train", to_lower=True):
        """
        Preprocess the dataset.

        Parameters
        ----------
        set : the set to preprocess (it can be train, dev or test)
        to_lower : boolean indicating if the dataset should be converted to lowercase
        """
        # lower the tokens in a given set
        if set == "train" or set == "dev" or set == "test":
            for doc in getattr(self, set)[0]:
                    if to_lower:
                        for i in range(len(doc)):
                            doc[i] = doc[i].lower()
        else:
            raise ValueError("Invalid set name. It should be train, dev or test.")