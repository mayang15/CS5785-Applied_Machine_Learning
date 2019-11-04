
# coding: utf-8

# In[291]:


import numpy as np
from numpy import linalg
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import string


# In[347]:


## (a) load data
def load_data(filename):
    sentences = []
    labels = []
    for line in open("sentiment labelled sentences/" + filename):
        sentences.append(line.strip('\n').split('\t')[0])
        labels.append(line.strip('\n').split('\t')[1])
#     print("dataset: "+str(filename))
#     print("total sentences: " + str(len(labels)))
#     print("positive sentences: " + str(labels.count('1')))
#     print("negative sentences: " + str(labels.count('0')))
    return sentences, labels


# In[346]:


## (b) preprocessing
def preprocessing(sentences, lowercase, strip_punctuation, lemmatization, remove_stop_words):
    trainset = sentences
    
    if lowercase:
        # lowercase all of the words
        #print("preprocessing - lowercase")
        trainset_0 = []
        for sentence in trainset:
            trainset_0.append(sentence.lower())
        trainset = trainset_0
        #print(trainset[0])
    
    if strip_punctuation:
        # strip punctuation
        #print("preprocessing - strip punctuation")
        trainset_0 = []
        for sentence in trainset:
            trainset_0.append(sentence.translate(str.maketrans('', '', string.punctuation)))
        trainset = trainset_0
        #print(trainset[0])
    
    if lemmatization:
        # lemmatization of all the words
        #print("preprocessing - lemmatization")
        ps = PorterStemmer()
        trainset_0 = []
        for sentence in trainset:
            words = sentence.split(' ')
            sentence_out = ""
            for word in words:
                sentence_out = sentence_out + ps.stem(word) + ' '
            trainset_0.append(sentence_out)
        trainset = trainset_0
        #print(trainset[0])
    
    if remove_stop_words:
        # remove stop words
        #print("preprocessing - remove stop words")
        trainset_0 = []
        #stop_words = set(stopwords.words('english'))
        stop_words = set(['the','and','or','a','an','for','that','is','are','were','was'])
        for sentence in trainset:
            word_tokens = word_tokenize(sentence) 
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            trainset_0.append(" ".join(filtered_sentence))
            #print(" ".join(filtered_sentence))
        trainset = trainset_0
        #print(trainset[0])
    
    # split sentences
    for i in range(len(trainset)):
        trainset[i] = trainset[i].split(' ')
    #print(trainset[0])
    
    return trainset


# In[268]:


## (c) split training and testing set
def split_data(sentences, labels):
    one_indexs = [i for i,x in enumerate(labels) if x == '1']
    zero_indexs = [i for i,x in enumerate(labels) if x == '0']
    trainset_1 = [sentence for i,sentence in enumerate(sentences) if i in one_indexs[:400]]
    trainset_0 = [sentence for i,sentence in enumerate(sentences) if i in zero_indexs[:400]]
    testset_1 = [sentence for i,sentence in enumerate(sentences) if i in one_indexs[400:]]
    testset_0 = [sentence for i,sentence in enumerate(sentences) if i in zero_indexs[400:]]
    trainset = trainset_1 + trainset_0
    testset = testset_1 + testset_0
    trainlabels = np.concatenate((np.ones((400,1)),np.zeros((400,1))), axis = 0)
    testlabels = np.concatenate((np.ones((100,1)),np.zeros((100,1))), axis = 0)
    return trainset, testset, trainlabels, testlabels


# In[289]:


def generate_trainset_testset():
    trainset = []
    testset = []
    trainlabels = []
    testlabels = []
    for filename in ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]:
        sentences, labels = load_data(filename)
        sentences = preprocessing(sentences,1,1,1,1)
        trainset0, testset0, trainlabels0, testlabels0 = split_data(sentences, labels)
        trainset.extend(trainset0)
        testset.extend(testset0)
        trainlabels.extend(trainlabels0)
        testlabels.extend(testlabels0)
    return trainset, testset, trainlabels, testlabels


# In[368]:


## (d) Bag of words model - extract features
def bag_of_words():
    trainset, testset, trainlabels, testlabels = generate_trainset_testset()

    # build dictionary of unique words
    dict_key = []
    for sentence in trainset:
        dict_key.extend(sentence)
    dict_key = list(set(dict_key))
    # my_dict = my_dict.fromkeys(dict_key, 0)

    # extract features
    train_features = np.zeros((len(trainset), len(dict_key)))
    for i in range(len(trainset)):
        for word in trainset[i]:
            train_features[i, dict_key.index(word)] = train_features[i, dict_key.index(word)] + 1

    test_features = np.zeros((len(testset), len(dict_key)))
    for i in range(len(testset)):
        for word in testset[i]:
            if word in dict_key:
                test_features[i, dict_key.index(word)] = test_features[i, dict_key.index(word)] + 1

    return dict_key, train_features, test_features, trainlabels, testlabels
    
#     print("shape of train_features: " + str(np.shape(train_features)))
#     print("shape of test_features: "+ str(np.shape(test_features)))

#     print("sample trainset review 0:")
#     print("sum of feature vector: " + str(sum(train_features[0,:])))
#     print("indexs of non zero element in feature vector: " + str([i for i,x in enumerate(train_features[0,:]) if x != 0]))
#     print("non zero words: " + str([x for i,x in enumerate(dict_key) if train_features[0,i] != 0]))

#     print("sample trainset review 3:")
#     print("sum of feature vector: " + str(sum(train_features[3,:])))
#     print("indexs of non zero element in feature vector: " + str([i for i,x in enumerate(train_features[3,:]) if x != 0]))
#     print("non zero words: " + str([x for i,x in enumerate(dict_key) if train_features[3,i] != 0]))


# In[271]:


## (e) Post-processing
def L2_norm(features):
    post_features = features
    for i in range(np.shape(post_features)[0]):
        #print(np.sum(post_features[i,:]))
        #print(linalg.norm(post_features[i,:]))
        post_features[i,:] = post_features[i,:] / linalg.norm(post_features[i,:])
        #print(features[i,:])
    return post_features

def standardize(features):
    post_features = features
    post_features = post_features - np.mean(post_features,axis=0)
    feat_var = np.var(post_features, axis=0)
    for i in range(np.shape(post_features)[1]):
        post_features[:,i] = post_features[:,i] / feat_var[i]
        # print(feat_var[i])
    return post_features


# In[369]:


## (f) sentiment prediction
def sentiment_prediction(dict_key, md, train_features,trainlabels, test_features, testlabels):
    if md == "NB":
        model = GaussianNB()
    elif md == "LR":
        model = LR()
    clf = model.fit(train_features, np.array(trainlabels))
    pred_labels = clf.predict(test_features).reshape(600,1)
    # print(sum(abs(np.array(clf.predict(test_features)).reshape(600,1) - np.array(testlabels))))
    print("score: " + str(clf.score(test_features, np.array(testlabels))))

    cm = confusion_matrix(testlabels, pred_labels, labels=[1,0])
    print("confusion matrix:")
    print("      pred_1,pred_0")
    print("true_1  " + str(cm[0,:]))
    print("true_0  " + str(cm[1,:]))
    print("TPR=" + str(cm[0,0]/300))
    print("FPR=" + str(cm[1,0]/300))

    if md == "LR":
        sorted_w_ind = np.argsort(-abs(clf.coef_[0,:]))
        print()
        for i in sorted_w_ind[:10]:
            print(dict_key[i], clf.coef_[0,i])


# In[370]:


## (g) N-gram model
def n_gram(n):
    trainset, testset, trainlabels, testlabels = generate_trainset_testset()
    
    # build dictionary of n-grams
    dict_key = []
    for sentence in trainset:
        if len(sentence) >= n:
            for i in range(len(sentence) - n + 1):
                dict_key.append(" ".join(sentence[i:i+n]))
    dict_key = list(set(dict_key))

    # extract features
    train_features = np.zeros((len(trainset), len(dict_key)))
    for i in range(len(trainset)):
        sentence = trainset[i]
        if len(sentence) >= n:
            for j in range(len(sentence) - n + 1):
                feat = " ".join(sentence[j:j+n])
                train_features[i, dict_key.index(feat)] = train_features[i, dict_key.index(feat)] + 1
    
    test_features = np.zeros((len(testset), len(dict_key)))
    for i in range(len(testset)):
        sentence = testset[i]
        if len(sentence) >= n:
            for j in range(len(sentence) - n + 1):
                feat = " ".join(sentence[j:j+n])
                if feat in dict_key:
                    test_features[i, dict_key.index(feat)] = test_features[i, dict_key.index(feat)] + 1
    
    return dict_key, train_features, test_features, trainlabels, testlabels


# In[371]:


## (h) PCA for bag of words model
def PCA_bow(features, n):
    C = np.dot(features.T, features) / (np.shape(features)[0] - 1)
    eig_vals, eig_vecs = np.linalg.eig(C)
    print(np.shape(eig_vecs[:,:n]))
    pca_features = np.dot(features, eig_vecs[:,:n])   
    return pca_features


# In[372]:


dict_key, train_features, test_features, trainlabels, testlabels = bag_of_words()
pca_train_features = PCA_bow(train_features, 10)
pca_test_features = PCA_bow(test_features, 10)
sentiment_prediction(dict_key, "LR", pca_train_features,trainlabels, pca_test_features, testlabels)


# In[355]:


pca_train_features = PCA_bow(train_features, 50)
pca_test_features = PCA_bow(test_features, 50)
sentiment_prediction(dict_key, "LR", pca_train_features,trainlabels, pca_test_features, testlabels)


# In[356]:


pca_train_features = PCA_bow(train_features, 100)
pca_test_features = PCA_bow(test_features, 100)
sentiment_prediction(dict_key, "LR", pca_train_features,trainlabels, pca_test_features, testlabels)


# In[354]:


# dict_key, train_features, test_features, trainlabels, testlabels = n_gram(2)
# dict_key, trainset, testset, trainlabels, testlabels = bag_of_words()
sentiment_prediction(dict_key, "LR", train_features,trainlabels, test_features, testlabels)


# In[362]:


from sklearn.decomposition import PCA
pca_train_features = PCA(n_components=100).fit_transform(train_features)
pca_test_features = PCA(n_components=100).fit_transform(test_features)
sentiment_prediction(dict_key, "LR", pca_train_features,trainlabels, pca_test_features, testlabels)


# In[373]:


dict_key, train_features, test_features, trainlabels, testlabels = n_gram(2)
# dict_key, trainset, testset, trainlabels, testlabels = bag_of_words()
sentiment_prediction(dict_key, "LR", train_features,trainlabels, test_features, testlabels)

