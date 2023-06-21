from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from config4FinalTfIdfAsurion import FLAGS
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from nltk import ngrams, FreqDist
from matplotlib import pyplot,rcParams
rcParams.update({'figure.autolayout': True})

def preProc(trainTsvPath, testTsvPath):
    with open(trainTsvPath, "r", encoding="utf-8", newline='') as trainTsvFile:
        tsvTrainReader = csv.reader(trainTsvFile, delimiter='\t')
        X = []
        Y = []
        # the list of tokens is needed for eda
        tokens = []
        for t in tsvTrainReader:
            tokenListPerLine = t[1].split(",")
        # join the comma-separated tokens for future use by tfidf vectorizer
            X.append(" ".join(tokenListPerLine))
            Y.append(t[0])
            tokens += tokenListPerLine
    with open(testTsvPath, "r", encoding="utf-8", newline='') as testTsvFile:
        # change separator to "," as test.tsv does not have tabs in it
        csvTestReader = csv.reader(testTsvFile)
        XTest = []
        for c in csvTestReader:
            XTest.append(" ".join(c))
            tokens += c
    return X, Y, XTest, tokens

def eda(Y,tokens,maxN, plotFreqSize, plotWidth, plotHeight):
    pyplot.tight_layout()
    labelFreqPlot =pyplot.figure(figsize=(plotWidth, plotHeight))
    # check whether we get a binary classification or multiclass classification problem.
    # also for checking whether the classes are balanced.
    pyplot.hist(Y)
    pyplot.gca().set(title='Label Frequency Histogram', xlabel='Label', ylabel='Frequency');
    pyplot.show()
    labelFreqPlot.savefig("labelFreqPlot.png")
    # check whether some words are repeated such that tfidf vectors would be a good option for representing the text data
    # check whether some n-grams are repeated sufficient times such that including them as features in vector representations would be considered a good idea
    for n in range(1,maxN+1):
        ngramFreqPlot = pyplot.figure(figsize=(plotWidth, plotHeight))
        ngramFreq = FreqDist(ngrams(tokens,n))
        ngramFreq.plot(plotFreqSize)
        pyplot.gca().set(title=str(n) +'-gram Frequency Graph', xlabel=str(n)+'-gram', ylabel='Frequency');
        pyplot.show()
        ngramFreqPlot.savefig(str(n)+"gramFreqPlot.png")
    return()

def trainNValid(X,Y,maxN, split, seed):
    XTrain, XVal, YTrain, YVal = train_test_split(X, Y, test_size=split, random_state=seed)
    vectorizer = TfidfVectorizer(ngram_range=(1, maxN))
    vectorizer.fit(XTrain)
    XTrainVec = vectorizer.transform(XTrain)
    XValVec = vectorizer.transform(XVal)
    classifier = KNeighborsClassifier()
    classifier.fit(XTrainVec, YTrain)
    YPred = classifier.predict(XValVec)
    return classification_report(YVal, YPred)

def cvNPredict(X,Y,XTest,split,maxN):
    # hyper-parameter tuning and cross-validation
    params =  {
    'n_neighbors': (1,21, 2),
    'leaf_size': (20,40,1),
    'p': (1,2),
    'weights': ('uniform', 'distance')  }
    gridObj = GridSearchCV(KNeighborsClassifier(), params, cv=int(1/split),
                           scoring='f1_macro', n_jobs=-1, verbose=5)
    vectorizer = TfidfVectorizer(ngram_range=(1, maxN))
    vectorizer.fit(X)
    XVec = vectorizer.transform(X)
    XTestVec= vectorizer.transform(XTest)
    gridObj.fit(XVec, Y)
    means = gridObj.cv_results_['mean_test_score']
    print ("cross validation results")
    for mean, params in zip(means, gridObj.cv_results_['params']):
        print("%0.3f for %r"
              % (mean, params))
    # train a knn classifier with the best parameters from grid search and cross-validation
    classifier = KNeighborsClassifier()
    classifier.set_params(**gridObj.best_params_)
    classifier.fit(XVec, Y)
    return (classifier.predict(XTestVec))




if __name__ == '__main__':
    X, Y, XTest, tokens = preProc(FLAGS.trainTsvPath, FLAGS.testTsvPath)
    eda(Y,tokens,FLAGS.maxN, FLAGS.plotFreqSize, FLAGS.plotWidth, FLAGS.plotHeight)
    print ("classification report before cross validation")
    print(trainNValid(X,Y,FLAGS.maxN, FLAGS.split, FLAGS.seed))
    with open(FLAGS.predictTsvPath, 'w', newline='') as predictTsvFile:
        predictWriter = csv.writer(predictTsvFile, delimiter='\t')
        for y,x in zip(cvNPredict(X, Y, XTest, FLAGS.split, FLAGS.maxN), XTest):
            predictWriter.writerow([y, ",".join(x.split())])
