import pdb, os, torch, sklearn, pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
		
def loadTrainingData(langTokens, dataLocation, trainingLines = 1000):
	langDict = {}
	for lang in langTokens:
		with open(dataLocation + lang, encoding="utf8") as f:
			langDict[lang] = [next(f).strip() for x in range(trainingLines)]
	
	for key, phraseList in langDict.items():
		for i, line in enumerate(phraseList):
			langDict[key][i] = [line, key]
		if "mixList" in locals():
			mixList += langDict[key]
		else:
			mixList = langDict[key]
			
	df = pandas.DataFrame(mixList, columns = ["phrase" , "token"])
	df = sklearn.utils.shuffle(df)
	return df
	
def trainModel(trainingData): # vectorizer and model from https://github.com/danielv775/Natural-Language-Identification-Graduate-Project
	tfidf_vect = TfidfVectorizer(analyzer="char", ngram_range=(1,3))
	model = sklearn.naive_bayes.MultinomialNB()
	text_clf = Pipeline([("tfidf", tfidf_vect), ("clf", model),])
	xTrain, yTrain, xTest, yTest = splitData(trainingData, .2)
	
	text_clf.fit(xTrain, yTrain)
	
	predictions = text_clf.predict(xTest)
	print("Tested model accuracy of " + str(accuracy_score(yTest, predictions)))
	
	return text_clf
	
def splitData(df, testPct):
	xTrain = df["phrase"]
	yTrain = df["token"]
	
	testNum = len(xTrain) - int(testPct * len(xTrain))
	xTest = xTrain[testNum:len(xTrain)]
	xTrain = xTrain[0:testNum]
	yTest = yTrain[testNum:len(yTrain)]
	yTrain = yTrain[0:testNum]
	
	return xTrain, yTrain, xTest, yTest
	
def testModel(text_clf, testDataLocation, predictionLocation):
	with open(testDataLocation, encoding="utf8") as f:
		content = [x.strip() for x in f.readlines()]
	
	testData = pandas.Series(content)
	preds = text_clf.predict(testData)
	
	with open(predictionLocation, 'w') as f:
		for item in preds:
			f.write("%s\n" % item)

def task1():
	try:
		clf = joblib.load("langid/task1CLF.pkl")
	except FileNotFoundError:
		trainingData = loadTrainingData(["en" , "es", "pt"], "trainingData/task1/data.", 20000)
		clf = trainModel(trainingData)
		joblib.dump(clf, "langid/task1CLF.pkl")
	
	testModel(clf, "langid/langid.test", "langid/task1Preds")
	
def task2():
	try:
		clf = joblib.load("langid/task2CLF.pkl")
	except FileNotFoundError:
		trainingData = loadTrainingData(["pt-br" , "pt-pt"], "trainingData/task2/data.", 20000)
		clf = trainModel(trainingData)
		joblib.dump(clf, "langid/task2CLF.pkl")
	
	testModel(clf, "langid/langid-variants.test", "langid/task2Preds")