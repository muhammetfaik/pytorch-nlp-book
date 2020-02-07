from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns


vocab = {time, fruit, flies, like, a, an, arrow, banana}
corpus = ['Time flies flies like an arrow.',
'Fruit flies like a banana.']
one_hot_vectorizer = CountVectorizer(binary=True)
one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
sns.heatmap(one_hot, annot=True,
cbar=False, xticklabels=vocab,
yticklabels=['Sentence 2'])

