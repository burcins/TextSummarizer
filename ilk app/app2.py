import gradio as gr
import wikipedia
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.stem import WordNetLemmatizer
from heapq import nlargest
import warnings


warnings.filterwarnings("ignore")

def get_wiki_summary(inp):
    text = wikipedia.summary(inp)
    print('\033[1m' + "Original Text Fetched from Wikipedia" + '\033[0m')

    print(text)

    stopwords = list(STOP_WORDS)

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(str(token).lower()) for token in nltk.word_tokenize(text) if str(token) not in punctuation and str(token).lower() not in stopwords and len(token) >1]
    word_counts = {}

    for token in tokens:
        if token in word_counts.keys():
            word_counts[token] += 1
        else:
            word_counts[token] = 1

        

    sentence_scores = {}

    for sentence in nltk.sent_tokenize(text):
        sentence_scores[sentence] = 0
        for wrd in nltk.word_tokenize(sentence):
            if lemmatizer.lemmatize(str(wrd).lower()) in word_counts.keys():
                sentence_scores[sentence] += word_counts[lemmatizer.lemmatize(str(wrd).lower())]

    summary_length = 0

    if len(sentence_scores) > 5 :
        summary_length = int(len(sentence_scores)*0.20)
    else:
        summary_length = int(len(sentence_scores)*0.50)
        
    summary = str()

    for sentence in nltk.sent_tokenize(text):
        for i in range(0,summary_length):
            if str(sentence).find(str(nlargest(summary_length, sentence_scores, key = sentence_scores.get)[i])) == 0:
                summary += str(sentence).replace('\n','')
                summary += ' '
                
                
    print('\033[1m' + "Summarized Text" + '\033[0m')

    return summary

if __name__ == '__main__':
    gr.Interface(fn=get_wiki_summary, inputs=gr.inputs.Textbox(label="Requested Topic from Wikipedia    :   "), outputs="text").launch(inline=False, share=True)
