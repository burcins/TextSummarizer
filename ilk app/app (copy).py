import gradio as gr
from gradio.mix import Parallel, Series
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
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import PegasusForConditionalGeneration

warnings.filterwarnings("ignore")

def get_wiki_original_text(inp):
    text = wikipedia.summary(inp)
    return text


def  get_wiki_summary_by_pegasus(inp):
    text = wikipedia.summary(inp)
    tokenizer = gr.Interface.load("huggingface/google/pegasus-xsum") 
    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
    summary = model.generate(**tokens)
    return tokenizer.decode(summary)
    


def get_wiki_summary_by_lem(inp):
    text = wikipedia.summary(inp)

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


def get_wiki_summary_by_tfidf(inp):
    text = wikipedia.summary(inp)

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3))
    
    all_sentences = [str(sent) for sent in nltk.sent_tokenize(text)]
    sentence_vectors = tfidf_vectorizer.fit_transform(all_sentences)
    
    sentence_scores_vector = np.hstack(np.array(sentence_vectors.sum(axis=1)))

    sentence_scores = dict(zip(all_sentences, sentence_scores_vector))

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
                
                
    return summary



desc =  """This interface allows you to summarize Wikipedia contents. Only requirement is to write the topic and it collects content by fetching from Wikipedia. For summarization this model uses 2 different extractive summarization methods and the number of sentences in the output depends on the length of the original text."""


sample = [['Europe'],['Great Depression'],['Crocodile Dundee']]


iface = Parallel(gr.Interface(fn=get_wiki_original_text, inputs=gr.inputs.Textbox(label="Text"), outputs="text", description='Original Text'),
                 gr.Interface(fn=get_wiki_summary_by_lem, inputs=gr.inputs.Textbox(label="Text"), outputs="text", description='Summary 1'),
                 gr.Interface(fn=get_wiki_summary_by_tfidf, inputs=gr.inputs.Textbox(label="Text"), outputs="text", description='Summary 2'),
                 gr.Interface(fn=get_wiki_summary_by_pegasus, inputs=gr.inputs.Textbox(label="Text"), outputs="text", description='Summary 3'),
                 title= 'Text Summarizer', 
                 description = desc,
                 examples=sample, 
                 inputs = gr.inputs.Textbox(label="Text"))

iface.launch(inline = False)