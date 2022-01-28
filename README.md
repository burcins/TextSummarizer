# TextSummarizer


Summarizing texts is one of the sub-fields of NLP, and as the name suggests, it is aims to summarize relatively longer texts or articles. In this notebook my plan is to work on this task by using different approaches. I used Wikipedia to find texts by using wikipedia package. There are two main approaches to summarize text in literature, which are Extractive and Abstractive.

Extractive text summarization serve most important sentences as the summary of the article based on occurance counts of words.
Abstractive method generates its own sentences from text by using transformers (decoder -encoders), For this approaches there are several pre-trained models to use. 

To clarify, let's suppose we have a text consisting of 5 different sentences from A to E.<br>
A.B.C.D.E.<br>
Extractive == > B.D.<br>
Abstractive == > F.G.<br>

I was planning to implement both approaches to summarize by using several methods. 
For the extractive approach, I tried 3 different methods, first one is lemmitizing the words for reaching the roots of the words, so ı can count the word occurences. After obtaining word counts, I scored all sentences by summing the scores of the words contained in the sentence.  In the second method I used the same approach, but this  time by skimming the words to count word roots. And the last extractive method that I used is to calculate TF-IDF scores of the texts by calculating from 1-gram to 3-ngram word counts. To calculate IDF I treated each sentence as different documents. I deployed 2 extractive methods to the huggingface spaces page, you can follow this link below;

<p align="center">
  <b>Model Link:</b><br>
  <a href="https://huggingface.co/spaces/Burcin/ExtractiveSummarizer">TRY TO SUMMARIZE</a> 
</p>

For abstractive approach, my local machine did not allow me to import and use pre-trained models, but I still kept their scripts in the notebook without outputs to show the method. And I am planning to deploy it for trials. 

But for the extractive part, for making trials with different contents, 

