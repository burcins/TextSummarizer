# TextSummarizer


Summarizing texts is one of the sub-fields of NLP, and as the name suggests, it is aims to summarize relatively longer texts or articles. In this notebook my plan is to work on this task by using different approaches. I used Wikipedia to find texts by using wikipedia package. There are two main approaches to summarize text in literature, which are Extractive and Abstractive.

Extractive text summarization serve most important sentences as the summary of the article based on occurance counts of words.
Abstractive method generates its own sentences from text by using transformers (decoder -encoders), For this approaches there are several pre-trained models to use. In this project I will try

To clarify, let's suppose we have a text consisting of 5 different sentences from A to E.
A.B.C.D.E.
Extractive == > B.D.
Abstractive == > F.G.

I was planning to implement both approaches to summarize by using several methods. However my local machine did not allow me to import and use pre-trained models, but I still kept their scripts without outputs to show the method. 

For making trials with different contents, I deployed the extractive summarization model to the huggingface spaces page, you can follow this link below;

[TRY TO SUMMARIZE](https://huggingface.co/spaces/Burcin/ExtractiveSummarizer)


<p align="center">
  <b>Model Link:</b><br>
  <a href="https://huggingface.co/spaces/Burcin/ExtractiveSummarizer">TRY TO SUMMARIZE</a> 
</p>
