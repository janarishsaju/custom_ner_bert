


**Janarish Saju C**

AI/ML Engineer

Named Entity Recognition

**10th December 2022**
# **OVERVIEW**
Named-entity recognition (NER) is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text
# **GOALS** 
1. EDA (Exploratory Data Analysis)
1. How many solutions can you think of and why are you choosing your version of the solution?
1. Error Analysis
# **COLDSTART ANALYSIS**
There  are several NER libraries for implementations using Python.

1. **BERT: <https://huggingface.co/>**
1. **spaCy: [https://spacy.io/usage/linguistic-features ](https://spacy.io/usage/linguistic-features)**
1. NLTK: [https://www.nltk.org/book/ch07.html ](https://www.nltk.org/book/ch07.html)
1. Stanford CoreNLP: <https://stanfordnlp.github.io/CoreNLP/>
1. Polyglot: [http://polyglot.readthedocs.io/en/latest/NamedEntityRecognition.html ](http://polyglot.readthedocs.io/en/latest/NamedEntityRecognition.html)
1. Apache OpenNLP: <https://opennlp.apache.org/>

Among those I go with the  first two **BERT and spaCy,** these are the most favorite of mine

There exists other popular frameworks such ***OpenAI, NLTK and Stanford CoreNLP*** as well.
## **Advantages & Disadvantages**


|**Methods**|**Advantages**|**Disadvantages**|
| :- | :- | :- |
|BERT|<p>Time Efficiency</p><p>Pretrained with large datasets</p><p>Knowledge Transfer (Transfer Learning)</p>|<p>Computationally Cost Expensive</p><p>Not so good for Domain Based NER</p>|
|spaCy|<p>Faster as it is built with C++ in low level</p><p>Good for Domain Based NER</p>|<p>Require more training data</p><p>More Complexity in Data Structures</p>|
|NLTK|Good for Base Level Analysis|Requires Implementations from Scratch|
|Stanford Core NLP|<p></p><p>No Idea, Since I never experienced these tools for any of my former projects</p>|
|OpenNLP||

# **EXPLORATORY DATA ANALYSIS**
*(\*written here all the necessary steps carried out in the shared code)*
## ***1. Text Statistics***
- Average Word Length
- Histogram and Bar Charts
- Most Influenced Words
- Most Influential Entities
- Most Dominant Entity Labels
- N-gram Exploration

*(\*From the above analysis we can see the corpus influences more about the Social Networks, Twitter, Facebook, Youtube, Please see the Colab Notebook)*
## ***2. Outlier Analysis*** 
- Found some outliers/uncommon behaviors when relating the entities with special characters. *(See the Colab Notebook)*
## ***3. Assumptions & Thought***
- As discussed in Outlier Analysis. Need to take care of the following
  - Mismatch in name taggings
  - Special Characters + Entities Overlapping

`                                                                                    `*(See the Colab Notebook)*

- Data Augmentation
  - An effective approach if we have lesser training samples
- Data Annotation
  - Better data annotation pipeline tool avoid faulty datasets from Human Level
- Ensembled Algorithms
  - Effectively utilizing ensemble algorithms archives better performance.

*(\*Discussed in last section)*
# **IMPLEMENTATIONS**
*(\*written here all the necessary steps carried out in the shared code)*
## **BERT NER**
## **1. Data Preprocessing Steps**
- Data read/import
- Handle data encoding issue
- Data conversion as per model requirements
- Data partition
## **2. Feature Engineering Steps**
- Unique input and output label features
- Encode the labels to Numeric representation
- Tokenize and embed the datasets
## **3. Model Initialization**
- Initialize the BERT model
- Define the Task Name
- Define the Tokenizer method
## **4. Hyper Parameter Turning**
- The following parameters were used
  - evaluation\_strategy = "epoch",
  - learning\_rate=1e-4,
  - per\_device\_train\_batch\_size=16,
  - per\_device\_eval\_batch\_size=16,
  - num\_train\_epochs=6,
  - weight\_decay=1e-5,
## **5. Train the Model**
- Train the model with the below metrics
  - Train\_dataset,
  - Eval\_dataset,
  - Tokenizer,
  - Compute\_metrics
## **6. Evaluate the Model**
- Evaluation done based on the 20 percent of data extracted for validation purposes from the training data.
## **7. Error Analysis**
- Accuracy on Validation Dataset
- Confusion Matrix / Cross Table
- Precision, Recall, F-Measure
- K Fold Cross Validation can be applied for advanced analysis.

*(\* It is explicitly seen that the entity **I-Location, B-Location and O** have more mismatches. We should analyze and look deep into those entities. Please see the Colab Notebook)*
## **8. Prediction Module**
- Read the test data from disk
- Handle Encoding and Alignment issues
- Data Conversion
- Feed the Converted Test Data to the fine turned model and Get Predictions
- Get ***Label Predictions*** using ArgMax function
- Get ***Probabilistic Prediction Scores*** using  SoftMax function
- Store every results in a DataFrame
## **9. Export the Results**
- Export test results in text file separated by "\t"
# **CONCLUSION**
- BERT has the advantage over other Machine Learning and Deep Learning models.
- As it is a transformer technique pretrained with huge datasets.
- And it save us a lot of time for training
- Although it has a disadvantage, Heavier BERT model is computationally expensive
# **ADDITIONAL RESEARCH & FUTURE EXPERIMENTS**
## **More Ideas for Making Stronger NER Formatting Models**
## ***1. Replace Pretrained embeddings with Contextual Embeddings such as BERT or ELMo***
` `[*https://github.com/huggingface/pytorch-pretrained-BERT*](https://github.com/huggingface/pytorch-pretrained-BERT)
## ***2. Combine Embeddings with Character Level, CNNs or RNNs for handling unseen words***
*https://eli.thegreenplace.net/2018/understanding-how-to-implement-a-character-based-rnn-language-model/*
## ***3. Combine Linguistic Features with your Embeddings***
[*https://spacy.io/usage/linguistic-features*](https://spacy.io/usage/linguistic-features)
## ***4. Add Self-Attention Mechanisms to your RNN***
[*https://towardsdatascience.com/deep-learning-for-named-entity-recognition-2-implementing-the-state-of-the-art-bidirectional-lstm-4603491087f1*](https://towardsdatascience.com/deep-learning-for-named-entity-recognition-2-implementing-the-state-of-the-art-bidirectional-lstm-4603491087f1)
##
## **REFERENCES & URLs**
## **Online Sources:**

1. <https://github.com/dmoonat/Named-Entity-Recognition/blob/main/Fine_tune_NER.ipynb>
1. https://medium.com/@andrewmarmon/fine-tuned-named-entity-recognition-with-hugging-face-bert-d51d4cb3d7b5
1. <https://pub.towardsai.net/top-5-approaches-to-named-entity-recognition-ner-in-2022-38afdf022bf1>
1. https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools
##

## **Google Colab:**
**BERT NER:** 

<https://colab.research.google.com/drive/19qNHO9E618JP6IVeY6DtvlPx3Bwplhiw?authuser=2#scrollTo=ISV1dQxoKrbk>

**Exploratory Data Analysis:**

<https://colab.research.google.com/drive/16-J6qeLLmEJf0lerHKS1w0rmztAq_nIg?authuser=2#scrollTo=iu65GkEF8ueH>

## **GitHub:**
https://github.com/janarishsaju/ner\_bert.git
