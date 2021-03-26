# Practical Natural Language Processing

Notes and tests for the book Practical Natural Language Processing (O'Reilly)

## Part I. Foundations

### NLP Pipeline

Data Acquisition

Text Cleaning

	- Unicode normalization
	- Spell correction
		- Keyboard errors (fat finger)
		- OCR errors
		- Which character to replace first?
			- keyboard
				- inner key's first
				- ??? (statistically)
			- OCR
				- ?? (statistically, depending on the source docs)


Pre-processing

	- Text -> [Sentence Tokenization] -> Sentences
	- Sentence
		- Lowercasting
		- Removal punction
		- Stemming
		- Lemmatization
		- POS tagging
		- Parsing (Syntax resolution)
		- Correference resolution

Text representation & Feature Engineering

	- ML / NLP interpretability, can be broken easily with manual action.
	- Deep Learning non interpretable.

Modeling

	- Spacy's rule-based matching: https://explosion.ai/demos/matcher
	- ML
		- Stacking & Ensemble
		- Transfer learning (pre-trained models)

Evaluation
	- Intrinsic (ML & AI concepts, e.g. precision, recall, F1 score...)
	- Extrinsic (does the model solve the business problem?)

Deployment
	- Web service
	- Batch processing

Monitoring and Model Updating
	- 

### Text representation

Vector Space Models
	- A way to represent proximity between words and their relations (cos / Euclidean distance)
	- |V|: number of words in the vocabulary
	- |NG|: number of N-grams in the vocabulary

Problem: Out of Vocabulary (OOV)
	- How to handle a word not previously seen?

One-hot encoding
	- Size of text = N * |V|, where N is the number of words in the text.

Bag of Words (BoW) (Size |V|) : 
	- Binary: a text's vector contains a `1` in each position of the word. 
	- Counting aparitions: a text's vector contains the number of aparitions.

Bag of N-grams (Size |NG|)

TF-IDF: 
	- TF(t, d) = # appearances of `t` in `d` / # terms in `d`
	- IDF (t) = log (# docs / # docs with t)
	- TF-IDF(t, d) = TF(t, d) * IDF(t)
	- Use this score instead in BoW models

Problems: 
	- Discrete representations
	- High dimensional
	- Cannot handle OOV problem. 

Distributional vs. Distributed Repr

- One-hot, BoW, BoN, etc.	- Word2vec, Doc2Vec
- High dimensionality 		- Low dimensions
- Sparse vectors			- Dense vectors

Word2Vec
 
Parameters to decide (kind of art... )
	- Dimensionality: 50 - 500
	- Context window
	- CBOW vs. Skipgram

Doc2Vec: 
	- SUM or AVG individual word embeddings
	- Gensim generates a vector for a full text

OOV approach: 
	- Remove words not previously seen in the training, which can lead to poor performance if overlap is <80%.


How to build your own word embeddings: 

```python
from gensim.models import Word2Vec
corpus = [
	['dog','bites','man'], 
	["man", "bites" ,"dog"],
	["dog","eats","meat"],
	["man", "eats","food"]
]

# Training the CBOW model
model_cbow = Word2Vec(corpus, min_count=1,sg=0) #using CBOW Architecture for trainnig

# Training the skipgram model
model_skipgram = Word2Vec(corpus, min_count=1,sg=1)#using skipGram Architecture for training

```


Universal text representations

- Building language models: predicting the next word for a given sequence of words
	- Recurrent Neural Networks
	- BERT
	- ELMo

	- Language Models with transformers: https://arxiv.org/pdf/1904.09408.pdf
	- Embeddings in NLP: https://www.aclweb.org/anthology/2020.coling-tutorials.2.pdf

Visualize embeddings 
	- t-SNE (dimension reduction)
	- TensorBoard: https://www.tensorflow.org/tutorials/text/word_embeddings


## Part II. Essentials 

### Text Classification

- Types: binary, multiclass and multilabel 
- Applications: Content classification (search engines), E-commerce, Customer support, etc. 


Poor performance possible reasons: 

- Sparse feature vectors 
	-> reduce dimensionality
- Unbalanced (skewed) classes
	-> oversample lower categories
	-> undersample higher categories
	-> Imbalace-Learn Python lib
- Use different classifiers: Naive Bayes, Logistic Regression, Support Vector Machines, 
- Improve pre-processing pipeline & feature extraction
	-> BoW
	-> Embeddings (pre-trained vs. custom-trained)	
		-> compare vocabulary overlap, if >80%, use pre-trained. 
	-> FastText
		!! Model size can be >500MB
- Tune classifier's parameters & hyperparams


Algorithms
- Word embeddings
- Subword embeddings (n-gram embeddings), e.g. FastText
- Document Embeddings (doc2vec)
- Deep Learning
	- Pretrain models vs. Trained embeddings
		- Pretrained model input layer: Embedding(..., embeddings_initializer=Constant(embedding_matrix), ..., trainable=False)
		- Trained embeddings input layer: Embedding(MAX_NUM_WORDS, 128)


Interpreting ML blackboxes

- LIME (Local Interpretable Model-agnostic Explanations)
- https://github.com/marcotcr/lime
- https://github.com/practical-nlp/practical-nlp/blob/master/Ch4/09_Lime_RNN.ipynb


Training with no data
- Use Snorkel to generate data with "weak supervision": https://www.snorkel.org/
- Crowdsourcing (captcha, aws turk, etc.)


Adapting to new domains
- Active learning
	- Train the model with available data
	- Use it to predict new data
	- Unsure predictions, ask for human help
	- Retrain the model
- Use Prodigy for active learning https://prodi.gy/

- Transfer learning
	- Start with a large pretrained language model of the source domain (e.g. Wikipedia)
	- Fine-tune this model with target language's unlabeled data
	- Train a classifier on the labeled target domain data by extracting feature representations from the fine-tuned language model.
- Use ULMFiT (https://nlp.fast.ai/classification/2018/05/15/introducing-ulmfit.html)


Practical advice
- Establish strong baselines, get a MVP fast with simple, but robust techniques. State-of-art techniques might only improve a little bit. 
- Balance training data: collecting more data, resampling, undersample, weight balancing. 
- Combine models and human rules
- Make it work, make it better. Get a simple model first, deploy it and then focus on improvement when the whole pipeline is working. 
- Use the wisdom of many. Consider combining the results of multiple classifiers. 


### Information extraction

Main tasks: 

- Basic 
	- Keyword / Keyphrase extraction (KPE)
		- Textacy, TextRank
	- Named Entity Recognition (NER)
		- gazzeeter (look-up table)
		- NLP regex (stamford, spacy)
		- Sequence Classification (conditional random fields)
			- BIO notation (Begin, Inside, Other)
		- Existing libraries for NER
			- Custom entities -> Active learning -> Augmented data
		- Scratch NER system -> Annotated data set
	- Named Entity Disambiguation and Linking (Apple as Apple, Inc. and not other company with apple.)
		- NED
		- NEL
			- Recommended to use off-the-shelf / pay-per-use services
			- https://www.dbpedia-spotlight.org/
	- Relation Extraction
		- What is a relation? -> Domain specific
		- Train two models for given two entities of the text
			- Whether two entities in a text are related (binary classification)
			- What is the relation between them (multiclass classification)
			- https://allennlp.org/
		- Hard to develop from scratch. Rely on Watson API
- Advanced:
	- Event extraction
		- Supervised training to build a Sequence tagging, multilevel classifier model. Identify various events over time periods and connect them, building a temporally ordered graph. 
		- No general solutions
		- Start with rule-based systems + weak supervision, then move to ML approaches.
	- Temporal information extraction
		- https://duckling.wit.ai/
		- https://nlp.stanford.edu/software/sutime.shtml
		- http://natty.joestelmach.com/
		- https://pypi.org/project/parsedatetime/
		- https://github.com/mojombo/chronic
	- Template filling
		- Model text generation for "slot-filling" problem.
		- Having pre-defined templates, create a two-step model: 
			- Identifying if a model follows a template
			- Identifying the slots of the pattern

### Chatbots

- Taxonomy
	- Exact answer or FAQ bot with limited conversations
	- Flow-based bot
	- Open-ended bot

- Categories
	- Goal-oriented
	- Chitchats

- NLU tasks:
	- Sentiment detection
	- Named entity extraction
	- Correference resolution
	- Dialog act classification
		- How a UQ plays a role in the context of dialog
		- Example: identify a "yes/no" question

- Terminology
	- Dialog act or intent: main goal of the UQ
	- Slot or entity
	- Dialog state or context

- Response generation
	- Fixed responses
	- Use of templates
	- Automatic generation

- Alternative pipelines
	- End-to-end: train seq2seq model saves complexity in having to maintain multiple modules. 
	- Goal-oriented + seq2seq generation -> reinforcement learning

- Rasa NLU
	- https://rasa.com/


### Topics in Brief 


#### Search 

- Types:
	- Generic search engines, e.g. Google
	- Enterprise search engines

- Components
	- Crawler
	- Indexer
	- Searcher
	- Feedback

- NLP pipeline
	- Crawling / Data acquisition
	- Text normalization: tokenization, lowercase, stop words, stemming, etc... 
	- Indexing
		- TF-IDF
		- BERT -> https://www.elastic.co/es/blog/text-similarity-search-with-vectors-in-elasticsearch
	- Query processing and execution
		- Same text normalization
	- Feedback and ranking

#### Topic modelling

- Extract the "key" words (aka topics) present in a text corpus without prior knowledge about it.

- Algorithms
	- Latent Dirichlet allocation (LDA) (most used in practice)
	- Latent Semantic Analysis (LSA) 
	- Probabilistic latent semantic analysis (PLSA)

- Number of topics "k" is a parameter of the algorithm


#### Text summarization

- Create a coherent summary that captues the key ideas in a larger text. 

- Common tasks: 
	- Extractive vs. abstractive: selecting important sentences vs. generating an abstract
	- Query-focused vs. query-independent
	- Single document vs. multi-document

- Most common one: single-doc + query-independent + extractive
	- Lib: https://pypi.org/project/sumy/
	- https://rare-technologies.com/text-summarization-with-gensim/


#### Recommender Systems for Textual Data

- Types: 
	- Collaborative filtering: netflix example
	- Content-based filtering

- Content-based recommenders: 
	- Topic based
	- Doc2vec

- Evaluation using A/B tests


#### Machine Translation

- informal to grammatically-correct language can be understood as an MT problem. 
	- "am gud" -> "I am good"

- Don't build your own MT system, instead use APIs / pay-per-use services
	- Keep a translation memory cache to save requests


#### Question-answering Systems

- What type of question?
- What type of answer? 
- Extract information from candidate documents

- Approaches
	- ML: index question - answers using embeddings. Search by similarity
	- knowledge-based QA: map UQ to the database. E.g. watson


## Part III. Applied 

### Social media

Text in social media is different from other sources. 

Applications

- Trending topic detection
- Opinion mining
- Sentiment detection
- Rumor/fake news detection
- Adult content filtering
- Customer support

"Canonical" text satisfies: 
- Single language
- Single script
- Formal
- Gramatically correct
- Few or no spelling errors
- Mostly text-like (non emojis, images, etc.)

Social media text contains: 
- extrem brevity
- nonstandard spellings
- hashtags
- emojis
- new words
- acrynyms
- code-mixing, transliteration

We call it a new language, Social Media Text Data (SMTD)

Difficulties
- no grammar: inconsistence/absence of punctuation, emojis, incorrect spelling, character repetition, abbreviations make the steps of tokenization, POS tagging, sentence boundaries hard to accomplish
- non-standard spelling: e.g. tomorrow -> 2mrw, tmw, tomarrow, etc.
- multilingual: multiple languages are mixed in the same sentences
- transliteration: using the letters from another alphabet to represent the phonetics of a different one. 
- special characters: emojis, gifs, non-ascii, hashtags, etc. 
- ever-evolving vocabulary: new words are added very quickly in SMTD. This problem is known as Out Of Vocabulary problem in NLP systems. In 2014 - 2016 there were 10% of new words every month. 
- length of text: shorter texts to save space and type faster. 
- noisy data: social media contains ads, spam, promoted content, etc. This unwanted data has to be filtered before. 

#### NLP Applications for social media

##### Word cloud

It is an image containing the most important (frequent) words in a given set of texts. 
Pipeline: 
- Tokenize
- Remove stop words
- Sort remaining words in descending order of frequency
- Take top _k_ words and plot them

[twokenize](https://github.com/aritter/twitter_nlp/blob/master/python/twokenize.py) is a tool specialized in SMTD tokenization. 

##### Trending topics

Use Twitter API to get the trending topics for a given WOEID (where on earth identifier). 

##### Sentiment analysis

Use [TextBlob](https://github.com/sloria/TextBlob) to obtain polarity and subjectivity

#### Preprocessing SMTD

- Remove markup elements (HTML, XML, XHML), use [Beautiful Soup](https://pypi.org/project/beautifulsoup4/)
- Handling non-text data: normalize to UTF-8
- Handling apostrophes: expand them to their full form
- Handling emojis: removing them could imply a loss of meaning, instead replace them with their corresponding text. To do so, use [Demoji](https://pypi.org/project/demoji/)
- Split-joined words: camelcase hashtags
- URL removal: use regex `http\S+`. 
- Nonstandard spellings: e.g. "yesssss" or "sssssh". Use some facts like a letter never repeats 3 times in a row

##### Text representation for SMTD

BoW and TF-IDF do not work well with SMTD. Pretrained models do not work well, due to the differences in the vocabulary. 
Solution:
- Use pretrained models on SMTD like the one drlm Stanford NLP group. 
- Use better tokenizer, e.g. twokenize
- Train your own embeddings

New words are added constantly in SMTD, so models get their performance reduced as production and trining data reduce their overlapping. 

To avoid the permanent OOV problem in SMTD, character n-gram embedding models have been created. When a word is OOV, split it into its n-grams and use the embeddings to form  the embedding of the new word. fastText has already done this. 

##### Social media and Customer support

Companies receive lots of contacts which can be differentiated into two big groups: actionable messages and noise. Building a model to distinguish them out would follow the pipeline: 
- Collect a labeled dataset
- Clean
- Pre-process
- Tokenize
- Represent
- Train 
- Test
- Deploy

#### Memes and Fake News

Memes can be classified into: 
- Content-based: uses content to match with other memes of similar patterns that have already been identified. 
- Behavior-based: a viral post with high initial activity. 

Fake news have some approaches: 
- Fact verification using external sources. This has the problem that some parts of the sentences may be wrong, but we do not know which ones. 
- Classifying fake vs. real: a basic classifier can do badly due to the fact that fake news could hardly be distinguished from real news. 

### E-Commerce and Retail

The following techniques can be A/B tested to see their impact on sales, click-through, time spent on one webpage, etc. 

#### Attribute extraction

- Direct: the information is directly available in the text to be analyzed (usually product description)
	- Regexes with a list of brands and attributes
	- Other possibility are seq2seq models (BIO notation), but they need complex features: 
	 	- Characteristic features: token-based features, letter case, length, character composition...
	 	- Locational features: positional aspect of the token in the input sequence, num tokens before, ratio pos token/total num tokens...
	 	- Contextual features: neighboring tokens, next/prev token, POS tag...
- Derived: the information is not present in the text and must be derived from the context.
	- Text classification to infer external categories

#### Product categorization 

- Process of dividing products into groups. Generally e-commerce has pre-defined broad categories of products. 
- Hierarchical text classification: 
 	- Higher levels -> rule-based models
 	- Lower levels -> ML models
 - APIs from Semantics3, eBay and Lucidworks

#### Product enrichment

- Product title or description are incorrect or incomplete. 
- Templating titles to fit the taxonomy

#### Product deduplication and matching

- For multiple reasons, there might be duplicated products in the database. 
- Locating these products can be done through multiple techniques:
	- Attribute match: overlapping of attributes, string similarity, etc. Abbreviations should be avoided using a good Product enrichment. 
	- Title match: considering n-grams counting, sentence embeddings similarity, Siamese network. 
	- Image match: pixel-to-pixel match, feature map matching, 

### Review analysis

- Star rating and text rating may not be correlated. 
- It is important to understand bad reviews, especially. 
- Important keywords in bad reviews are useful 

#### Aspect-level sentiment analysis

- Aspect: semantically rich, concept-centric collection of words that indicates certain properties or characteristics of the product. They may not be only about the product, but delivery, presentation, return, etc. 
	- Usually defined by the retailer based on the preferred granularity level

- Supervised approach:
	- Assumes all aspects are known
	- If a sentence contains one of the aspect's word assigns the sentence to the aspect. 
	- Sentiment analysis done at sentence level
- Unsupervised approach: 
	- Topic modelling to identify latent topics in the reviews (LDA). Topics -> aspects. Group sentences about topics. 
	- Clustering of sentence representations. Gives better results with fewer data. 

#### Connecting overall ratings to aspects

- Latent rating regression analysis (LARA)
- Assumption: A final rating is a weighted sum of individual aspect-level sentiments. 
- Goal: estimate those weights. Those weights will indicate how important each aspect is for the customer. 

#### Understanding aspects

- Even though we have a general view of the sentiment behind an aspect, if we wanted to perform a general action to improve it we will not know how. 
- If the number of sentences in a given aspect is huge, text summarization can be a good idea. 
- LexRank is an algorithm similar to PageRank, linking sentences by similarity. Then picks the sentence in the center and presents a list of sentences as a summary.  

### Recommendations for e-commerce

- Similarity of products can be defined as content-based or user profile-based. 
- Complements & Substitutes: 
	- Complements are products that are typically bought togethe. 
	- Substitutes are products that are bought instead of the other. 
- Product interrelationships
	- Latent attribute extraction from reviews
	- Product linking


## Healthcare, Finance, and Law

### Healthcare

- Structuring unstructured textual medical data can help automating some parts of the process, e.g. question-answering systems to decrease time to look up relevant patient infomration. 
- Analyzing the effects of medication commented in social media can help detecting and tracking these effects and ensure that a drug is safe. 
- Chatbots can present a friendly interactive way to expert systems that help mental illnesses and other type of patients. 
- It is important to avoid blackboxes in diagnosis predicting models. To achieve this, attention, a concept in deep learning is used to understand what data points and incidents are most important for an outcome. https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/
- It is possible to use signals from social media to track the emotional state and mental balance of both particular individuals and acrros groups of individuals. 
- AWS Amazon Comprehend Medical and BioBERT are two good starting points. 

### Finance and Law

- Finantial sentiment analyzes experts opinion, social media posts, and news to see how the information will affect certain company's stock price. FinBERT.
- Risk assesments can benefit from NLP when processing more textual data about 'borrowers' and detecting incoherences when trying to assess credit risk.
- Legal documents can be improved using NLP for tasks such as case referencing, brief preparation, document review, contract design, background analysis, etc. Most of the advances are protected by patents. 


## Part IV. Bringing it all together

NLP pipeline: 
- Data Acquisition
- Text Cleaning
- Pre-processing
- Text representation & Feature Engineering
- Modeling
- Evaluation
- **Deployment**
- **Monitoring and Model Updating**

Let's focus on the *deployment* and *monitoring & updating the model* steps from the NLP pipeline.

### Questions before starting the NLP project

- What kind of data do we need for training the NLP system? Where do we get this data from? These questions are important at the start and also later as the model matures.
- How much data is available? If it’s not enough, what data augmentation techniques can we try?
- How will we label the data, if necessary?
- How will we quantify the performance of our model? What metrics will we use to do that?
- How will we deploy the system? Using API calls over the cloud, or a monolith system, or an embedded module on an edge device?
- How will the predictions be served: streaming or batch process?
- Would we need to update the model? If yes, what will the update frequency be: daily, weekly, monthly?
- Do we need a monitoring and alerting mechanism for model performance? If yes, what kind of mechanism do we need and how will we put it in place?

### Deployment

- Model packaging: where to persist the model for easy access. Famous pre-trained models typically weight more than 2GB, which makes it harder and more expensive to keep in a service. 
- Model serving: make it available as a web service for other services to consume. 
- Model scaling: models as a service should be capable to scale with respect to request traffic or batch size.

### Building and maintaining a mature system

- Covariate shift: the fact that the production data evolves from the one used for training the model and it leads to peformance drop of the model. 
- Manage complexity of a model while making sure it is also maintainable. 
	- Finding better features: 
		- build a simple model first, start with simple representaitons or use prior knowledge to build the features. 
		- Feature selection is also important and statistical methods can be used to find redundant or irrelevant features. 
			- Wrapper methods: train a model with a subset of the features and compare its performance with the original model. Computational expensive
			- Filter methods: use some sort of proxy measure instead of the error rate to rank and score features.
		- DeepLearning models feature selection is automated.
	- Iterating existing models: 
		- set up a process to periodically retrain and update the existing model and deploy the new model in production. 
		- How do we know this new model is better than the existing one? Internal validation against a "gold standard test set" or external validation using charts and A/B tests on the performance. 
		- Incremental rollout of the new model. 
	- Code and model reproducibility
		- Versioning code with GIT
		- Versioning data with "Data version control" https://dvc.org/
		- Store model settings along with the model
		- Use the same seed wherever random initialization is used. 
		- Note all steps explicitly
	- Troubleshooting and testing
		- Run the model on train, validation and test datasets keeping the deviation of results low. K-fold cross validation is often used to verify model performance. 
		- Test the model for edge cases, e.g. double or triple negation for sentiment analysis. 
		- Analyze the mistakes the model is making. 
		- Keep track of statistics of the features like mean, median, sd, distribution plots, etc. Deviations from these features is a red-flag. 
		- Create dashboards for tracking model metrics and create an alert mechanism on them in case there are any deviations in the metrics. 
		- Interpretability: It's always good to know what a model is doing inside. Use LIME: https://github.com/marcotcr/lime, attention networks and Shapley. 
	- Monitoring: Automatic retrain can add bugs or malfunction to the model. To ensure this doesn't happen we need to: 
		- Monitor model performance reularly: percentile of response time for web services, and task time for batch processing systems. 
		- Store model parameters, behavior, and KPIs (external validation)
		- Run the stored metrics through an anormaly detection system that can alert changes in normal behavior. Actions for sudden performance drop should also be taken into account. 
	- Minimizing technical debt
		- We may have scenarios where we don’t know if the incremental improvements justify the complexity.
		- Drop out unused or rare features 
		- Choose a simpler model that has performance comparable to a much more complex model if you want to minimize technical debt.
	- Automating the ML process
		- From finding better features to version control of datasets, these practices are manual and effort intensive. Driven by the ultimate goal of building intelligent machines and reducing manual effort, AutoML has been created. This is an area of ML to make it more accessible. 
		- AutoML is itself essentially “doing machine learning using machine learning”
		- `auto-sklearn`: automatic accuracy of 98% on MNIST dataset
		- Google AutoML: https://cloud.google.com/natural-language/automl/docs/features
		- AutoCompete framework: https://arxiv.org/pdf/1507.02188.pdf & https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle

### The Data Science process

Two popular processes in the NLP & Data Science industry are: 

#### Knowledge Discovery and Data Mining (KDD)

- Understanding the domain and the goals of the problem to solve
- Target dataset creation
- Data pre-processing
- Data reduction
- Choosing the data mining task: regression, classification, clustering (depends on the domain)
- Choosing the data mining algorithm: SVM, random forests, CNNs, etc. 
- Data mining: algorithm + dataset = predictive models
- Interpretation
- Consolidation: deploy the model into an existing system, document and generate reports

#### Microsoft Team Data Science Process

Current (2017) standard process for executing and delivering advanced analytics solutions. Features:
- A data science life cycle definition: Business understanding, data acquisitioin and understanding, modeling, deployment, and customer acceptance. 
- A standardized project structure
- An infrastructure for project execution
- Tools for data science, version control, data exploration, and modeling

https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/

![image](https://user-images.githubusercontent.com/8902219/112701804-e64adc80-8e91-11eb-9738-3393dbfa1aaa.png)

### Making AI succeed at your organization

- 85% of AI projects fail
- An ideal AI team is formed by: 
	1. Scientists who build models, preferrably if they have worked in industry after graduate school
	2. Engineers who operationalize and maintain models and understand scale and data pipelines
	3. Leaders who manage AI teams and strategize, who have been individual contributor scientists in the past. 

- Right problem and right expectations: 
	- Given a task. The possibilities are many. It’s important to work with the stakeholders first to clearly define the task.
	- The ready availability of a lot of data does not make something an AI problem by default; many problems can be solved using engineering and rule-based and human-in-the-loop approaches.
	- Most stakeholders coming from the world of software engineering treat a wrong output as a bug and are not willing to accept anything that’s not 100% correct.

- Data and timing
	- Quality of data: high quality data means structured, homogeneous, cleaned, and free of noise and outliers. 
	- Quantity of data: 
		- Text classification using Naive Bayes or random forests 3000 datapoints per class is a good starting point.
	- Data labeling: having a human-validated set of labels is time-consuming and expensive process. It is often a continuous process. 

- A good process	
	- Set up the right metrics: beyond accuracy, recall, etc. business metrics alogn with AI metrics. 
	- Start simple, establish strong baselines: do not apply state-of-the-art models right away unless strictly necessary. 
	- Make it work, make it better: complete a whole project cycle having an acceptable model quickly. 
	- Keep shorter turnaround cycles: build models quickly and present results to stakeholders frequently. This helps raise any red flags early and get early feedback.

- Other aspects
	- Cost of compute
	- Blindly following state-of-the-art
	- Return of Investment: AI is expensive and the gains must be estimated first. 
	- Full automation is hard

![image](https://user-images.githubusercontent.com/8902219/112702983-9ec64f80-8e95-11eb-8189-fe05bc4a2d98.png)
