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

Feature Engineering

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



