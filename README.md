# HAN

This is an elegant implementation of HAN for Sentiment Classification Task.

The original paper is [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

The core codes are from this repo: [textClassifierHATT](https://github.com/richliao/textClassifier/blob/master/textClassifierHATT.py), 
I refactor the original code to make it easier to understand.

# Preparation

First, check your python version and related libraries:

```
Python >= 3.6
numpy
pandas
re
bs4
pickle
sklearn
gensim
nltk
keras
tensorflow
```
Then, make sure that you alreay have the pre-trained word-vector files: [glove.6B.zip](https://nlp.stanford.edu/projects/glove/) or 
[GoogleNews-vectors-negative300.bin](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)

Next, get your training data. In this case, I use IMDB movie reviews(known as SST-2) for training. One thing you should notice is that 
we should pass a **tsv** file, the format of this file must look like this:
>id	sentiment	review
>
>"5814_8"	1	"With all this stuff going down at the moment with MJ i've started listening to his music, 
>
>watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. 
>
>Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties 
>
>just to maybe make up my mind whether he is guilty or innocent."

which means that the head of this file is "**id\tsentiment\treview**", and the data starts from second line.

# How to use
Here are some examples:
```
python HAN.py --help
```
and you will get an output like picture below(if everything goes well):
![output](https://github.com/Larry955/HAN/blob/master/imgs/args.png)

```
"""
This means: 
The path of your training data is ./train_data.tsv
The path of your embedding path is ./*.bin
The epoch is 20
"""
python HAN.py --full_data_path=train_data.tsv --embedding_path=GoogleNews-vectors-negative300.bin --epoch=20

python HAN.py -d=train_data.tsv -s=GoogleNews-vectors-negative300.bin --epoch=20  # Same as command above
```


```
"""
This means:
You have alreay trained your data, and you will get your training data from pickle file(processed_data.pickle in this case)
Moreover, you have saved the model you trained, and you will load your model from a *h5 file(model.h5 in this case)
"""
python HAN.py --training_data_ready --model_ready
```

# Results
![output](https://github.com/Larry955/HAN/blob/master/imgs/HAN_glove_300dim.png)
