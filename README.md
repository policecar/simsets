## simsets

Predicting semantic relations with set algebra.

--
Subject-Verb-Object (SVO) constructs extracted from [Google's syntactic n-grams](https://commondatastorage.googleapis.com/books/syntactic-ngrams/index.html) as well as their flipped version Object-Verb-Subject were used to compute context and similarity features for the [BLESS noun pairs](https://sites.google.com/site/geometricalmodels/shared-evaluation) using the [JoBim framework](http://sourceforge.net/projects/jobimtext/). The resulting feature matrices are combined in various set-algebraic operations to predict the semantic relation between pairs of words; e.g. hyper( alligator, chordate ), mero( alligator, jaw ), coord( alligator, crocodile ), random( alligator, addition ). 

For some background on the underlying methodological and theoretical framework, see, e.g., [statistical semantics](http://aclweb.org/aclwiki/index.php?title=Statistical_Semantics)


### The data/ directory

Dir | Content
:----|:-----
ctx | the word contexts observed for word pairs ( pairs ) and the single words ( w1, w2 )
sim | similar words and similar word pairs respectively
lda | a topic model of the data
bless | labeled word pairs


### Usage

```python
python2.7 play_bless.py
```


### References

[Turney 2013](arxiv.org/pdf/1401.8269v1.pdf) – similarity differences for textual entailment ( set minus )  
[Mikolov 2013](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality) – vector arithmetics with word representations ( element-wise addition )  
