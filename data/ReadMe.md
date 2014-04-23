## simsets

Setting: 'Subject verb object' (SVO) snippets from [Google's syntactic n-grams](https://commondatastorage.googleapis.com/books/syntactic-ngrams/index.html) as well as their flipped version 'object verb subject' (OVS) were used to compute context and similarity features for the [BLESS noun pairs](https://sites.google.com/site/geometricalmodels/shared-evaluation) using the [JoBim framework](http://sourceforge.net/projects/jobimtext/).  

( s. e.g. [statistical semantics](http://aclweb.org/aclwiki/index.php?title=Statistical_Semantics) for some background on the underlying linguistic hypothesis )  

--
Directory structure:

Dir | Content
:----|:-----
ctx | the word contexts observed for word pairs ( pairs ) and the single words ( w1, w2 )
sim | similar words and similar word pairs respectively
lda | a topic model of the data
bless | labeled word pairs
