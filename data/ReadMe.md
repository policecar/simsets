## simsets

Background: 'Subject verb object' (SVO) snippets from [Google's syntactic n-grams](https://commondatastorage.googleapis.com/books/syntactic-ngrams/index.html) as well as their flipped version 'object verb subject' (OVS) were used to compute context and similarity features for the [BLESS noun pairs](https://sites.google.com/site/geometricalmodels/shared-evaluation) using the [JoBim framework](http://sourceforge.net/projects/jobimtext/).  

Directory structure:

dir | content
:----|:-----
ctx | the word contexts used ( cf. [statistical semantics](http://aclweb.org/aclwiki/index.php?title=Statistical_Semantics) ) for single words ( w1, w2 ) and for word pairs ( pairs ) respectively
sim | similar words resp. word pairs
lda | topic model of the data
bless | labeled word pairs
