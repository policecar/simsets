
## simsets

Input data: subject verb object (SVO) snippets from Google's syntactic n-grams as well as their flipped cousins object verb subject (OVS).

Context: there are contexts of single nouns and context between noun pairs. Noun pairs are denoted (X,Y) and analogously the first noun aka left argument is named X, the right one Y.

Directory structure:

ctx/ 		the word contexts used ( cf. distributional similarity ). for single words ( w1, w2 ) 
			and for word pairs ( pairs ) respectively.
sim/ 		similar words and word pairs for single words and word pairs respectively
lda/ 		topic model of the data
bless/ 		labeled word pairs