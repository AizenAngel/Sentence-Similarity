# Sentence-Similarity

<!-- ## Introduction
This repository implements method for calculating similarities between small sentences. Implementation is divided in 2 files: 
* `main.py` - for implementing functionalities
* `main.ipynb` - for testing -->

## Introduction
Sentences consist of *words*, which are combined and give them different meanings. So we should pay special attention to words. But not only that. Word order is also very important. As stated in the paper, let's look at the following example:  
* ***A quick brown dog jumps over the lazy fox.***  
* ***A quick brown fox jumps over the lazy dog.***  

Words in these sentences belong to the same corpus and only order of words *fox* and *dog* is reverted. Naive implementation would consider these 2 sentences to be the same because of the words used in them. 
So our implementation will be focused on the following: 

* *semantic word similarity*
* *order similarity*

## Semantic word similarity
Semantic similarity is one of the bigger chalanges in this implementation.
We can easily check if 2 words are the same based on their characters, but can we tell when two words are similar (have similar meaning)? Let's take a look at the following graph:

![semantic graph](./Images/semantic_similarity_graph.png)
Given 2 words `w1` and `w2`, we want to find their semantic similarity `s(w1,w2)`. We can do this by calculating the shortest paths between words in this graph. For example, shortest distance between words *boy* and *girl* is 4 (path from: male - person - female - girl).  
Another example shows that distance between *boy* and *teacher* is 6, but also that distance between *boy* and *animal* is 4. So, word *boy* is semanticly closer to word *animal* than to word *teacher*, which makes no sense. One easy way to fix this, is to take into consideration the depth of the root for two words in question, because words in upper nodes of our graph are more abstract, and have more meanings attached to them. 
Finally, semantic similarity will look like this:   
`s(w1, w2) = f(l, h)`  
* l - shortest path distance between words
* h - depth of the common root of these two words. 

We will assume that above formula could be written like:  
`s(w1, w2) = f1(l) * f2(h)`

### Shortest path distance
Having previous counterexample for distance similarity excluded (for *animal*, *teacher* and *boy*), it makes complete sense that shorter the distance between words, the more similar they are. So, `f(l)` should be monotonically decreasing function as *l* increases. We will use the following function:  
<code>f1(l) = e<sup>-αl</sup> </code>

### Depth of the common root
Here we will add more meaning to bigger depth, so unlike the `Shortest path distance` section, `f2(h)` will be monotonically increasing function with the respect to the parameter h. We will use hyperbolic tangent function:

<code>f2(h) = (e<sup>βh</sup> - e<sup>-βh</sup>) / (e<sup>βh</sup> + e<sup>-βh</sup>) </code>



Calculate similarity between diferent sentences. Based on: [this work](https://www.researchgate.net/publication/232645326_Sentence_Similarity_Based_on_Semantic_Nets_and_Corpus_Statistics#:~:text=The%20semantic%20similarity%20of%20two,database%20and%20from%20corpus%20statistics.&text=Experiments%20on%20two%20sets%20of,significant%20correlation%20to%20human%20intuition.)
