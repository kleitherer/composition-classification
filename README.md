# composition-classification

This is my CS109 Final Project. My goal is to use probability to determine when a musical score is stylistically inconsistent with its attributed composer. Based on quantifiable features in the score, like pitch intervals, pitch range, tempo, chromaticity, etc, we can model the likelilhood that this piece actually was written by a specific composer. This has huge implications in identifying misattributed composers from history, especially for classical compositions when many women's works were attributed to their husbands. 

Here is our high level process:

First, the success in our attempt to uncover hidden histories rests on the assumption that the features we use for training really can be used to capture a composer's identity. So, to validate this assumption, we employed clustering methodst to ensure 

we trained a simple classifier that outputs the posterior distribution given an observation of features. I.E. given that we see this range of pitches, this key, this number of accidentals, what is the probability that the composer is ___? 


Second, since we're able to group 


This has been done before 

# Feature Extraction, using kNN classification and correlation matrix

We cannot use sklearn's built in features and that's where the novelty of this project comes in. We'll be using symbolic data to characterize our pieces, and then we'll use kNN classifcation to validate the features we've extracted.
If neighbors make sense (Beethoven near Beethoven, harpists near harpists), you extracted meaningful features.

You can show plots:

“These five closest pieces to Piece X are all harpist works → strong harp idiom signal.”

This is intuitive to non-technical readers.


YES, you should include a correlation matrix — and YES, in this context each feature is treated as a random variable.
A correlation matrix is exactly the right tool to understand your feature space, diagnose redundancy, and justify modeling decisions.

This is exactly the setup where:
	•	Each column = a random variable
	•	Each row = a sample drawn from a distribution over feature-vectors


D. Helps you choose between global vs. local features

You can visually show:
	•	global features cluster / correlate together
	•	local features (interval distributions) behave differently

This supports your narrative that local features carry more stylistic identity.




# Experiments:

models to test from sklearn:


# Real world extension - solving the mystery of Dussek's Sonata in C Minor

How? We'll be running a statistical test to see whether composer-style models trained mostly on men accidentally treat women’s compositions as stylistic OUTLIERS, or misclassify them more often than men’s.

