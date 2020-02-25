Implements a Naïve Bayse classifer for categorizing movie reviews as either positive or negative. The dataset of movie reviews can be found here: https://ai.stanford.edu/~amaas/data/sentiment/. A negative review has a score ≤ 4 out of 10, and a positive
review has a score ≥ 7 out of 10.

DISCLAIMER: I only implemented the NaiveBayesClassifier.java and CrossValidation.java files. The rest was provided by my professor. 

To run the program:

```
java SentimentAnalysis <mode> <trainFilename> [<testFilename> | <K>]
```

mode is integer from 0 to 3. If mode is 0 or 1, there are two arguments, mode and trainFilename. If mode is 2, the third argument is testFilename. If mode is 3, the third argument is K, the number of folds for cross validation.