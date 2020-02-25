import java.util.List;
import java.util.Map;
import java.util.HashMap;

/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifier implements Classifier {
    private Map<Label, Integer> docPerLabel;
    private Map<Label, Integer> wordPerLabel;

    private Map<String, Integer> posWords;
    private Map<String, Integer> negWords;

    private double docs;
    private double vocab;

    /**
     * Trains the classifier with the provided training data and vocabulary size
     */
    @Override
    public void train(List<Instance> trainData, int v) {
        this.docPerLabel = this.getDocumentsCountPerLabel(trainData);
        this.wordPerLabel = this.getWordsCountPerLabel(trainData);

        this.negWords = this.getWordOccurrences(Label.NEGATIVE, trainData);
        this.posWords = this.getWordOccurrences(Label.POSITIVE, trainData);

        this.docs = trainData.size();
        this.vocab = v;
    }

    /*
     * Counts the number of words for each label
     */
    @Override
    public Map<Label, Integer> getWordsCountPerLabel(List<Instance> trainData) {
        Map<Label, Integer> map = new HashMap<>();
        int posCount = 0;
        int negCount = 0;

        for (Instance inst : trainData) {
            if (inst.label == Label.NEGATIVE) {
                negCount += inst.words.size();
            }
            else {
                posCount += inst.words.size();
            }
        }

        map.put(Label.NEGATIVE, negCount);
        map.put(Label.POSITIVE, posCount);


        return map;
    }


    /*
     * Counts the total number of documents for each label
     */
    @Override
    public Map<Label, Integer> getDocumentsCountPerLabel(List<Instance> trainData) {
        Map<Label, Integer> map = new HashMap<>();
        Integer neg = 0;
        Integer pos = 0;

        for (Instance inst: trainData) {
            if (inst.label == Label.POSITIVE) {
                pos++;
            }
            else {
                neg++;
            }
        }

        map.put(Label.POSITIVE, pos);
        map.put(Label.NEGATIVE, neg);

        return map;
    }

    private Map<String, Integer> getWordOccurrences(Label label, List<Instance> trainData) {
        Map<String, Integer> values = new HashMap<>();

        for (Instance inst: trainData) {
            if (inst.label == label) {
                List<String> words = inst.words;
                for (String w: words) {
                    if (values.containsKey(w)) {
                        int count = values.get(w);
                        values.replace(w, count + 1);
                    }
                    else {
                        values.put(w, 1);
                    }
                }
            }
        }

        return values;
    }


    /**
     * Returns the prior probability of the label parameter, i.e. P(POSITIVE) or P(NEGATIVE)
     */
    private double p_l(Label label) {
        if (this.docs == 0) {
//
            return 0;
        }

        double count = this.docPerLabel.get(label);
        return count / this.docs;
    }

    /**
     * Returns the smoothed conditional probability of the word given the label, i.e. P(word|POSITIVE) or
     * P(word|NEGATIVE)
     */
    private double p_w_given_l(String word, Label label) {
//
        double num = 0.0;
        double denom = this.wordPerLabel.get(label);

        if (label == Label.NEGATIVE) {
            num = this.negWords.getOrDefault(word, 0);
        }
        else {
            num = this.posWords.getOrDefault(word, 0);
        }

        double double1 = 1;
        double num2 = num + double1;
        double num3 = (this.vocab * double1) + denom;

        if (num3 == 0.0) {
            return 0.0;
        }

        return num2 / num3;
    }

    /**
     * Classifies an array of words as either POSITIVE or NEGATIVE.
     */
    @Override
    public ClassifyResult classify(List<String> words) {
        ClassifyResult result = new ClassifyResult();
        double negCond = 0;
        Map<Label, Double> map = new HashMap<>();
        double posCond = 0;

        double negProbLabel = this.p_l(Label.NEGATIVE) == 0.0 ? 0.0 : Math.log(this.p_l(Label.NEGATIVE));

        for (String w: words) {
            negCond += Math.log(this.p_w_given_l(w, Label.NEGATIVE));
        }
        map.put(Label.NEGATIVE, negProbLabel + negCond);

        double posProbLabel = this.p_l(Label.POSITIVE) == 0.0 ? 0.0 : Math.log(this.p_l(Label.POSITIVE));

        for (String w: words) {
            posCond += Math.log(this.p_w_given_l(w, Label.POSITIVE));
        }
        map.put(Label.POSITIVE, posProbLabel + posCond);

        if (map.get(Label.NEGATIVE) > map.get(Label.POSITIVE)) {
            result.label = Label.NEGATIVE;
        }
        else {
            result.label = Label.POSITIVE;
        }

        result.logProbPerLabel = map;
        return result;
    }


}