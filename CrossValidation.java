import java.util.List;
import java.io.*;
import java.util.*;
import java.util.HashMap;
import java.util.Map;

public class CrossValidation {
    /*
     * Returns the k-fold cross validation score of classifier clf on training data.
     */
    static double kFoldScore(Classifier clf, List<Instance> trainData, int k, int v) {
        if (k < 2) {
            return 0;
        }

        List<Instance> testingSet;
        List<Instance> trainingSet;
        ArrayList<Double> values = new ArrayList<>();

        for (int i = 0; i < k; i++) {
            testingSet = CrossValidation.getTestingSet(trainData, k, i);
            trainingSet = CrossValidation.getTrainingSet(trainData, k, i);
            clf.train(trainingSet, v);
            values.add(CrossValidation.validation(clf, testingSet));
        }

        double sum = 0;
        for (Double value: values) {
            sum += value;
        }

        return sum / k;
    }

    private static List<Instance> getTestingSet(List<Instance> trainData, int k, int fold) {
        List<Instance> values = new ArrayList<>();

        int size = trainData.size() / k;
        int idx = size * fold;

        for (int i = idx; i < (idx + size); i++) {
            values.add(trainData.get(i));
        }

        return values;
    }

    private static List<Instance> getTrainingSet(List<Instance> trainData, int k, int fold) {
        List<Instance> values = new ArrayList<>();
        int size = trainData.size() / k;
        int firstIdx = size * fold;
        int lastIdx = firstIdx + size;

        for (int i = 0; i < trainData.size(); i++) {
            if (!(i >= firstIdx && i < lastIdx)) {
                values.add(trainData.get(i));
            }
        }

        return values;
    }

    private static double validation(Classifier clf, List<Instance> testingSet) {
        if (testingSet.size() == 0) {
            return 0;
        }

        double correct = 0;

        for (Instance inst: testingSet) {
            ClassifyResult results = clf.classify(inst.words);
            if (results.label == inst.label) {
                correct ++;
            }
        }
        return correct / testingSet.size();
    }
}