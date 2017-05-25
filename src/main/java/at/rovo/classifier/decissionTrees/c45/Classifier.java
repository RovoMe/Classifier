package at.rovo.classifier.decissionTrees.c45;

/**
 * A classifier which can classify test data.
 *
 * @author Ping He
 * @author Xiaohua Xu
 */
public interface Classifier
{
    /**
     * Classify the test data.
     *
     * @param testData the test data to be classified
     * @return the classification results
     * @see DecisionTree
     */
    String[] classify(String[][] testData);

    /**
     * Get the classification error on the test data.
     *
     * @param testData the test data to be classified
     * @return the classification error on the test data.
     * @see DecisionTree
     */
    int getTestError(String[][] testData);

    /**
     * Get the classification error ratio on the test data.
     *
     * @param testData the test data to be classified
     * @return the classification error on the test data.
     * @see DecisionTree
     */
    double getTestErrorRatio(String[][] testData);
}