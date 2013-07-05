/**
 * @(#)Classifier.java        1.5.2 09/03/29
 */
 
package at.rovo.classifier.decissionTrees.c45;

/**
 * A classifier which can classify test data.
 *
 * @author 	    Ping He
 * @author 	    Xiaohua Xu
 */
public interface Classifier
{
	/**
	 * Classify the test data.
	 *
	 * @param testData the test data to be classified
	 * @return the classification results
	 * @see ml.classifier.dt.DecisionTree
	 */
	public String[] classify(String[][] testData);
	
	/**
	 * Get the classification error on the test data.
	 *
	 * @param testData the test data to be classified
	 * @return the classification error on the test data.
	 * @see ml.classifier.dt.DecisionTree
	 */
	public int getTestError(String[][] testData);
	
	/**
	 * Get the classification error ratio on the test data.
	 *
	 * @param testData the test data to be classified
	 * @return the classification error on the test data.
	 * @see ml.classifier.dt.DecisionTree
	 */
	public double getTestErrorRatio(String[][] testData);
}