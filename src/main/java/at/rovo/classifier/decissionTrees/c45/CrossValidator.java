/**
 * @(#)CrossValidator.java        1.5.2 09/03/29
 */

package at.rovo.classifier.decissionTrees.c45;

import java.util.Random;
import at.rovo.classifier.dataset.DataSet;
import at.rovo.classifier.dataset.UciDataSet;
import at.rovo.classifier.decissionTrees.c45.DecisionTree;
import at.rovo.classifier.decissionTrees.c45.util.Statistics;

/**
 * Cross validate a classifier to get its average classification performance.
 *
 * @author 	    Ping He
 * @author 	    Xiaohua Xu
 */
public class CrossValidator {
	/* The whole cross validation data, used to generates different train and test data.*/
	private String[][] crossValidationData;
	/* The number of fold to cross validate.*/
	private int fold;
	/* The changeable dataSet constructed in different folds of cross validation.*/
	private DataSet dataSet;
	/**
	 * One recorded result of the cross validation.
	 * For decision tree classifier, the size of constructed need to be recorded.
	 */
	private float[] treeSizes;
	/**
	 * The other recorded result of the cross validation:
	 * the test error ratios in different folds of cross validation.
	 */
	private float[] testErrorRatios;

	/**
	 * Initialize a cross validator with the specified fold of cross validation.
	 * <p>
	 * It first shuffles, clusters and distributes the provided cross validation data,
	 * so that the class-uniformly distributed data can be produced.<br>
	 * Then a classifier is built on the train data in each fold, followed with
	 * the test data classified using the classifier to get its test error ratio.
	 * </p>
	 *
	 * @param dataSet The data set to be cross validated
	 * @param fold    The folds of cross validation
	 */
	public CrossValidator(DataSet dataSet, int fold) {
		// Load the data for cross validation
		this.dataSet = dataSet;
		this.crossValidationData = new String[dataSet.getCaseCount()][];
		System.arraycopy(dataSet.getTrainData(), 0, crossValidationData, 0, dataSet.getCaseCount());
		this.fold = fold;

		// Shuffle the crossValidationData to random sequence
		shuffle();
		// Group the shuffled crossValidationData with the same class attribute values together
		cluster();
		// Distribute the clustered crossValidationData to make sure each class is uniformly scattered for fair cross validation
		distribute();
		// Cross validate
		crossValidate();
	}

	/**
	 * Initialize a cross validator of the default 10 fold cross validation
	 * @param dataSet The data set to be cross validated
	 */
	public CrossValidator(DataSet dataSet) {
		this(dataSet, 10);
	}

	/**
	 * Make a random perturbation of the whole cross validation data
	 */
	private void shuffle() {
		Random random = new Random();
		for(int i = crossValidationData.length-1; i > 0; i --) {
			int selectedIndex = random.nextInt(i+1);
			swap(crossValidationData, selectedIndex, i);
		}
	}

	/**
	 * Group the cross validation data with the same class attribute values together.
	 * <p>
	 * For example, the sequence of the original class attributes values is {a b a c b},
	 * then after clustering, it becomes {a a b b c}
	 * </p>
	 */
	private void cluster() {
		int classIndex = dataSet.getClassAttributeIndex();

		int first = 0, last;
		int max = crossValidationData.length;
		while(first < max) {
			last = first;
			for(int i = first + 1; i < max; i ++) {
				if(crossValidationData[i][classIndex].equals(crossValidationData[last][classIndex])) {
					last ++;
					if(i == last) continue;
					swap(crossValidationData, i, last);
				}
			}
			first = last + 1;
		}
	}

	/**
	 * Distributes the cross validation data uniformly in different class values.
	 * It is used to produce uniform class distribution for each fold of cross validation.
	 *
	 * For example, the original sequence of the class value is {a a a b b b c c c c},
	 * after distribution, it becomes {a b c c | a b c | a b c}
	 * (suppose 3 fold cross validation).
	 */
	private void distribute() {
    	String[][] result = new String[crossValidationData.length][];
    	int count = 0;

    	for(int i = 0; i < fold; i ++) {
    		for(int j = i; j < crossValidationData.length; j += fold) {
    			result[count ++] = crossValidationData[j];
    		}
    	}

		// update the cross validation data
    	this.crossValidationData = result;
    }

	/**
	 * Partition the cross validation data to different train data and test data in different
	 * folds, build a classifier for each train data and use it to classify the corresponding
	 * test data.
	 * The cross validation result of each fold is all recorded.
	 */
	private void crossValidate() {
		// The number of cases which cannot be equally distributed into all the folds
		int fraction = crossValidationData.length % fold;
		// The default size of the test data
 		int testSize = crossValidationData.length / fold + 1;

		// Ready to record the error ratio of each fold
		float[] testErrorRatios = new float[fold];
		// Ready to record the
		float[] treeSizes = new float[fold];

		// The start index of the train data in the crossValidationData
		int start = 0;
		int attributeCount = dataSet.getAttributeCount();
		// Execute tree construction and test data evaluation for each fold of cross validation
 		for(int i = 0; i < fold; i ++) {
			/* For the folds belonging to [0, fraction), their testSize equals the default testSize
			 *     i.e. crossValidationData.length / fold + 1;
			 * For the folds >= fraction, their testSize becomes one less
			 *     i.e. crossValidationData.length / fold;
			 */
			if(i == fraction) testSize = crossValidationData.length / fold;
			int trainSize = crossValidationData.length-testSize;

			// Train data used for tree construction (it changes with cross validation)
			String[][] trainData = new String[trainSize][attributeCount];
			// Test data used for classification (it changes with cross validation)
			String[][] testData = new String[testSize][attributeCount];

			// Partition the train data and the test data for the current fold
			int trainCount = 0;
			int testCount = 0;
			for(int k = start; (trainCount + testCount) < crossValidationData.length; k ++) {
				// Traverse the whole crossValidationData cyclically
				int index = k % crossValidationData.length;
				// Copy the train data
				if(trainCount < trainSize) {
					trainData[trainCount ++] = crossValidationData[index];
				}
				// Copy the test data
				else {
					testData[testCount ++] = crossValidationData[index];
				}
			}

			// Renew the data set for the classifier construction in different folds
			dataSet.setTrainData(trainData);
			dataSet.addColumnSetView();

			// Construct the tree classifier
			TreeClassifier tree = new DecisionTree(dataSet);
			tree.prune();

			// Compute the tree size and the test error ratio of the constructed tree classifier
			int size = tree.size();
	    	int error = tree.getTestError(testData);
	    	float errorRatio = (float)(100.0f * error / testCount);

			// Record the tree size and the test error ratio
			testErrorRatios[i] = errorRatio;
			treeSizes[i] = size;

			// Ready to partite the train data and the test data for the next fold
			start += testSize;
		}

		// Record the tree size and the test error ratio in the CrossValidator instance
		this.testErrorRatios = testErrorRatios;
		this.treeSizes = treeSizes;
	}

	/**
	 * Change the position of two String arrays in a 2D string array
	 */
	private static void swap(String[][] str, int i, int j) {
		String[] temp = str[i];
		str[i] = str[j];
		str[j] = temp;
	}

	/**
	 * Get the sizes of the classifier built in all the folds of cross validation.
	 */
	public float[] getTreeSizes() {
		return this.treeSizes;
	}

	/**
	 * Get the test error ratios of the classifier built in all the folds of cross validation.
	 */
	public float[] getTestErrorRatios() {
		return this.testErrorRatios;
	}

	/**
	 * Get the fold of cross validation
	 */
	public int getFold() {
		return this.fold;
	}

	/**
	 * Cross validate a data set with the specified data set name.
	 * <br>
	 * Usage: java CrossValidator dataSetName [fold]
	 */
	public static void main(String[] args) {
		CrossValidator cross;

		if(args.length > 2 || args[0].equalsIgnoreCase("-h")
                           || args[0].equalsIgnoreCase("-help")){
            System.out.println("Usage: java CrossValidator dataSetName [fold]");
			return;
		}
		else if(args.length == 1) {
			String dataSetName = args[0];
			cross = new CrossValidator(new UciDataSet(dataSetName));
		}
		else{
			String dataSetName = args[0];
			int fold = Integer.parseInt(args[1]);
			cross = new CrossValidator(new UciDataSet(dataSetName), fold);
		}

		float[] treeSizes = cross.getTreeSizes();
		float[] testErrorRatios = cross.getTestErrorRatios();

		// Print the tree size, test error ratios and their corresponding standard deviations
		System.out.printf("%-15s %-15s\n", "Size(std)", "Ratio(std)");
		System.out.printf("%.1f(%.2f)       %.1f%%(%.2f)\n\n",
   		                   Statistics.mean(treeSizes),
						   Statistics.std(treeSizes),
						   Statistics.mean(testErrorRatios),
						   Statistics.std(testErrorRatios)
						 );
	}
}
