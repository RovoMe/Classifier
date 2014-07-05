package at.rovo.classifier.naiveBayes;

import java.io.Serializable;
import java.util.Collection;
import at.rovo.classifier.TrainingData;

/**
 * <p>Abstract base class for a naive Bayes training data.</p>
 * <p>Training a naive Bayes classifier is basically counting the number of f.e.
 * words in a document and incrementing the count for these words in the respective
 * category.</p>
 * <p>As the style of storing trained data can have a huge impact on performance
 * on large data sets, this implementation provides possibilities to change the
 * behavior as needed.</p>
 * 
 * @param <F> The type of the features or words
 * @param <C> The type of the categories or classes
 * 
 * @author Roman Vottner
 */
public abstract class NBTrainingData<F,C> implements Serializable, TrainingData<F,C>
{
	/**
	 * <p>Unique identifier necessary for serialization</p>
	 */
	private static final long serialVersionUID = -2101815681608863601L;
	
	/**
	 * <p>Initializes a package-private instance of an abstract training data
	 * object for a naive Bayes classifier.</p>
	 */
	NBTrainingData()
	{

	}
	
	/**
	 * <p>A factory method for initializing the appropriate training method 
	 * according to the provided {@link TrainingDataStorageMethod} object.</p>
	 * 
	 * @param method Defines the style of storing the trained data internally
	 * @return An initialized and ready-to-be-trained naive Bayes classifier
	 */
	static <F,C> NBTrainingData<F,C> create(TrainingDataStorageMethod method)
	{
		if (TrainingDataStorageMethod.MAP.equals(method))
			return new NBMapTrainingData<>();
		else if (TrainingDataStorageMethod.LIST.equals(method))
			return new NBListTrainingData<>();
		else
			return null;
	}
	
	/**
	 * <p>Increments the count of a feature in a specific category.</p>
	 * 
	 * @param feature The feature whose count should be incremented
	 * @param category The category the feature to increment belongs to
	 */
	public abstract void incrementFeature(F feature, C category);
	
	/**
	 * <p>Increments the sample-size for a specific category.</p>
	 * <p>A sample here is f.e. a whole sentence classified as 'in' or 'out',
	 * while every word in the sentence is a specific feature.</p>
	 * 
	 * @param category The category whose number of samples should be incremented
	 */
	public abstract void incrementNumberOfSamplesForCategory(C category);
	
	/**
	 * <p>Counts the number of times a certain feature appeared in all categories</p>
	 * 
	 * @param feature The feature of interest
	 * @return The number of times this feature appeared in all categories
	 */
	protected abstract long getFeatureCount(F feature);
	
	/**
	 * <p>Returns the number of categories used throughout the training of the
	 * classifier.</p>
	 * @return The number of categories
	 */
	protected abstract int getNumberOfCategories();
	
	/**
	 * <p>Returns the total number of features trained.</p>
	 * 
	 * @return The total number of features trained
	 */
	protected abstract long getTotalNumberOfFeatures();
	
	/**
	 * <p>The number of samples for a specific category</p>
	 * 
	 * @param category The category whose trained samples should be returned
	 * @return The number of samples trained for this category
	 */
	public abstract long getNumberOfSamplesForCategory(C category);
	
	/**
	 * <p>The total number of samples trained</p>
	 * 
	 * @return The number of all samples trained
	 */
	public abstract long getTotalNumberOfSamples();
	
	/**
	 * <p>The number of times a feature has appeared in a category</p>
	 * 
	 * @param feature The feature of interest
	 * @param category The category the feature should be in (f.e. marked as 
	 *                 'good' entry)
	 * @return The absolute count of the feature among all examples labeled
	 *         as the specified category
	 */
	public abstract int getFeatureCount(F feature, C category);
	
	/**
	 * <p>Checks if a certain category was trained previously.</p>
	 * 
	 * @param category The category which should be checked
	 * @return True if the category was previously trained and it therefore is
	 *         available, false otherwise.
	 */
	protected abstract boolean containsCategory(C category);
	
	/**
	 * <p>Returns all categories (classes) which have previously been trained.</p>
	 * 
	 * @return The categories available after training
	 */
	protected abstract Collection<C> getCategories();
}
