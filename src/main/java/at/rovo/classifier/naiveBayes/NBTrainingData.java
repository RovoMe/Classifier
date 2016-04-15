package at.rovo.classifier.naiveBayes;

import at.rovo.classifier.TrainingData;
import java.io.Serializable;
import java.util.Collection;

/**
 * Abstract base class for a naive Bayes training data.
 * <p>
 * Training a naive Bayes classifier is basically counting the number of f.e. words in a document and incrementing the
 * count for these words in the respective category.
 * <p>
 * As the style of storing trained data can have a huge impact on performance on large data sets, this implementation
 * provides possibilities to change the behavior as needed.
 *
 * @param <F>
 *         The type of the features or words
 * @param <C>
 *         The type of the categories or classes
 *
 * @author Roman Vottner
 */
public abstract class NBTrainingData<F, C> implements Serializable, TrainingData<F, C>
{
    /**
     * Unique identifier necessary for serialization
     */
    private static final long serialVersionUID = -2101815681608863601L;

    /**
     * Initializes a package-private instance of an abstract training data object for a naive Bayes classifier.
     */
    NBTrainingData()
    {

    }

    /**
     * A factory method for initializing the appropriate training method according to the provided {@link
     * TrainingDataStorageMethod} object.
     *
     * @param method
     *         Defines the style of storing the trained data internally
     *
     * @return An initialized and ready-to-be-trained naive Bayes classifier
     */
    static <F, C> NBTrainingData<F, C> create(TrainingDataStorageMethod method)
    {
        if (TrainingDataStorageMethod.MAP.equals(method))
        {
            return new NBMapTrainingData<>();
        }
        else if (TrainingDataStorageMethod.LIST.equals(method))
        {
            return new NBListTrainingData<>();
        }
        else
        {
            return null;
        }
    }

    /**
     * Increments the count of a feature in a specific category.
     *
     * @param feature
     *         The feature whose count should be incremented
     * @param category
     *         The category the feature to increment belongs to
     */
    public abstract void incrementFeature(F feature, C category);

    /**
     * Increments the sample-size for a specific category.
     * <p>
     * A sample here is f.e. a whole sentence classified as 'in' or 'out', while every word in the sentence is a
     * specific feature.
     *
     * @param category
     *         The category whose number of samples should be incremented
     */
    public abstract void incrementNumberOfSamplesForCategory(C category);

    /**
     * Counts the number of times a certain feature appeared in all categories.
     *
     * @param feature
     *         The feature of interest
     *
     * @return The number of times this feature appeared in all categories
     */
    protected abstract long getFeatureCount(F feature);

    /**
     * Returns the number of categories used throughout the training of the classifier.
     *
     * @return The number of categories
     */
    protected abstract int getNumberOfCategories();

    /**
     * Returns the total number of features trained.
     *
     * @return The total number of features trained
     */
    protected abstract long getTotalNumberOfFeatures();

    /**
     * The number of samples for a specific category.
     *
     * @param category
     *         The category whose trained samples should be returned
     *
     * @return The number of samples trained for this category
     */
    public abstract long getNumberOfSamplesForCategory(C category);

    /**
     * The total number of samples trained.
     *
     * @return The number of all samples trained
     */
    public abstract long getTotalNumberOfSamples();

    /**
     * The number of times a feature has appeared in a category.
     *
     * @param feature
     *         The feature of interest
     * @param category
     *         The category the feature should be in (f.e. marked as 'good' entry)
     *
     * @return The absolute count of the feature among all examples labeled as the specified category
     */
    public abstract int getFeatureCount(F feature, C category);

    /**
     * Checks if a certain category was trained previously.
     *
     * @param category
     *         The category which should be checked
     *
     * @return True if the category was previously trained and it therefore is available, false otherwise.
     */
    protected abstract boolean containsCategory(C category);

    /**
     * Returns all categories (classes) which have previously been trained.
     *
     * @return The categories available after training
     */
    protected abstract Collection<C> getCategories();
}
