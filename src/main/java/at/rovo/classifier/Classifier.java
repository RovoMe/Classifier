package at.rovo.classifier;

import java.io.File;
import java.io.Serializable;
import java.util.List;

/**
 * Core class for any classifier. It requires extending classes to implement some typical methods for any classifier as
 * <code>train()</code> and <code>classify()</code>.
 * <p>
 * Further it is able to save and load its data from and to disk, therefore it requires that categories as well as
 * features have to implement {@link Serializable}.
 *
 * @param <F>
 *         Type of the features this classifier has to deal with
 * @param <C>
 *         Type of the categories this classifier supports
 *
 * @author Roman Vottner
 */
public abstract class Classifier<F extends Serializable, C extends Serializable>
{
    /** The actual training data which it uses to classify features **/
    protected TrainingData<F, C> trainingData = null;
    /** The data to test the trained model against with **/
    protected TrainingData<F, C> testData = null;

    /** Name of the classifier **/
    private String name = "";

    /**
     * Instantiates necessary data structures for subclasses.
     */
    public Classifier()
    {

    }

    /**
     * Instantiates necessary data structures for subclasses and assign a name to the classifier.
     *
     * @param name
     *         The name of the classifier
     */
    public Classifier(String name)
    {
        this.name = name;
    }

    /**
     * Sets the name of the classifiers' instance.
     *
     * @param name
     *         New name of the classifier
     */
    public void setName(String name)
    {
        this.name = name;
    }

    /**
     * Returns the name of the classifiers' instance.
     *
     * @return Name of the classifier
     */
    public String getName()
    {
        return this.name;
    }

    /**
     * Persists a {@link Classifier}s data object via java object serialization to a file in a defined directory.
     *
     * @param directory
     *         The directory the {@link Classifier} should be saved in
     * @param name
     *         The name of the {@link File} which will hold the bytes of the persisted object.
     */
    public void saveData(File directory, String name)
    {
        this.trainingData.saveData(directory, name);
    }

    /**
     * Loads data from a previous classification into the classifier.
     *
     * @param serializedObject
     *         A reference to the {@link File} representing the serialized object
     *
     * @return true if the data could be loaded; false otherwise
     */
    public abstract boolean loadData(File serializedObject);

    /**
     * Trains the classifier for a single feature.
     * <p>
     * The category is used to label the item/s, f.e. to be a good ('good') example or to be within ('in') an articles
     * content. This is later on used to decide if a certain feature is more likely to be in a certain category or not.
     *
     * @param item
     *         A {@link String} contains only one word that should be trained for a given category.
     * @param category
     *         The label of the item, f.e. 'good' or 'in', the classifier should use to decide later on what results to
     *         predict
     */
    public abstract void train(F item, C category);

    /**
     * Trains the classifier with the specified items to be in a certain category.
     * <p>
     * The category is used to label the item/s, f.e. to be a good ('good') example or to be within ('in') an articles
     * content. This is later on used to decide if a certain feature is more likely to be in a certain category or not.
     *
     * @param items
     *         An array of {@link String} containing all the words that should be trained for a given category.
     * @param category
     *         The label of the item, f.e. 'good' or 'in', the classifier should use to decide later on what results to
     *         predict
     */
    public abstract void train(F[] items, C category);

    /**
     * Trains the classifier with the specified items to be in a certain category.
     * <p>
     * The category is used to label the item/s, f.e. to be a good ('good') example or to be within ('in') an articles
     * content. This is later on used to decide if a certain feature is more likely to be in a certain category or not.
     *
     * @param items
     *         A {@link List} of {@link String}s containing all the words that should be trained for a given category.
     * @param category
     *         The label of the item, f.e. 'good' or 'in', the classifier should use to decide later on what results to
     *         predict
     */
    public abstract void train(List<F> items, C category);

    /**
     * Tries to predict the category a certain item is most likely to be in
     *
     * @param item
     *         The item which category should be predicted
     *
     * @return The predicted category the item is most likely to be in
     */
    public abstract C classify(F item);

    /**
     * Tries to predict the category certain items are most likely to be in
     *
     * @param items
     *         The items which category should be predicted
     *
     * @return The predicted category the items are most likely to be in
     */
    public abstract C classify(F[] items);

    /**
     * Divides the dataset into a training and a test set. The latter is used to estimate the classification accuracy of
     * the trained model.
     */
    public void crossValidation()
    {

    }
}
