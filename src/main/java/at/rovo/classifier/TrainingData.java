package at.rovo.classifier;

import java.io.File;

/**
 * Specifies certain methods common to the training process of classifiers.
 *
 * @param <F>
 *         The type of the feature to train
 * @param <C>
 *         The type of the category a feature will be associated with
 *
 * @author Roman Vottner
 */
public interface TrainingData<F, C>
{
    /**
     * Persists a {@link Classifier}s data object via java object serialization to a file in a defined directory.
     *
     * @param directory
     *         The directory the training data should be saved in
     * @param name
     *         The name of the {@link File} which will hold the bytes of the persisted object.
     */
    void saveData(File directory, String name);

    /**
     * Loads data from a previous classification into the classifier.
     *
     * @param serializedObject
     *         A reference to the {@link File} representing the serialized object
     *
     * @return true if the data could be loaded; false otherwise
     */
    boolean loadData(File serializedObject);
}
