package at.rovo.classifier.naiveBayes;

import java.util.List;
import java.util.Map;

/**
 * Defines the form the training data used by the naive Bayes implementation is stored internally.
 *
 * @author Roman Vottner
 */
public enum TrainingDataStorageMethod
{
    /**
     * Uses a map structure to store the training data.
     * <p>
     * The form of the data to be stored corresponds to the following structure: <code>{'in':{'word1':num1,'word2':num2},
     * 'out':{'word1':num3,'word3':num4}}</code>
     * <p>
     * Where <em>in</em> and <em>out</em> are the classes and <em>word1<em> till <em>word3</em> are features and
     * <em>num1</em> till <em>num4</em> are the occurrences of the respective feature in all training examples for the
     * respective trained class.
     */
    MAP,

    /**
     * Stores the actual occurrences of words contained in an example in a {@link List} structure which is furthermore
     * stored in a List which is a container for all training samples.
     * <p>
     * A mapping between a word and the words position in the list is stored in a {@link Map} where the key is the word
     * and the value is the position of that word in the list that stores the occurrences of words per example.
     */
    LIST
}
