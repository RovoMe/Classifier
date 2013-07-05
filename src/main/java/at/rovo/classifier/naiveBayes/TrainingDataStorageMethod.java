package at.rovo.classifier.naiveBayes;

/**
 * <p>
 * Defines the form the training data used by the naive Bayes implementation is
 * stored internally.
 * </p>
 * 
 * @author Roman Vottner
 * 
 */
public enum TrainingDataStorageMethod
{
	/**
	 * <p>
	 * Uses a map structure to store the training data.
	 * </p>
	 * <p>
	 * The form of the data to be stored corresponds to the following structure:
	 * </p>
	 * <code>{'in':{'word1':num1,'word2':num2}, 
	 * 'out':{'word1':num3,'word3':num4}}</code>
	 * <p>
	 * Where <em>in</em> and <em>out</em> are the classes and
	 * <em>word1<em> till <em>word3</em> are features and <em>num1</em> till
	 * <em>num4</em> are the occurrences of the respective feature in all
	 * training examples for the respective trained class.
	 * </p>
	 */
	MAP,

	/**
	 * <p>
	 * Stores the actual occurrences of words contained in an example in a
	 * {@link List} structure which is furthermore stored in a List which is a
	 * container for all training samples.
	 * </p>
	 * <p>
	 * A mapping between a word and the words position in the list is stored in
	 * a {@link Map} where the key is the word and the value is the position of
	 * that word in the list that stores the occurrences of words per example.
	 * </p>
	 */
	LIST
}
