package at.rovo.classifier.naiveBayes;

import java.io.Serializable;
import java.util.Hashtable;
import java.util.Map;
import at.rovo.classifier.Classifier;

/**
 * <p>The base class for all naive Bayes implementations.</p>
 * <p><em>{@link #create}</em> is a factory method to initialize the
 * appropriate naive Bayes classifier based on the provided 
 * {@link ProbabilityCalculation} argument.</p>
 * 
 * @param <F> Type of the features this classifier has to deal with
 * @param <C> Type of the categories this classifier supports
 * 
 * @author Roman Vottner
 */
public abstract class NaiveBayes<F extends Serializable, C extends Serializable> extends Classifier<F,C>
{
	/** Defines threshold for classification */
	private Map<C, Double> threshold = new Hashtable<C, Double>();
	
	/**
	 * <p>Hides the constructor except for child-classes. This forces external
	 * classes to instantiate new objects via the factory method.</p>
	 */
	protected NaiveBayes()
	{
		super();
	}
	
	/**
	 * <p>Hides the constructor except for child-classes. This forces external
	 * classes to instantiate new objects via the factory method.</p>
	 */
	protected NaiveBayes(String name)
	{
		super(name);
	}
	
	/**
	 * <p>Sets the threshold for a certain category</p>
	 * 
	 * @param category The category the threshold should be applied to
	 * @param t The new value of the threshold
	 */
	public void setThreshold(C category, double t)
	{
		this.threshold.put(category, t);
	}
	
	/**
	 * <p>Returns the threshold for the specified category of the current instance.</p>
	 * 
	 * @param category The category the threshold is set for
	 * @return The current threshold of the specified category
	 */
	public double getThreshold(C category)
	{
		if (!this.threshold.containsKey(category))
			return 1.0;
		return this.threshold.get(category);
	}
	
	/**
	 * <p>Calculates the a-posterior probability of a certain item to be within
	 * a specific category.</p>
	 * 
	 * @param category The category the item should be in
	 * @param item The item whose probability should be calculated to 
	 *             be in a certain category
	 * @return The probability of the item being in the provided category
	 */
	public abstract double getProbability(C category, F item);
	
	/**
	 * <p>Calculates the a-posterior probability for certain items to be within
	 * a specific category.</p>
	 * 
	 * @param category The category the item should be in
	 * @param items The item whose probability should be calculated to 
	 *              be in a certain category
	 * @return The probability of the item being in the provided category
	 */
	public abstract double getProbability(C category, F[] items);
	
	/**
	 * <p>Initializes the appropriate naive Bayes instance based on the provided
	 * <em>{@link ProbabilityCalculation}</em> argument.</p>
	 * 
	 * @param pc The type of naive Bayes classifier to instantiate
	 * @return The initialized naive Bayes classifier
	 */
	public static <F extends Serializable, C extends Serializable> NaiveBayes<F,C> create(ProbabilityCalculation pc, TrainingDataStorageMethod method)
	{
		if (ProbabilityCalculation.NORMAL.equals(pc))
			return new NormalNaiveBayes<F,C>(method);
		else if (ProbabilityCalculation.WEIGHTED.equals(pc))
			return new WeightedNaiveBayes<F,C>(method);
		else if (ProbabilityCalculation.SMOOTHED.equals(pc))
			return new SmoothedNaiveBayes<F,C>(method);
		else
			return new EvenLikelihoodNaiveBayes<F,C>(method);
	}
}
