package at.rovo.classifier.naiveBayes;

import java.io.File;
import java.io.Serializable;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * <p>Implementation of a naive Bayes classifier which uses probabilities extracted
 * from training examples to divide certain candidates into their appropriate classes.</p>
 * <p>The current implementation only supports a multinomial event model.</p>
 * <p>This implementation was originally based on the code presented in "Programming 
 * Collective Intelligence" by Toby Segaran (ISBN: 978-0-596-52932-1; 2007) but has
 * changed massively since the beginning.</p>
 * 
 * @author Roman Vottner
 */
public class NormalNaiveBayes<F extends Serializable, C extends Serializable> extends NaiveBayes<F,C>
{
	protected static Logger logger = LogManager.getLogger(NormalNaiveBayes.class.getName());
	
	// for faster lookup
	/** Contains the probability for each category */
	private Map<C, Double> catProb = null;
	protected NBTrainingData<F,C> trainingData = null;
	protected TrainingDataStorageMethod method = null;
	
	/**
	 * <p>Creates a new default instance of this class.</p>
	 */
	protected NormalNaiveBayes(TrainingDataStorageMethod method)
	{
		super();
		this.method = method;
		this.trainingData = NBTrainingData.create(method);
		super.trainingData = this.trainingData;
		this.catProb = new Hashtable<C,Double>();
	}
	
	/**
	 * <p>Create a new instance of this class and sets its name to the value of 
	 * the provided argument.</p>
	 * 
	 * @param name The name of this instance
	 */
	protected NormalNaiveBayes(String name)
	{
		super(name);
		this.catProb = new Hashtable<C,Double>();
	}
				
	@Override
	public boolean loadData(File serializedObject)
	{
		return this.trainingData.loadData(serializedObject);
	}
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//                               P R O B A B I L I T Y    S E C T I O N
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	 * <p>Calculates the a-priori probability for a certain category.</p>
	 * <p>It therefore counts all entries labeled with this category and
	 * divides the sum through the total number of entries.</p>
	 * 
	 * @param category The category whose probability should be calculated
	 * @return The probability of a certain category
	 */
	public double getCategoryProbability(C category)
	{
		// P(C) = number of items in category / total number in all categories
		if (this.catProb.get(category) == null)
			this.catProb.put(category, (double)this.trainingData.getNumberOfSamplesForCategory(category) 
					/ this.trainingData.getTotalNumberOfSamples());
		if (logger.isDebugEnabled())
			logger.debug("   P('"+category+"') = "+this.trainingData.getNumberOfSamplesForCategory(category)+
					"/"+this.trainingData.getTotalNumberOfSamples()+" = "+this.catProb.get(category));
		return this.catProb.get(category);
	}
	
	/**
	 * <p>Calculates the probability for a certain feature.</p>
	 * <p>This method runs through every defined category and sums up the 
	 * conditional probability of the feature given the category <em>P(X|C)</em> 
	 * times the probability of the category <em>P(C)</em> itself.</p>
	 * 
	 * @param feature The feature whose probability should be calculated
	 * @return The probability of a certain feature
	 */
	public double getFeatureProbability(F feature)
	{
		// P(F) = P(F|C1)*P(C1) + ... + P(F|Cn)*P(Cn)
		String output = null;
		if (logger.isDebugEnabled())
			output = "   P('"+feature+"') = ";
		double p = 0;
		// Runs through every defined category
		// and sums up the conditional probability P(X|C) times the probability of the category P(C)
		for (C category : this.trainingData.getCategories())
		{
			if (logger.isDebugEnabled())
				output += "P('"+feature+"'|'"+category+"')*P('"+category+"') + ";
			p += this.getConditionalProbability(feature, category) * this.getCategoryProbability(category);
		}
		if (logger.isDebugEnabled())
			logger.debug(output.substring(0,output.length()-3)+": "+p);
		return p;
	}
	
	/**
	 * <p>Calculates the probability for a certain set of features.</p>
	 * 
	 * @param feature The feature whose probability should be calculated
	 * @return The combined probability of the provided features
	 */
	public double getFeatureProbability(F[] features)
	{
		// P(F1,F2) = P(F1|C1)*P(F2|C1)*P(C1) + P(F1|C2)*P(F2|C2)*P(C2) + ... + P(F1|Cn)*P(F2|Cn)*P(Cn)
		double p = 0;
		for (C category : this.trainingData.getCategories())
		{
			double _p = this.getCategoryProbability(category);
			for (F feature : features)
			{
				_p *= this.getConditionalProbability(feature, category);
			}
			p += _p;
		}
		return p;
	}
		
	/**
	 * <p>Returns the conditional probability of a word given its 
	 * classification-category <em>[Pr(word|classification)]</em>.</p>
	 * 
	 * @param feature Feature or word the probability should be calculated for
	 * @param category The category the feature/word have to be in
	 * @return The probability of the feature given its category
	 */
	public double getConditionalProbability(F feature, C category)
	{
		// P(F|C)
		String output = null;
		if (logger.isDebugEnabled())
			output = "   P('"+feature+"'|'"+category+"') = ";
		long samplesForCategory = this.trainingData.getNumberOfSamplesForCategory(category);
		if (samplesForCategory == 0)
			return 0.;
		// The total number of times this feature appeared in this
		// category divided by the total number of items in this category
		int featCount = this.trainingData.getFeatureCount(feature, category);
		double result = (double)featCount/samplesForCategory;
		if (logger.isDebugEnabled())
			logger.debug(output+featCount+"/"+samplesForCategory+" = "+result);
		return result;
	}
		
	/**
	 * <p>Returns the conditional probability for words given their 
	 * classification-category <em>[Pr(word1,word2|classification)]</em>.</p>
	 * 
	 * @param features Features or words the probability should be calculated for
	 * @param category The category the features/words have to be in
	 * @return The probability of the features given their category
	 */
	public double getConditionalProbability(F[] features, C category)
	{
		// http://en.wikipedia.org/wiki/Naive_Bayes_classifier
		// P(F1,F2|C) = P(F1|C)*P(F2|C,F1) but as F1 and F2 are statistically 
		//                                 independent (naive assumption)
		//              P(F1|C)*P(F2|C)
		String output = null;
		if (logger.isDebugEnabled())
		{
			output = "   P(";
			for (F feature : features)
				output += "'"+feature+"',";
			output = output.substring(0, output.length()-1);
			output += "|'"+category+"') = ";
		}
		double prob = 1;
		for (F feature : features)
			prob *= this.getConditionalProbability(feature, category);
		if (logger.isDebugEnabled())
			logger.debug(output+prob);
		return prob;
	}
		
	@Override
	public double getProbability(C category, F item)
	{
		// given a specific word (f.e. 'money'), what's the probability that it fits into a specific category (f.e. 'good')
		// P('good'|'money') = [P('money'|'good')*P('good')]/P('money')
		//
		// given a specific document, what's the probability that it fits into this category
		// P(Category|Document) = P(Document|Category)*P(Category)/P(Document)
		//
		// Further P(F) = P(F|C1) + ... + P(F|Cn)
		//
		// P(C|F) = [P(F|C)*P(C)]/P(F)		
		if (this.trainingData.containsCategory(category))
		{
			double catProb = this.getCategoryProbability(category);
			double condProb = this.getConditionalProbability(item, category);
			double featProb = this.getFeatureProbability(item);

			if (featProb == 0)
				return 0.;
			return condProb*catProb / featProb;
		}
		else
		{
			// we have never seen this concept before
			return 0.;
		}
	}
	
	@Override
	public double getProbability(C category, F[] items)
	{
		// http://en.wikipedia.org/wiki/Naive_Bayes_classifier
		// given specific words, what's the probability that they fit into this category
		// P('good'|'money', 'casino') = P('money','casino'|'good')*P('good') / P('money','casino') 
		//                             = P('money','casino'|'good')*P('good') / 
		//                                 [P('money'|'good')*P('casino'|'good')*P('good') +
		//                                  P('money'|'bad')*P('casino'|'bad')*P('bad')]
		//
		// P(C|F1,F2) = [P(F1,F2|C)*P(C)] / P(F1, F2)			
		if (this.trainingData.containsCategory(category))
		{
			double catProb = this.getCategoryProbability(category);
			double condProb = this.getConditionalProbability(items, category);
			double featProb = this.getFeatureProbability(items);
			
			if (featProb == 0)
				return 0.;
			return condProb*catProb / featProb;
		}
		else
		{
			// we haven't seen this category yet
			return 0.;
		}
	}
	
	@Override
	public void train(F item, C category)
	{
		if (this.trainingData == null)
			this.trainingData = NBTrainingData.create(this.method);
		// increment the count for every feature with this category
		this.trainingData.incrementFeature(item, category);
		this.trainingData.incrementNumberOfSamplesForCategory(category);
	}
	
	@Override
	public void train(F[] items, C category)
	{
		if (this.trainingData == null)
			this.trainingData = NBTrainingData.create(this.method);
		for (F item : items)
			this.trainingData.incrementFeature(item, category);
		this.trainingData.incrementNumberOfSamplesForCategory(category);
	}
	
	@Override
	public void train(List<F> items, C category)
	{
		if (this.trainingData == null)
			this.trainingData = NBTrainingData.create(this.method);
		for (F item : items)
			this.trainingData.incrementFeature(item, category);
		this.trainingData.incrementNumberOfSamplesForCategory(category);
	}
	
	@Override
	public C classify(F item)
	{
		Map<C, Double> probs = new Hashtable<C, Double>();
		// find the category with the highest probability
		double max = 0.0f;
		C best = null;
		for (C cat : this.trainingData.getCategories())
		{
			probs.put(cat, this.getProbability(cat, item));
			if (probs.get(cat) > max)
			{
				max = probs.get(cat);
				best = cat;
			}
		}
		
		// Make sure the probability exceeds threshold*next best
		for (C cat : probs.keySet())
		{
			if (cat==best) 
				continue;
			if (probs.get(cat)*this.getThreshold(best)>probs.get(best))
				return null;
		}
		return best;
	}
	
	@Override
	public C classify(F[] items)
	{
		Map<C, Double> probs = new Hashtable<C, Double>();
		// find the category with the highest probability
		double max = 0.0f;
		C best = null;
		for (C cat : this.trainingData.getCategories())
		{
			probs.put(cat, this.getProbability(cat, items));
			if (probs.get(cat) > max)
			{
				max = probs.get(cat);
				best = cat;
			}
		}
		
		// Make sure the probability exceeds threshold*next best
		for (C cat : probs.keySet())
		{
			if (cat==best) 
				continue;
			if (probs.get(cat)*this.getThreshold(best)>probs.get(best))
				return null;
		}
		return best;
	}
}
