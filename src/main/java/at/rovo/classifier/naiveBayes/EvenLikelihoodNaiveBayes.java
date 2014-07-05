package at.rovo.classifier.naiveBayes;

import java.io.Serializable;

public class EvenLikelihoodNaiveBayes<F extends Serializable, C extends Serializable> extends NormalNaiveBayes<F,C>
{
	protected EvenLikelihoodNaiveBayes(TrainingDataStorageMethod method)
	{
		super(method);
	}
	
	protected EvenLikelihoodNaiveBayes(String name)
	{
		super(name);
	}
	
	/**
	 * <p>Calculates the a-posterior probability of a certain item to be within
	 * a specific category. This method threats yet unknown elements with even
	 * likelihood, so returns 0.5 for two categories, but calculates the normal 
	 * probability for items that are already included in the 'vocabulary' at least
	 * once.</p>
	 * <p>This modification was proposed by Jeff Pasternack and Dan Roth in the
	 * paper 'Extracting Article Text from the Web with Maximum Subsequence 
	 * Segmentation'.</p>
	 * 
	 * @param item The item whose probability should be calculated to 
	 *             be in a certain category
	 * @param category The category the item should be in
	 * @return The probability of the item being in the provided category
	 */
	@Override
	public double getProbability(C category, F item)
	{
		if (this.trainingData.getFeatureCount(item)==0)
		{
			return 0.5;
		}
		
		return super.getProbability(category, item);
	}	
}
