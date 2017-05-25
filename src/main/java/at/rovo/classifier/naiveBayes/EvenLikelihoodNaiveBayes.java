package at.rovo.classifier.naiveBayes;

import java.io.Serializable;

/**
 * This class calculates the a-posterior probability of items based on its affiliation to a certain category based on
 * the number of occurrences of that item in the respective category. Items, which have not yet occurred in the
 * calculation, are treated with even likelihood among each category. The probability of these items is therefore 1/n
 * where n is the number of prior known categories.
 *
 * @param <F>
 *         The type of the features or words
 * @param <C>
 *         The type of the categories or classes
 */
public class EvenLikelihoodNaiveBayes<F extends Serializable, C extends Serializable> extends NormalNaiveBayes<F, C>
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
     * Calculates the a-posterior probability of a certain item to be within a specific category. This method threats
     * yet unknown elements with even likelihood, so returns 0.5 for two categories, but calculates the normal
     * probability for items that are already included in the 'vocabulary' at least once.
     * <p/>
     * This modification was proposed by Jeff Pasternack and Dan Roth in the paper 'Extracting Article Text from the Web
     * with Maximum Subsequence Segmentation'.</p>
     *
     * @param items
     *         The item whose probability should be calculated to be in a certain category
     * @param category
     *         The category the item should be in
     *
     * @return The probability of the item being in the provided category
     */
    @Override
    public double getProbability(C category, F ... items)
    {
        if (items.length == 1)
        {
            if (this.trainingData.getFeatureCount(items[0]) == 0)
            {
                return 0.5;
            }
        }

        return super.getProbability(category, items);
    }
}
