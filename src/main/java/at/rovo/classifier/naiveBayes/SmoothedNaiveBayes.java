package at.rovo.classifier.naiveBayes;

import java.io.Serializable;

@SuppressWarnings("unused")
public class SmoothedNaiveBayes<F extends Serializable, C extends Serializable> extends NormalNaiveBayes<F, C>
{
    private double smoothingPrior = 0.0;

    protected SmoothedNaiveBayes(TrainingDataStorageMethod method)
    {
        super(method);
    }

    protected SmoothedNaiveBayes(String name)
    {
        super(name);
    }

    public void setSmoothingPrior(double smoothingPrior)
    {
        this.smoothingPrior = smoothingPrior;
    }

    public double getSmoothingPrior()
    {
        return this.smoothingPrior;
    }

    /**
     * Returns the conditional probability of a word given its classification-category
     * <em>[Pr(word|classification)]</em>.
     * <p>
     * This method allows passing an additional argument, the <em>smoothingPrior</em> which prevents zero probabilities
     * in further computations. A <em>smoothingPrior</em> of 0 does not affect the outcome while a
     * <em>smoothingPrior</em> between 0 and 1 is called a Lidstone smoothing while a <em>smoothingPrior</em> of 1 is
     * called a Laplace smoothing.
     *
     * @param feature
     *         Feature or word the probability should be calculated for
     * @param category
     *         The category the feature/word have to be in
     *
     * @return The probability of the feature given its category
     */
    @Override
    public double getConditionalProbability(F feature, C category)
    {
        // P(F|C)
        StringBuilder output = null;
        if (LOG.isDebugEnabled())
        {
            output = new StringBuilder();
            output.append("   P('");
            output.append(feature);
            output.append("'|'");
            output.append(category);
            output.append("') = ");
        }
        long samplesForCategory = this.trainingData.getNumberOfSamplesForCategory(category);
        if (samplesForCategory == 0)
        {
            return 0.;
        }
        // The total number of times this feature appeared in this
        // category divided by the total number of items in this category
        // application of the multinomial event model
        int featCount = this.trainingData.getFeatureCount(feature, category);
        double result = ((double) featCount + this.smoothingPrior) /
                        (samplesForCategory + this.smoothingPrior * this.trainingData.getTotalNumberOfFeatures());
        if (LOG.isDebugEnabled() && output != null)
        {
            LOG.debug("{}{}/{} = {}", output.toString(), featCount, samplesForCategory, result);
        }
        return result;
    }

    /**
     * Returns the conditional probability for words given their classification-category
     * <em>[Pr(word1,word2|classification)]</em>.
     * <p>
     * This method allows passing an additional argument, the <em>smoothingPrior</em> which prevents zero probabilities
     * in further computations. A <em>smoothingPrior</em> of 0 does not affect the outcome while a
     * <em>smoothingPrior</em> between 0 and 1 is called a Lidstone smoothing while a <em>smoothingPrior</em> of 1 is
     * called a Laplace smoothing.
     *
     * @param features
     *         Features or words the probability should be calculated for
     * @param category
     *         The category the features/words have to be in
     *
     * @return The probability of the features given their category
     */
    @Override
    public double getConditionalProbability(F[] features, C category)
    {
        // http://en.wikipedia.org/wiki/Naive_Bayes_classifier
        // P(F1,F2|C) = P(F1|C)*P(F2|C,F1) but as F1 and F2 are statistically
        //                                 independent (naive assumption)
        //              P(F1|C)*P(F2|C)
        StringBuilder output = null;
        if (LOG.isDebugEnabled())
        {
            output = new StringBuilder();
            output.append("   P(");
            for (F feature : features)
            {
                output.append("'");
                output.append(feature);
                output.append("',");
            }
            output.delete(output.length() - 1, output.length());
            output.append("|'");
            output.append(category);
            output.append("') = ");
        }
        double prob = 1;
        for (F feature : features)
        {
            prob *= this.getConditionalProbability(feature, category);
        }
        if (LOG.isDebugEnabled() && output != null)
        {
            LOG.debug("{}{}", output.toString(), prob);
        }
        return prob;
    }
}
