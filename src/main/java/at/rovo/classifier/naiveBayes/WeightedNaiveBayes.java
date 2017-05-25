package at.rovo.classifier.naiveBayes;

import java.io.Serializable;

/**
 * This implementation of a naive Bayes classifier calculates the probability of certain features or words among
 * categories or classes these features occur. In contrast to {@link NormalNaiveBayes}, this implementation calculates
 * the weighted probability for certain words or features. The weighted probability is the sum of each conditional
 * probability of a word to appear in a certain category times the probability of that respective category
 *
 * @param <F>
 *         The type of the features or words
 * @param <C>
 *         The type of the categories or classes
 */
@SuppressWarnings("unused")
public class WeightedNaiveBayes<F extends Serializable, C extends Serializable> extends NormalNaiveBayes<F, C>
{
    protected WeightedNaiveBayes(TrainingDataStorageMethod method)
    {
        super(method);
    }

    protected WeightedNaiveBayes(String name)
    {
        super(name);
    }

    /**
     * Calculates the weighted Probability for a certain feature.
     * <p/>
     * This method runs through every defined category and sums up the conditional probability of the feature given the
     * category <em>P(X|C)</em> times the probability of the category <em>P(C)</em> itself.
     *
     * @param feature
     *         The feature whose probability should be calculated
     *
     * @return The probability of a certain feature
     */
    @Override
    public double getFeatureProbability(F feature)
    {
        // P(F) = P(F|C1)*P(C1) + ... + P(F|Cn)*P(Cn)
        double p = 0;
        // Runs through every defined category
        // and sums up the conditional probability P(X|C) times the probability of the category P(C)
        for (C category : this.trainingData.getCategories())
        {
            p += this.getConditionalProbability(feature, category) * this.getCategoryProbability(category);
        }
        return p;
    }

    /**
     * Calculates the weighted Probability for a certain set of features.
     *
     * @param features
     *         The feature whose probability should be calculated
     *
     * @return The probability of a certain feature
     */
    @Override
    public double getFeatureProbability(F[] features)
    {
        // P(F1,F2) = P(F1|C1)*P(F2|C1)*P(C1) + ... + P(F1|Cn)*P(F2|Cn)*P(Cn)
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
     * As probabilities may behave extreme on small sample data (f.e. only a single occurrence of a word yields either a
     * total agreement or a total rejection for a certain class) or on features that occur rarely in training examples,
     * assumed probabilities are used to soften the effect on small data or rare features.
     * <p/>
     * The idea behind assumed probabilities is to grant a feature the same chance to be within every category from the
     * start on and move it towards a certain category through training. For a classifier containing two categories 0.5
     * is a good assumed probability.
     * <p/>
     * Moreover a weight can be introduced to increase the importance of certain features. A weight of 1 gives the
     * assumed probability the same weight as one word.
     *
     * @param feature
     *         Feature or word the probability should be calculated for
     * @param category
     *         The category the feature/word have to be in
     * @param weight
     *         The weight of the assumed probability to take influence on the importance of a certain feature. A weight
     *         of 1 means it has the same weight as one word.
     * @param assumedProb
     *         The assumed probability of a feature if this feature has not yet been seen before. Typically this
     *         parameter is set to 0.5 for a two-sided category like 'in' and 'out' or 'good' and 'bad' as this feature
     *         should be equally possible to be in either of these categories
     *
     * @return The weighted probability of a feature given a certain category
     */
    public double getConditionalProbability(F feature, C category, double weight, double assumedProb)
    {
        // P(F|C)
        //
        // (weight*assumeProb + count*condProb) / (count+weight)
        //
        // Count the number of times this feature has appeared in all categories
        long count = this.trainingData.getFeatureCount(feature);
        double condProb = super.getConditionalProbability(feature, category);
        double prob = (weight * assumedProb + count * condProb) / (count + weight);
        if (LOG.isDebugEnabled())
        {
            LOG.debug(
                    "   wP('{}'|'{}') = (weight={}*assumedProb={} + count={}*condProb={}) / (count={}+weight={}) = {}",
                    feature, category, weight, assumedProb, count, condProb, count, weight, prob);
        }
        return prob;
    }

    /**
     * As probabilities may behave extreme on small sample data (f.e. only a single occurrence of a word yields either a
     * total agreement or a total rejection for a certain class) or on features that occur rarely in training examples,
     * assumed probabilities are used to soften the effect on small data or rare features.
     * <p/>
     * The idea behind assumed probabilities is to grant a feature the same chance to be within every category from the
     * start on and move it towards a certain category through training. This method assigns a base probability of
     * 1/'numbers of categories' to every feature.
     * <p/>
     * Moreover a weight can be introduced to increase the importance of certain features. This method assigns a weight
     * of 1 to the assumed probability which gives it the same weight as one word.
     *
     * @param feature
     *         Feature or word the probability should be calculated for
     * @param category
     *         The category the feature/word have to be in
     *
     * @return The weighted probability of a feature given a certain category
     */
    @Override
    public double getConditionalProbability(F feature, C category)
    {
        // P(F|C)
        return this.getConditionalProbability(feature, category, 1.0, 1. / this.trainingData.getNumberOfCategories());
    }

    /**
     * Calculates the conditional probability of multiple features given their category - <em>Pr(feature1,
     * feature2|category)</em>.
     * <p/>
     * This is where NaiveBayes does derive its name from as items (e.g. words) are naively considered to be independent
     * form each other which in reality is not quite true.
     * <p/>
     * It therefore multiplies the weighted probabilities for each feature contained in item with the provided
     * category.
     *
     * @param features
     *         List of features whose conditional probability for a certain given category should be calculated
     * @param category
     *         The category used to calculate the probability for the needed features
     * @param weight
     *         The weight of the assumed probability to take influence on the importance of a certain feature. A weight
     *         of 1 means it has the same weight as one word.
     * @param assumedProb
     *         The assumed probability of a feature if this feature has not yet been seen before. Typically this
     *         parameter is set to 0.5 for a two-sided category like 'in' and 'out' or 'good' and 'bad' as this feature
     *         should be equally possible to be in either of these categories
     *
     * @return The conditional probability <em>P(X,Y|C)</em> of all the features given their category
     */
    public double getConditionalProbability(F[] features, C category, double weight, double assumedProb)
    {
        // P(F1,F2|C) = P(F1|C)*P(F2|C)
        // Multiply the probabilities of all the features together
        double p = 1.0;
        for (F feature : features)
        {
            //			if (this.trainingData.getData().get(category).getFeatures().containsKey(feature))
            if (this.trainingData.getFeatureCount(feature, category) > 0)
            {
                p *= this.getConditionalProbability(feature, category, weight, assumedProb);
            }
            else
            {
                p *= 1. / this.trainingData.getNumberOfCategories();
            }
        }
        return p;
    }

    /**
     * Calculates the conditional probability of multiple features given their category - <em>Pr(feature1,
     * feature2|category)</em>.
     * <p/>
     * This is where NaiveBayes does derive its name from as items (e.g. words) are naively considered to be independent
     * form each other which in reality is not quite true.
     * <p/>
     * It therefore multiplies the weighted probabilities for each feature contained in item with the provided
     * category.
     *
     * @param features
     *         List of features whose conditional probability for a certain given category should be calculated
     * @param category
     *         The category used to calculate the probability for the needed features
     *
     * @return The conditional probability <em>P(X,Y|C)</em> of all the features given their category
     */
    @Override
    public double getConditionalProbability(F[] features, C category)
    {
        // P(F1,F2|C) = P(F1|C)*P(F2|C)
        // Multiply the probabilities of all the features together
        double p = 1.0;
        for (F s : features)
        {
            p *= this.getConditionalProbability(s, category);
        }
        return p;
    }

    //	/**
    //	 * Calculates the a-posterior probability of a certain item to be within a specific category.
    //	 *
    //	 * @param item The item whose probability should be calculated to
    //	 *             be in a certain category
    //	 * @param category The category the item should be in
    //	 * @return The probability of the item being in the provided category
    //	 */
    //	@Override
    //	public double getProbability(C category, F item)
    //	{
    //		// given specific word, what's the probability that it fits into this category
    //		// wP('good'|'money') = [wP('money'|'good')*P('good')] / wP('money')
    //		//                    = [wP('money'|'good')*P('good')] /
    //		//                        [wP('money'|'good')*P('good') + wP('money'|'bad')*P('bad')]
    //		//
    //		// wP(C|F) = [wP(F|C)*P(C)] / wP(F)
    //		if (this.trainingData.containsCategory(category))
    //		{
    //			double catProb = this.getCategoryProbability(category);
    //			double condProb = this.getConditionalProbability(item, category);
    //			double featProb = this.getFeatureProbability(item);
    //
    //			if (featProb == 0)
    //			{
    //				return 1. / this.trainingData.getNumberOfCategories();
    //			}
    //			return condProb*catProb / featProb;
    //		}
    //		else
    //		{
    //			// we haven't seen this category yet
    //			return 1./this.trainingData.getNumberOfCategories();
    //		}
    //	}

    /**
     * Calculates the a-posterior probability for certain items to be within a specific category.
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
        // given specific words, what's the probability that they fit into this category
        // wP('good'|'money', 'casino') = [wP('money','casino'|'good')*P('good')] / wP('money','casino')
        //                              = [wP('money'|'good')*P('casino'|'good')*P('good')] /
        //                                  [wP('money'|'good')*wP('casino'|'good')*P('good') +
        //                                   wP('money'|'bad')*wP('casino'|'bad')*P('bad')]
        //
        // wP(C|F1,F2) = [wP(F1,F2|C)*P(C)] / wP(F1, F2)
        if (this.trainingData.containsCategory(category))
        {
            // probability for a certain category
            double catProb = this.getCategoryProbability(category);
            double condProb = this.getConditionalProbability(items, category);
            double featProb = this.getFeatureProbability(items);

            if (featProb == 0)
            {
                return 1. / this.trainingData.getNumberOfCategories();
            }
            return condProb * catProb / featProb;
        }
        else
        {
            // we haven't seen this category yet
            return 1. / this.trainingData.getNumberOfCategories();
        }
    }
}
