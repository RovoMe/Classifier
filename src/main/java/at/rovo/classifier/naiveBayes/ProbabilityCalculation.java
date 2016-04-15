package at.rovo.classifier.naiveBayes;

/**
 * Specifies the desired naive Bayes probability calculation method. <p> The methods differ in their behavior of
 * returning special probabilities for unknown or sparse features.</p>
 *
 * @author Roman Vottner
 */
public enum ProbabilityCalculation
{
    /**
     * The probability of a feature will be between 0 and 1 with unknown features set to 0. Sparse features may act
     * extreme. Moreover if a feature has no occurrence in one class it will return 0. This may have effects on
     * multiplied probabilities.
     */
    NORMAL,

    /**
     * Probabilities are weighted depending on the number of occurrences found so far. An unknown feature will have
     * equal probability among all known classes. Sparse features will lie around the medium.
     */
    WEIGHTED,

    /**
     * Will prevent 0 probabilities for features with no occurrences in a certain class, but the probability of new,
     * sparse features will not be placed around the medium.
     */
    SMOOTHED,

    /**
     * Will set the probability of unknown features to 0.5, but will not modify the probability of sparse features or
     * features that do not occur within a specific class.
     */
    EVEN_LIKELIHOOD
}
