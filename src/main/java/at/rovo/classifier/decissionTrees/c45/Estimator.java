package at.rovo.classifier.decissionTrees.c45;

/**
 * An estimator specifically used to estimate extra error ratio for c4.5's
 * error-based pruning strategy.
 *
 * @author Ping He
 * @author Xiaohua Xu
 * @see DecisionTree#prune()
 */
public class Estimator
{
    /** The confidence level table which can be used to interpolate the user specified confidence value **/
    private static final float[] confidenceLevels = {0f, 0.001f, 0.005f, 0.01f, 0.05f, 0.10f, 0.2f, 0.4f, 1.00f};
    /** The standard deviation table used to interpolate the user specified standard deviation **/
    private static final float[] standardDeviations = {4.0f, 3.09f, 2.58f, 2.33f, 1.65f, 1.28f, 0.84f, 0.25f, 0.00f};

    /** The default confidence value **/
    private static float CONFIDENCE = Parameter.CONFIDENCE;
    /** The default squared standard deviation **/
    private static float SQUAREDSTD = Parameter.SQUAREDSTD;

    private Estimator()
    {
    }

    /**
     * Get the pruning confidence level
     */
    public static float getConfidence()
    {
        return CONFIDENCE;
    }

    /**
     * Set the pruning confidence level with the specified value.
     */
    public static void setConfidence(float confidence)
    {
        // Testify the validation of input argument
        if (confidence < 0 || confidence > 1)
        {
            throw new IllegalArgumentException("Confidence level must belong to (0,1)");
        }
        CONFIDENCE = confidence;

        // Find the index range that the specified CONFIDENCE belongs to
        int i = 0;
        while (CONFIDENCE > confidenceLevels[i])
        {
            i++;
        }

        // Interpolate the standard standardDeviations of the specified CONFIDENCE
        float std = standardDeviations[i - 1] +
                    (standardDeviations[i] - standardDeviations[i - 1]) * (CONFIDENCE - confidenceLevels[i - 1]) /
                    (confidenceLevels[i] - confidenceLevels[i - 1]);
        // The squared value of std
        SQUAREDSTD = std * std;
    }

    /**
     * Estimate the extra error ratio for the specified basic error ratio of the
     * data. Transplanted from Quinlan's c4.5(Release 8).
     *
     * @param totalWeight
     *         the total weight of data distributed on the tree node.
     * @param leafError
     *         the leaf error of the tree node when considered as a leaf node.
     *
     * @return The estimated extra error ratio of the data.
     */
    public static float getExtraError(float totalWeight, float leafError)
    {
        if (leafError < Parameter.PRECISION)
        {
            return totalWeight * (1 - (float) Math.exp(Math.log(CONFIDENCE) / totalWeight));
        }
        else if (leafError < 1 - Parameter.PRECISION)
        {
            float val = totalWeight * (1 - (float) Math.exp(Math.log(CONFIDENCE) / totalWeight));
            return val + leafError * (getExtraError(totalWeight, 1.0f) - val);
        }
        else if (leafError + 0.5f >= totalWeight)
        {
            return 0.67f * (totalWeight - leafError);
        }
        else
        {
            float Pr = (leafError + 0.5f + SQUAREDSTD / 2
                        + (float) Math.sqrt(SQUAREDSTD * ((leafError + 0.5f) * (1 - (leafError + 0.5f) / totalWeight) + SQUAREDSTD / 4))
                       ) / (totalWeight + SQUAREDSTD);
            return (totalWeight * Pr - leafError);
        }
    }
}
