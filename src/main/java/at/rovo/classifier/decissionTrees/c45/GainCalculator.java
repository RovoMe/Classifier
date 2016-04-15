/**
 * @(#)GainCalculator.java 1.5.2 09/03/29
 */

package at.rovo.classifier.decissionTrees.c45;

/**
 * A calculator specifically used to compute Gain, GainRatio and Entropy criteria for different data distribution.
 *
 * @author Ping He
 * @author Xiaohua Xu
 */
public class GainCalculator
{

    private GainCalculator()
    {
    }

    /**
     * Compute Gain for the specified data distribution
     *
     * @param stateEntropy
     *         the entropy of the data before it splits on an attribute
     * @param branchDistribution
     *         the weight distribution of the data in different attribute branch values
     * @param classDistribution
     *         the weight distribution of the data in different classes and different attribute branch values.
     * @param unknownRatio
     *         the ratio of the missing data
     *
     * @return Gain value of the specified data distribution
     */
    public static float computeGain(float stateEntropy, float[] branchDistribution, float[][] classDistribution,
                                    float unknownRatio)
    {
        int reasonableSubset = 0;
        for (int i = 1; i < branchDistribution.length; i++)
        {
            if (branchDistribution[i] > Parameter.MINWEIGHT - Parameter.PRECISION)
            {
                reasonableSubset++;
            }
        }
        if (reasonableSubset < 2)
        {
            return -Parameter.EPSILON;
        }

        float x = computeStandardEntropy(branchDistribution, classDistribution);
        return (1 - unknownRatio) * (stateEntropy - x);
    }

    /**
     * Compute Entropy of the specified data distribution
     *
     * @param classDistribution
     *         the weight distribution of the data in different classes and different attribute branch values.
     * @param knownWeight
     *         the total weight of the known data
     *
     * @return Entropy of the specified data distribution
     */
    public static float computeStateEntropy(float[][] classDistribution, float knownWeight)
    {

        float informationSum = 0.0f;
        for (int i = 0; i < classDistribution[0].length; i++)
        {
            float classFrequency = 0.0f;
            for (int j = 1; j < classDistribution.length; j++)
            {
                classFrequency += classDistribution[j][i];
            }
            informationSum += classFrequency * log(classFrequency);
        }

        return (knownWeight * log(knownWeight) - informationSum) / knownWeight;
    }

    /**
     * Compute splitInfo for the specified data distribution
     *
     * @param branchDistribution
     *         the weight distribution of the data in different attribute branch values
     * @param totalWeight
     *         the total weight of the data
     *
     * @return splitInfo value of the data distribution
     */
    public static float computeSplitInfo(float[] branchDistribution, float totalWeight)
    {
        return computeTotalInformation(branchDistribution, totalWeight) / totalWeight;
    }

    /**
     * Compute the entropy of the specified branch distribution
     *
     * @param branchDistribution
     *         the weight distribution of the cases in different branch values
     * @param totalWeight
     *         the total weight of all the cases
     *
     * @return the splitInfo evaluation
     */
    private static float computeTotalInformation(float[] branchDistribution, float totalWeight)
    {
        float informationSum = 0;
        for (float frequency : branchDistribution)
        {
            informationSum += frequency * log(frequency);
        }
        return totalWeight * log(totalWeight) - informationSum;
    }

    /**
     * Compute GainRatio for the provided Gain and splitInfo value
     *
     * @param gain
     *         Gain of the data
     * @param splitInfo
     *         splitInfo of the data
     * @param minGain
     *         the threshold value as a valid Gain
     *
     * @return GainRatio of the data
     */
    public static float computeGainRatio(float gain, float splitInfo, float minGain)
    {
        if (gain >= minGain - Parameter.PRECISION && splitInfo > Parameter.PRECISION)
        {
            return gain / splitInfo;
        }
        return -Parameter.EPSILON;
    }

    /**
     * Compute log<sub>2</sub>(value)
     */
    public static float log(float value)
    {
        return (value <= 0) ? 0.0f : (float) Math.log10(value) / log2;
    }

    /**
     * Compute the entropy of the data after it splits on a attribute
     *
     * @param branchDistribution
     *         the weight distribution of the cases in different branch values
     * @param classDistribution
     *         the weight distribution of the cases in different classes among each different branch value.
     */
    private static float computeStandardEntropy(float[] branchDistribution, float[][] classDistribution)
    {
        float informationSum = 0.0f;
        float knownWeight = 0f;

        for (int i = 1; i < branchDistribution.length; i++)
        {
            float infoPart = (branchDistribution[i] == 0) ? 0 :
                    computeTotalInformation(classDistribution[i], branchDistribution[i]);
            informationSum += infoPart;
            knownWeight += branchDistribution[i];
        }

        return informationSum / knownWeight;
    }

    // the value of log10(2)
    private static float log2 = (float) Math.log10(2.0);
}