/**
 * @(#)DiscreteAttributeDelegate.java 1.5.2 09/03/29
 */

package at.rovo.classifier.decissionTrees.c45;

import at.rovo.classifier.dataset.DiscreteAttribute;
import java.util.Arrays;

/**
 * A delegate of a discrete attribute, containing some essential processed information of the discrete attribute to
 * speed up the tree building process.
 *
 * @author Ping He
 * @author Xiaohua Xu
 */
public class DiscreteAttributeDelegate extends AttributeDelegate
{
    /**
     * All the attributes share a common cases array and weight array The original sequence of the train data.
     */
    private int[] cases;
    /**
     * The weight of the train data
     */
    private float[] weight;
    /**
     * The branch index of each train data's value on the corresponding attribute
     */
    private int[] branch;
    /**
     * The corresponding discrete attribute of the delegate
     */
    private DiscreteAttribute attribute;

    /**
     * Initialize a delegate for the specified discrete attribute.
     * <p>
     * The initialization of a discrete attribute delegate is a preprocessing of the attribute values on the discrete
     * attribute. It mainly extracts the branch indices of the attribute values according to their nominal values.
     * Branch 0 is especially kept for the missing data.
     *
     * @param attribute
     *         The corresponding discrete attribute
     */
    public DiscreteAttributeDelegate(DiscreteAttribute attribute)
    {
        super();

        this.attribute = attribute;
        // Determine the branch index of each data
        String[] nominalValues = attribute.getNominalValues();
        String[] data = attribute.getData();
        this.branch = new int[data.length];
        for (int i = 0; i < data.length; i++)
        {
            if (data[i].equals("?"))
            {
                branch[i] = 0;
                setHasMissingData(true);
            }
            // branch 0 is kept for missing data
            else
            {
                for (int j = 0; j < nominalValues.length; j++)
                {
                    if (nominalValues[j].equals(data[i]))
                    {
                        branch[i] = j + 1;
                    }
                }
            }
        }
    }

    public void setCasesWeight(int[] casesValue, float[] weightValue)
    {
        this.cases = casesValue;
        this.weight = weightValue;
    }

    /**
     * @return If the attribute is evaluated as an invalid test attribute, then <i>null</i> is returned;<br> Otherwise,
     * a 1-by-4 float array with<br> &nbsp;&nbsp;&nbsp; the 1<sup>st</sup> element recording the Gain,<br>
     * &nbsp;&nbsp;&nbsp; the 2<sup>nd</sup> element recording the splitInfo and<br> &nbsp;&nbsp;&nbsp; the
     * 3<sup>rd</sup> and 4<sup>th</sup> elements recording invalid indices to keep consistency with
     * ContinuousAttributeDelegate.
     */
    public float[] evaluate(int first, int last, AttributeDelegate classAttributeDelegate)
    {
        // This variable records the total weight of the [first last) cases
        float totalWeight = 0.0f;
        // Get the number of split branches if the corresponding attribute is
        // evaluated as a test attribute
        int branchCount = attribute.getNominalValuesCount();
        // This variable records the weight distribution of the [first last)
        // cases in different branches of the current attribute
        float[] branchDistri = new float[branchCount + 1];
        // This variable records the weight distribution of the [first last)
        // cases in different classes of the different branches of the attribute
        float[][] branchClassDistri = new float[branchCount + 1][classAttributeDelegate.getBranchCount()];
        // The minimal weight of the known cases
        float minKnownWeight = Parameter.MINWEIGHT;

        // Initialize branchDistri and branchClassDistri
        Arrays.fill(branchDistri, 0);
        for (float[] distri : branchClassDistri)
        {
            Arrays.fill(distri, 0f);
        }

        // Compute branchDistri and its branchClassDistri
        // Here branch index 0 means missing data
        for (int i = first; i < last; i++)
        {
            totalWeight += weight[cases[i]];
            int branchIndex = branch[cases[i]];
            branchDistri[branchIndex] += weight[cases[i]];
            // The class attribute has no missing value
            int classLabel = classAttributeDelegate.getClassBranch(cases[i]);
            branchClassDistri[branchIndex][classLabel] += weight[cases[i]];
        }

        // Compute the weight of the known cases
        float knownWeight = totalWeight - branchDistri[0];
        // If there is too much missing data on this attribute, return nothing
        // to try the next attribute
        if (knownWeight < 2 * minKnownWeight)
        {
            return null;
        }

        // Compute the ratio of the unknown weight
        float unknownRatio = branchDistri[0] / totalWeight;

        // Construct the result array recording gain and splitInfo
        // float[] result = new float[4];
        /**
         * Compute the entropy of the tree node as a Leaf Then compute the gain
         * and splitInfo of the tree node as an InternalNode with the current
         * attribute as its test attribute
         */
        float stateEntropy = GainCalculator.computeStateEntropy(branchClassDistri, knownWeight);
        float gain = GainCalculator.computeGain(stateEntropy, branchDistri, branchClassDistri, unknownRatio);
        float splitInfo = GainCalculator.computeSplitInfo(branchDistri, totalWeight);

        // The last two -1 are filled for the consistency output with
        // ContinuousAttributeDelegate's evaluate
        // It means there are no valid split ranks for discrete attribute
        return new float[] {gain, splitInfo, -1, -1};
    }

    public int groupForward(int first, int last, int groupBranch, float[] branchDistri)
    {
        // The first branch is kept for missing data
        int branchIndex = groupBranch + 1;

        int i, j;
        for (i = first, j = last - 1; i <= j; )
        {
            while (i <= j && branch[cases[i]] == branchIndex)
            {
                branchDistri[branchIndex] += weight[cases[i]];
                i++;
            }
            while (i <= j && branch[cases[j]] != branchIndex)
            {
                j--;
            }

            if (i <= j)
            {
                int tmp = cases[i];
                cases[i] = cases[j];
                cases[j] = tmp;

                branchDistri[branchIndex] += weight[cases[i]];
                i++;
                j--;
            }
        }

        return i;
    }

    public int groupBackward(int first, int last)
    {
        int i, j;
        int branchIndex = 0;

        for (i = last - 1, j = first; i >= j; )
        {
            while (i >= j && branch[cases[i]] == branchIndex)
            {
                i--;
            }
            while (i >= j && branch[cases[j]] != branchIndex)
            {
                j++;
            }

            if (i >= j)
            {
                int tmp = cases[i];
                cases[i] = cases[j];
                cases[j] = tmp;

                i--;
                j++;
            }
        }
        return i + 1;
    }

    /**
     * Get the branch index of the class attribute value of the specified data.
     */
    public int getClassBranch(int caseIndex)
    {
        // Class attribute never has missing data, therefore branch 0 is not
        // valid
        return branch[caseIndex] - 1;
    }

    public int getBranchCount()
    {
        return attribute.getNominalValuesCount();
    }
}