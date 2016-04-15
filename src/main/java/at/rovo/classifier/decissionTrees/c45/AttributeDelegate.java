/**
 * @(#)AttributeDelegate.java 1.5.2 09/03/29
 */

package at.rovo.classifier.decissionTrees.c45;

/**
 * A delegate of an attribute, containing some essential processed information about the attribute to speed up the tree
 * building process.
 *
 * @author Ping He
 * @author Xiaohua Xu
 */
public abstract class AttributeDelegate
{
    /** Whether there is any missing data on the attribute **/
    protected boolean hasMissingData;

    /**
     * Initialize an attribute delegate
     */
    protected AttributeDelegate()
    {
        // Before we find missing data, it is false
        this.hasMissingData = false;
    }

    /**
     * Retrieve whether there is any missing data on the attribute
     */
    public boolean hasMissingData()
    {
        return hasMissingData;
    }

    /**
     * Set whether there is any missing data on the attribute
     */
    public void setHasMissingData(boolean value)
    {
        hasMissingData = value;
    }

    /**
     * Set the original sequence and weight for each data.
     */
    public abstract void setCasesWeight(int[] cases, float[] weight);

    /**
     * Evaluate the Gain and splitInfo value for the specified data when it splits on the attribute.
     *
     * @param first
     *         the begin (inclusive) index of the data to be evaluated
     * @param last
     *         the end (exclusive) index of the data to be evaluated
     * @param classAttributeDelegate
     *         the delegate of the class attribute. It helps when computing data's class distribution.
     *
     * @return The evaluation result
     */
    protected abstract float[] evaluate(int first, int last, AttributeDelegate classAttributeDelegate);

    /**
     * Group the data with the specified branch value forward and compute its branch weight.
     *
     * @param first
     *         the begin (inclusive) index of the data to be grouped
     * @param last
     *         the end (exclusive) index of the data to be grouped
     * @param groupBranch
     *         For discrete attribute, the branch index to be grouped; For continuous attribute, -1 for missing data,
     *         otherwise the rank of the cut value.
     * @param branchDistri
     *         Actually an output of this method, recording the weight of each branch.
     *
     * @return The boundary index before which the specified data is grouped
     */
    protected abstract int groupForward(int first, int last, int groupBranch, float[] branchDistri);

    /**
     * Group the data with missing value on the attribute backward. <p> The reason for grouping missing data backward is
     * to narrow the grouping range for the next branch. </p>
     *
     * @param first
     *         the begin (inclusive) index of the data to be grouped
     * @param last
     *         the end (exclusive) index of the data to be grouped
     *
     * @return The boundary index after which the data is grouped
     */
    protected abstract int groupBackward(int first, int last);

    /**
     * Get the number of branches if the attribute is selected as the test attribute. <p> When an attribute is selected
     * as the test attribute, missing value is not taken as a valid branch value. </p>
     */
    protected abstract int getBranchCount();

    /**
     * Get the branch index of the class attribute value of the specified data. <p> Only supported by discrete attribute
     * delegates. <br> Only used by class attribute delegate. </p>
     *
     * @param caseIndex
     *         the index of the specified data
     *
     * @return The branch index of the class value of the specified data.
     */
    protected int getClassBranch(int caseIndex)
    {
        throw new UnsupportedOperationException("Only Supported By The Class Attribute!");
    }

    /**
     * Find the rank of the cut value in the test attribute. <p> Only supported by continuous attribute delegates. </p>
     *
     * @param preSplitRank
     *         the begin (inclusive) rank from which the search of cut should start
     * @param splitRank
     *         the end (inclusive) rank to which the search of cut should stop
     *
     * @return The rank of the cut value
     */
    protected int findCutRank(int splitRank, int preSplitRank)
    {
        throw new UnsupportedOperationException("Only Supported By Continuous Attribute!");
    }

    /**
     * Find the cut value of the test attribute when provided with its rank. <p> Only supported by continuous attribute
     * delegates. </p>
     *
     * @param cutRank
     *         the rank of the cut
     *
     * @return The cut value of the test attribute
     */
    protected float findCut(int cutRank)
    {
        throw new UnsupportedOperationException("Only Supported By Continuous Attribute!");
    }
}