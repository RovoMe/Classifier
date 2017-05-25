package at.rovo.classifier.decissionTrees.c45;

import at.rovo.classifier.dataset.Attribute;
import at.rovo.classifier.dataset.ContinuousAttribute;
import at.rovo.classifier.dataset.DataSet;
import at.rovo.classifier.dataset.DiscreteAttribute;
import at.rovo.classifier.decissionTrees.c45.tree.InternalNode;
import at.rovo.classifier.decissionTrees.c45.tree.LeafNode;
import at.rovo.classifier.decissionTrees.c45.tree.TreeNode;
import at.rovo.classifier.decissionTrees.c45.tree.TreeNodeContent;
import java.util.Arrays;

class TreeBuilder
{
    /** The sequence of the cases used for tree construction **/
    private int[] cases;
    /** The weight of each case used for tree construction **/
    private float[] weight;
    /** The number of the candidate test attributes **/
    private int candidateTestAttrCount;
    /** Whether the attributes are candidate for test attribute selection **/
    private boolean[] isCandidateTestAttr;

    /** The loaded data set **/
    private DataSet dataSet;
    /** The delegate of attributes to assist tree building and tree pruning **/
    private AttributeDelegate[] attributeDelegates;
    /** The root node **/
    private TreeNode root;

    /**
     * Initialize a tree builder which build a decision tree.
     */
    TreeBuilder(DataSet dataSet, AttributeDelegate[] attributeDelegates)
    {
        this.dataSet = dataSet;
        this.attributeDelegates = attributeDelegates;

        // Create Attribute Delegate objects
        this.attributeDelegates = new AttributeDelegate[dataSet.getAttributeCount()];
        int attributeIndex = 0;
        for (Attribute attribute : dataSet.getAttributes())
        {
            if (attribute instanceof ContinuousAttribute)
            {
                this.attributeDelegates[attributeIndex] =
                        new ContinuousAttributeDelegate((ContinuousAttribute) attribute);
            }
            else
            {
                this.attributeDelegates[attributeIndex] = new DiscreteAttributeDelegate((DiscreteAttribute) attribute);
            }
            attributeIndex++;
        }

        // Initialize the qualification of candidate test attributes
        candidateTestAttrCount = dataSet.getAttributeCount() - 1;
        this.isCandidateTestAttr = new boolean[dataSet.getAttributeCount()];
        Arrays.fill(isCandidateTestAttr, true);
        isCandidateTestAttr[dataSet.getClassAttributeIndex()] = false;

        // Initialize the data sequence and their weight
        initializeCasesWeight();

        this.root = constructTreeNode(0, dataSet.getCaseCount());
    }

    /**
     * Returns the loaded data set
     *
     * @return The data set
     */
    public DataSet getDataSet()
    {
        return this.dataSet;
    }

    /**
     * Returns the attribute delegates
     *
     * @return The attribute delegates
     */
    public AttributeDelegate[] getAttributeDelegates()
    {
        return this.attributeDelegates;
    }

    /**
     * Returns the root node created by this object
     *
     * @return The created root node
     */
    public TreeNode getRootNode()
    {
        return this.root;
    }

    /**
     * Initialize the sequence of the train data from 1 to n, and initialize
     * their weight with all 1.0.
     */
    void initializeCasesWeight()
    {
        int caseCount = this.dataSet.getCaseCount();
        this.cases = new int[caseCount];
        for (int i = 0; i < this.cases.length; i++)
        {
            this.cases[i] = i;
        }
        this.weight = new float[caseCount];
        Arrays.fill(weight, 1.0f);

        // All the attribute delegates share the same cases and weight array
        for (AttributeDelegate attributeDelegate : this.attributeDelegates)
        {
            attributeDelegate.setCasesWeight(this.cases, this.weight);
        }
    }

    /**
     * Construct tree node from top to bottom.
     *
     * @param first
     *         The start(inclusive) index of the train data used for tree node construction.
     * @param last
     *         The end(exclusive) index of the train data used for tree node construction.
     *
     * @return The constructed tree node.
     */
    private TreeNode constructTreeNode(int first, int last)
    {
        // Construct an initial Leaf tree node
        TreeNodeContent content = createContent(first, last);
        // float errorAsLeafNode = content.getErrorAsLeafNode();
        // If any of the leaf conditions is satisfied, return the Leaf tree node
        if (content.satisfyLeafNode(Parameter.MINWEIGHT) || this.candidateTestAttrCount <= 0)
        {
            return new LeafNode(content);
        }

        // Select a test attribute from all the candidate attributes
        int[] testAttributeInfo = new int[2];
        Attribute testAttribute = selectTestAttribute(first, last, testAttributeInfo);
        int testAttributeIndex, cutRank;
        float cut;
        // If no attribute is selected as the final test attribute, return Leaf
        if (testAttribute == null)
        {
            return new LeafNode(content);
        }
        else
        {
            testAttributeIndex = testAttributeInfo[0];
            cutRank = testAttributeInfo[1];
        }
        // Change the type the tree node to an InternalNode
        InternalNode node = new InternalNode(content, testAttribute);

        AttributeDelegate testAttributeDelegate = this.attributeDelegates[testAttributeIndex];
        // Record the class weight distribution of the selected test attribute
        float[] testBranchDistri;
        // If the test attribute is a discrete attribute
        if (!(testAttribute instanceof ContinuousAttribute))
        {
            // 0 is kept for missing data
            testBranchDistri = new float[((DiscreteAttribute) testAttribute).getNominalValuesCount() + 1];
            // A discrete attribute can not be test attribute again in its offspring tree nodes
            this.isCandidateTestAttr[testAttributeIndex] = false;
            this.candidateTestAttrCount--;
        }
        else
        {
            // 0 is kept for missing data
            testBranchDistri = new float[2 + 1];
            cut = testAttributeDelegate.findCut(cutRank);
            node.setCut(cut);
            node.setCutRank(cutRank);
        }

        /*
         * 'missingBegin' records the begin index of the missing data if there is any, otherwise it coordinates with
         * beginIndex; 'groupBegin' records the begin index to group the cases for one branch 'nextGroupBegin' records
         * the begin index group the cases for next branch
         */
        int missingBegin = first;
        int groupBegin = first;

        TreeNode aChild;
        // Group the missing data to the most front
        if (testAttributeDelegate.hasMissingData())
        {
            groupBegin = testAttributeDelegate.groupForward(first, last, -1, testBranchDistri);
        }
        // Classify the [first last) cases to the branches of the test attribute except for the last branch, to
        // construct the children tree nodes
        for (int index = 0; index < testBranchDistri.length - 1; index++)
        {
            // For a continuous attribute, the group criterion is cutRank;
            // For a discrete attribute, the group criterion is the branch value(or index)
            int split = testAttribute instanceof ContinuousAttribute ? cutRank : index;

            // For the first several branches, we need to group the specified branch values forward near "groupBegin"
            // and compute its branch weight
            int nextGroupBegin;
            if (index < testBranchDistri.length - 2)
            {
                nextGroupBegin = testAttributeDelegate.groupForward(groupBegin, last, split, testBranchDistri);
            }
            // For the last branch, the "nextGroupBegin" must be last and its branch weight must be the rest weight of
            // the total weight.
            else
            {
                nextGroupBegin = last;
                float lastWeight = content.getTrainWeight();
                for (int j = 0; j < testBranchDistri.length - 1; j++)
                {
                    lastWeight -= testBranchDistri[j];
                }
                testBranchDistri[testBranchDistri.length - 1] = lastWeight;
            }

            // If there is no cases distributed in this branch, construct a Leaf
            if (groupBegin == nextGroupBegin)
            {
                // Add a child with its parent's class
                aChild = new LeafNode(new TreeNodeContent(0, null, content.getClassification(), 0));
            }
            // If the test attribute contains missing data and at the same time there are some cases distributed in this
            // branch
            else if (groupBegin > missingBegin)
            {
                // Compute the weight ratio of this branch
                float ratio = testBranchDistri[index + 1] / (content.getTrainWeight() - testBranchDistri[0]);
                // Update the weight of the cases with unknown value on this test attribute with the above ratio
                for (int i = missingBegin; i < groupBegin; i++)
                {
                    this.weight[this.cases[i]] *= ratio;
                }

                // Construct a child tree node for this branch recursively
                aChild = constructTreeNode(missingBegin, nextGroupBegin);

                // Restore the original sequence of the cases after the recursive construction
                missingBegin = testAttributeDelegate.groupBackward(missingBegin, nextGroupBegin);
                // Restore the weight of the unknown cases for the next iteration
                for (int i = missingBegin; i < nextGroupBegin; i++)
                {
                    this.weight[this.cases[i]] /= ratio;
                }
            }
            // If the test attribute contains no missing data and at the same time some cases are distributed in this
            // branch
            else
            {
                aChild = constructTreeNode(groupBegin, nextGroupBegin);
                // When there is no missing data, missingBegin moves together with groupBegin
                missingBegin = nextGroupBegin;
            }
            // For next branch, group from nextGroupBegin index
            groupBegin = nextGroupBegin;

            node.addChild(aChild);
            // System.out.println("node = null ? " + (node == null));
        }

        // After the recursion construction of its offspring tree nodes, the qualification of this discrete attribute as
        // a candidate test attribute should be restored
        if (!(testAttribute instanceof ContinuousAttribute))
        {
            this.isCandidateTestAttr[testAttributeIndex] = true;
            this.candidateTestAttrCount++;
        }
        // Choose to be a Leaf or InternalNode
        if (node.getTrainError() - content.getErrorAsLeafNode() >= -Parameter.PRECISION)
        {
            return new LeafNode(content);
        }

        return node;
    }

    /**
     * Create a tree node content with the specified data.
     *
     * @param first
     *         The start(inclusive) index of the train data used for creating the tree node content.
     * @param last
     *         The end(exclusive) index of the train data used for creating the tree node content.
     *
     * @return the created tree node content.
     */
    private TreeNodeContent createContent(int first, int last)
    {
        // Compute the total weight of the cases from first to last
        float totalWeight = 0;
        AttributeDelegate classAttributeDelegate = this.attributeDelegates[this.dataSet.getClassAttributeIndex()];
        // Compute the weight distribution of the cases in different classes
        float[] totalClassDistri = new float[this.dataSet.getClassCount()];
        Arrays.fill(totalClassDistri, 0);

        for (int i = first; i < last; i++)
        {
            int classLabel = classAttributeDelegate.getClassBranch(this.cases[i]);
            totalClassDistri[classLabel] += this.weight[cases[i]];
        }

        // Find the index of the class with maximal weight distribution
        int maxClassIndex = 0;
        for (int i = 0; i < totalClassDistri.length; i++)
        {
            totalWeight += totalClassDistri[i];
            if (totalClassDistri[i] > totalClassDistri[maxClassIndex])
            {
                maxClassIndex = i;
            }
        }

        // Get the different class values of the dataSet
        String[] classValues = this.dataSet.getClassValues();
        String classification = classValues[maxClassIndex];

        // Compute the errorAsLeaf and construct an initial Leaf tree node
        float leafError = totalWeight - totalClassDistri[maxClassIndex];

        return new TreeNodeContent(totalWeight, totalClassDistri, classification, leafError);
    }

    /**
     * Select a test attribute from the candidate test attributes.
     *
     * @param first
     *         The start(inclusive) index of the train data used for the selection of test attribute.
     * @param last
     *         The end(exclusive) index of the train data used for the selection of test attribute.
     * @param testAttrInfo
     *         Actually an output of this method, its 1<sup>st</sup> element recording the index of the test attribute,
     *         its 2<sup>nd</sup> element recording the rank of the cut value if the test attribute is a continuous
     *         attribute. If there is no test attribute selected, this array keeps empty.
     *
     * @return The selected test attribute.<br> If there is no test attribute selected, null is returned.
     */
    private Attribute selectTestAttribute(int first, int last, int[] testAttrInfo)
    {
        // Ready to record the gain and splitInfo of each available attribute
        float[] gain = new float[this.candidateTestAttrCount];
        float[] splitInfo = new float[this.candidateTestAttrCount];
        // For continuous attributes, ready to record the two neighboring ranks of the best split
        int[] splitRank = new int[this.candidateTestAttrCount];
        int[] preSplitRank = new int[this.candidateTestAttrCount];

        float averageGain = 0;
        // The number of the candidate test attributes with comparable Gain values
        int feasibleTestAttr = 0;

        AttributeDelegate classAttributeDelegate = this.attributeDelegates[this.dataSet.getClassAttributeIndex()];
        int gainIndex = 0;
        int attrIndex = 0;
        // Evaluate Gain and SplitInfo for each attribute
        for (AttributeDelegate attributeDelegate : this.attributeDelegates)
        {
            // Omit the unavailable attribute
            if (!this.isCandidateTestAttr[attrIndex])
            {
                attrIndex++;
                continue;
            }

            /*
             * Evaluate Gain and SplitInfo for each attribute.
             * For discrete attributes, just evaluate its nominal values as the test branches.
             * For continuous attributes, select the split test with the maximal Gain value
             */
            float[] evaluation = attributeDelegate.evaluate(first, last, classAttributeDelegate);

            // If the current attribute is valid as test attribute here
            if (evaluation != null)
            {
                gain[gainIndex] = evaluation[0];
                splitInfo[gainIndex] = evaluation[1];
                splitRank[gainIndex] = (int) evaluation[2];
                preSplitRank[gainIndex] = (int) evaluation[3];
            }

            // If the current attribute is feasible
            if (gain[gainIndex] > 0 && attributeDelegate.getBranchCount() < 0.3 * (this.dataSet.getCaseCount() + 1))
            {
                // Increase the number of feasible test attributes
                feasibleTestAttr++;
                // Prepare to compute the average Gain
                averageGain += gain[gainIndex];
            }

            gainIndex++;
            attrIndex++;
        }

        // Compute the average Gain value
        // If there is no feasible test attribute, than average Gain is set very big
        averageGain = ((feasibleTestAttr == 0) ? 100000 : averageGain / feasibleTestAttr);

        // Ready to select the test attribute with the maximal GainRatio
        float bestValue = -Parameter.EPSILON;
        int bestAttrIndex = -1;
        // If the test attribute is continuous, we need to record the two ranks which produce the split value
        int winSplitIndex = -1, winPreSplitIndex = -1;

        /*
         * Select the best test attribute with the maximal GainRatio value attrIndex records the index of the attributes
         * gainIndex records the index of the filled gain array
         */
        gainIndex = 0;
        attrIndex = 0;
        Attribute testAttribute = null;
        for (Attribute attribute : dataSet.getAttributes())
        {
            // neglect the unavailable attributes
            if (!this.isCandidateTestAttr[attrIndex])
            {
                attrIndex++;
                continue;
            }
            // neglect the attributes with Gain less than 0
            if (gain[gainIndex] <= -Parameter.EPSILON)
            {
                gainIndex++;
                attrIndex++;
                continue;
            }
            // compute the GainRatio value for feasible candidate attributes
            float gainRatio = GainCalculator.computeGainRatio(gain[gainIndex], splitInfo[gainIndex], averageGain);

            // Update the best attribute
            if (gainRatio >= bestValue + Parameter.PRECISION)
            {
                // Record the best test attribute index and its GainRatio value
                bestAttrIndex = attrIndex;
                bestValue = gainRatio;
                testAttribute = attribute;

                // If the selected test attribute is continuous, record the split ranks as well
                if (testAttribute instanceof ContinuousAttribute)
                {
                    winSplitIndex = splitRank[gainIndex];
                    winPreSplitIndex = preSplitRank[gainIndex];
                }
            }
            gainIndex++;
            attrIndex++;
        }
        // If no test attribute is selected
        if (testAttribute != null)
        {
            testAttrInfo[0] = bestAttrIndex;
            // If the test attribute is continuous, record its cutRank
            if (testAttribute instanceof ContinuousAttribute)
            {
                testAttrInfo[1] = this.attributeDelegates[bestAttrIndex].findCutRank(winSplitIndex, winPreSplitIndex);
            }
        }
        return testAttribute;
    }
}