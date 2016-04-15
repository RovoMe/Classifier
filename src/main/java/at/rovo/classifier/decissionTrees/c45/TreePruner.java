package at.rovo.classifier.decissionTrees.c45;

import at.rovo.classifier.dataset.DataSet;
import at.rovo.classifier.decissionTrees.c45.tree.InternalNode;
import at.rovo.classifier.decissionTrees.c45.tree.LeafNode;
import at.rovo.classifier.decissionTrees.c45.tree.TreeNode;
import at.rovo.classifier.decissionTrees.c45.tree.TreeNodeContent;
import java.util.Arrays;

class TreePruner
{
    /** The sequence of the cases used for tree construction **/
    private int[] cases;
    /** The weight of each case used for tree construction **/
    private float[] weight;

    /** The loaded data set **/
    private DataSet dataSet;
    /** The delegate of attributes to assist tree building and tree pruning **/
    private AttributeDelegate[] attributeDelegates;
    /** The root of the decision tree **/
    private TreeNode root;
    /** The decision tree **/
    private DecisionTree dt;

    /**
     * Initialize a tree pruner which prunes the built decision tree
     */
    TreePruner(DataSet dataSet, AttributeDelegate[] attributeDelegates, TreeNode root, DecisionTree dt)
    {
        this.dataSet = dataSet;
        this.attributeDelegates = attributeDelegates;
        this.root = root;
        this.dt = dt;

        // ReInitialize the data sequence and their weight
        int caseCount = dataSet.getCaseCount();
        this.cases = new int[caseCount];
        for (int i = 0; i < this.cases.length; i++)
        {
            this.cases[i] = i;
        }
        this.weight = new float[caseCount];
        Arrays.fill(this.weight, 1.0f);

        // Reset the cases and weight array of all attributes delegate
        // objects
        for (AttributeDelegate attributeDelegate : attributeDelegates)
        {
            attributeDelegate.setCasesWeight(this.cases, this.weight);
        }

		/* float errorAfterPrune = */
        ebpPrune(this.root, 0, caseCount, true);
    }

    /**
     * Returns the loaded data set.
     *
     * @return The data set
     */
    public DataSet getDataSet()
    {
        return this.dataSet;
    }

    /**
     * Returns the attribute delegates.
     *
     * @return The attribute delegates
     */
    public AttributeDelegate[] getAttributeDelegates()
    {
        return this.attributeDelegates;
    }

    /**
     * Returns the root node created by this object.
     *
     * @return The created root node
     */
    public TreeNode getRootNode()
    {
        return this.root;
    }

    /**
     * Prune the decision tree from top to bottom with EBP strategy.
     *
     * @param node
     *         The current tree node to be pruned
     * @param first
     *         The start(inclusive) index of the train data used for pruning.
     * @param last
     *         The end(exclusive) index of the train data used for pruning.
     * @param update
     *         whether the current pruning is a trial to retrieve the error after pruning (update = false) or an actual
     *         pruning (update = true).
     *
     * @return The estimated error after completing pruning the subtree started from the current tree node.
     */
    private float ebpPrune(TreeNode node, int first, int last, boolean update)
    {
        TreeNodeContent content = createContent(first, last, node);
        float estimatedLeafError = content.getErrorAsLeafNode();
        // If this is an actual pruning instead of an error-estimation,
        // reset the tree node information
        if (update)
        {
            node.setContent(content);
        }
        InternalNode internalNode;
        // If the current tree node is a Leaf, its pruning is finished
        if (node instanceof LeafNode)
        {
            return estimatedLeafError;
        }
        else
        {
            internalNode = (InternalNode) node;
        }

        /*
         * Begin to estimate the errors of each branch to get the
         * errorAsInternalNode of the tree node
         */

        // The estimated test error of the tree node as an InternalNode
        float estimatedTreeError = 0;
        // The branch index with the maximal weight distribution
        int maxBranch = -1;
        // The maximal branch weight
        float maxBranchWeight = 0;

        // The index of the test attribute on the tree node
        int testAttributeIndex =
                indexOf(internalNode.getTestAttribute().getName(), this.dataSet.getMetaData().getAttributeNames());
        AttributeDelegate testAttributeDelegate = this.attributeDelegates[testAttributeIndex];
        int testBranchCount = testAttributeDelegate.getBranchCount();

        // Record the class weight distribution of the selected test
        // attribute
        float[] branchDistri = new float[testBranchCount + 1];
        /*
         * 'missingBegin' records the begin index of the missing data if
         * there is any, otherwise it coordinates with beginIndex;
         * 'groupBegin' records the begin index to group the cases for
         * one branch 'nextGroupBegin' records the begin index group the
         * cases for next branch
         */
        int missingBegin = first;
        int groupBegin = first;

        // Group the missing data to the most front
        if (testAttributeDelegate.hasMissingData())
        {
            groupBegin = testAttributeDelegate.groupForward(first, last, -1, branchDistri);
        }
        // Classify the [first last) cases to the branches of the test
        // attribute
        // except for the last branch, to construct the children tree
        // nodes
        for (int index = 0; index < testBranchCount; index++)
        {
            // For a continuous attribute, the group criterion is
            // cutRank;
            // For a discrete attribute, the group criterion is the
            // branch value(or index)
            int split =
                    testAttributeDelegate instanceof ContinuousAttributeDelegate ? internalNode.getCutRank() : index;

            // For the first several branches, we need to group the
            // specified branch values forward
            // near "groupBegin" and compute its branch weight
            int nextGroupBegin;
            if (index < testBranchCount - 1)
            {
                nextGroupBegin = testAttributeDelegate.groupForward(groupBegin, last, split, branchDistri);
            }
            // For the last branch, the "nextGroupBegin" must be last
            // and its branch weight must be
            // the rest weight of the total weight.
            else
            {
                nextGroupBegin = last;
                float lastWeight = content.getTrainWeight();
                for (int j = 0; j < branchDistri.length - 1; j++)
                {
                    lastWeight -= branchDistri[j];
                }
                branchDistri[branchDistri.length - 1] = lastWeight;
            }

            // If there is no cases distributed in this branch, omit
            if (groupBegin == nextGroupBegin)
            {
                continue;
            }
            // If there is missing data
            else if (groupBegin > missingBegin)
            {
                // Compute the weight ratio of this branch
                float ratio = branchDistri[index + 1] / (content.getTrainWeight() - branchDistri[0]);
                // split the weight of the missing data with by
                // multiplying the ratio
                for (int i = missingBegin; i < groupBegin; i++)
                {
                    this.weight[this.cases[i]] *= ratio;
                }

                // Accumulate the estimated errorAsInternalNode
                estimatedTreeError += ebpPrune(internalNode.getChildAt(index), missingBegin, nextGroupBegin, update);

                // Restore the original sequence of the cases after the
                // recursive construction
                missingBegin = testAttributeDelegate.groupBackward(missingBegin, nextGroupBegin);
                // Restore the weight of the missing data with by
                // dividing the ratio
                for (int i = missingBegin; i < nextGroupBegin; i++)
                {
                    this.weight[this.cases[i]] /= ratio;
                }
            }
            else
            {
                estimatedTreeError += ebpPrune(internalNode.getChildAt(index), missingBegin, nextGroupBegin, update);
                // When there is no missing data, missingBegin moves
                // together with groupBegin
                missingBegin = nextGroupBegin;
            }
            // For next branch, group from nextGroupBegin index
            groupBegin = nextGroupBegin;

            // Select the biggest branch with maximal weight for
            // branchError estimation
            if (branchDistri[index + 1] > maxBranchWeight)
            {
                maxBranchWeight = branchDistri[index + 1];
                maxBranch = index;
            }
        }

        // If this sentence is not present, it will lead to significant
        // pruning!
        // Do not evaluate doubled subtree raising (i.e. subtree-raising
        // of subtree-raising)
        if (!update)
        {
            return estimatedTreeError;
        }

        // Estimate the subtree-raising error
        float estimatedBranchError = ebpPrune(internalNode.getChildAt(maxBranch), first, last, false);

        TreeNode parent = internalNode.getParent();
        // Select a strategy with the minimal error
        if (estimatedLeafError <= estimatedBranchError + 0.1 && estimatedLeafError <= estimatedTreeError + 0.1)
        {
            LeafNode newNode = new LeafNode(content);
            if (parent != null)
            {
                int childIndex = parent.indexOfChild(internalNode);
                parent.setChildAt(childIndex, newNode);
            }
            else
            {
                this.dt.setRoot(newNode);
            }

            node = newNode;
        }
        else if (estimatedBranchError <= estimatedTreeError + 0.1)
        {
            ebpPrune(internalNode.getChildAt(maxBranch), first, last, true);
            TreeNode newNode = node.getChildAt(maxBranch);
            if (parent != null)
            {
                int childIndex = parent.indexOfChild(internalNode);
                parent.setChildAt(childIndex, newNode);
            }
            else
            {
                this.dt.setRoot(newNode);
            }

            node = newNode;
        }
        else
        {
            node.setTrainError(estimatedTreeError);
        }

        return node.getTrainError();
    }

    /**
     * Recreate a tree node content with the specified data based on the tree node's existing content.
     *
     * @param first
     *         The start(inclusive) index of the train data used for creating the tree node content.
     * @param last
     *         The end(exclusive) index of the train data used for creating the tree node content.
     * @param node
     *         The tree node whose content need to be recreated.
     *
     * @return The recreated tree node content.
     */
    private TreeNodeContent createContent(int first, int last, TreeNode node)
    {
        // Compute the total weight and its class distribution of [first
        // last) prune cases
        float totalWeight = 0;
        AttributeDelegate classAttributeDelegate = this.attributeDelegates[dataSet.getClassAttributeIndex()];
        float[] totalClassDistri = new float[this.dataSet.getClassCount()];
        Arrays.fill(totalClassDistri, 0);
        for (int i = first; i < last; i++)
        {
            int classLabel = classAttributeDelegate.getClassBranch(this.cases[i]);
            totalClassDistri[classLabel] += this.weight[cases[i]];
        }

        // Find the original classification of the tree node
        String nodeClassification = node.getContent().getClassification();
        String[] classValues = this.dataSet.getClassValues();
        int maxClassIndex = indexOf(nodeClassification, classValues);

        // Find the most probable classification of the prune data on
        // the current tree node
        for (int i = 0; i < totalClassDistri.length; i++)
        {
            totalWeight += totalClassDistri[i];
            if (totalClassDistri[i] > totalClassDistri[maxClassIndex])
            {
                maxClassIndex = i;
            }
        }

        String classification = classValues[maxClassIndex];

        // Estimate the leafError of the tree node with the [first last)
        // prune data
        float basicLeafError = totalWeight - totalClassDistri[maxClassIndex];
        float extraLeafError = Estimator.getExtraError(totalWeight, basicLeafError);
        float estimatedLeafError = basicLeafError + extraLeafError;

        return new TreeNodeContent(totalWeight, totalClassDistri, classification, estimatedLeafError);
    }

    /**
     * Find the index of a String value in a String array.
     */
    private int indexOf(String target, String[] from)
    {
        for (int i = 0; i < from.length; i++)
        {
            if (from[i].equals(target))
            {
                return i;
            }
        }
        return -1;
    }
}