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

/**
 * A decision tree built with C4.5 algorithm.
 *
 * @author Ping He
 * @author Xiaohua Xu
 */
public class DecisionTree implements TreeClassifier
{
    /** The root of the decision tree **/
    private TreeNode root;
    /** The DataSet contains all the information in the input files **/
    private DataSet dataSet;
    /** The delegate of attributes to assist tree building and tree pruning **/
    private AttributeDelegate[] attributeDelegates;

    /**
     * Build a decision tree with the specified data set
     */
    public DecisionTree(DataSet dataSet)
    {
        this.dataSet = dataSet;
        build();
        this.root.setName(dataSet.getName());
    }

    public int size()
    {
        return treeSize(root);
    }

    /**
     * Compute the number of tree nodes in the subtree started from the
     * specified tree node.
     */
    private int treeSize(TreeNode root)
    {
        if (root instanceof LeafNode)
        {
            return 1;
        }

        int sum = 0;
        int childrenCount = root.getChildrenCount();
        for (int i = 0; i < childrenCount; i++)
        {
            sum += treeSize(root.getChildAt(i));
        }
        return sum + 1;
    }

    @Override
    public int getTrainError()
    {
        return getTestError(dataSet.getTrainData());
    }

    @Override
    public TreeNode getRoot()
    {
        return root;
    }

    @Override
    public void setRoot(TreeNode root)
    {
        this.root = root;
    }

    @Override
    public int getTestError(String[][] testData)
    {
        String[] classificationResults = classify(testData);
        int testError = 0;
        int classAttributeIndex = dataSet.getClassAttributeIndex();
        for (int i = 0; i < classificationResults.length; i++)
        {
            if (!classificationResults[i].equals(testData[i][classAttributeIndex]))
            {
                testError++;
            }
        }
        return testError;
    }

    @Override
    public double getTestErrorRatio(String[][] testData)
    {
        return 1.0 * getTestError(testData) / testData.length;
    }

    @Override
    public String[] classify(String[][] testData)
    {
        // Ready to record the classification results
        String[] results = new String[testData.length];

        // Get the number of different class values
        int numberOfClasses = dataSet.getClassCount();
        String[] classValues = dataSet.getClassValues();

        // Initialize the test error
        float[] testClassDistribution = new float[numberOfClasses];
        // TreeNode node = root;

        for (int testIndex = 0; testIndex < testData.length; testIndex++)
        {
            // Classify the test data into a specific class
            // Initialize the probability of the test data belonging to each
            // class value as 0
            Arrays.fill(testClassDistribution, 0.0f);

            // Classify a single test data from top to bottom
            classifyDownward(root, testData[testIndex], testClassDistribution, 1.0f);

            // Select the branch whose probability is the greatest as the
            // classification of the test data
            float max = -1.0f;
            int maxIndex = -1;
            for (int i = 0; i < testClassDistribution.length; i++)
            {
                if (testClassDistribution[i] > max)
                {
                    maxIndex = i;
                    max = testClassDistribution[i];
                }
            }
            results[testIndex] = classValues[maxIndex];
        }

        return results;
    }

    /**
     * Classify a test data from top to bottom from one tree node to its
     * offspring (if there is any).
     *
     * @param node The current tree node classify the test data
     * @param record The test data with its attribute values extracted
     * @param testClassDistribution Actually the output of this method,
     *                              recording the weight distribution of the
     *                              test data in different class values.
     * @param weight The weight of the test data on the current tree node
     */
    private void classifyDownward(TreeNode node, String[] record, float[] testClassDistribution, float weight)
    {
        TreeNodeContent content = node.getContent();
        if (node instanceof LeafNode)
        {
            // If there is no train data distributed on this tree node,
            // then add the weight of the test data to its corresponding class
            // branch
            if (content.getTrainWeight() <= 0)
            {
                // Get the branch index of the tree node's classification
                int classificationIndex = indexOf(content.getClassification(), dataSet.getClassValues());
                testClassDistribution[classificationIndex] += weight;
            }
            // Otherwise, distribute the weight of the test data with the
            // coefficient
            // of trainClassDistri[classValueIndex]/trainWeight
            else
            {
                float[] trainClassDistribution = content.getTrainClassDistribution();
                for (int i = 0; i < testClassDistribution.length; i++)
                {
                    testClassDistribution[i] += weight * trainClassDistribution[i] / content.getTrainWeight();
                }
            }
        }
        // If the current tree node is an InternalNode
        else
        {
            // if the test attribute value of the test data is not missing, then
            // pass it to its child tree node for classification.
            Attribute testAttribute = ((InternalNode) node).getTestAttribute();
            int testAttributeIndex = indexOf(testAttribute.getName(), dataSet.getMetaData().getAttributeNames());
            if (!record[testAttributeIndex].equals("?"))
            {
                int branchIndex = findChildBranch(record[testAttributeIndex], (InternalNode) node);
                classifyDownward(node.getChildAt(branchIndex), record, testClassDistribution, weight);
            }
            /*
             * If the test attribute value of the test data is missing or not
             * exists in declaration, the test data is then passed to all the
             * children tree nodes with the partitioned weight of
             * (weight*children[childindex].getTrainWeight()/trainWeight)
             */
            else
            {
                TreeNode[] children = node.getChildren();
                for (TreeNode child : children)
                {
                    TreeNodeContent childContent = child.getContent();
                    float childWeight = weight * childContent.getTrainWeight() / content.getTrainWeight();
                    classifyDownward(child, record, testClassDistribution, childWeight);
                }
            }
        }
    }

    /**
     * Find the branch index of the child tree node to which the parent tree
     * node should classify the test data to.
     *
     * @param value The attribute value of the test data on the parent tree
     *              node's test attribute
     * @param node The parent tree node which need to classify the test data to
     *             its offspring.
     */
    private int findChildBranch(String value, InternalNode node)
    {
        Attribute testAttribute = node.getTestAttribute();
        // If the test attribute is continuous, find the branch of the test data
        // belong to by comparing its test attribute value and the cut value.
        if (testAttribute instanceof ContinuousAttribute)
        {
            float continValue = Float.parseFloat(value);
            return (continValue < (node.getCut() + Parameter.PRECISION)) ? 0 : 1;
        }
        else
        {
            // If the test attribute is discrete, find the branch whose value is
            // the same as the test attribute value of the test data
            String[] nominalValues = ((DiscreteAttribute) testAttribute).getNominalValues();
            for (int i = 0; i < nominalValues.length; i++)
            {
                if (nominalValues[i].equals(value))
                {
                    return i;
                }
            }
            // Not Found the test attribute value
            return -1;
        }
    }

    /**
     * Build a decision tree.
     */
    public void build()
    {
        TreeBuilder tb = new TreeBuilder(this.dataSet, this.attributeDelegates);

        this.dataSet = tb.getDataSet();
        this.attributeDelegates = tb.getAttributeDelegates();
        this.root = tb.getRootNode();
    }

    /**
     * Prune the built decision tree.
     */
    public void prune()
    {
        TreePruner tp = new TreePruner(this.dataSet, this.attributeDelegates, this.root, this);

        this.dataSet = tp.getDataSet();
        this.attributeDelegates = tp.getAttributeDelegates();
        this.root = tp.getRootNode();
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
