/**
 * @(#)TreeNodeContent.java 1.5.2 09/03/29
 */
package at.rovo.classifier.decissionTrees.c45.tree;

import java.util.Formatter;

/**
 * The content of a tree node
 *
 * @author Ping He
 * @author Xiaohua Xu
 */
public class TreeNodeContent
{
    /** The total weight of the data distributed on the tree node **/
    private float trainWeight;
    /** The class distribution of the data distributed on the tree node **/
    private float[] trainClassDistribution;
    /** The most probable classification of the tree node **/
    private String classification;
    /** Every TreeNode can be a leaf and has the corresponding leaf error **/
    private float errorAsLeafNode;

    /**
     * Initialize a tree node.
     *
     * @param trainWeight
     *         The total weight of the data distributed on the tree node
     * @param trainClassDistribution
     *         The class distribution of the data distributed on the tree node
     * @param classification
     *         The classification of the tree node
     * @param errorAsLeafNode
     *         The classification error of the tree node as a leaf node
     */
    public TreeNodeContent(float trainWeight, float[] trainClassDistribution, String classification,
                           float errorAsLeafNode)
    {
        this.trainWeight = trainWeight;
        this.trainClassDistribution = trainClassDistribution;
        this.classification = classification;
        this.errorAsLeafNode = errorAsLeafNode;
    }

    /**
     * Query whether the tree node content qualifies a leaf node's content.
     *
     * @param minKnownWeight
     *         The minimal known weight
     */
    public boolean satisfyLeafNode(float minKnownWeight)
    {
        if (errorAsLeafNode == 0 || trainWeight < 2 * minKnownWeight)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    /**
     * Get the total weight of the data distributed on the tree node
     *
     * @return The training weight
     */
    public float getTrainWeight()
    {
        return this.trainWeight;
    }

    /**
     * Get the class distribution of the data distributed on the tree node
     *
     * @return The class distribution
     */
    public float[] getTrainClassDistribution()
    {
        return this.trainClassDistribution;
    }

    /**
     * Get the classification of the tree node
     */
    public String getClassification()
    {
        return this.classification;
    }

    /**
     * Get the classification error of the tree node as a leaf node
     */
    public float getErrorAsLeafNode()
    {
        return this.errorAsLeafNode;
    }

    /**
     * Set the total weight of the data distributed on the tree node
     */
    public void setTrainWeight(float trainWeight)
    {
        this.trainWeight = trainWeight;
    }

    /**
     * Set the class distribution of the data distributed on the tree node
     */
    public void setTrainClassDistribution(float[] trainClassDistribution)
    {
        this.trainClassDistribution = trainClassDistribution;
    }

    /**
     * Set the classification of the tree node
     */
    public void setClassification(String classification)
    {
        this.classification = classification;
    }

    /**
     * Set the classification error of the tree node as a leaf node
     */
    public void setErrorAsLeaf(float errorAsLeafNode)
    {
        this.errorAsLeafNode = errorAsLeafNode;
    }

    /**
     * The String exhibition of the tree node content
     */
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append(classification).append("(");
        Formatter formatter = new Formatter(sb);
        formatter.format("%.1f", trainWeight);
        if (errorAsLeafNode > 0)
        {
            formatter.format("/%.1f", errorAsLeafNode);
        }
        sb.append(")");
        formatter.close();
        return sb.toString();
    }
}