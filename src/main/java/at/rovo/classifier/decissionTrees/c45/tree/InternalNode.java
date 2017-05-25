package at.rovo.classifier.decissionTrees.c45.tree;

import at.rovo.classifier.dataset.Attribute;
import at.rovo.classifier.dataset.ContinuousAttribute;
import at.rovo.classifier.dataset.DiscreteAttribute;

/**
 * An internal tree node.
 * <p/>
 * An internal node is a super set of a tree node. It not only contains a basic tree node content, it also contains some
 * information about its test attribute.
 *
 * @author Ping He
 * @author Xiaohua Xu
 */
public class InternalNode extends TreeNode
{
    /** The train error of the tree node as an internal tree node **/
    private float errorAsInternalNode;
    /** The test attribute of the internal tree node **/
    private Attribute testAttribute;
    /**
     * The cut value of the test on the internal tree node if its test
     * attribute is continuous
     **/
    private float cut;
    /**
     * The rank of the cut value among all the attribute values of the train
     * data on the test attribute. This is used for the speedup of C45
     * algorithm.
     **/
    private int cutRank;

    /**
     * Initialize an internal tree node.
     *
     * @param content
     *         The content of the tree node
     * @param testAttribute
     *         The test attribute of the internal node
     */
    public InternalNode(TreeNodeContent content, Attribute testAttribute)
    {
        super(content);

        errorAsInternalNode = 0;
        this.testAttribute = testAttribute;
        if (testAttribute instanceof ContinuousAttribute)
        {
            this.children = new TreeNode[2];
        }
        else
        {
            this.children = new TreeNode[((DiscreteAttribute) testAttribute).getNominalValuesCount()];
        }
        this.errorAsInternalNode = 0;
    }

    @Override
    public float getTrainError()
    {
        return errorAsInternalNode;
    }

    @Override
    public void setTrainError(float errorAsInternalNode)
    {
        this.errorAsInternalNode = errorAsInternalNode;
    }

    /**
     * Get the cut value of the internal node, if its test attribute is continuous
     *
     * @return The value that cuts the internal node
     */
    public float getCut()
    {
        return this.cut;
    }

    /**
     * Set the cut value of the internal node, if its test attribute is continuous
     *
     * @param cut
     *         The value to cut the node
     */
    public void setCut(float cut)
    {
        this.cut = cut;
    }

    /**
     * Get the rank of the cut value of the internal node, if its test attribute is continuous
     *
     * @return the rank of the cut value
     */
    public int getCutRank()
    {
        return this.cutRank;
    }

    /**
     * Set the rank of the cut value of the internal node, if its test attribute is continuous
     *
     * @param cutRank
     *         The rank to cut the value of the internal node
     */
    public void setCutRank(int cutRank)
    {
        this.cutRank = cutRank;
    }

    /**
     * Get the test attribute of the internal node
     *
     * @return The test attribute
     */
    public Attribute getTestAttribute()
    {
        return this.testAttribute;
    }

    /**
     * Set the test attribute of the internal node
     *
     * @param attribute
     *         The attribute to test
     */
    public void setTestAttribute(Attribute attribute)
    {
        this.testAttribute = attribute;
    }

    /**
     * The String exhibition of the internal node
     */
    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();

        if (!isRoot())
        {
            Attribute parentTestAttribute = ((InternalNode) parent).getTestAttribute();
            int childIndex = parent.indexOfChild(this);

            sb.append(parentTestAttribute.getName());
            if (parentTestAttribute instanceof ContinuousAttribute)
            {
                sb.append(childIndex == 0 ? " <= " : " > ").append(((InternalNode) parent).getCut());
            }
            else
            {
                sb.append(" = ").append(((DiscreteAttribute) parentTestAttribute).getNominalValues()[childIndex]);
            }
        }
        else
        {
            sb.append(name);
        }

        return sb.toString();
    }
}