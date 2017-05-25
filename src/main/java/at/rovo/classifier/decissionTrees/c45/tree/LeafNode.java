package at.rovo.classifier.decissionTrees.c45.tree;

import at.rovo.classifier.dataset.Attribute;
import at.rovo.classifier.dataset.ContinuousAttribute;
import at.rovo.classifier.dataset.DiscreteAttribute;

/**
 * A leaf tree node.
 * <p/>
 * A leaf node is a subset of a tree node, because it does not have any child tree node.
 *
 * @author Ping He
 * @author Xiaohua Xu
 */
public class LeafNode extends TreeNode
{
    /**
     * Initialize a leaf node with the specified tree node content
     *
     * @param content
     *         The content of the tree node
     */
    public LeafNode(TreeNodeContent content)
    {
        super(content);
    }

    @Override
    public float getTrainError()
    {
        return content.getErrorAsLeafNode();
    }

    @Override
    public void setTrainError(float error)
    {
        content.setErrorAsLeaf(error);
    }

    /**
     * The String exhibition of the leaf node
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
            sb.append(" : ");
        }
        else
        {
            sb.append(name);
        }

        sb.append(content.getClassification());

        return sb.toString();
    }
}