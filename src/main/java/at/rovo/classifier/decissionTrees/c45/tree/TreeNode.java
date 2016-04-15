/**
 * @(#)TreeNode.java 1.5.3 09/04/22
 */
package at.rovo.classifier.decissionTrees.c45.tree;

/**
 * A tree node of a tree
 *
 * @author Ping He
 * @author Xiaohua Xu
 */
public abstract class TreeNode
{
    /** The content of the tree node **/
    protected TreeNodeContent content;
    /**
     * The name of the tree node. <p> Only root has a specified name of the tree. </p>
     **/
    protected String name;
    /** The parent tree node **/
    protected TreeNode parent;
    /** The children tree nodes **/
    protected TreeNode[] children;
    /** The number of the registered children tree nodes **/
    protected int childCount;

    /**
     * Initialize a tree node with the specified tree node content
     */
    protected TreeNode(TreeNodeContent content)
    {
        this.content = content;
        this.childCount = 0;
    }

    /**
     * Get the content of the tree node
     *
     * @return The content of this instance
     */
    public TreeNodeContent getContent()
    {
        return this.content;
    }

    /**
     * Set the content of the tree node
     *
     * @param content
     *         Sets the new content of this instance
     */
    public void setContent(TreeNodeContent content)
    {
        this.content = content;
    }

    /**
     * Get the name of the tree node
     *
     * @return The name of this instance
     */
    public String getName()
    {
        if (name != null)
        {
            return this.name;
        }
        else
        {
            return this.toString();
        }
    }

    /**
     * Set the name of the tree node
     *
     * @param name
     *         The new name of this instance
     */
    public void setName(String name)
    {
        this.name = name;
    }

    /**
     * Get the parent tree node
     *
     * @return The parent of this instance
     */
    public TreeNode getParent()
    {
        return this.parent;
    }

    /**
     * Set the parent tree node
     *
     * @param parent
     *         The new parent of this instance
     */
    public void setParent(TreeNode parent)
    {
        this.parent = parent;
    }

    /**
     * Add a child tree node
     *
     * @param aChild
     *         A new child of this instance
     */
    public void addChild(TreeNode aChild)
    {
        children[childCount++] = aChild;
        aChild.setParent(this);
        setTrainError(getTrainError() + aChild.getTrainError());
    }

    /**
     * Get all the children tree nodes
     *
     * @return The children of this instance
     */
    public TreeNode[] getChildren()
    {
        return this.children;
    }

    /**
     * Set all the children tree nodes
     *
     * @param children
     *         The new children of this instance
     */
    public void setChildren(TreeNode[] children)
    {
        this.children = children;
    }

    /**
     * Get the index<sup>th</sup> child tree node
     *
     * @param index
     *         The index of the child to return
     *
     * @return The child element at the index<sup>th</sup> position
     */
    public TreeNode getChildAt(int index)
    {
        return children[index];
    }

    /**
     * Set the index<sup>th</sup> child tree node
     *
     * @param index
     *         The position to add the new child at
     * @param child
     *         The new child to add at the index<sup>th</sup> position
     */
    public void setChildAt(int index, TreeNode child)
    {
        children[index] = child;
        child.setParent(this);
    }

    /**
     * Get the index of the specified tree node in the children tree nodes array
     *
     * @param child
     *         The specified tree node whose child index is to be retrieved
     *
     * @return The index of the specified child node
     */
    public int indexOfChild(TreeNode child)
    {
        for (int i = 0; i < children.length; i++)
        {
            if (children[i] == child)
            {
                return i;
            }
        }
        return -1;
    }

    /**
     * Get the number of the children tree nodes
     *
     * @return The number of children
     */
    public int getChildrenCount()
    {
        return children.length;
    }

    /**
     * Query whether the tree node is root node
     *
     * @return True if this instance is the root node, false otherwise
     */
    public boolean isRoot()
    {
        return parent == null;
    }

    /**
     * Query whether the tree node is leaf node
     *
     * @return True if this instance is a leaf node, false otherwise
     */
    public boolean isLeaf()
    {
        return (children == null || children.length == 0);
    }

    /**
     * Get the train error of the tree node
     *
     * @return If there is no missing data distributed on the tree node, the train error should be an integer; <br>
     * Otherwise, the train error should be an estimated <i>float</i> value computed by some statistical method.
     */
    public abstract float getTrainError();

    /**
     * Set the train error of the tree node
     *
     * @param error
     *         The new train error
     */
    public abstract void setTrainError(float error);
}
