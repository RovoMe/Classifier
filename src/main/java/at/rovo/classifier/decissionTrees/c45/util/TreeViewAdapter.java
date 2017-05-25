package at.rovo.classifier.decissionTrees.c45.util;

/**
 * An adapter of the different views of trees.
 *
 * @author Xiaohua Xu
 * @author Ping He
 */
public class TreeViewAdapter implements TreeView
{
    /** The head of the tree view **/
    protected StringBuilder head;
    /** The body of the tree view **/
    protected StringBuilder body;
    /** The tail of the tree view **/
    protected StringBuilder tail;

    /**
     * Initialize an empty tree view
     */
    protected TreeViewAdapter()
    {
        head = new StringBuilder();
        body = new StringBuilder();
        tail = new StringBuilder();
    }

    /**
     * Returns the head of the tree view
     */
    @Override
    public String getHead()
    {
        return head.toString();
    }

    /**
     * Returns the tail of the tree view
     */
    @Override
    public String getTail()
    {
        return tail.toString();
    }

    /**
     * Returns the body the tree view
     */
    @Override
    public String getBody()
    {
        return body.toString();
    }

    /**
     * Sets the new head of the tree view
     *
     * @param newHead The new head of the tree view
     */
    @Override
    public void setHead(String newHead)
    {
        head.setLength(0);
        head.append(newHead);
    }

    /**
     * Sets the new body of the tree view
     *
     * @param newBody The new body of the tree view
     */
    @Override
    public void setBody(String newBody)
    {
        body.setLength(0);
        body.append(newBody);
    }

    /**
     * Sets the new tail of the tree view
     *
     * @param newTail The new Tail of the tree view
     */
    @Override
    public void setTail(String newTail)
    {
        tail.setLength(0);
        tail.append(newTail);
    }

    /**
     * Inserts a new line at the start of the body
     *
     * @param line The line to insert at the start of the body
     */
    @Override
    public TreeView insert(String line)
    {
        body.insert(0, line + TreeView.CR);
        return this;
    }

    /**
     * Appends a new line at the end of the body
     *
     * @param line The line to append to the body of this instance
     */
    @Override
    public TreeView append(String line)
    {
        body.append(line).append(TreeView.CR);
        return this;
    }

    /**
     * Appends the body of <em>treeView</em> to the end of the body of this
     * instance.
     * <p>
     * If <em>treeView</em> is not of the same type as the current instance a
     * {@link UnsupportedOperationException} will be thrown.
     *
     * @param treeView The element whose body should be appended to the body of
     *                 the current instance
     */
    @Override
    public TreeView union(TreeView treeView)
    {
        if (this.getClass() != treeView.getClass())
        {
            throw new UnsupportedOperationException();
        }

        body.append(treeView.getBody());
        return this;
    }

    /**
     * The whole tree view.
     */
    @Override
    public String toString()
    {
        return head.toString() + body.toString() + tail.toString();
    }
}