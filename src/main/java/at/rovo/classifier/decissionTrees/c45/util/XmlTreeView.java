package at.rovo.classifier.decissionTrees.c45.util;

import at.rovo.classifier.decissionTrees.c45.tree.LeafNode;
import at.rovo.classifier.decissionTrees.c45.tree.Tree;
import at.rovo.classifier.decissionTrees.c45.tree.TreeNode;

/**
 * The xml view of trees.
 *
 * @author Xiaohua Xu
 * @author Ping He
 */
public class XmlTreeView extends TreeViewAdapter
{
    /** The root of the tree to be viewed **/
    private TreeNode root;

    /**
     * Initialize a xml view for the specified tree.
     */
    public XmlTreeView(Tree tree)
    {
        this(tree.getRoot());
    }

    /**
     * Initialize a xml view for the tree with the specified tree node as its
     * root.
     */
    public XmlTreeView(TreeNode root)
    {
        this.root = root;
        initHead();
        initBody();
        initTail();
    }

    /**
     * Initialize the head of the tree view.
     */
    private void initHead()
    {
        setHead("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>" + TreeView.CR + TreeView.CR + "<classifier>" + TreeView.CR);
    }

    /**
     * Initialize the body of the tree view.
     */
    private void initBody()
    {
        StringBuilder buffer = new StringBuilder();
        String prefix = TreeView.LEVEL_PREFIX + TreeView.LEVEL_GAP + TreeView.LEVEL_GAP;
        preorderToBuffer(root, buffer, prefix, TreeView.LEVEL_GAP);
        setBody(buffer.toString());
    }

    /**
     * Initialize the tail of the tree view.
     */
    private void initTail()
    {
        setTail("</classifier>");
    }

    /**
     * Correct the expression of the symbols in the xml output.
     */
    private String correct(String value)
    {
        value = value.replace("&", "&amp;");
        value = value.replace("<", "&lt;");
        value = value.replace(">", "&gt;");
        value = value.replace("\"", "&quot;");
        value = value.replace("'", "&apos;");

        return value;
    }

    /**
     * Construct the xml view of the tree started from the specified tree node
     * preorderly.
     */
    private void preorderToBuffer(TreeNode node, StringBuilder buffer, String prefix, String gap)
    {
        buffer.append(prefix).append("<").append(getNodeType(node)).append(">").append(TreeView.CR);
        buffer.append(prefix).append(gap).append("<content>").append(correct(node.toString())).append("</content>")
                .append(TreeView.CR);

        if (node instanceof LeafNode)
        {
            buffer.append(prefix).append("</").append(getNodeType(node)).append(">").append(TreeView.CR);
            return;
        }

        for (TreeNode child : node.getChildren())
        {
            preorderToBuffer(child, buffer, prefix + gap, gap);
        }

        buffer.append(prefix).append("</").append(getNodeType(node)).append(">").append(TreeView.CR);
    }

    /**
     * The node type as metadata to be stored in xml tree.
     */
    private String getNodeType(TreeNode node)
    {
        if (node.isRoot())
        {
            return "root";
        }
        else if (node.isLeaf())
        {
            return "leaf";
        }
        else
        {
            return "internal";
        }
    }

}