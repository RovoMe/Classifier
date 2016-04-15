/**
 * @(#)HtmlTreeView.java 1.5.2 09/03/29
 */

package at.rovo.classifier.decissionTrees.c45.util;

import at.rovo.classifier.decissionTrees.c45.tree.LeafNode;
import at.rovo.classifier.decissionTrees.c45.tree.Tree;
import at.rovo.classifier.decissionTrees.c45.tree.TreeNode;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * The html view of trees.
 *
 * @author Xiaohua Xu
 * @author Ping He
 */
public class HtmlTreeView extends TreeViewAdapter
{
    /** The number of trees the current tree view contains **/
    private static int count = 0;
    /** The root of the tree to be viewed **/
    private TreeNode root;

    /**
     * Initialize a html view for the specified tree.
     */
    public HtmlTreeView(Tree tree)
    {
        this(tree.getRoot());
    }

    /**
     * Initialize a html view for the tree with the specified tree node as its root.
     *
     * @param root
     *         The root node which acts as a parent for this instance
     */
    public HtmlTreeView(TreeNode root)
    {
        this.root = root;
        initHead();
        initBody();
        initTail();
    }

    /**
     * Initialize the head of the tree view.
     */
    public void initHead()
    {
        String title = root.getName();
        StringBuilder buffer = new StringBuilder();

        buffer.append("<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" ")
                .append("\"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\">").append(TreeView.CR)
                .append(TreeView.CR);
        buffer.append("<html>").append(TreeView.CR).append(TreeView.CR);
        buffer.append("<head>").append(TreeView.CR);
        buffer.append(TreeView.LEVEL_GAP).append("<title>").append(title).append("</title>").append(TreeView.CR);
        buffer.append(TreeView.LEVEL_GAP)
                .append("<link rel=\"StyleSheet\" href=\"dTree/dtree.css\" type=\"text/css\" />").append(TreeView.CR);
        buffer.append(TreeView.LEVEL_GAP).append("<script type=\"text/javascript\" src=\"dTree/dtree.js\"></script>")
                .append(TreeView.CR);
        buffer.append("</head>").append(TreeView.CR).append(TreeView.CR);
        buffer.append("<body>").append(TreeView.CR).append(TreeView.CR);
        setHead(buffer.toString());
    }

    /**
     * Initialize the body of the tree view.
     */
    public void initBody()
    {
        StringBuilder buffer = new StringBuilder();
        buffer.append("<div class=\"dtree\">").append(TreeView.CR);
        buffer.append(TreeView.LEVEL_GAP).append("<p><a href=\"javascript: dt").append(count)
                .append(".openAll();\">open all</a> ").append("| <a href=\"javascript: dt").append(count)
                .append(".closeAll();\">close all</a></p>").append(TreeView.CR);
        buffer.append(TreeView.LEVEL_GAP).append("<script type=\"text/javascript\">").append(TreeView.CR);
        buffer.append(TreeView.LEVEL_GAP).append(TreeView.LEVEL_GAP).append("dt").append(count)
                .append(" = new dTree('dt").append(count).append("');").append(TreeView.CR);

        // add all nodes in depth-first order into the map and use their number
        // of adding to the map as value
        Map<TreeNode, Integer> map = new HashMap<>();
        preorderToMap(root, map);

        Set<TreeNode> keySet = map.keySet();
        for (TreeNode node : keySet)
        {
            buffer.append(TreeView.LEVEL_GAP).append(TreeView.LEVEL_GAP).append("dt").append(count).append(".add(")
                    .append(map.get(node)).append(", ");
            if (node.isRoot())
            {
                buffer.append("-1").append(", '").append(correct(node.toString())).append("');").append(TreeView.CR);
            }
            else
            {
                buffer.append(correct(map.get(node.getParent()).toString())).append(", '")
                        .append(correct(node.toString())).append("');").append(TreeView.CR);
            }

        }

        buffer.append(TreeView.LEVEL_GAP).append(TreeView.LEVEL_GAP).append("document.write(dt").append(count)
                .append(")").append(TreeView.CR);
        buffer.append(TreeView.LEVEL_GAP).append(TreeView.LEVEL_GAP).append("dt").append(count).append(".openAll()")
                .append(TreeView.CR);
        buffer.append(TreeView.LEVEL_GAP).append("</script>").append(TreeView.CR);
        buffer.append("</div>").append(TreeView.CR).append(TreeView.CR);
        setBody(buffer.toString());

        count++;
    }

    /**
     * Initialize the tail of the tree view.
     */
    public void initTail()
    {
        StringBuilder buffer = new StringBuilder();
        buffer.append("</body>").append(TreeView.CR).append(TreeView.CR);
        buffer.append("</html>").append(TreeView.CR);
        setTail(buffer.toString());
    }

    /**
     * Correct the expression of the symbol "'" in the html output.
     *
     * @return The corrected value
     */
    private String correct(String value)
    {
        return value.replace("'", "\\'");
    }

    /**
     * Map the tree nodes in the tree started from the specified tree node to a hash map preorderly.
     *
     * @param from
     *         The source node to add to the map in depth-first order
     * @param map
     *         The map to add the respective nodes with their id (number of elements added to the map)
     */
    private void preorderToMap(TreeNode from, Map<TreeNode, Integer> map)
    {
        //if (from == null) return;
        Integer id = map.size();
        map.put(from, id);

        if (from instanceof LeafNode)
        {
            return;
        }

        for (TreeNode child : from.getChildren())
        {
            preorderToMap(child, map);
        }
    }
}