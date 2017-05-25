package at.rovo.classifier.decissionTrees.c45.tree;

/**
 * The interface of a tree
 *
 * @author Ping He
 * @author Xiaohua Xu
 */
public interface Tree
{
    /** Get the total number of tree nodes in the tree **/
    int size();

    /** Get the root node of the tree **/
    TreeNode getRoot();

    /** Set the root node of the tree **/
    void setRoot(TreeNode newRoot);
}