/**
 * @(#)Tree.java        1.5.2 09/03/29
 */

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
	public int size();

	/** Get the root node of the tree **/
	public TreeNode getRoot();

	/** Set the root node of the tree **/
	public void setRoot(TreeNode newRoot);
}