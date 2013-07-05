/**
 * @(#)TreeView.java   1.5.2 09/03/29
 */
package at.rovo.classifier.decissionTrees.c45.util;

/**
 * An interface of the different views of trees.
 * 
 * @author Xiaohua Xu
 * @author Ping He
 */

public interface TreeView
{
	/** The prefix of the first line in the tree view **/
	public static String LEVEL_PREFIX = "";
	/** The incremental indent of the lines in the tree view **/
	public static String LEVEL_GAP = "  ";
	/** The line separator **/
	public static final String CR = System.getProperty("line.separator");

	/** 
	 * Get the head of the tree view.
	 */
	public String getHead();

	/** 
	 * Set the head of the tree view. 
	 * 
	 * @param head The new head of the tree view
	 */
	public void setHead(String head);

	/**
	 * Get the body of the tree view.
	 * 
	 * @return The body of the tree view
	 */
	public String getBody();

	/**
	 * Set the body of the tree view.
	 * 
	 * @param body The new body of the tree view
	 */
	public void setBody(String body);

	/**
	 * Get the tail of the tree view.
	 * 
	 * @return The tail of the tree view
	 */
	public String getTail();

	/**
	 * Set the tail of the tree view.
	 * 
	 * @param tail The new tail of the tree view
	 */
	public void setTail(String tail);

	/**
	 * Insert the specified information before the body of the tree view.
	 * 
	 * @param line The information to add at the beginning of the body
	 * @return The modified TreeView instance
	 */
	public TreeView insert(String line);

	/**
	 * Append the specified information after the body of the tree view.
	 * 
	 * @param line Appends the information at the end of the body
	 * @return The modified TreeView instance
	 */
	public TreeView append(String line);

	/**
	 * Unite the specified tree view into the current tree view.
	 * 
	 * @param treeView Appends the body of <em>treeView</em> to the end of this
	 *                 instance's body
	 */
	public TreeView union(TreeView treeView);

}