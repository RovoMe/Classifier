package at.rovo.classifier.svm.struct;

/**
 * A node corresponds to a certain feature. It contains a index field which can be compared to the features position in
 * a word-vector, and a value field which may be represented as a weight or scaling factor.
 *
 * @author Chih-Chung Chang, Chih-Jen Lin
 */
public class Node implements java.io.Serializable
{
	/** Unique identifier necessary for serialization **/
	private static final long serialVersionUID = -9130518857022242321L;
	/** The index of a feature inside a word-vector f.e. **/
	public int index;
	/** The weight or scale of a certain feature **/
	public double value;
}
