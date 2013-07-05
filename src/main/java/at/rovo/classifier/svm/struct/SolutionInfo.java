package at.rovo.classifier.svm.struct;

/**
 * <p>Information about solution except alpha</p>
 * 
 * @author Chih-Chung Chang, Chih-Jen Lin
 */
public class SolutionInfo
{
	/** The optimal objective value of the dual SVM **/
	public double obj;
	/** −b in the decision function **/
	public double rho;
	/** Upper bound for rho **/
	public double upper_bound_p;
	/** Upper bound for nu **/
	public double upper_bound_n;
	public double r; // for Solver_NU
}
