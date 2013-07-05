//
// svm_model
//
package at.rovo.classifier.svm;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import at.rovo.classifier.svm.kernel.Kernel;
import at.rovo.classifier.svm.struct.Node;
import at.rovo.classifier.svm.struct.Parameter;

/**
 * <p></p>
 * 
 * @author Chih-Chung Chang, Chih-Jen Lin
 */
public class Model implements java.io.Serializable
{
	protected static Logger logger = LogManager.getLogger(Model.class.getName());
	
	/** Unique identifier necessary for serialization **/
	private static final long serialVersionUID = 3286349814636287449L;
	/** The parameters passed to the application **/
	public Parameter param;
	/** The number of classes. This is 2 in regression or one class SVM **/
	public int nrClass;
	/** The number of instances trained **/
	public int numInstances; // total #SV
	/** SVs (SV[l]) **/
	public Node[][] SV;
	/** Coefficients for SVs in decision functions (sv_coef[k-1][l]) **/
	public double[][] svCoef; 
	/** constants in decision functions (rho[k*(k-1)/2]) **/
	public double[] rho;
	/** pairwise probability information **/
	public double[] probA;
	public double[] probB;
	/** sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to 
	 * indicate SVs in the training set **/
	public int[] svIndices;

	// for classification only

	/** label of each class (label[k]) **/
	public int[] label;
	/** number of SVs for each class (nSV[k]) nSV[0] + nSV[1] + ... + nSV[k-1] = l **/
	public int[] nSV;
	
	/**
	 * <p>Loads a model from the file and returns it as an object.</p>
	 * 
	 * @param modelFileName The name of the file o load the data from
	 * @return The parsed model
	 * @throws IOException If the file could not be opened or read
	 */
	public static Model load(String modelFileName) throws IOException
	{
		// read parameters

		Model model = new Model();
		Parameter param = new Parameter();
		model.param = param;
		model.rho = null;
		model.probA = null;
		model.probB = null;
		model.label = null;
		model.nSV = null;
		
		BufferedReader fp = new BufferedReader(new FileReader(modelFileName));

		while (true)
		{
			String cmd = fp.readLine();
			String arg = cmd.substring(cmd.indexOf(' ') + 1);

			if (cmd.startsWith("svm_type"))
			{
				int i;
				for (i = 0; i < SVMType.length(); i++)
				{
					if (arg.equals(SVMType.get(i).toString()))
					{
						param.svmType = SVMType.get(i);
						break;
					}
				}
				if (i == SVMType.length())
				{
					System.err.print("unknown svm type.\n");
					fp.close();
					return null;
				}
			}
			else if (cmd.startsWith("kernel_type"))
			{
				int i;
				for (i = 0; i < KernelType.length(); i++)
				{
					if (arg.equals(KernelType.get(i).toString()))
					{
						param.kernelType = KernelType.get(i);
						break;
					}
				}
				if (i == KernelType.length())
				{
					System.err.print("unknown kernel function.\n");
					fp.close();
					return null;
				}
			}
			else if (cmd.startsWith("degree"))
				param.degree = Integer.parseInt(arg);
			else if (cmd.startsWith("gamma"))
				param.gamma = Double.parseDouble(arg);
			else if (cmd.startsWith("coef0"))
				param.coef0 = Double.parseDouble(arg);
			else if (cmd.startsWith("nr_class"))
				model.nrClass = Integer.parseInt(arg);
			else if (cmd.startsWith("total_sv"))
				model.numInstances = Integer.parseInt(arg);
			else if (cmd.startsWith("rho"))
			{
				int n = model.nrClass * (model.nrClass - 1) / 2;
				model.rho = new double[n];
				StringTokenizer st = new StringTokenizer(arg);
				for (int i = 0; i < n; i++)
					model.rho[i] = Double.parseDouble(st.nextToken());
			}
			else if (cmd.startsWith("label"))
			{
				int n = model.nrClass;
				model.label = new int[n];
				StringTokenizer st = new StringTokenizer(arg);
				for (int i = 0; i < n; i++)
					model.label[i] = Integer.parseInt(st.nextToken());
			}
			else if (cmd.startsWith("probA"))
			{
				int n = model.nrClass * (model.nrClass - 1) / 2;
				model.probA = new double[n];
				StringTokenizer st = new StringTokenizer(arg);
				for (int i = 0; i < n; i++)
					model.probA[i] = Double.parseDouble(st.nextToken());
			}
			else if (cmd.startsWith("probB"))
			{
				int n = model.nrClass * (model.nrClass - 1) / 2;
				model.probB = new double[n];
				StringTokenizer st = new StringTokenizer(arg);
				for (int i = 0; i < n; i++)
					model.probB[i] = Double.parseDouble(st.nextToken());
			}
			else if (cmd.startsWith("nr_sv"))
			{
				int n = model.nrClass;
				model.nSV = new int[n];
				StringTokenizer st = new StringTokenizer(arg);
				for (int i = 0; i < n; i++)
					model.nSV[i] = Integer.parseInt(st.nextToken());
			}
			else if (cmd.startsWith("SV"))
			{
				break;
			}
			else
			{
				System.err.print("unknown text in model file: [" + cmd + "]\n");
				fp.close();
				return null;
			}
		}

		// read sv_coef and SV

		int m = model.nrClass - 1;
		int l = model.numInstances;
		model.svCoef = new double[m][l];
		model.SV = new Node[l][];

		for (int i = 0; i < l; i++)
		{
			String line = fp.readLine();
			StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");

			for (int k = 0; k < m; k++)
				model.svCoef[k][i] = Double.parseDouble(st.nextToken());
			int n = st.countTokens() / 2;
			model.SV[i] = new Node[n];
			for (int j = 0; j < n; j++)
			{
				model.SV[i][j] = new Node();
				model.SV[i][j].index = Integer.parseInt(st.nextToken());
				model.SV[i][j].value = Double.parseDouble(st.nextToken());
			}
		}

		fp.close();
		return model;
	}
	
	/**
	 * <p>Saves a trained model and stores the data into a file 
	 * <em>modelFileName</em> is pointing to.</p>
	 * <p>The data will be stored in the following format:</p>
	 * <code>
	 * svm_type string<br/>
	 * kernel_type string<br/>
	 * gamma double<br/>
	 * nr_class int<br/>
	 * total_sv int<br/>
	 * rho double<br/>
	 * label int<br/>
	 * nr_sv int<br/>
	 * SV<br/>
	 * double int:double int:double ...<br/>
	 * double int:double int:double ...
	 * </code>
	 * <p>Note that <em>label</em> and <em>nr_sv</em> are stored for every class.
	 * So if there are two classes there will be two entries separated by a comma
	 * for <em>label</em> and <em>nr_sv</em>.</p>
	 * <p>The entries for SV always are lead by the class label of the sample
	 * and followed by the enumeration of the features, where the part before the
	 * semicolon (:) is the index part and the part after the semicolon the value
	 * part for each feature. The features should be sorted in an ascending order
	 * by the index part.</p>
	 * 
	 * @param modelFileName
	 * @throws IOException
	 */
	public void save(String modelFileName) throws IOException
	{
		DataOutputStream fp = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(modelFileName)));

		Parameter param = this.param;

		fp.writeBytes("svm_type " + param.svmType + "\n");
		fp.writeBytes("kernel_type " + param.kernelType	+ "\n");

		if (KernelType.POLYNOMIAL.equals(param.kernelType))
			fp.writeBytes("degree " + param.degree + "\n");

		if (KernelType.POLYNOMIAL.equals(param.kernelType)
				|| KernelType.RBF.equals(param.kernelType)
				|| KernelType.SIGMOID.equals(param.kernelType))
			fp.writeBytes("gamma " + param.gamma + "\n");

		if (KernelType.POLYNOMIAL.equals(param.kernelType)
				|| KernelType.SIGMOID.equals(param.kernelType))
			fp.writeBytes("coef0 " + param.coef0 + "\n");

		int nr_class = this.nrClass;
		fp.writeBytes("nr_class " + nr_class + "\n");
		int l = this.numInstances;
		fp.writeBytes("total_sv " + l + "\n");
		fp.writeBytes("rho");
		for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
			fp.writeBytes(" " + this.rho[i]);
		fp.writeBytes("\n");
		
		if (this.label != null)
		{
			fp.writeBytes("label");
			for (int i = 0; i < nr_class; i++)
				fp.writeBytes(" " + this.label[i]);
			fp.writeBytes("\n");
		}

		if (this.probA != null) // regression has probA only
		{
			fp.writeBytes("probA");
			for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
				fp.writeBytes(" " + this.probA[i]);
			fp.writeBytes("\n");
		}
		if (this.probB != null)
		{
			fp.writeBytes("probB");
			for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
				fp.writeBytes(" " + this.probB[i]);
			fp.writeBytes("\n");
		}

		if (this.nSV != null)
		{
			fp.writeBytes("nr_sv");
			for (int i = 0; i < nr_class; i++)
				fp.writeBytes(" " + this.nSV[i]);
			fp.writeBytes("\n");
		}

		fp.writeBytes("SV\n");
		double[][] sv_coef = this.svCoef;
		Node[][] SV = this.SV;

		for (int i = 0; i < l; i++)
		{
			for (int j = 0; j < nr_class - 1; j++)
				fp.writeBytes(sv_coef[j][i] + " ");

			Node[] p = SV[i];
			if (KernelType.PRECOMPUTED.equals(param.kernelType))
				fp.writeBytes("0:" + (int) (p[0].value));
			else
				for (int j = 0; j < p.length; j++)
					fp.writeBytes(p[j].index + ":" + p[j].value + " ");
			fp.writeBytes("\n");
		}

		fp.close();
	}
	
	/**
	 * <p>Returns the SVM type of the model.</p>
	 * 
	 * @return The SVM type of the model
	 */
	public SVMType getSVMType()
	{
		return this.param.svmType;
	}

	/**
	 * <p>Returns the number of classes the model contains.</p>
	 * 
	 * @return The number of trained classes
	 */
	public int getNrClass()
	{
		return this.nrClass;
	}
	
	/**
	 * <p>Fills a provided array of labels with the labels contained in this 
	 * model.</p>
	 * 
	 * @param label The array that should be filled with labels contained in
	 *              the model
	 */
	public void getLabels(int[] label)
	{
		if (this.label != null)
			for (int i = 0; i < this.nrClass; i++)
				label[i] = this.label[i];
	}

	/**
	 * <p>Fills a provided array of indices with the indices contained in this
	 * model.</p>
	 * 
	 * @param indices The array that should be filled with indices contained in
	 *                the model
	 */
	public void getSVIndices(int[] indices)
	{
		if (this.svIndices != null)
			for (int i = 0; i < this.numInstances; i++)
				indices[i] = this.svIndices[i];
	}

	/**
	 * <p>Returns the total number of support vectors.</p>
	 * 
	 * @return The number of support vectors
	 */
	public int getNrSV()
	{
		return this.numInstances;
	}
	
	/**
	 * <p>Returns the probability for a regression based model.</p>
	 * 
	 * @return The probability if the model is a regression model, 0 instead
	 */
	public double getSVRProbability()
	{
		if ((SVMType.EPSILON_SVR.equals(this.param.svmType) 
				|| SVMType.NU_SVR.equals(this.param.svmType))
				&& this.probA != null)
			return this.probA[0];
		else
		{
			System.err.print("Model doesn't contain information for SVR probability inference\n");
			return 0;
		}
	}
	
	/**
	 * <p>Predicts the class a certain sample belongs to.</p>
	 * 
	 * @param x A sample consisting of multiple features
	 * @return The predicted class the sample belongs to
	 */
	public double predict(Node[] x)
	{
		int nr_class = this.nrClass;
		double[] decValues;
		if (SVMType.ONE_CLASS.equals(this.param.svmType)
				|| SVMType.EPSILON_SVR.equals(this.param.svmType)
				|| SVMType.NU_SVR.equals(this.param.svmType))
			decValues = new double[1];
		else
			decValues = new double[nr_class * (nr_class - 1) / 2];
		return predictValues(x, decValues);
	}
	
	/**
	 * <p>Predicts the values for each feature inside the node by solving a 
	 * quadratic programming problem with linear constraints by utilizing
	 * a kernel function which transforms the problem into a further space to
	 * simplify the problem.</p>
	 * <p>This approach can be thought of as instead of drawing a non-linear 
	 * separation line (winding line) between the classes, the kernel function 
	 * modifies the the space in a form that the separation line is a linear 
	 * line and the classes adept to the new space instead.</p>
	 * 
	 * @param x The sample containing the features to predict values for
	 * @param decValues An array to be filled with values by the method to enable
	 *                  decision support for choosing the best support vectors
	 * @return The predicted class the sample belongs to
	 * @see http://www.dtreg.com/svm.htm
	 */
	double predictValues(Node[] x, double[] decValues)
	{
		int i;
		if (SVMType.ONE_CLASS.equals(this.param.svmType)
				|| SVMType.EPSILON_SVR.equals(this.param.svmType)
				|| SVMType.NU_SVR.equals(this.param.svmType))
		{
			double[] sv_coef = this.svCoef[0];
			double sum = 0;
			// application of the kernel function for each feature
			for (i = 0; i < this.numInstances; i++)
				sum += sv_coef[i] * Kernel.function(x, this.SV[i], this.param);
			sum -= this.rho[0];
			decValues[0] = sum;

			if (SVMType.ONE_CLASS.equals(this.param.svmType))
				return (sum > 0) ? 1 : -1;
			else
				return sum;
		}
		else
		{
			int nr_class = this.nrClass;
			int l = this.numInstances;

			// application of the kernel function for each feature
			double[] kvalue = new double[l];
			for (i = 0; i < l; i++)
				kvalue[i] = Kernel.function(x, this.SV[i], this.param);

			int[] start = new int[nr_class];
			start[0] = 0;
			for (i = 1; i < nr_class; i++)
				start[i] = start[i - 1] + this.nSV[i - 1];

			int[] vote = new int[nr_class];
			for (i = 0; i < nr_class; i++)
				vote[i] = 0;

			int p = 0;
			for (i = 0; i < nr_class; i++)
				for (int j = i + 1; j < nr_class; j++)
				{
					double sum = 0;
					int si = start[i];
					int sj = start[j];
					int ci = this.nSV[i];
					int cj = this.nSV[j];

					int k;
					double[] coef1 = this.svCoef[j - 1];
					double[] coef2 = this.svCoef[i];
					for (k = 0; k < ci; k++)
						sum += coef1[si + k] * kvalue[si + k];
					for (k = 0; k < cj; k++)
						sum += coef2[sj + k] * kvalue[sj + k];
					sum -= this.rho[p];
					decValues[p] = sum;

					if (decValues[p] > 0)
						++vote[i];
					else
						++vote[j];
					p++;
				}

			int vote_max_idx = 0;
			for (i = 1; i < nr_class; i++)
				if (vote[i] > vote[vote_max_idx])
					vote_max_idx = i;

			return this.label[vote_max_idx];
		}
	}

	public double predictProbability(Node[] x, double[] prob_estimates)
	{
		if ((SVMType.C_SVC.equals(this.param.svmType) || SVMType.NU_SVC.equals(this.param.svmType))
				&& this.probA != null && this.probB != null)
		{
			int i;
			int nr_class = this.nrClass;
			double[] dec_values = new double[nr_class * (nr_class - 1) / 2];
			this.predictValues(x, dec_values);

			double min_prob = 1e-7;
			double[][] pairwise_prob = new double[nr_class][nr_class];

			int k = 0;
			for (i = 0; i < nr_class; i++)
				for (int j = i + 1; j < nr_class; j++)
				{
					pairwise_prob[i][j] = Math.min(Math.max(sigmoidPredict(
							dec_values[k], this.probA[k], this.probB[k]),
							min_prob), 1 - min_prob);
					pairwise_prob[j][i] = 1 - pairwise_prob[i][j];
					k++;
				}
			multiclassProbability(nr_class, pairwise_prob, prob_estimates);

			int prob_max_idx = 0;
			for (i = 1; i < nr_class; i++)
				if (prob_estimates[i] > prob_estimates[prob_max_idx])
					prob_max_idx = i;
			return this.label[prob_max_idx];
		}
		else
			return this.predict(x);
	}
	
	private double sigmoidPredict(double decision_value, double A, double B)
	{
		double fApB = decision_value * A + B;
		if (fApB >= 0)
			return Math.exp(-fApB) / (1.0 + Math.exp(-fApB));
		else
			return 1.0 / (1 + Math.exp(fApB));
	}
	
	// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
	private void multiclassProbability(int k, double[][] r, double[] p)
	{
		int t, j;
		int iter = 0, max_iter = Math.max(100, k);
		double[][] Q = new double[k][k];
		double[] Qp = new double[k];
		double pQp, eps = 0.005 / k;

		for (t = 0; t < k; t++)
		{
			p[t] = 1.0 / k; // Valid if k = 1
			Q[t][t] = 0;
			for (j = 0; j < t; j++)
			{
				Q[t][t] += r[j][t] * r[j][t];
				Q[t][j] = Q[j][t];
			}
			for (j = t + 1; j < k; j++)
			{
				Q[t][t] += r[j][t] * r[j][t];
				Q[t][j] = -r[j][t] * r[t][j];
			}
		}
		for (iter = 0; iter < max_iter; iter++)
		{
			// stopping condition, recalculate QP,pQP for numerical accuracy
			pQp = 0;
			for (t = 0; t < k; t++)
			{
				Qp[t] = 0;
				for (j = 0; j < k; j++)
					Qp[t] += Q[t][j] * p[j];
				pQp += p[t] * Qp[t];
			}
			double max_error = 0;
			for (t = 0; t < k; t++)
			{
				double error = Math.abs(Qp[t] - pQp);
				if (error > max_error)
					max_error = error;
			}
			if (max_error < eps)
				break;

			for (t = 0; t < k; t++)
			{
				double diff = (-Qp[t] + pQp) / Q[t][t];
				p[t] += diff;
				pQp = (pQp + diff * (diff * Q[t][t] + 2 * Qp[t])) / (1 + diff)
						/ (1 + diff);
				for (j = 0; j < k; j++)
				{
					Qp[j] = (Qp[j] + diff * Q[t][j]) / (1 + diff);
					p[j] /= (1 + diff);
				}
			}
		}
		if (iter >= max_iter)
			if (logger.isDebugEnabled())
				logger.debug("Exceeds max_iter in multiclass_prob\n");
	}
	
	public int checkProbabilityModel()
	{
		if (((SVMType.C_SVC.equals(this.param.svmType) || SVMType.NU_SVC.equals(this.param.svmType)) && this.probA != null && this.probB != null)
				|| ((SVMType.EPSILON_SVR.equals(this.param.svmType) || SVMType.NU_SVR.equals(this.param.svmType)) && this.probA != null))
			return 1;
		else
			return 0;
	}
}
