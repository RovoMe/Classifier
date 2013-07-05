package at.rovo.classifier.svm.struct;

import at.rovo.classifier.svm.KernelType;
import at.rovo.classifier.svm.SVMType;

/**
 * <p></p>
 * 
 * @author Chih-Chung Chang, Chih-Jen Lin
 */
public class Parameter implements Cloneable, java.io.Serializable
{
	/**
	 * 
	 */
	private static final long serialVersionUID = -4329004352172259479L;
	
	/** The SVM type to be used for training or classification **/
	public SVMType svmType;
	/** The kernel function used for training or classification **/
	public KernelType kernelType;
	public int degree; // for poly
	/** Specifies the radius of the RBF function **/
	public double gamma; // for poly/rbf/sigmoid
	public double coef0; // for poly/sigmoid

	// these are for training only
	public double cache_size; // in MB
	public double eps; // stopping criteria
	/** 
	 * <p>Specifies how many outliers are taken into account. It therefore 
	 * specifies the importance of outliers in respect to the margin.</p>
	 * <p>The larger C is the less the final training error will be. But on 
	 * increasing C too much one risks losing the generalization properties of 
	 * the classifier, because it will try to fit as best as possible all the 
	 * training points (including the possible errors of the dataset).</p>
	 * <p>In addition a large C, usually increases the time needed for training.
	 * </p>
	 */
	public double C; // for C_SVC, EPSILON_SVR and NU_SVR
	public int nrWeight; // for C_SVC
	public int[] weightLabel; // for C_SVC
	public double[] weight; // for C_SVC
	public double nu; // for NU_SVC, ONE_CLASS, and NU_SVR
	/** Measured the cost of the errors on the training points. These are zero 
	 * for all points that are inside the "epsilon intensive" band **/
	public double p; // for EPSILON_SVR
	public int shrinking; // use the shrinking heuristics
	public int probability; // do probability estimates
	public int crossValidation;
	public int nrFold;
	public String inputFileName;
	public String modelFileName;

	public Object clone()
	{
		try
		{
			return super.clone();
		}
		catch (CloneNotSupportedException e)
		{
			return null;
		}
	}
	
	public static Parameter create(String[] argv)
	{
		Parameter param = new Parameter();
		// default values
		param.svmType = SVMType.C_SVC;
		param.kernelType = KernelType.RBF;
		param.degree = 3;
		param.gamma = 0; // 1/num_features
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 100;
		param.C = 1;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.probability = 0;
		param.nrWeight = 0;
		param.weightLabel = new int[0];
		param.weight = new double[0];
		param.crossValidation = 0;

		return update(param, argv);
	}
	
	public static Parameter update(Parameter param, String[] argv)
	{
		int i;
		// parse options
		for (i = 0; i < argv.length; i++)
		{
			if (argv[i].charAt(0) != '-')
				break;
			if (++i >= argv.length)
				return param;
			switch (argv[i - 1].charAt(1))
			{
				case 's':
					param.svmType = SVMType.get(argv[i]);
					break;
				case 't':
					param.kernelType = KernelType.get(argv[i]);
					break;
				case 'd':
					param.degree = Integer.parseInt(argv[i]);
					break;
				case 'g':
					param.gamma = Double.parseDouble(argv[i]);
					break;
				case 'r':
					param.coef0 = Double.parseDouble(argv[i]);
					break;
				case 'n':
					param.nu = Double.parseDouble(argv[i]);
					break;
				case 'm':
					param.cache_size = Double.parseDouble(argv[i]);
					break;
				case 'c':
					param.C = Double.parseDouble(argv[i]);
					break;
				case 'e':
					param.eps = Double.parseDouble(argv[i]);
					break;
				case 'p':
					param.p = Double.parseDouble(argv[i]);
					break;
				case 'h':
					param.shrinking = Integer.parseInt(argv[i]);
					break;
				case 'b':
					param.probability = Integer.parseInt(argv[i]);
					break;
				case 'v':
					param.crossValidation = 1;
					param.nrFold = Integer.parseInt(argv[i]);
					if (param.nrFold < 2)
					{
						System.err.print("n-fold cross validation: n must >= 2\n");
						return null;
					}
					break;
				case 'w':
					++param.nrWeight;
					{
						int[] old = param.weightLabel;
						param.weightLabel = new int[param.nrWeight];
						System.arraycopy(old, 0, param.weightLabel, 0,	param.nrWeight - 1);
					}
	
					{
						double[] old = param.weight;
						param.weight = new double[param.nrWeight];
						System.arraycopy(old, 0, param.weight, 0, param.nrWeight - 1);
					}
	
					param.weightLabel[param.nrWeight - 1] = Integer.parseInt(argv[i - 1].substring(2));
					param.weight[param.nrWeight - 1] = Double.parseDouble(argv[i]);
					break;
				default:
					System.err.print("Unknown option: " + argv[i - 1] + "\n");
					return param;
			}
		}

		// determine if filenames have been passed
		if (i >= argv.length)
			return param;

		param.inputFileName = argv[i];

		if (i < argv.length - 1)
			param.modelFileName = argv[i + 1];
		else
		{
			int p = argv[i].lastIndexOf('/');
			++p; // whew...
			param.modelFileName = argv[i].substring(p) + ".model";
		}
		
		return param;	
	}

}
