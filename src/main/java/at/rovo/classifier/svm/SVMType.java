package at.rovo.classifier.svm;

public enum SVMType
{
	/**
	 * <p>C-support vector classification (Boser et al., 1992; Cortes and Vapnik,
	 * 1995) with C > 0 being the regularization parameter of the classification.</p>
	 */
	C_SVC(0)
	{
		public String toString()
		{
			return "c_svc"; 
		}
	},
	/**
	 * <p>ν-support vector classification (Schölkopf et al., 2000) with ν being
	 * an upper bound on the fraction of training errors and a lower bound of 
	 * the fraction of support vectors.</p>
	 */
	NU_SVC(1)
	{
		public String toString()
		{
			return "nu_svc"; 
		}
	},
	/**
	 * <p>One class SVM (Schölkopf et al., 2001) for estimating the support of 
	 * a high-dimensional distribution.</p>
	 */
	ONE_CLASS(2)
	{
		public String toString()
		{
			return "one_class"; 
		}
	},
	/**
	 * <p>Epsilon-support vector regression (Vapnik, 1998)</p>
	 */
	EPSILON_SVR(3)
	{
		public String toString()
		{
			return "epsilonc_svr"; 
		}
	}, 
	/**
	 * <p>ν-support vector regression (Schölkopf et al., 2000) where parameter
	 * v controls the number of support vectors used.</p>
	 */
	NU_SVR(4)
	{
		public String toString()
		{
			return "nu_svr"; 
		}
	};
	private int value;
	
	private SVMType(int value)
	{
		this.value = value;
	}
	
	/**
	 * <p>Returns the ordinal index of the current SVM type.</p>
	 * 
	 * @return The ordinal index of the SVM type
	 */
	public int valueOf()
	{
		return this.value;
	}
	
	/**
	 * <p>Returns the number of known SVM types.</p>
	 * 
	 * @return The number of known SVM types
	 */
	public static int length()
	{
		return 5;
	}
	
	/**
	 * <p>Returns a SVMType based on the ordinal index of the type.</p>
	 * 
	 * @param i The ordinal index of the SVM type
	 * @return The SVM type corresponding to the ordinal index
	 */
	public static SVMType get(int i)
	{
		switch (i)
		{
			case 0:
				return SVMType.C_SVC;
			case 1:
				return SVMType.NU_SVC;
			case 2:
				return SVMType.ONE_CLASS;
			case 3:
				return SVMType.EPSILON_SVR;
			case 4:
				return SVMType.NU_SVR;
			default:
				return SVMType.C_SVC;
		}
	}
	
	/**
	 * <p>Returns a SVM type based on the name of the type.</p>
	 * 
	 * @param s The name of the SVM type
	 * @return The SVM type corresponding to the provided name
	 */
	public static SVMType get(String s)
	{
		switch (s)
		{
			case "c_svc":
			case "0":
				return SVMType.C_SVC;
			case "nu_svc":
			case "1":
				return SVMType.NU_SVC;
			case "one_class":
			case "2":
				return SVMType.ONE_CLASS;
			case "epsilon_svr":
			case "3":
				return SVMType.EPSILON_SVR;
			case "nu_svr":
			case "4":
				return SVMType.NU_SVR;
			default:
				return SVMType.C_SVC;
		}
	}
}
