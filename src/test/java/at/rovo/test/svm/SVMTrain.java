package at.rovo.test.svm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;
import at.rovo.classifier.svm.SVM;
import at.rovo.classifier.svm.struct.Node;
import at.rovo.classifier.svm.struct.Parameter;

/**
 * <p>Example class on how to use the SVM classifier to train data from a sample
 * file and persist a trained model.</p>
 * <p>This class is based on svm_train class of libSVM created by Chih-Chung 
 * Chang and Chih-Jen Lin but was refactored strongly to fit into the available
 * classification library.</p>
 * 
 * @author Chih-Chung Chang, Chih-Jen Lin, Roman Vottner
 */
class SVMTrain
{	
	/**
	 * <p>Prints a simple help menu in case a wrong command was entered.</p>
	 */
	private static void exitWithHelp()
	{
		System.out.print("Usage: SVMTrain [options] training_set_file [model_file]\n"
						+ "options:\n"
						+ "-s svm_type : set type of SVM (default 0)\n"
						+ "	0 -- C-SVC		(multi-class classification)\n"
						+ "	1 -- nu-SVC		(multi-class classification)\n"
						+ "	2 -- one-class SVM\n"
						+ "	3 -- epsilon-SVR	(regression)\n"
						+ "	4 -- nu-SVR		(regression)\n"
						+ "-t kernel_type : set type of kernel function (default 2)\n"
						+ "	0 -- linear: u'*v\n"
						+ "	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
						+ "	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
						+ "	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
						+ "	4 -- precomputed kernel (kernel values in training_set_file)\n"
						+ "-d degree : set degree in kernel function (default 3)\n"
						+ "-g gamma : set gamma in kernel function (default 1/num_features)\n"
						+ "-r coef0 : set coef0 in kernel function (default 0)\n"
						+ "-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
						+ "-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
						+ "-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
						+ "-m cachesize : set cache memory size in MB (default 100)\n"
						+ "-e epsilon : set tolerance of termination criterion (default 0.001)\n"
						+ "-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
						+ "-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
						+ "-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
						+ "-v n : n-fold cross validation mode\n"
						+ "-q : quiet mode (no outputs)\n");
		System.exit(1);
	}

	/**
	 * <p>Parses the arguments passed to the application as well as the training
	 * data contained in the corresponding file.</p>
	 * <p>After checking the validity of the parameters it either trains the 
	 * SVM model or it executes a cross validation.</p>
	 * 
	 * @param argv The arguments passed to the application
	 * @throws IOException If the training data file could not be opened or read
	 */
	private void run(String argv[]) throws IOException
	{
		// parses the arguments passed to the application
		Parameter param = Parameter.create(argv);
		if (param == null)
		{
			exitWithHelp();
			return;
		}
		SVM svm = new SVM(param);
				
		BufferedReader fp = new BufferedReader(new FileReader(param.inputFileName));
		// read the test data from the file
		while (true)
		{
			String line = fp.readLine();
			if (line == null)
				break;

			StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");

			// first entry in the line is always the classification
			double cat = Double.parseDouble(st.nextToken());
			// run through the rest of the tokens and create a new feature for
			// every pair of index:value
			// index could be the index of a word inside a word-vector, while a
			// value of 1 may indicate that this feature is available
			int m = st.countTokens() / 2;
			Node[] node = new Node[m];
			for (int j = 0; j < m; j++)
			{
				node[j] = new Node();
				node[j].index = Integer.parseInt(st.nextToken());
				node[j].value = Double.parseDouble(st.nextToken());
			}
			svm.train(node, cat);
		}
		fp.close();
		
		if (param.crossValidation != 0)
		{
			svm.crossValidation(param);
		}
		else
		{
			svm.save();
		}
	}

	public static void main(String argv[]) throws IOException
	{
		SVMTrain t = new SVMTrain();
		t.run(argv);
	}
}
