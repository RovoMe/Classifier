package at.rovo.classifier.svm.struct;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import at.rovo.classifier.TrainingData;
import at.rovo.classifier.svm.KernelType;

/**
 * <p>A <em>Problem</em> is a container for already parsed samples. A Sample 
 * moreover exists of the predefined class-label {@link #y} and a set of features
 * where each feature contains an index and a value field {@link #x}</p>
 * 
 * @author Chih-Chung Chang, Chih-Jen Lin
 */
public class Problem implements Serializable, TrainingData<Number,Node[][]>
{
	/** Unique identifier necessary for serialization **/
	private static final long serialVersionUID = 7396001955832580392L;
	/** The number of instances trained **/
	public int numInstances;
	/** Contains the labels or classifications **/
	public List<Double> y;
	/** Contains the set of features the model should be trained with**/
	public List<Node[]> x;
	/** **/
	private int maxIndex;
	
	public Problem()
	{
		x = new ArrayList<Node[]>();
		y = new ArrayList<Double>();
	}
	
	public void add(Double label, Node[] features)
	{
		this.x.add(features);
		this.y.add(label);
		this.numInstances++;	
		
		this.maxIndex = Math.max(maxIndex, features[features.length-1].index);
	}
	
	public int getMaxIndex()
	{
		return this.maxIndex;
	}
	
	/**
	 * <p>Creates a problem statement from the data provided as test data.</p>
	 * <p>Therefore the test data is parsed. The first entry in every line is
	 * the actual classification result. A value of 1 indicates a positive 
	 * classification while a value of -1 indicates a negative one.</p>
	 * <p>The remaining entries in the line are pairs of index:value, where index
	 * may represent the index of a word inside a word-vector while a value of 1
	 * indicates its positive occurrence in the document. By default a value of
	 * 0 is taken for features not mentioned yet to reduce the necessity to 
	 * provide every single feature.</p>
	 * 
	 * @param param The parameters provided to the application
	 * @return The parsed test data converted to a Problem instance to train
	 *         the classifier
	 * @throws IOException If the training file cannot be read
	 */
	public static Problem create(String inputFileName, Parameter param) throws IOException
	{
		BufferedReader fp = new BufferedReader(new FileReader(inputFileName));
		List<Double> y = new ArrayList<Double>(); // will hold the classification
		List<Node[]> x = new ArrayList<Node[]>(); // will hold the features
		int maxIndex = 0;

		// read the test data from the file
		while (true)
		{
			String line = fp.readLine();
			if (line == null)
				break;

			StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");

			// first entry in the line is always the classification
			y.add(Double.parseDouble(st.nextToken()));
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
			if (m > 0)
				maxIndex = Math.max(maxIndex, node[m - 1].index);
			x.add(node);
		}
		fp.close();

		// Create a problem instance and assign the parsed and converted data
		// to the problem instance
		Problem prob = new Problem();
		prob.numInstances = y.size();
		prob.x = x;
		prob.y = y;

		// normalizes the radius used for RBF f.e.
		if (param.gamma == 0 && maxIndex > 0)
			param.gamma = 1.0 / maxIndex;

		// check if the format for a pre-computed kernel type is appropriate
		if (KernelType.PRECOMPUTED.equals(param.kernelType))
		{
			for (int i = 0; i < prob.numInstances; i++)
			{
				if (prob.x.get(i)[0].index != 0)
				{
					System.err.print("Wrong kernel matrix: first column must be 0:sample_serial_number\n");
					System.exit(1);
				}
				if ((int) prob.x.get(i)[0].value <= 0 || (int) prob.x.get(i)[0].value > maxIndex)
				{
					System.err.print("Wrong input format: sample_serial_number out of range\n");
					System.exit(1);
				}
			}
		}

		return prob;
	}
}
