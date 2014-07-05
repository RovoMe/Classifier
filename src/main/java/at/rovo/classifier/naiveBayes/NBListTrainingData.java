package at.rovo.classifier.naiveBayes;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import at.rovo.classifier.TrainingData;

/**
 * <p>
 * A naive Bayes training data implementation which stores trained data in
 * {@link List} structures. Only the word vector itself is defined as a map to
 * enhance the lookup-time for the position of features within the occurrences
 * list.
 * </p>
 * <p>
 * This implementation will store a list for all found categories while
 * training. A separate list will keep track of the number of trained samples
 * for each category. This value differs from the number of features per
 * category as samples relate to f.e. whole sentences while features relate to
 * words only.
 * </p>
 * <p>
 * The actual number of occurrences for a certain feature is stored in a nested
 * list structure.
 * </p>
 * <p>
 * Example:
 * </p>
 * <code>
 * categories: ['in','out']<br/>
 * catCount: [1,1]<br/>
 * wordVector: {word1:pos1,word2:po2,word3:pos3}<br/>
 * occurrences: [[num00,num01,0],[num10,0,num12]]
 * </code>
 * <p>
 * The example above contains two sample sentences. The first sentence contains
 * <em>word1</em> num00 times and <em>word2</em> num01 times while sentence two
 * contains <em>word1</em> num10 times and <em>word3</em> num12 times.
 * </p>
 * <p>
 * pos1 to pos3 in the wordVector represent the positions of the respective
 * feature in the nested lists.
 * </p>
 * 
 * @param <F>
 *            The type of the features or words
 * @param <C>
 *            The type of the categories or classes
 * 
 * @author Roman Vottner
 */
public class NBListTrainingData<F, C> extends NBTrainingData<F, C>
{
	/** The logger of this class **/
	private static Logger LOG = LogManager.getLogger(NBListTrainingData.class);
	/** Unique identifier necessary for serialization **/
	private static final long serialVersionUID = -2101815681608863601L;
	/** Stores the categories trained **/
	private List<C> categories = null;
	/** Stores the number of samples trained for each category **/
	private List<Integer> catCount = null;
	/** Stores the features and their index **/
	private Map<F, Integer> wordVector = null;
	/** Stores the actual occurrences of a feature in a certain category **/
	private List<List<Integer>> occurrences = null;

	/**
	 * <p>
	 * Initializes a package-private instance of an abstract training data
	 * object for a naive Bayes classifier.
	 * </p>
	 */
	NBListTrainingData()
	{
		this.categories = new ArrayList<>();
		this.catCount = new ArrayList<>();
		this.wordVector = new Hashtable<>();
		this.occurrences = new ArrayList<>();
	}

	@Override
	public void incrementFeature(F feature, C category)
	{
		// check if the category is already available
		if (!this.categories.contains(category))
		{
			this.categories.add(category);
			this.catCount.add(0);
			List<Integer> list = new ArrayList<>();
			this.occurrences.add(list);
			// we have to fill the occurrences of the new category up to the
			// length of the other categories
			if (this.categories.size() > 1)
			{
				int size = this.occurrences.get(0).size();
				for (int i = 0; i < size; i++)
					list.add(0);
			}
		}
		int index = this.categories.indexOf(category);

		// we do not know this feature yet, so add it and make sure the feature
		// is also added to the list of the other categories with 0 occurrences
		if (!this.wordVector.containsKey(feature))
		{
			for (int i = 0; i < this.categories.size(); i++)
				this.occurrences.get(i).add(0);

			// the new feature was added at the end of the list
			int pos = this.occurrences.get(0).size() - 1;
			this.wordVector.put(feature, pos);
			this.occurrences.get(index).set(pos, 1);
		}
		else
		{
			Integer pos = this.wordVector.get(feature);
			int val = this.occurrences.get(index).get(pos);
			this.occurrences.get(index).set(pos, ++val);
		}
	}

	@Override
	public void incrementNumberOfSamplesForCategory(C category)
	{
		int index = this.categories.indexOf(category);
		if (index != -1)
			this.catCount.set(index, this.catCount.get(index) + 1);
	}

	@Override
	protected int getNumberOfCategories()
	{
		return this.categories.size();
	}

	@Override
	protected long getTotalNumberOfFeatures()
	{
		return this.wordVector.size();
	}

	@Override
	public long getNumberOfSamplesForCategory(C category)
	{
		int index = this.categories.indexOf(category);
		if (index == -1)
			return 0;
		return this.catCount.get(index);
	}

	@Override
	public long getTotalNumberOfSamples()
	{
		int num = 0;
		for (C cat : this.categories)
			num += this.getNumberOfSamplesForCategory(cat);
		return num;
	}

	@Override
	public int getFeatureCount(F feature, C category)
	{
		int index = this.categories.indexOf(category);
		if (index == -1)
			return 0;
		Integer pos = this.wordVector.get(feature);
		if (pos != null)
			return this.occurrences.get(index).get(pos);
		return 0;
	}

	@Override
	protected long getFeatureCount(F feature)
	{
		int sum = 0;
		for (C cat : this.categories)
			sum += this.getFeatureCount(feature, cat);
		return sum;
	}

	@Override
	protected boolean containsCategory(C category)
	{
		return this.categories.contains(category);
	}

	@Override
	protected List<C> getCategories()
	{
		return this.categories;
	}
	
	@Override
	public void saveData(File directory, String name)
	{
		try (ObjectOutput object = new ObjectOutputStream(
				new BufferedOutputStream(
						new FileOutputStream(directory.getAbsoluteFile()+"/"+name))))
		{
			object.writeObject(this);
		}
		catch (IOException e)
		{
			LOG.error(e);
		}
	}
	
	@Override
	@SuppressWarnings("unchecked")
	public boolean loadData(File serializedObject)
	{
		NBListTrainingData<F,C> data = null;
		try (ObjectInputStream ois = new ObjectInputStream(
				new BufferedInputStream(
						new FileInputStream(serializedObject))))
		{
			Object obj = ois.readObject();
			if (obj instanceof TrainingData)
			{
				data = (NBListTrainingData<F, C>)obj;
				LOG.info("Found trained data for: {}", data);
			}
			else
				LOG.error("File is not a valid data object for this classifier!");
		}
		catch (IOException | ClassNotFoundException e)
		{
			LOG.error("Error while loading classifier data", e);
		}
		
		if (data != null)
		{
			this.catCount = data.catCount;
			this.categories = data.categories;
			this.occurrences = data.occurrences;
			this.wordVector	= data.wordVector;
			return true;
		}
		return false;
	}
}
