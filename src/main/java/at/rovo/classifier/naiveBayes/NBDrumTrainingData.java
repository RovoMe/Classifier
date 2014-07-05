package at.rovo.classifier.naiveBayes;

import at.rovo.caching.drum.Drum;
import at.rovo.caching.drum.IDispatcher;
import at.rovo.caching.drum.IDrum;
import at.rovo.caching.drum.data.ObjectSerializer;
import at.rovo.caching.drum.util.DrumUtil;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * <p>
 * A naive Bayes training data implementation which uses DRUM to store the
 * training data inside a backing cache.
 * </p>
 */
public class NBDrumTrainingData<F, C> extends NBTrainingData<F, C> implements IDispatcher<DrumTrainingData<F, C>, ObjectSerializer<F>>
{

	private final HashMap<C, Integer> availableCategories = new HashMap<>();
	private final Class<F> featureClass;
	private final Class<C> categoryClass;
	private final IDrum<DrumTrainingData<F, C>, ObjectSerializer<F>> drum;
	private final Map<C, Integer> numSamplesPerCategory = new HashMap<>();
	private final List<DrumTrainingData<F, C>> checkedTrainingData = new ArrayList<>();

	public NBDrumTrainingData(List<C> categories, Class<C> categoryClass, Class<F> featureClass)
	{
		this.categoryClass = categoryClass;
		this.featureClass = featureClass;

		for (C cat : categories)
		{
			availableCategories.put(cat, 0);
		}

		try
		{
			drum = (Drum<DrumTrainingData<F,C>, ObjectSerializer<F>>)new Drum.Builder("classifier", DrumTrainingData.class, ObjectSerializer.class)
					.numBucket(2)
					.bufferSize(1024)
					.dispatcher(this)
					.build();
		}
		catch (Exception ex)
		{
			throw new RuntimeException("Could not initialize backing DRUM cache", ex);
		}
	}

	@Override
	public void incrementFeature(F feature, C category)
	{
		Map<C, Integer> categories = new HashMap<>(availableCategories);
		categories.put(category, categories.get(category)+1);
		DrumTrainingData<F, C> trainingData = new DrumTrainingData<>(feature, featureClass, categories, categoryClass);
		drum.appendUpdate(DrumUtil.hash(feature), trainingData, new ObjectSerializer<>(feature, featureClass));
	}

	@Override
	public void incrementNumberOfSamplesForCategory(C category)
	{
		Integer count = this.numSamplesPerCategory.get(category);
		if (null == count)
		{
			this.numSamplesPerCategory.put(category, 1);
		}
		else
		{
			this.numSamplesPerCategory.put(category, count+1);
		}
	}

	@Override
	protected long getFeatureCount(F feature)
	{
		drum.check(DrumUtil.hash(feature));


		DrumTrainingData<F, C> data;

		long count = 0;
		for (C category : availableCategories)
		{
			count += data.getCategoryCount(category);
		}
		return count;
	}

	@Override
	protected int getNumberOfCategories()
	{
		return 0;
	}

	@Override
	protected long getTotalNumberOfFeatures()
	{
		return 0;
	}

	@Override
	public long getNumberOfSamplesForCategory(C category)
	{
		return 0;
	}

	@Override
	public long getTotalNumberOfSamples()
	{
		return 0;
	}

	@Override
	public int getFeatureCount(F feature, C category)
	{
		return 0;
	}

	@Override
	protected boolean containsCategory(C category)
	{
		return false;
	}

	@Override
	protected Collection<C> getCategories()
	{
		return null;
	}

	@Override
	public void saveData(File directory, String name)
	{

	}

	@Override
	public boolean loadData(File serializedObject)
	{
		return false;
	}



	@Override
	public void uniqueKeyCheck(Long key, ObjectSerializer<F> aux)
	{
		
	}

	@Override
	public void duplicateKeyCheck(Long key, DrumTrainingData<F, C> value, ObjectSerializer<F> aux)
	{

	}

	@Override
	public void uniqueKeyUpdate(Long key, DrumTrainingData<F, C> value, ObjectSerializer<F> aux)
	{
		// do nothing
	}

	@Override
	public void duplicateKeyUpdate(Long key, DrumTrainingData<F, C> value, ObjectSerializer<F> aux)
	{
		// do nothing
	}

	@Override
	public void update(Long key, DrumTrainingData<F, C> value, ObjectSerializer<F> aux)
	{
		// do nothing
	}
}
