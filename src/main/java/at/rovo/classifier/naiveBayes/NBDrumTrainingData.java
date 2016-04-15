package at.rovo.classifier.naiveBayes;

import at.rovo.caching.drum.Drum;
import at.rovo.caching.drum.DrumBuilder;
import at.rovo.caching.drum.DrumException;
import at.rovo.caching.drum.data.ObjectSerializer;
import at.rovo.caching.drum.util.DataStoreLookup;
import at.rovo.caching.drum.util.DrumUtils;
import at.rovo.common.Pair;
import java.io.File;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * A naive Bayes training data implementation which uses DRUM to store the training data inside a backing cache.
 */
public class NBDrumTrainingData<F, C> extends NBTrainingData<F, C>
{

    private final static Logger LOG = LogManager.getLogger(MethodHandles.lookup().lookupClass());

    private final Class<F> featureClass;
    private final Class<C> categoryClass;
    private final Drum<DrumTrainingData<F, C>, ObjectSerializer<F>> drum;

    private final HashMap<C, Integer> availableCategories = new HashMap<>();
    private final Map<C, Integer> numSamplesPerCategory = new HashMap<>();
    private int numFeatures = 0;

    private final DataStoreLookup<DrumTrainingData<F, C>> dataStoreLookup;

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
            DrumBuilder<DrumTrainingData<F, C>, ObjectSerializer<F>> builder =
                    new DrumBuilder<>("classifier", DrumTrainingData.class, ObjectSerializer.class);
            drum = builder.numBucket(2).bufferSize(1024).build();
            dataStoreLookup = new DataStoreLookup<>("classifier");
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
        categories.put(category, categories.get(category) + 1);
        DrumTrainingData<F, C> trainingData = new DrumTrainingData<>(feature, featureClass, categories, categoryClass);
        drum.appendUpdate(DrumUtils.hash(feature), trainingData, new ObjectSerializer<>(feature, featureClass));
        numFeatures++;
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
            this.numSamplesPerCategory.put(category, count + 1);
        }
    }

    @Override
    protected long getFeatureCount(F feature)
    {
        drum.check(DrumUtils.hash(feature));

        try
        {
            long hash = DrumUtils.hash(feature);
            Pair<Long, DrumTrainingData<F, C>> result = dataStoreLookup.findEntry(hash, DrumTrainingData.class);
            DrumTrainingData<F, C> data = result.getLast();

            long count = 0;
            for (C category : availableCategories.keySet())
            {
                count += data.getCategoryCount(category);
            }
            return count;
        }
        catch (IOException | DrumException ex)
        {
           LOG.error("Could not lookup feature " + feature + " due to: " + ex.getLocalizedMessage(), ex);
        }
        return 0;
    }

    @Override
    protected int getNumberOfCategories()
    {
        return availableCategories.size();
    }

    @Override
    protected long getTotalNumberOfFeatures()
    {
        return this.numFeatures;
    }

    @Override
    public long getNumberOfSamplesForCategory(C category)
    {
        return this.numSamplesPerCategory.get(category);
    }

    @Override
    public long getTotalNumberOfSamples()
    {
        long total = 0L;
        for (C cat : this.numSamplesPerCategory.keySet()) {
            total += this.numSamplesPerCategory.get(cat);
        }
        return total;
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
}
