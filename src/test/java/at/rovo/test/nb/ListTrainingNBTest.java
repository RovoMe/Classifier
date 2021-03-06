package at.rovo.test.nb;

import at.rovo.classifier.naiveBayes.NormalNaiveBayes;
import at.rovo.classifier.naiveBayes.TrainingDataStorageMethod;
import java.lang.invoke.MethodHandles;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ListTrainingNBTest extends NormalNaiveBayes<String, String>
{
    private static Logger LOG = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

    public ListTrainingNBTest()
    {
        super(TrainingDataStorageMethod.LIST);
    }

    @Before
    public void sampleTrain()
    {
        String[] items = "Nobody owns the water".split("\\W");
        this.train(items, "good");
        items = "the quick rabbit jumps fences".split("\\W");
        this.train(items, "good");
        items = "buy pharmaceuticals now".split("\\W");
        this.train(items, "bad");
        items = "make quick money at the online casino".split("\\W");
        this.train(items, "bad");
        items = "the quick brown fox jumps".split("\\W");
        this.train(items, "good");
    }

    @Test
    public void testCategoryCount()
    {
        long catCount = this.trainingData.getNumberOfSamplesForCategory("good");
        if (LOG.isDebugEnabled())
        {
            LOG.debug("Category 'good' contained in examples: " + catCount);
        }
        // 3 sentences are labeled as good
        Assert.assertEquals("Category 'good' contained in examples ", 3L, catCount);

        catCount = this.trainingData.getNumberOfSamplesForCategory("bad");
        if (LOG.isDebugEnabled())
        {
            LOG.debug("Category 'bad' contained in examples: " + catCount);
        }
        // 2 sentences are labeled as bad
        Assert.assertEquals("Category 'bad' contained in examples ", 2L, catCount);

        catCount = this.trainingData.getNumberOfSamplesForCategory("notInThere");
        if (LOG.isDebugEnabled())
        {
            LOG.debug("Category 'notInThere' contained in examples: " + catCount);
        }
        // 0 sentences are labeled as notInThere
        Assert.assertEquals("Category 'notInThere' contained in examples ", 0L, catCount);
    }

    @Test
    public void testFeatureCount()
    {
        long featCount = this.trainingData.getFeatureCount("quick", "good");
        if (LOG.isDebugEnabled())
        {
            LOG.debug("Feature 'quick' contained in 'good' examples: " + featCount);
        }
        // 2 sentences labeled as good contain quick
        Assert.assertEquals("Feature 'quick' contained in 'good' examples ", 2L, featCount);

        featCount = this.trainingData.getFeatureCount("quick", "bad");
        if (LOG.isDebugEnabled())
        {
            LOG.debug("Feature 'quick' contained in 'bad' examples: " + featCount);
        }
        // only 1 sentence labeled as bad contains quick
        Assert.assertEquals("Feature 'quick' contained in 'bad' examples ", 1L, featCount);

        featCount = this.trainingData.getFeatureCount("notInThere", "good");
        if (LOG.isDebugEnabled())
        {
            LOG.debug("Feature 'notInThere' contained in 'good' examples: " + featCount);
        }
        // 0 sentences labeled as good contain notInThere
        Assert.assertEquals("Feature 'notInThere' contained in 'good' examples ", 0L, featCount);

        featCount = this.trainingData.getFeatureCount("notInThere", "bad");
        if (LOG.isDebugEnabled())
        {
            LOG.debug("Feature 'notInThere' contained in 'bad' examples: " + featCount);
        }
        // 0 sentences labeled as good contain notInThere
        Assert.assertEquals("Feature 'notInThere' contained in 'bad' examples ", 0L, featCount);

        featCount = this.trainingData.getFeatureCount("notInThere", "noCategory");
        if (LOG.isDebugEnabled())
        {
            LOG.debug("Feature 'notInThere' contained in 'noCategory' examples: " + featCount);
        }
        // 0 sentences labeled as good contain notInThere
        Assert.assertEquals("Feature 'notInThere' contained in 'noCategory' examples ", 0L, featCount);
    }

    @Test
    public void testAbsoluteCount()
    {
        long totalCount = this.trainingData.getTotalNumberOfSamples();
        if (LOG.isDebugEnabled())
        {
            LOG.debug("Total count: " + totalCount);
        }
        // there are exactly 5 test entries
        Assert.assertEquals("Total count ", 5L, totalCount);
    }
}
