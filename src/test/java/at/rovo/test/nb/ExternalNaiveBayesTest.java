package at.rovo.test.nb;

import org.junit.Assert;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Before;
import org.junit.Test;
import at.rovo.classifier.naiveBayes.NaiveBayes;
import at.rovo.classifier.naiveBayes.ProbabilityCalculation;
import at.rovo.classifier.naiveBayes.TrainingDataStorageMethod;

import java.lang.invoke.MethodHandles;

public class ExternalNaiveBayesTest
{
	private static Logger LOG = LogManager.getLogger(MethodHandles.lookup().lookupClass());

	private NaiveBayes<String,String> nb = null;
	
	@Before
	public void sampleTrain()
	{	
		this.nb = NaiveBayes.create(ProbabilityCalculation.WEIGHTED, TrainingDataStorageMethod.MAP);
		
		String[] items = "Nobody owns the water".split("\\W");
		this.nb.train(items, "good");
		items = "the quick rabbit jumps fences".split("\\W");
		this.nb.train(items, "good");
		items = "buy pharmaceuticals now".split("\\W");
		this.nb.train(items, "bad");
		items = "make quick money at the online casino".split("\\W");
		this.nb.train(items, "bad");
		items = "the quick brown fox jumps".split("\\W");
		this.nb.train(items, "good");
	}
	
	@Test
	public void testClassification()
	{
		String category = this.nb.classify("quick rabbit".split("\\W"));
		if (LOG.isDebugEnabled())
			LOG.debug("classify 'quick rabbit' as: "+category);
		Assert.assertEquals("classify 'quick rabbit' as good failed", "good", category);
		
		category = this.nb.classify("quick money".split("\\W"));
		if (LOG.isDebugEnabled())
			LOG.debug("classify 'quick money' as: "+category);
		Assert.assertEquals("classify 'quick money' as bad failed", "bad", category);
		
		if (LOG.isDebugEnabled())
			LOG.debug("Setting Threshold to 3.0");
		this.nb.setThreshold("bad", 3.0f);
		
		category = this.nb.classify("quick money".split("\\W"));
		if (LOG.isDebugEnabled())
			LOG.debug("classify 'quick money' as: "+category);
		Assert.assertNull("classify 'quick money' as null failed", category);
		
		if (LOG.isDebugEnabled())
			LOG.debug("Train sample data 10 times");
		for (int i=0; i<10; i++)
			this.sampleTrain();
		
		category = this.nb.classify("quick money".split("\\W"));
		if (LOG.isDebugEnabled())
			LOG.debug("classify 'quick money' as: "+category);
		Assert.assertEquals("classify 'quick money' as bad failed", "bad", category);
	}
}
