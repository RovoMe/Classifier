package at.rovo.test.nb;

import org.junit.Assert;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Before;
import org.junit.Test;
import at.rovo.classifier.naiveBayes.NormalNaiveBayes;
import at.rovo.classifier.naiveBayes.TrainingDataStorageMethod;

import java.lang.invoke.MethodHandles;

public class NormalNaiveBayesTest extends NormalNaiveBayes<String, String>
{
	private static Logger LOG = LogManager.getLogger(MethodHandles.lookup().lookupClass());
		
	public NormalNaiveBayesTest()
	{
		super(TrainingDataStorageMethod.MAP);
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
	public void testFeatureProbability()
	{	
		// 'quick' occurred in 3 out of 5 examples
		double p = this.getFeatureProbability("quick");
		if (LOG.isDebugEnabled())
			LOG.debug("P('quick'): "+p);
		Assert.assertEquals("P('quick') ", 3./5, p, 0.);
		
		// 'money' occurred in 1 out of 5 examples
		p = this.getFeatureProbability("money");
		if (LOG.isDebugEnabled())
			LOG.debug("P('money'): "+ p);
		Assert.assertEquals("P('money') ", 1./5, p, 0.);
		
		// 'jumps' occurred in 2 out of 5 examples
		p = this.getFeatureProbability("jumps");
		if (LOG.isDebugEnabled())
			LOG.debug("P('jumps'): "+ p);
		Assert.assertEquals("P('jumps') ", 2./5, p, 0.000001);
		
		// 'notInThere' occurred in 0 out of 5 examples
		p = this.getFeatureProbability("notInThere");
		if (LOG.isDebugEnabled())
			LOG.debug("P('notInThere'): "+ p);
		Assert.assertEquals("P('notInThere') ", 0./5, p, 0.);
		
		p = this.getFeatureProbability("quick money".split("\\W"));
		if (LOG.isDebugEnabled())
			LOG.debug("P('quick','money'): "+p);
		// P('quick','money') = P('quick'|'good')*P('money'|'good')*P('good') + 
		//                      P('quick'|'bad')*P('money'|'bad')*P('bad')
		// P('quick'|'good') = 2/3 - 2 out of 3 samples labeled as 'good' contain quick
		// P('quick'|'bad') =  1/2 - 1 out of 2 samples labeled as 'bad' contain quick
		// P('money'|'good') = 0/3 - 'money' does not occur in 'good' samples
		// P('money'|'bad') = 1/2  - 'money' is contained in 1 'bad' examples 
		// P('good') = 3/5 - 3 out of 5 samples are labeled as 'good'
		// P('bad') = 2/5 - 2 out of 5 samples are labeled as 'bad'
		//
		// [(2/3)*(0/3)*(3/5)+(1/2)*(1/2)*(2/5)] = 0.1
		Assert.assertEquals("P('quick','money') ", 0.1, p, 0.);
		
		p = this.getFeatureProbability("quick notInThere".split("\\W"));
		if (LOG.isDebugEnabled())
			LOG.debug("P('quick','notInThere'): "+p);
		
		// P('quick','notInThere') = P('quick'|'good')*P('notInThere'|'good')*P('good') + 
		//                           P('quick'|'bad')*P('notInThere'|'bad')*P('bad')
		// ...
		// P('notInThere'|'good') = 0/3
		// P('notInThere'|'bad') = 0/2
		//
		// [(2/3)*(0/3)*(3/5)+(1/2)*(0/2)*(2/5)] = 0
		Assert.assertEquals("P('quick','notInThere') ", 0., p, 0.);
		
		p = this.getFeatureProbability("quick rabbit".split("\\W"));
		if (LOG.isDebugEnabled())
			LOG.debug("P('quick','rabbit'): "+p);
		// P('quick', 'rabbit') = P('quick'|'good')*P('rabbit'|'good')*P('good') +
		//                        P('quick'|'bad')*P('rabbit'|'bad')*P('bad')
		// P('quick'|'good') = 2/3 - 2 out of 3 samples labeled as 'good' contain 'quick'
		// P('quick'|'bad') =  1/2 - 1 out of 2 samples labeled as 'bad' contain 'quick'
		// P('rabbit'|'good') = 1/3 - 'rabbit' is contained in 1 good samples
		// P('rabbit'|'bad') = 0/2  - 'rabbit' does not occur in bad samples 
		//
		// P('good') = 3/5 - 3 out of 5 samples are labeled as 'good'
		// P('bad') = 2/5 - 2 out of 5 samples are labeled as 'bad'
		//
		// [(2/3)*(1/3)*(3/5)+(1/2)*(0/2)*(2/5)] = 0.133333333333
		Assert.assertEquals("P('quick','rabbit') ", 0.1333333333333333, p, 0.);
	}
	
	@Test
	public void testConditionalProbabilities()
	{	
		double featProb = this.getConditionalProbability("quick", "good");
		if (LOG.isDebugEnabled())
			LOG.debug("P('quick'|'good'): " + featProb);
		// 'quick' is included in 2 of the 3 training sets which have been labeled as good
		Assert.assertEquals("P('quick'|'good') ", 2.0/3.0, featProb, 0.);
		
		featProb = this.getConditionalProbability("quick", "bad");
		if (LOG.isDebugEnabled())
			LOG.debug("P('quick'|'bad'): " + featProb);
		// 'quick' is included in 1 of the 2 training sets which have been labeled as bad
		Assert.assertEquals("P('quick'|'bad') ", 1.0/2.0, featProb, 0.);
		
		featProb = this.getConditionalProbability("jumps", "good");
		if (LOG.isDebugEnabled())
			LOG.debug("P('jumps'|'good'): " + featProb);
		// 'jumps' is included in 2 of the 3 training sets which have been labeled as good
		Assert.assertEquals("P('jumps'|'good') ", 2.0/3.0, featProb, 0.);
		
		featProb = this.getConditionalProbability("jumps", "bad");
		if (LOG.isDebugEnabled())
			LOG.debug("P('jumps'|'bad'): " + featProb);
		// 'jumps' is included in 0 of the 2 training sets which have been labeled as bad
		Assert.assertEquals("P('jumps'|'bad') ", 0.0/2.0, featProb, 0.);
		
		featProb = this.getConditionalProbability("notInThere", "good");
		if (LOG.isDebugEnabled())
			LOG.debug("P('notInThere'|'good'): " + featProb);
		// 'notInThere' is included in 0 of the 3 training sets which have been labeled as good
		Assert.assertEquals("P('notInThere'|'good') ", 0.0/3.0, featProb, 0.);
		
		featProb = this.getConditionalProbability("notInThere", "bad");
		if (LOG.isDebugEnabled())
			LOG.debug("P('notInThere'|'bad'): " + featProb);
		// 'notInThere' is included in 0 of the 2 training sets which have been labeled as bad
		Assert.assertEquals("P('notInThere'|'bad') ", 0.0/2.0, featProb, 0.);
		
		featProb = this.getConditionalProbability("quick", "unsure");
		if (LOG.isDebugEnabled())
			LOG.debug("P('quick'|'unsure'): " + featProb);
		// 'quick' is included in 0 of the 0 training sets which have been labeled as unsure
		Assert.assertEquals("P('quick'|'unsure') ", 0.0, featProb, 0.);
		
		featProb = this.getConditionalProbability("quick money".split("\\W"), "good");
		if (LOG.isDebugEnabled())
			LOG.debug("P('quick','money'|'bad'): "+featProb);
		Assert.assertEquals("P('quick','money'|'bad') ", (2./3.)*(0./3.), featProb, 0.);
		
		featProb = this.getConditionalProbability("quick money".split("\\W"), "bad");
		if (LOG.isDebugEnabled())
			LOG.debug("P('quick','money'|'bad'): "+featProb);
		Assert.assertEquals("P('quick','money'|'bad') ", (1./2.)*(1./2.), featProb, 0.);
		
		featProb = this.getConditionalProbability("quick notInThere".split("\\W"), "bad");
		if (LOG.isDebugEnabled())
			LOG.debug("P('quick','notInThere'|'bad'): "+featProb);
		Assert.assertEquals("P('quick','notInThere'|'bad') ", (1./2.)*(0./2.), featProb, 0.);
	}
		
	@Test
	public void testCategoryProbability()
	{
		// P(C)
		double catProb = this.getCategoryProbability("good");
		if (LOG.isDebugEnabled())
			LOG.debug("Probability of category 'good': "+catProb);
		// category count / total count
		Assert.assertEquals("Probability of category 'good' ", 3./5., catProb, 0.);
		
		catProb = this.getCategoryProbability("bad");
		if (LOG.isDebugEnabled())
			LOG.debug("Probability of category 'good': "+catProb);
		// category count / total count
		Assert.assertEquals("Probability of category 'good' ", 2./5., catProb, 0.);
		
		catProb = this.getCategoryProbability("notInThere");
		if (LOG.isDebugEnabled())
			LOG.debug("Probability of category 'good': "+catProb);
		// category count / total count
		Assert.assertEquals("Probability of category 'good' ", 0, catProb, 0.);
	}

	
	@Test
	public void testProbability()
	{	
		// quick has 2 occurrences in good examples and total 3 occurrences
		double p = this.getProbability("good", "quick");
		if (LOG.isDebugEnabled())
			LOG.debug("nb-P('good'|'quick'): "+p);
		Assert.assertEquals("nb-P('good'|'quick') ", 2./3, p, 0.);
		
		// quick has 1 occurrence in bad examples and total 3 occurrences
		p = this.getProbability("bad", "quick");
		if (LOG.isDebugEnabled())
			LOG.debug("nb-P('bad'|'quick'): "+p);
		Assert.assertEquals("nb-P('bad'|'quick') ", 1./3, p, 0.00005); // rounding-difference - result is: 0.33333333333333337
		
		// P(C|F) = P(F|C)*P(C)/P(F)
		//        = P(F|C)*P(C)/(P(F|C1)*P(C1)+...+P(F|Cn)*P(Cn))
		//        = P('notInThere'|'good')*P('good')/[P('notInThere'|'good')*P('good')+P('notInThere'|'bad')*P('bad')]
		// P('notInThere'|'good') = 0/3
		// P('good') = 3/5
		// P('notInThere'|'bad') = 0/2
		// P('bad') = 2/5
		// (0/3*3/5) / (0/3*3/5 + 0/2*2/5) = 0
		p = this.getProbability("good", "notInThere");
		if (LOG.isDebugEnabled())
			LOG.debug("nb-P('good'|'notInThere'): "+p);
		Assert.assertEquals("nb-P('good'|'notInThere') ", 0., p, 0.);
		
		p = this.getProbability("notExisting", "quick");
		if (LOG.isDebugEnabled())
			LOG.debug("nb-P('notExisting'|'quick'): "+ p);
		Assert.assertEquals("nb-P('notExisting'|'quick') ", 0., p, 0.);
		
		p = this.getProbability("notExisting", "notInThere");
		if (LOG.isDebugEnabled())
			LOG.debug("nb-P('notExisting'|'notInThere'): "+ p);
		Assert.assertEquals("nb-P('notExisting'|'notInThere') ", 0., p, 0.);
			
		String[] notInThere = new String[] { "notInThere", "notInThereToo" };
		p = this.getProbability("notExisting", notInThere);
		if (LOG.isDebugEnabled())
			LOG.debug("nb-P('notExisting'|'notInThere','notInThereToo'): "+p);
		Assert.assertEquals("nb-P('notExisting'|'notInThere','notInThereToo') ", 0., p, 0.);
		
		String[] words = new String[] { "quick", "rabbit" };
		p = this.getProbability("good", words);
		if (LOG.isDebugEnabled())
			LOG.debug("nb-P('good'|'quick','rabbit'): "+p);

//		11:16:50 DEBUG NaiveBayes.getCategoryProbability() -    P('good') = 3/5 = 0.6
//		11:16:50 DEBUG NaiveBayes.getConditionalProbability() -    P('quick'|'good') = 2/3 = 0.6666666666666666
//		11:16:50 DEBUG NaiveBayes.getConditionalProbability() -    P('rabbit'|'good') = 1/3 = 0.3333333333333333
//		11:16:50 DEBUG NaiveBayes.getConditionalProbability() -    P('quick','rabbit'|'good') = 0.2222222222222222
//		11:16:50 DEBUG NaiveBayes.getCategoryProbability() -    P('good') = 3/5 = 0.6
//		11:16:50 DEBUG NaiveBayes.getConditionalProbability() -    P('quick'|'good') = 2/3 = 0.6666666666666666
//		11:16:50 DEBUG NaiveBayes.getConditionalProbability() -    P('rabbit'|'good') = 1/3 = 0.3333333333333333
//		11:16:50 DEBUG NaiveBayes.getCategoryProbability() -    P('bad') = 2/5 = 0.4
//		11:16:50 DEBUG NaiveBayes.getConditionalProbability() -    P('quick'|'bad') = 1/2 = 0.5
//		11:16:50 DEBUG NaiveBayes.getConditionalProbability() -    P('rabbit'|'bad') = 0/2 = 0.0
//		11:16:50 DEBUG TestNaiveBayes.testProbability() - P('good'|'quick','rabbit') 1.0000000000000002
//		
		// Weka: 0.884; own result 1.0000000000000002
		double prob1 = this.getProbability("good", "quick rabbit".split("\\W"));
		if (LOG.isDebugEnabled())
			LOG.debug("P('good'|'quick','rabbit') "+prob1);
		Assert.assertEquals("P('good'|'quick','rabbit') ", p, prob1, 0.);
//		
//		11:16:50 DEBUG NaiveBayes.getCategoryProbability() -    P('bad') = 2/5 = 0.4
//		11:16:50 DEBUG NaiveBayes.getConditionalProbability() -    P('quick'|'bad') = 1/2 = 0.5
//		11:16:50 DEBUG NaiveBayes.getConditionalProbability() -    P('rabbit'|'bad') = 0/2 = 0.0
//		11:16:50 DEBUG NaiveBayes.getConditionalProbability() -    P('quick','rabbit'|'bad') = 0.0
//		11:16:50 DEBUG NaiveBayes.getCategoryProbability() -    P('good') = 3/5 = 0.6
//		11:16:50 DEBUG NaiveBayes.getConditionalProbability() -    P('quick'|'good') = 2/3 = 0.6666666666666666
//		11:16:50 DEBUG NaiveBayes.getConditionalProbability() -    P('rabbit'|'good') = 1/3 = 0.3333333333333333
//		11:16:50 DEBUG NaiveBayes.getCategoryProbability() -    P('bad') = 2/5 = 0.4
//		11:16:50 DEBUG NaiveBayes.getConditionalProbability() -    P('quick'|'bad') = 1/2 = 0.5
//		11:16:50 DEBUG NaiveBayes.getConditionalProbability() -    P('rabbit'|'bad') = 0/2 = 0.0
//		11:16:50 DEBUG TestNaiveBayes.testProbability() - P('bad'|'quick','rabbit'): 0.0
//		
		// Weka: 0.116, own result 0.0
		double prob2 = this.getProbability("bad", "quick rabbit".split("\\W"));
		if (LOG.isDebugEnabled())
			LOG.debug("P('bad'|'quick','rabbit'): "+prob2);
		
		Assert.assertEquals("Testing the sum of probability and complementary probability",1., prob1+prob2, 0.00000000003);
	}
	
	
	@Test
	public void testClassification()
	{
		String category = this.classify("quick rabbit".split("\\W"));
		if (LOG.isDebugEnabled())
			LOG.debug("classify 'quick rabbit' as: "+category);
		Assert.assertEquals("classify 'quick rabbit' as", "good", category);
		
		category = this.classify("quick money".split("\\W"));
		if (LOG.isDebugEnabled())
			LOG.debug("classify 'quick money' as: "+category);
		Assert.assertEquals("classify 'quick money' as", "bad", category);
	}
}
