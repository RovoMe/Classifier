package at.rovo.test.nb;

import org.junit.Assert;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Before;
import org.junit.Test;
import at.rovo.classifier.naiveBayes.TrainingDataStorageMethod;
import at.rovo.classifier.naiveBayes.WeightedNaiveBayes;

public class WeightedNaiveBayesTest extends WeightedNaiveBayes<String,String>
{
	private static Logger logger = LogManager.getLogger(WeightedNaiveBayesTest.class.getName());
		
	public WeightedNaiveBayesTest()
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
	public void testConditionalProbability()
	{
		// P(X|C)
		double wP = this.getConditionalProbability("quick", "good");
		if (logger.isDebugEnabled())
			logger.debug("wP('quick'|'good'): "+wP);
		// (weight*assumeProb + count*featProb) / (count+weight)
		// (1*0.5+count*featProb) / (count+1)
		// 'quick' is in 3 of those 5 example sentences --> count = 3
		// 'quick' is included in 2 of 3 sentence labeled as 'good' --> 2/3
		//
		// (1*0.5+3*2/3) / (3+1) = 0.625
		Assert.assertEquals("wP('quick'|'good') ", 0.625, wP, 0.);
		
		wP = this.getConditionalProbability("quick", "bad");
		if (logger.isDebugEnabled())
			logger.debug("wP('quick'|'bad'): "+wP);
		// (weight*assumeProb + count*featProb) / (count+weight)
		// (1*0.5+count*featProb) / (count+1)
		// 'quick' is in 3 of those 5 example sentences --> count = 3
		// 'quick' is included in 1 of 3 sentence labeled as 'bad' --> 1/2
		//
		// (1*0.5+3*1/2) / (3+1) = 0.5
		Assert.assertEquals("wP('quick'|'bad') ", 0.5, wP, 0.);
		
		wP = this.getConditionalProbability("jumps", "good");
		if (logger.isDebugEnabled())
			logger.debug("wP('jumps'|'good'): "+wP);
		// (weight*assumeProb + count*featProb) / (count+weight)
		// (1*0.5+count*featProb) / (count+1)
		// 'jumps' is in 2 of those 5 example sentences --> count = 2
		// 'jumps' is included in 2 of 3 sentence labeled as 'good' --> 2/3
		//
		// (1*0.5+2*2/3) / (2+1) = 0.6111111111
		Assert.assertEquals("wP('jumps'|'good') ", 0.611111111, wP, 0.0000001);
		
		wP = this.getConditionalProbability("jumps", "bad");
		if (logger.isDebugEnabled())
			logger.debug("wP('jumps'|'bad'): "+wP);
		// (weight*assumeProb + count*featProb) / (count+weight)
		// (1*0.5+count*featProb) / (count+1)
		// 'quick' is in 3 of those 5 example sentences --> count = 3
		// 'quick' is included in 0 of 3 sentence labeled as 'bad' --> 0
		//
		// (1*0.5+2*0) / (2+1) = 1/6 = 0.166666667
		Assert.assertEquals("wP('jumps'|'bad') ", 1./6, wP, 0.);
		
		wP = this.getConditionalProbability("notInThere", "good");
		if (logger.isDebugEnabled())
			logger.debug("wP('notInThere'|'good'): "+wP);
		// (weight*assumeProb + count*featProb) / (count+weight)
		// (1*0.5+count*featProb) / (count+1)
		// 'notInThere' is in 0 of those 5 example sentences --> count = 0
		// 'notInThere' is included in 0 of 3 sentence labeled as 'good' --> 0/3
		//
		// (1*0.5+0*0/3) / (0+1) = 0.5
		Assert.assertEquals("wP('notInThere'|'good') ", 0.5, wP, 0.);
		
		wP = this.getConditionalProbability("quick", "notExisting");
		if (logger.isDebugEnabled())
			logger.debug("wP('quick'|'notExisting'): "+wP);
		// (weight*assumeProb + count*featProb) / (count+weight)
		// (1*0.5+count*featProb) / (count+1)
		// 'quick' is in 3 of those 5 example sentences --> count = 3
		// 'quick' is included in 0 of 0 sentence labeled as 'notExisting' --> 0/0
		//
		// (1*0.5+3*0) / (3+1) = 0.125
		Assert.assertEquals("wP('quick'|'notExisting') ", 0.125, wP, 0.);
		
		wP = this.getConditionalProbability("money", "good");
		if (logger.isDebugEnabled())
			logger.debug("wP('money'|'good'): " + wP);
		// (weight*assumeProb + count*featProb) / (count+weight)
		// (1*0.5+count*featProb) / (count+1)
		// money is only in 1 of those 5 example sentences --> count = 1
		// money is not included in any sentence labeled as good --> 0/3 = 0
		//
		// (1*0.5+1*0)/(1+1) = 0.25
		Assert.assertEquals("wP('money'|'good') ", 0.25, wP, 0.);
		
		this.sampleTrain();
		wP = this.getConditionalProbability("money", "good");
		if (logger.isDebugEnabled())
			logger.debug("Weighted probability for 'money' in 'good' examples after further sample training: " +wP);
		// After train of the same samples again - we have now
		// * 10 sentences - 6 labeled as good, 4 as bad
		// * 2 sentences contain money - 0 labeled as good, 2 as bad
		//
		// (weight*assumeProb + count*featProb) / (count+weight)
		// (1*0.5+count*featProb) / (count+1)
		// money is only in 2 of those 10 example sentences --> count = 2
		// money is not included in any sentence labeled as good --> 0/6 = 0
		//
		// (1*0.5+2*0)/(2+1) = 1/6 = 0.16666..
		Assert.assertEquals("Weighted probability for 'money' in 'good' examples after further sample training ", 1./6, wP, 0.);
	}	
	
	@Test
	public void testConditionalProbability2()
	{
		// P(X,Y|C)
		double wP = this.getConditionalProbability("quick money".split("\\W"), "good");
		if (logger.isDebugEnabled())
			logger.debug("wP('quick','money'|'good'): "+wP);
		// (weight*assumeProb + count*featProb) / (count+weight)
		// (1*0.5+count*featProb) / (count+1)
		// 'quick' is in 3 of those 5 example sentences --> count = 3
		// 'quick' is included in 2 of 3 sentence labeled as 'good' --> 2/3
		// 'money' is in 1 of those 5 example sentences --> count = 1
		// 'money' is included in 0 of 3 sentences labeled as 'good' --> 0/3
		// p = wP('quick', 'good') * wP('money', 'good')
		//
		// (1*0.5+3*(2/3))/(3+1) * (1*0.5+1*(0/3))/(1+1) = 0.625 * 0.25 = 0.15625
		Assert.assertEquals("wP('quick','money'|'good') ", 0.15625, wP, 0.);
		
		wP = this.getConditionalProbability("quick money".split("\\W"), "bad");
		if (logger.isDebugEnabled())
			logger.debug("wP('quick','money'|'bad'): "+wP);
		// (weight*assumeProb + count*featProb) / (count+weight)
		// (1*0.5+count*featProb) / (count+1)
		// 'quick' is in 3 of those 5 example sentences --> count = 3
		// 'quick' is included in 1 of 5 sentence labeled as 'bad' --> 1/2
		// 'money' is in 1 of those 5 example sentences --> count = 1
		// 'money' is included in 1 of 2 sentences labeled as 'bad' --> 1/2
		// p = wP('quick', 'bad') * wP('money', 'bad')
		//
		// (1*0.5+3*(1/2))/(3+1) * (1*0.5+1*(1/2))/(1+1) = 0.5 * 0.5 = 0.25
		Assert.assertEquals("wP('quick','money'|'bad') ", 0.25, wP, 0.);
	}
	
	@Test
	public void testCategoryProbability()
	{
		// P(C)
		double catProb = this.getCategoryProbability("good");
		if (logger.isDebugEnabled())
			logger.debug("Probability of category 'good': "+catProb);
		// category count / total count
		Assert.assertEquals("Probability of category 'good' ", new Double(3./5.), new Double(catProb));
		
		catProb = this.getCategoryProbability("bad");
		if (logger.isDebugEnabled())
			logger.debug("Probability of category 'good': "+catProb);
		// category count / total count
		Assert.assertEquals("Probability of category 'good' ", new Double(2./5.), new Double(catProb));
		
		catProb = this.getCategoryProbability("notInThere");
		if (logger.isDebugEnabled())
			logger.debug("Probability of category 'good': "+catProb);
		// category count / total count
		Assert.assertEquals("Probability of category 'good' ", new Double(0), new Double(catProb));
	}
	
	@Test
	public void testProbability()
	{
		// P(C|F) = P(F|C)*P(C)/P(F)
		//        = wP(F|C)*P(C)/(P(F|C1)*P(C1)+...+P(F|Cn)*P(Cn))
		//        = wP('notInThere'|'good')*P('good')/(wP('notInThere'|'good')*P('good')+wP('notInThere'|'bad')*P('bad'))
		// P('notInThere'|'good') = 0/3 = 0
		// P('notInThere'|'bad') = 0/2 = 0
		// P('good') = 3/5 = 0.6
		// P('bad') = 2/5 = 0.4
		// wP('notInThere'|'good') = ((1*0.5+0*(0/3)) / (0+1)) = 0.5
		// wP('notInThere'|'bad') =  ((1*0.5+0*(0/2)) / (0+1)) = 0.5
		//
		// (0.5*0.6) / (0.5*0.6 + 0.5*0.4) = 0.6 
		double p = this.getProbability("good", "notInThere");
		if (logger.isDebugEnabled())
			logger.debug("nb-wP('good'|'notInThere'): "+p);
		Assert.assertEquals("nb-wP('good'|'notInThere') ", 0.6, p, 0.);
		
		// (0.5*0.4) / (0.5*0.6 + =.5*0.4) = 0.4
		double p2 = this.getProbability("bad", "notInThere");
		if (logger.isDebugEnabled())
			logger.debug("nb-wP('notInThere'|'bad'): "+p2);
		Assert.assertEquals("nb-wP('notInThere'|'bad') ", 0.4, p2, 0.);
		Assert.assertEquals("nb-wP('good'|'notInThere')+nb-wP('notInThere'|'bad') ", 1., p+p2, 0.);
		
		// P(C|F1,F2) = P(F1,F2|C)*P(C)/P(F1,F2)
		//            = wP(F1|C)*wP(F2|C)*P(C) / [wP(F1|C1)*wP(F2|C1)*P(C1) +...+ wP(F1|Cn)*wP(F2|Cn)*P(Cn)]
		//            = wP('notInThere'|'good')*wP('notInThereToo'|'good')*P('good') / 
		//                  [wP('notInThere'|'good')*wP('notInThereToo')*P('good') +
		//                   wP('notInThere'|'bad')*wP('notInThereToo'|'bad')*P('bad')]
		// P('notInThere'|'good') = 0/3
		// P('notInThereToo'|'good') = 0/3
		// P('notInThere'|'bad') = 0/2 = 0.5
		// p('notInThereToo'|'bad') =  0/2 = 0.5
		// P('good') = 3/5 = 0.6
		// P('bad') = 2/5 = 0.4
		// wP('notInThere'|'good') = ((1*0.5+0*(0/3)) / (0+1)) = 0.5
		// wP('notInThere'|'bad') =  ((1*0.5+0*(0/2)) / (0+1)) = 0.5
		// wP('notInThereToo'|'good') = (1*0.5+0*(0/3)) / (0+1)) = 0.5
		// wP('notInThereToo'|'bad') =  (1*0.5+0*(0/2)) / (0+1)) = 0.5
		//
		// (0.5*0.5*0.6) / (0.5*0.5*0.6 + 0.5*0.5*0.4) = 0.6
		String[] notInThere = new String[] { "notInThere", "notInThereToo" };
		double p4 = this.getProbability("good", notInThere);
		if (logger.isDebugEnabled())
			logger.debug("nb-wP('good'|'notInThere','notInThereToo'): "+p4);
		Assert.assertEquals("nb-wP('good'|'notInThere','notInThereToo') ", p, p4, 0.);
		
		// (0.5*0.5*0.4) / (0.5*0.5*0.6 + 0.5*0.5*0.4) = 0.4
		double p5 = this.getProbability("bad", notInThere);
		if (logger.isDebugEnabled())
			logger.debug("nb-wP('bad'|'notInThere','notInThereToo'): "+p5);
		Assert.assertEquals("nb-wP('bad'|'notInThere','notInThereToo') ", p2, p5, 0.);
		Assert.assertEquals("nb-wP('good'|'notInThere','notInThereToo')+nb-wP('bad'|'notInThere','notInThereToo') ", 1.,(p4+p5), 0.);
	}
	
	@Test
	public void testClassification()
	{
		String category = this.classify("quick rabbit".split("\\W"));
		if (logger.isDebugEnabled())
			logger.debug("classify 'quick rabbit' as: "+category);
		Assert.assertEquals("classify 'quick rabbit' as ", "good", category);
		
		category = this.classify("quick money".split("\\W"));
		if (logger.isDebugEnabled())
			logger.debug("classify 'quick money' as: "+category);
		Assert.assertEquals("classify 'quick money' as ", "bad", category);
		
		if (logger.isDebugEnabled())
			logger.debug("Setting Threshold to 3.0");
		this.setThreshold("bad", 3.0f);
		
		category = this.classify("quick money".split("\\W"));
		if (logger.isDebugEnabled())
			logger.debug("classify 'quick money' as: "+category);
		Assert.assertNull("classify 'quick money' as ", category);
		
		if (logger.isDebugEnabled())
			logger.debug("Train sample data 10 times");
		for (int i=0; i<10; i++)
			this.sampleTrain();
		
		category = this.classify("quick money".split("\\W"));
		if (logger.isDebugEnabled())
			logger.debug("classify 'quick money' as: "+category);
		Assert.assertEquals("classify 'quick money' as ", "bad", category);
	}
}
