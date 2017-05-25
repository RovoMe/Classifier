package at.rovo.test.dt;

import at.rovo.classifier.dataset.DataSetType;
import at.rovo.classifier.decissionTrees.c45.C45;
import at.rovo.classifier.decissionTrees.c45.DecisionTree;
import at.rovo.classifier.decissionTrees.c45.util.PlainTreeView;
import at.rovo.classifier.decissionTrees.c45.util.TreeView;
import java.net.URISyntaxException;
import java.net.URL;
import org.junit.Test;

public class C45Test
{
    @Test
    public void testC45() throws URISyntaxException
    {
        URL url = this.getClass().getResource("/iris.arff");
        String fileName = url.toURI().getPath();
        C45 c45 = new C45(fileName, DataSetType.ARFF);

        // C45 c45 = new C45("./src/test/resources/adult", DataSetType.UCI);
        DecisionTree tree = c45.getDecisionTree();
        // tree.prune();
        TreeView view = new PlainTreeView(tree);

        System.out.println(view);
    }
}
