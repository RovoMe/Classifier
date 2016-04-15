/**
 * @(#)C45.java 1.5.3 09/04/22
 */

package at.rovo.classifier.decissionTrees.c45;

import at.rovo.classifier.dataset.ArffDataSet;
import at.rovo.classifier.dataset.DataSet;
import at.rovo.classifier.dataset.DataSetType;
import at.rovo.classifier.dataset.UciDataSet;
import at.rovo.classifier.decissionTrees.c45.util.HtmlTreeView;
import at.rovo.classifier.decissionTrees.c45.util.PlainTreeView;
import at.rovo.classifier.decissionTrees.c45.util.Stopwatch;
import at.rovo.classifier.decissionTrees.c45.util.TreeView;
import at.rovo.classifier.decissionTrees.c45.util.XmlTreeView;

/**
 * The main class of Fast C4.5.
 *
 * @author Ping He
 * @author Xiaohua Xu
 */
public class C45
{
    /** The c45 decision tree **/
    private DecisionTree tree;

    /**
     * Build a decision tree with the specified data set files.
     *
     * @param dataSetName
     *         the base name of the .names, .data and .test files
     */
    public C45(String dataSetName, DataSetType type)
    {
        DataSet dataSet = null;
        if (DataSetType.UCI.equals(type))
        {
            dataSet = new UciDataSet(dataSetName);
        }
        else if (DataSetType.ARFF.equals(type))
        {
            dataSet = new ArffDataSet(dataSetName);
        }
        this.tree = new DecisionTree(dataSet);
    }

    /**
     * Get the decision tree.
     */
    public DecisionTree getDecisionTree()
    {
        return this.tree;
    }

    /**
     * Build a decision tree with the specified .names and .data files, prune it and print it in the specified manner.
     */
    public static void main(String[] args)
    {
        C45 c45 = null;
        String dataSetName;
        String output = "plain";
        int repeat = 0;
        TreeView view = null;
        String buildTime = "";

        // Illegal input
        if (args.length == 0)
        {
            usage();
            return;
        }
        // Ask for help
        if (args.length == 1)
        {
            if (args[0].equalsIgnoreCase("-h") || args[0].equalsIgnoreCase("-help"))
            {
                usage();
                return;
            }
        }

        dataSetName = args[0];
        // Interpret the options
        for (int i = 1; i < args.length - 1; i += 2)
        {
            if (args[i].equals("-output") || args[i].equals("-o"))
            {
                output = args[i + 1];
                continue;
            }

            if (args[i].equals("-repeat") || args[i].equals("-r"))
            {
                repeat = Integer.parseInt(args[i + 1]);
            }
        }

        Stopwatch.start();
        for (int i = 0; i < Math.max(1, repeat); i++)
        {
            c45 = new C45(dataSetName, DataSetType.UCI);
        }
        Stopwatch.stop();

        // Compute the averaged tree building time
        if (repeat > 0)
        {
            buildTime = "" + Stopwatch.runtime() / repeat + " ms";
        }

        if (c45 == null)
        {
            System.err.println("Invalid C45 found");
            System.exit(-1);
        }

        DecisionTree tree = c45.getDecisionTree();

        if (!output.equals("plain") && !output.equals("html") && !output.equals("xml"))
        {
            System.out.println("Waring:Unsupported Output!");
            output = "plain";
        }

        if (output.equals("plain"))
        {
            view = plainView(tree, buildTime);
        }

        if (output.equals("html"))
        {
            view = htmlView(tree, buildTime);
        }

        if (output.equals("xml"))
        {
            view = xmlView(tree, buildTime);
        }

        System.out.println(view);
    }

    private static TreeView plainView(DecisionTree tree, String buildTime)
    {
        TreeView v1 = new PlainTreeView(tree);
        v1.insert("Tree Before Pruning");
        v1.append("tree size : " + tree.size());
        v1.append("train error : " + tree.getTrainError());

        if (!buildTime.equals(""))
        {
            v1.append("build time : " + buildTime);
        }
        v1.append(TreeView.CR);

        tree.prune();

        TreeView v2 = new PlainTreeView(tree);
        v2.insert("Tree After Pruning");
        v2.append("tree size : " + tree.size());
        v2.append("train error : " + tree.getTrainError());

        return v1.union(v2);
    }

    private static TreeView htmlView(DecisionTree tree, String buildTime)
    {
        TreeView v1 = new HtmlTreeView(tree);
        v1.insert("<h2>Tree Before Pruning</h2>");
        v1.append("<p>tree size : " + tree.size() + "<br>");
        v1.append("train error : " + tree.getTrainError() + "<br>");

        if (!buildTime.equals(""))
        {
            v1.append("build time : " + buildTime + "</p>");
        }
        v1.append(TreeView.CR);

        tree.prune();

        TreeView v2 = new HtmlTreeView(tree);
        v2.insert("<h2>Tree After Pruning</h2>");
        v2.append("<p>tree size : " + tree.size() + "<br>");
        v2.append("train error : " + tree.getTrainError() + "</p>");

        return v1.union(v2);
    }

    private static TreeView xmlView(DecisionTree tree, String buildTime)
    {
        TreeView v1 = new XmlTreeView(tree);
        String prefix = TreeView.LEVEL_PREFIX + TreeView.LEVEL_GAP;
        String prefix2 = prefix + TreeView.LEVEL_GAP;

        v1.insert(prefix2 + "<comment>Before Pruning</comment>");
        v1.append(prefix2 + "<size>" + tree.size() + "</size>");
        v1.append(prefix2 + "<trainerror>" + tree.getTrainError() + "</trainerror>");
        if (!buildTime.equals(""))
        {
            v1.append(prefix2 + "<buildtime>" + buildTime + "</buildtime>");
        }

        v1.insert(prefix + "<tree>");
        v1.append(prefix + "</tree>");

        v1.insert(prefix + "<version>1.5.3</version>");
        v1.insert(prefix + "<algorithm>FastC45</algorithm>");

        tree.prune();

        TreeView v2 = new XmlTreeView(tree);
        v2.insert(prefix2 + "<description>After Pruning</description>");
        v2.append(prefix2 + "<size>" + tree.size() + "</size>");
        v2.append(prefix2 + "<trainerror>" + tree.getTrainError() + "</trainerror>");

        v2.insert(prefix + "<tree>");
        v2.append(prefix + "</tree>");

        return v1.union(v2);
    }

    private static void usage(String... messages)
    {
        System.out.println("Usage: java " + C45.class + " dataSetName [-output plain | html | xml  -repeat times]");
        for (String line : messages)
        {
            System.out.println(line);
        }
    }
}
