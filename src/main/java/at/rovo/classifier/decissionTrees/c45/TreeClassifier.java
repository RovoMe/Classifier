/**
 * @(#)TreeClassifier.java 1.5.2 09/03/29
 */

package at.rovo.classifier.decissionTrees.c45;

import at.rovo.classifier.decissionTrees.c45.tree.Tree;

/**
 * A tree classifier playing the roles of both a tree and a classifier.
 *
 * @author Ping He
 * @author Xiaohua Xu
 */
public interface TreeClassifier extends Tree, Classifier
{
    /**
     * Build a tree classifier. This method is usually called in a tree classifier's constructor.
     */
    void build();

    /**
     * Prune the tree classifier to improve its generalization ability. Different pruning strategies can lead to
     * different pruned trees.
     */
    void prune();

    /**
     * Get the classification error on the train data.
     */
    int getTrainError();
}