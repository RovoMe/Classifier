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
     * Build a tree classifier.
     * This method is usually called in a tree classifier's constructor.
     *
     * @see DecisionTree
     */
    void build();

    /**
     * Prune the tree classifier to improve its generalization ability.
     * Different pruning strategies can lead to different pruned trees.
     *
     * @see DecisionTree
     */
    void prune();

    /**
     * Get the classification error on the train data.
     *
     * @see DecisionTree
     */
    int getTrainError();
}