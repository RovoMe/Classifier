package at.rovo.classifier.svm.solver.instance;

import at.rovo.classifier.svm.struct.Parameter;
import at.rovo.classifier.svm.struct.Problem;
import at.rovo.classifier.svm.struct.SolutionInfo;

public abstract class SolveInstance
{
    public abstract void solve(Problem prob, Parameter param, double[] alpha, SolutionInfo si);
}
