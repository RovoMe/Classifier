package at.rovo.classifier.svm.solver;

import at.rovo.classifier.svm.struct.QMatrix;
import at.rovo.classifier.svm.struct.SolutionInfo;
import at.rovo.common.Pair;


//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
public final class Solver_NU extends Solver
{
    private SolutionInfo si;

    public void solve(int l, QMatrix Q, double[] p, byte[] y, double[] alpha, double Cp, double Cn, double eps,
                      SolutionInfo si, int shrinking)
    {
        this.si = si;
        super.solve(l, Q, p, y, alpha, Cp, Cn, eps, si, shrinking);
    }

    // return 1 if already optimal, return 0 otherwise
    @Override
    int selectWorkingSet(int[] working_set)
    {
        // return i,j such that y_i = y_j and
        // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
        // j: minimizes the decrease of obj value
        // (if quadratic coefficeint <= 0, replace it with tau)
        // -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

        double Gmaxp = -INF;
        double Gmaxp2 = -INF;
        int Gmaxp_idx = -1;

        double Gmaxn = -INF;
        double Gmaxn2 = -INF;
        int Gmaxn_idx = -1;

        int Gmin_idx = -1;
        double obj_diff_min = INF;

        Pair<Double, Integer> gmax = calculateGmax(Gmaxp, Gmaxp_idx);
        Gmaxp = gmax.getFirst();
        Gmaxp_idx = gmax.getLast();

        int ip = Gmaxp_idx;
        int in = Gmaxn_idx;
        float[] Q_ip = null;
        float[] Q_in = null;
        if (ip != -1) // null Q_ip not accessed: Gmaxp=-INF if ip=-1
        {
            Q_ip = Q.get_Q(ip, active_size);
        }
        if (in != -1)
        {
            Q_in = Q.get_Q(in, active_size);
        }

        for (int j = 0; j < active_size; j++)
        {
            if (y[j] == +1)
            {
                if (!isLowerBound(j))
                {
                    double grad_diff = Gmaxp + G[j];
                    if (G[j] >= Gmaxp2)
                    {
                        Gmaxp2 = G[j];
                    }
                    SelectWorkingSetBoundaryCheckResult checkResult =
                            new SelectWorkingSetBoundaryCheckResult(Gmin_idx, obj_diff_min, ip, Q_ip, j, grad_diff);
                    obj_diff_min = checkResult.getObj_diff_min();
                    Gmin_idx = checkResult.getGmin_idx();
                }
            }
            else
            {
                if (!isUpperBound(j))
                {
                    double grad_diff = Gmaxn - G[j];
                    if (-G[j] >= Gmaxn2)
                    {
                        Gmaxn2 = -G[j];
                    }
                    SelectWorkingSetBoundaryCheckResult checkResult =
                            new SelectWorkingSetBoundaryCheckResult(Gmin_idx, obj_diff_min, in, Q_in, j, grad_diff);
                    obj_diff_min = checkResult.getObj_diff_min();
                    Gmin_idx = checkResult.getGmin_idx();
                }
            }
        }

        if (Math.max(Gmaxp + Gmaxp2, Gmaxn + Gmaxn2) < eps)
        {
            return 1;
        }

        if (y[Gmin_idx] == +1)
        {
            working_set[0] = Gmaxp_idx;
        }
        else
        {
            working_set[0] = Gmaxn_idx;
        }
        working_set[1] = Gmin_idx;

        return 0;
    }

    @Override
    void doShrinking()
    {
        double Gmax1 = -INF; // max { -y_i * grad(f)_i | y_i = +1, i in
        // I_up(\alpha) }
        double Gmax2 = -INF; // max { y_i * grad(f)_i | y_i = +1, i in
        // I_low(\alpha) }
        double Gmax3 = -INF; // max { -y_i * grad(f)_i | y_i = -1, i in
        // I_up(\alpha) }
        double Gmax4 = -INF; // max { y_i * grad(f)_i | y_i = -1, i in
        // I_low(\alpha) }

        // find maximal violating pair first
        int i;
        for (i = 0; i < active_size; i++)
        {
            if (!isUpperBound(i))
            {
                if (y[i] == +1)
                {
                    if (-G[i] > Gmax1)
                    {
                        Gmax1 = -G[i];
                    }
                }
                else if (-G[i] > Gmax4)
                {
                    Gmax4 = -G[i];
                }
            }
            if (!isLowerBound(i))
            {
                if (y[i] == +1)
                {
                    if (G[i] > Gmax2)
                    {
                        Gmax2 = G[i];
                    }
                }
                else if (G[i] > Gmax3)
                {
                    Gmax3 = G[i];
                }
            }
        }

        if (!unshrink && Math.max(Gmax1 + Gmax2, Gmax3 + Gmax4) <= eps * 10)
        {
            unshrink = true;
            reconstructGradient();
            active_size = l;
        }

        for (i = 0; i < active_size; i++)
        {
            if (beShrunk(i, Gmax1, Gmax2, Gmax3, Gmax4))
            {
                active_size--;
                while (active_size > i)
                {
                    if (!beShrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
                    {
                        swapIndex(i, active_size);
                        break;
                    }
                    active_size--;
                }
            }
        }
    }

    @Override
    double calculateRho()
    {
        int nr_free1 = 0, nr_free2 = 0;
        double ub1 = INF, ub2 = INF;
        double lb1 = -INF, lb2 = -INF;
        double sum_free1 = 0, sum_free2 = 0;

        for (int i = 0; i < active_size; i++)
        {
            if (y[i] == +1)
            {
                CalculationRhoBoundCheckResult boundCheckResult =
                        new CalculationRhoBoundCheckResult(nr_free1, ub1, lb1, sum_free1, i);
                nr_free1 = boundCheckResult.getNr_free();
                sum_free1 = boundCheckResult.getSum_free();
                ub1 = boundCheckResult.getUb();
                lb1 = boundCheckResult.getLb();
            }
            else
            {
                CalculationRhoBoundCheckResult boundCheckResult =
                        new CalculationRhoBoundCheckResult(nr_free2, ub2, lb2, sum_free2, i);
                nr_free2 = boundCheckResult.getNr_free();
                sum_free2 = boundCheckResult.getSum_free();
                ub2 = boundCheckResult.getUb();
                lb2 = boundCheckResult.getLb();
            }
        }

        double r1, r2;
        if (nr_free1 > 0)
        {
            r1 = sum_free1 / nr_free1;
        }
        else
        {
            r1 = (ub1 + lb1) / 2;
        }

        if (nr_free2 > 0)
        {
            r2 = sum_free2 / nr_free2;
        }
        else
        {
            r2 = (ub2 + lb2) / 2;
        }

        si.r = (r1 + r2) / 2;
        return (r1 - r2) / 2;
    }

    private class CalculationRhoBoundCheckResult
    {
        private int nr_free;
        private double ub;
        private double lb;
        private double sum_free;

        private CalculationRhoBoundCheckResult(int nr_free, double ub, double lb, double sum_free, int i)
        {
            if (isLowerBound(i))
            {
                this.ub = Math.min(ub, G[i]);
            }
            else if (isUpperBound(i))
            {
                this.lb = Math.max(lb, G[i]);
            }
            else
            {
                this.nr_free = ++nr_free;
                this.sum_free = sum_free + G[i];
            }
        }

        private int getNr_free()
        {
            return nr_free;
        }

        private double getUb()
        {
            return ub;
        }

        private double getLb()
        {
            return lb;
        }

        private double getSum_free()
        {
            return sum_free;
        }
    }

    private class SelectWorkingSetBoundaryCheckResult
    {
        private int gmin_idx;
        private double obj_diff_min;

        private SelectWorkingSetBoundaryCheckResult(int gmin_idx, double obj_diff_min, int ip, float[] q_ip, int j,
                                                    double grad_diff)
        {
            this.gmin_idx = gmin_idx;

            if (grad_diff > 0)
            {
                double obj_diff;
                double quad_coef = QD[ip] + QD[j] - 2 * q_ip[j];
                if (quad_coef > 0)
                {
                    obj_diff = -(grad_diff * grad_diff) / quad_coef;
                }
                else
                {
                    obj_diff = -(grad_diff * grad_diff) / 1e-12;
                }

                if (obj_diff <= obj_diff_min)
                {
                    this.gmin_idx = j;
                    this.obj_diff_min = obj_diff;
                }
            }
        }

        private int getGmin_idx()
        {
            return gmin_idx;
        }

        private double getObj_diff_min()
        {
            return obj_diff_min;
        }
    }
}
