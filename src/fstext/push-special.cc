// fstext/push-special.cc

// Copyright 2012  Johns Hopkins University (authors: Daniel Povey,
//                 Ehsan Variani, Pegah Ghahrmani)

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "fstext/push-special.h"
#include "base/kaldi-error.h"
#include "matrix/matrix-lib.h"

namespace fst {

/*
  In pushing algorithm:
    Each state gets a potential, say p(s).  Initial state must have potential zero.
    The states have transition probabilities to each other, call these w(s, t), and
    final-probabilities f(s).  One special state is the initial state.
    These are real probabilities, think of them like HMM transition probabilities; we'll
    represent them as double.  Each state has a potential, p(s).
    After taking into account the potentials, the weights transform
       w(s, t) -->  w(s, t) / p(s) * p(t)
    and the final-probs transform
       f(s)  -->  f(s) / p(s).
    The initial state's potential is fixed at 1.0.
    Let us define a kind of normalizer for each state s as:
       n(s) = f(s) + \sum_t w(s, t),
    or taking into account the potentials, and treating the self-loop as special,
       
       n(s) =  f(s)/p(s) + \sum_t w(s, t) p(t) / p(s)
            =  w(s, s) + (1/p(s)) f(s) + \sum_{t != s} w(s, t) p(t).         (Eq. 1)
     
    This should be one if the FST is stochastic (in the log semiring).
    In fact not all FSTs can be made stochastic while preserving equivalence, so
    in "PushSpecial" we settle for a different condition: that all the n(s) be the
    same.  This means that the non-sum-to-one-ness of the FST is somehow smeared
    out throughout the FST.  We want an algorithm that makes all the n(s) the same,
    and we formulate it in terms of iteratively improving objective function.  The
    objective function will be the sum-of-squared deviation of each of the n(s) from
    their overall mean, i.e.
       \sum_s  (n(s) - n_{avg})^2
    where n_avg is the average of the n(s).  When we choose an s to optimize its p(s),
    we'll minimize this function, but while minimizing it we'll treat n_{avg} as
    a fixed quantitiy.  We can show that even though this is not 100% accurate, we
    still end up minimizing the objective function (i.e. we still decrease the
    sum-of-squared difference).
    
    Suppose we choose some s for which to tweak p(s). [naturally s cannot be the start
    state].  Firstly, we assume n_{avg} is a known and fixed quantity.  When we
    change p(s) we alter n(s) and also n(t) for all states t != s that have a transition
    into s (i.e. w(s, t) != 0).  Let's write p(s) for the current value of p(s),
    and p'(s) for the value we'll replace it with, and use similar notation for
    the n(s).  We'll write out the part of the objective function that involves p(s),
    and this is:

     F =  (n(s) - n_{avg})^2  +  \sum_{t != s}  (n(t) - n_{avg})^2.

    Here, n_{avg} is treated as fixed.  We can write n(s) as:
       n(s) = w(s, s) + k(s) / p(s)
    where k(s) = f(s) + \sum_{t != s) w(s, t) p(t),
    but note that if we have n(s) already, k(s) can be computed by:
         k(s) = (n(s) - w(s, s)) * p(s)

    We can write n(t) [for t != s] as:
       n(t) = j(t) + w(t, s)/p(t) p(s)
    and
       j(t) = w(t, t) + (1/p(t)) \sum_{u != s, u != t} w(t, u) p(u)
    but in practice if we have the normalizers n(t) up to date,
    we can compute it more efficiently as
       j(t) = n(t) - w(t, s)/p(t) p(s)                      (Eq. 2)
       

    Now let's imagine we did the substitutions for n(s) and n(t), and we'll
    write out the terms in F that are functions of p(s).  We have:

       F =                         k(s)^2  p(s)^{-2}
             + 2 k(s) (w(s, s) -  n_{avg}) p(s)^{-1}
             + [constant term that doesn't matter]
+ (\sum_t 2(j(t) - n_{avg})(w(t, s)/p(t))  p(s)
             + (\sum_t  (w(t, s)/p(t))^2 ) p(s)^2                 (Eq. 3)

    Note that the {-2} and {+2} power terms are both positive, and this means
    that F will get large as p(s) either gets too large or too small.  This is
    comforting because we want to minimize F.  Let us write the four coefficients
    above as c_{-2}, c_{-1}, c_1 and c_2.   The minimum of F can be found where
    the derivative of F w.r.t. p(s) is zero.  Here, let's just call it p for short.
    This will be where:
     d/dp  c_{-2} p^{-2} + c_{-1} p^{-1} + c_1 p + c_2 p^2  = 0
            -2 c_{-2} p^{-3} - c_{-1} p^{-2} + c_1 + c_2 p  = 0 .
    Technically we can solve this type of formula by means of the quartic equations,
    but these take up pages and pages.  Instead we'll use a one-dimensional form
    of Newton's method, computing the derivative and second derivative by differentiating the
    formula.

*/
            
class PushSpecialClass {
  typedef StdArc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;

 public:
  // Everything happens in the initializer.
  PushSpecialClass(VectorFst<StdArc> *fst,
                   float delta): fst_(fst) {
    num_states_ = fst_->NumStates();
    initial_state_ = fst_->Start();
    final_.Resize(num_states_);
    self_loop_.Resize(num_states_);
    foll_.resize(num_states_);
    pred_.resize(num_states_);
    p_.Resize(num_states_);
    p_.Set(1.0); // Potentials: default is one.
    for (StateId s = 0; s < num_states_; s++) {
      for (ArcIterator<VectorFst<StdArc> > aiter(*fst, s);
           !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (arc.nextstate == s) {
          self_loop_(s) += exp(-arc.weight.Value());
        } else {
          StateId t = arc.nextstate;
          double weight = exp(-arc.weight.Value());
          foll_[s].push_back(std::make_pair(t, weight));
          pred_[t].push_back(std::make_pair(s, weight));
        }
      }
      Weight final = fst_->Final(s);
      final_(s) = exp(-final.Value());
    }
    n_.Resize(num_states_);
    OptimizePotentials(delta);
    ModifyFst();
  }
 private:
  void ComputeNormalizers() {
    // compute n(s) for each s.
    // Note: this may produce NaN's if some p_ values were zero,
    // and the results are meaningless if some p_values were negative,
    // but this won't matter (ComputeObjf() makes sure to return
    // +infinity for the objective function if some p values were
    // negative).
    double n_tot = 0.0;
    for (StateId s = 0; s < num_states_; s++) {
      double n = self_loop_(s), p_s = p_(s), f_s = final_(s);
      for (size_t i = 0; i < foll_[s].size(); i++) {
        StateId t = foll_[s][i].first;
        double w_st = foll_[s][i].second;
        n += w_st * p_(t) / p_s;
      }
      n += f_s / p_s;
      n_(s) = n;
      n_tot += n;
    }
    n_avg_ = n_tot / num_states_;
  }

  double ComputeF(double c_minus_2, double c_minus_1, double c_plus_1,
                  double c_plus_2, double p) {
    return c_minus_2 / (p * p) +
        c_minus_1 / p +
        c_plus_1 * p +
        c_plus_2 * p * p;
  }

  double SolveForP(double c_minus_2, double c_minus_1, double c_plus_1,
                   double c_plus_2, double p) {
    int num_iter = 1000; // maximum number of iterations
    double threshold = 1.0e-07; // Threshold on the derivative;
    // below this, we accept the value.
    double c_1 = 1.0e-04, backtrack_constant = 0.5; // c_1 in Armijo rule;
    // backtrack_constant is what we multiply step size by each time.
    double p_new;
    double diff1; // first derivative
    double diff2; // second derivative
    double cur_f;
    int i;
    for (i = 0; i < num_iter; i++) {
      cur_f = ComputeF(c_minus_2, c_minus_1, c_plus_1, c_plus_2, p);
      diff1 = -2.0 * c_minus_2 * pow(p, -3)
          -1.0 * c_minus_1 * pow(p, -2)
          + c_plus_1
          + 2.0 * c_plus_2 * p;
      diff2 = 6.0 * c_minus_2 * pow(p, -4)
          + 2.0 * c_minus_1 * pow(p, -3)
          + 2.0 * c_plus_2;
      if (std::abs(diff1) < threshold) { // we're done.
        break;
      }
      // Compute proposed new value p_new [we'll then backtrack]
      if (diff2 > 0){
        p_new = p - (diff1)/(diff2);
      } else {
        if (diff1 > 0) {
          p_new = 0.1 * p;
        } else {
          p_new = 10 * p;
        }
      }
      if (p_new < 0.1*p) p_new = 0.1*p;
      if (p_new > 10*p) p_new = 10*p;

      // Do backtracking line search with the Armijo rule.
      while (ComputeF(c_minus_2, c_minus_1, c_plus_1, c_plus_2, p_new) >
             cur_f + c_1 * (p_new - p) * diff1) {
        p_new = p + backtrack_constant * (p_new - p);
        // We had to add the line below as it gets stuck in this loop
        // with differences between p and p_new that are at the limit of
        // roundoff.
        if (fabs(p_new - p) < 1.0e-10 * p) 
          return p;
      }
      p = p_new;
    }
    if (i == num_iter)
      KALDI_WARN << "Did not converge in line search.\n";
    return p;
  }


  double ComputeObjf() {
    // sum of squared diffs.
    // First, if any potentials are zero or negative this is invalid,
    // so we have to return +infinity.  This matters inside OptimizePotentialsOneIter(),
    // for the line search.
    for (StateId s = 0; s < num_states_; s++)
      if (p_(s) <= 0.0)
        return std::numeric_limits<double>::infinity();
    double ans = 0.0;
    for (StateId s = 0; s < num_states_; s++) {
      double diff = n_(s) - n_avg_;
      ans += diff * diff;
    }
    return ans;
  }
  
  void ComputeOneParameterGradient(StateId s,
                                   double *gradient) {
    if (s == initial_state_) { // This state is not allowed to have
      // its normalizer change.
      *gradient = 0.0;
      return;
    }
    // Optimize potential for state s.
    double n_s = n_(s), p_s = p_(s), w_s_s = self_loop_(s),
        k_s = (n_s - w_s_s) * p_s;
    if (k_s < 0.0) {
      if (k_s < -1.0e-04) // can't really be negative; allow rounding error.
        KALDI_WARN << "Negative k_s " << k_s;
      k_s = 0.0;
    }
    // coefficients in (Eq. 3) above.
    double c_minus_2 = k_s * k_s,
        c_minus_1 = 2.0 * k_s * (w_s_s - n_avg_),
        c_plus_1 = 0.0, c_plus_2 = 0.0; // these are sums over t.
    for (size_t i = 0; i < pred_[s].size(); i++) {
      StateId t = pred_[s][i].first;
      double w_ts = pred_[s][i].second, p_t = p_(t);
      double j_t = n_(t) - w_ts * p_s / p_t; // (Eq. 2)
      KALDI_ASSERT(!KALDI_ISNAN(j_t));
      if (j_t < 0.0) { // should not be negative.
        if (j_t < -1.0e-08) KALDI_WARN << "Negative j_t " << j_t;
        j_t = 0.0;
      }
      // all this is part of (Eq. 3):
      double tmp = w_ts / p_t;
      c_plus_2 += tmp * tmp;
      c_plus_1 += 2.0 * (j_t - n_avg_) * tmp;
    }
    // The objective function as a function of p is
    // c_minus_2 p^{-2} + c_minus_1 p^{-1} + c_plus_1 p + c_plus_2 p^2.
    // derivative is:
    // -2 c_minus_2 p^{-3} - c_minus_1 p^{-2} + c_plus_1 + 2 p c_plus_2
    *gradient = -2 * c_minus_2 / (p_s * p_s * p_s)
        - c_minus_1 / (p_s * p_s)
        + c_plus_1
        + 2 * p_s * c_plus_2;
  }

  
  void OptimizePotentials(float delta) {

    kaldi::LbfgsOptions opts;    
    kaldi::OptimizeLbfgs<double> opt_lbfgs(p_, opts);
    for (int num_iters = 0; true; num_iters++) {
      p_.CopyFromVec(opt_lbfgs.GetProposedValue());
      ComputeNormalizers();
      // Test convergence.
      bool converged = true;
      for (StateId s = 0; s < num_states_; s++)
        if (std::abs(n_(s) - n_avg_) > delta)
          converged = false;
      if (converged) {
        KALDI_VLOG(2) << "Did " << num_iters << " iterations of L-BFGS";
        return;
      } else { // Do the optimization step.
        double f = ComputeObjf();
        kaldi::Vector<double> gradient(num_states_);
        if (f != std::numeric_limits<double>::infinity())
          for (StateId s = 0; s < num_states_; s++)
            ComputeOneParameterGradient(s, &(gradient(s)));
        opt_lbfgs.DoStep(f, gradient);
      }
    }
  }


  // Modifies the FST weights and the final-prob to take account of these potentials.
  void ModifyFst() {
    // First get the potentials as negative-logs, like the values
    // in the FST.
    for (StateId s = 0; s < num_states_; s++) {
      p_(s) = -log(p_(s));
      KALDI_ASSERT(!isnan(p_(s)));
    }
    for (StateId s = 0; s < num_states_; s++) {
      for (MutableArcIterator<VectorFst<StdArc> > aiter(fst_, s);
           !aiter.Done(); aiter.Next()) {
        Arc arc = aiter.Value();
        StateId t = arc.nextstate;
        // w(s, t) <-- w(s, t) * p(t) / p(s).
        arc.weight = Weight(arc.weight.Value() + p_(t) - p_(s));
        aiter.SetValue(arc);
      }
      // f(s) <-- f(s) / p(s) = f(s) * (1.0 / p(s)).
      fst_->SetFinal(s, Times(fst_->Final(s).Value(), Weight(-p_(s))));
    }
  }

private:
  StateId num_states_;
  StateId initial_state_;
  kaldi::Vector<double> final_; // final-prob of each state.
  kaldi::Vector<double> self_loop_; // self-loop prob of each state: w(s, s).
  std::vector<std::vector<std::pair<StateId, double> > > foll_; // List of transitions
  // out of each state, to a *different* state.  The pair (t, w(s, t)).
  std::vector<std::vector<std::pair<StateId, double> > > pred_; // List of transitions
  // into each state, to a *different* state.  The pair (t, w(t, s)).
  kaldi::Vector<double> p_; // The potential of each state.
  kaldi::Vector<double> n_; // The normalizer n(s) for each state.
  double n_avg_; // The current average normalizer.
  
  VectorFst<StdArc> *fst_;
  
};




void PushSpecial(VectorFst<StdArc> *fst, float delta) {
  if (fst->NumStates() > 0)
    PushSpecialClass c(fst, delta); // all the work
    // gets done in the initializer.
}

  
} // end namespace fst.


