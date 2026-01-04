import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

open Real
open Set

theorem max_ln_div_x (x : ℝ) (hx : x > 0) : Real.log x / x ≤ Real.log (Real.exp 1) / Real.exp 1 := by
  have h_exp_pos : Real.exp 1 > 0 := Real.exp_pos 1
  have h_log_exp : Real.log (Real.exp 1) = 1 := Real.log_exp 1
  rw [h_log_exp, Real.exp_one]
  have h1 : Real.log 1 / 1 = 0 := by
    rw [Real.log_one, zero_div]
  have h2 : Real.log (Real.exp 1) / Real.exp 1 = 1 / Real.exp 1 := by
    rw [h_log_exp]
  have deriv_eq : ∀ x > 0, deriv (fun x : ℝ => Real.log x / x) x = (1 - Real.log x) / (x ^ 2) := by
    intro x hx_pos
    have h_diff : DifferentiableAt ℝ (fun x : ℝ => Real.log x / x) x := by
      refine (Real.differentiableAt_log hx_pos.ne.symm).div (differentiableAt_id' x) ?_
      exact hx_pos.ne.symm
    rw [deriv_div (Real.differentiableAt_log hx_pos.ne.symm) (differentiableAt_id' x) hx_pos.ne.symm]
    simp [deriv_log' hx_pos.ne.symm, deriv_id'']
    ring
  have critical_point : deriv (fun x : ℝ => Real.log x / x) (Real.exp 1) = 0 := by
    rw [deriv_eq (Real.exp 1) (Real.exp_pos 1)]
    rw [Real.log_exp]
    simp
  have mono_on : ∀ x ∈ Ioi (0 : ℝ), deriv (fun x : ℝ => Real.log x / x) x ≥ 0 ↔ x ≤ Real.exp 1 := by
    intro x hx_mem
    rw [deriv_eq x (mem_Ioi.mp hx_mem)]
    constructor
    · intro h
      have : 1 - Real.log x ≥ 0 := by
        contrapose! h
        have : x ^ 2 > 0 := pow_pos (mem_Ioi.mp hx_mem) 2
        exact div_neg_of_neg_of_pos (by linarith) this
      linarith [Real.log_le_sub_one_of_pos (mem_Ioi.mp hx_mem)]
    · intro h
      have : Real.log x ≤ 1 := by
        calc Real.log x ≤ Real.log (Real.exp 1) := Real.log_le_log (mem_Ioi.mp hx_mem) (Real.exp_pos 1) h
          _ = 1 := h_log_exp
      have : x ^ 2 > 0 := pow_pos (mem_Ioi.mp hx_mem) 2
      exact div_nonneg (by linarith) (by positivity)
  have concavity : ∀ x ∈ Ioi (0 : ℝ), x ≠ Real.exp 1 → deriv (fun x : ℝ => Real.log x / x) x > 0 → x < Real.exp 1 := by 
    intro x hx_mem hx_ne hderiv
    by_contra! H
    have : x ≥ Real.exp 1 := H
    have := mono_on x hx_mem |>.mpr this
    linarith
  have h3 : IsMaxOn (fun x : ℝ => Real.log x / x) (Icc (0 : ℝ) (Real.exp 1)) (Real.exp 1) := by
    refine ConvexOn.isMaxOn_of_deriv_nonneg (convex_Icc 0 (Real.exp 1)) ?_ (by exact ⟨Real.exp_pos 1, by linarith [Real.one_le_exp]⟩)
      (fun x hx => ?_) (by simp)
    · refine (Continuous.continuousOn ?_).div continuousOn_id fun x hx => ?_
      exact Real.continuousAt_log (mem_Icc.mp hx).left.ne.symm
      exact (mem_Icc.mp hx).left.ne.symm
    · rcases hx with ⟨hx_left, hx_right⟩
      have hx_pos : x > 0 := by linarith
      rw [deriv_eq x hx_pos]
      exact div_nonneg (by linarith [Real.log_le_sub_one_of_pos hx_pos]) (pow_pos hx_pos 2).le
  have h4 : ∀ x ≥ Real.exp 1, Real.log x / x ≤ 1 / Real.exp 1 := by
    intro x hx
    have hx_pos : x > 0 := by linarith [Real.exp_pos 1]
    have : Real.log x / x ≤ Real.log (Real.exp 1) / Real.exp 1 := by
      rw [h_log_exp]
      exact h3 (right_mem_Icc.mpr hx)
    rwa [h_log_exp] at this
  by_cases h : x ≤ Real.exp 1
  · exact h3 (mem_Icc.mpr ⟨hx.le, h⟩)
  · push_neg at h
    exact h4 x (by linarith)
