import Mathlib.Analysis.SpecialFunctions.Pow.Real
open Real

theorem max_ln_div_x (x : ℝ) (hx : x > 0) :
    (Real.log x) / x ≤ (Real.log (Real.exp 1)) / (Real.exp 1) := by
  have h1 : Real.log (Real.exp 1) = 1 := Real.log_exp 1
  rw [h1]
  have h2 : (Real.log x) / x ≤ 1 / Real.exp 1 := by
    by_cases hx' : x ≤ Real.exp 1
    · have h3 : Real.log x ≤ Real.log (Real.exp 1) := Real.log_le_log hx hx'
      have h4 : Real.log (Real.exp 1) = 1 := Real.log_exp 1
      rw [h4] at h3
      exact (div_le_div_right hx).mpr h3
    · push_neg at hx'
      have h3 : Real.log x ≤ Real.log (Real.exp 1) :=
        Real.log_le_log (by linarith) (by linarith)
      have h4 : Real.log (Real.exp 1) = 1 := Real.log_exp 1
      rw [h4] at h3
      exact (div_le_div_right hx).mpr h3
  exact h2