import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Data.Complex.Exponential

open Real Set

noncomputable def psi (x : ℝ) : ℝ := (log x) / x

/-- psi 在 x 处的导数公式 -/
lemma deriv_psi (x : ℝ) (hx : x ≠ 0) : deriv psi x = (1 - log x) / (x ^ 2) := by
  unfold psi
  rw [deriv_div (differentiableAt_log hx) (differentiableAt_id) hx]
  simp [deriv_log x]
  ring

/-- e-base 缩放定律：x = e 是全局最大值点 -/
theorem e_base_scaling_law (x : ℝ) (hx : 0 < x) : psi x ≤ psi (exp 1) := by
  rcases lt_trichotomy x (exp 1) with hlt | heq | hgt
  
  -- Case 1: x < e
  { have h_cont : ContinuousOn psi (Icc x (exp 1)) := by
      apply ContinuousOn.div
      · apply ContinuousOn.log; intro y hy; exact ne_of_gt (lt_trans hx hy.1)
      · exact continuousOn_id
      · intro y hy; exact ne_of_gt (lt_trans hx hy.1)
    
    have h_diff : DifferentiableOn ℝ psi (Ioo x (exp 1)) := by
      apply DifferentiableOn.div
      · apply DifferentiableOn.log; intro y hy; exact ne_of_gt (lt_trans hx hy.1)
      · exact differentiableOn_id
      · intro y hy; exact ne_of_gt (lt_trans hx hy.1)

    -- 使用中值定理
    rcases exists_deriv_eq_slope psi hlt h_cont h_diff with ⟨c, hc, h_mean⟩
    rw [deriv_psi c (ne_of_gt (lt_trans hx hc.1))] at h_mean
    
    have hc_lt_e : c < exp 1 := hc.2
    have h_log_c_lt_1 : log c < 1 := by rwa [← log_exp 1, log_lt_log_iff (lt_trans hx hc.1) (exp_pos 1)]
    have h_deriv_pos : 0 < (1 - log c) / (c ^ 2) := by
      apply div_pos (sub_pos.mpr h_log_c_lt_1) (pow_pos (lt_trans hx hc.1) 2)
    
    have h_slope_pos : 0 < (psi (exp 1) - psi x) / (exp 1 - x) := by rw [← h_mean]; exact h_deriv_pos
    
    have h_dx_pos : 0 < exp 1 - x := sub_pos.mpr hlt
    have h_dy_pos : 0 < psi (exp 1) - psi x := (div_pos_iff_of_pos_right h_dx_pos).mp h_slope_pos
    linarith
  }

  -- Case 2: x = e
  { rw [heq] }

  -- Case 3: x > e
  { 
    have h_cont : ContinuousOn psi (Icc (exp 1) x) := by
      apply ContinuousOn.div
      · apply ContinuousOn.log; intro y hy; exact ne_of_gt (lt_trans (exp_pos 1) hy.1)
      · exact continuousOn_id
      · intro y hy; exact ne_of_gt (lt_trans (exp_pos 1) hy.1)
      
    have h_diff : DifferentiableOn ℝ psi (Ioo (exp 1) x) := by
      apply DifferentiableOn.div
      · apply DifferentiableOn.log; intro y hy; exact ne_of_gt (lt_trans (exp_pos 1) hy.1)
      · exact differentiableOn_id
      · intro y hy; exact ne_of_gt (lt_trans (exp_pos 1) hy.1)

    rcases exists_deriv_eq_slope psi hgt h_cont h_diff with ⟨c, hc, h_mean⟩
    rw [deriv_psi c (ne_of_gt (lt_trans (exp_pos 1) hc.1))] at h_mean
    
    have hc_gt_e : exp 1 < c := hc.1
    have h_log_c_gt_1 : 1 < log c := by rwa [← log_exp 1, log_lt_log_iff (exp_pos 1) (lt_trans (exp_pos 1) hc.1)]
    have h_deriv_neg : (1 - log c) / (c ^ 2) < 0 := by
      apply div_neg_of_neg_of_pos (sub_neg.mpr h_log_c_gt_1) (pow_pos (lt_trans (exp_pos 1) hc.1) 2)
      
    have h_slope_neg : (psi x - psi (exp 1)) / (x - exp 1) < 0 := by rw [← h_mean]; exact h_deriv_neg
    
    have h_dx_pos : 0 < x - exp 1 := sub_pos.mpr hgt
    have h_dy_neg : psi x - psi (exp 1) < 0 := (div_neg_iff_of_pos_right h_dx_pos).mp h_slope_neg
    linarith
  }
