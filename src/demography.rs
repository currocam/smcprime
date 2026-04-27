use std::cmp::Ordering;
use thiserror::Error;

pub type Time = f64;

#[derive(Debug, Error)]
pub enum SMCPrimeError {
    #[error("Invalid demography: {0}")]
    InvalidDemography(String),
}

#[derive(Debug, Clone)]
pub struct Epoch {
    pub start_time: Time,
    pub lambda_start: f64, // 1/N0 at epoch start
    pub alpha: f64,        // growth rate (0.0 = constant)
}

impl Epoch {
    fn is_constant(&self) -> bool {
        self.alpha.abs() < f64::EPSILON
    }

    /// Invert the cumulative hazard: find t_coal > t1 such that k * H(t1, t_coal) = e.
    /// Returns None if coalescence cannot happen in this epoch (when the accumulated hazard is insufficient).
    pub fn invert(&self, t1: Time, e: f64, k: f64) -> Option<Time> {
        if self.is_constant() {
            Some(t1 + e / (k * self.lambda_start))
        } else {
            let a = self.alpha;
            let s = self.start_time;
            // From e = k * H(t1, t_coal):
            // e * a / (k * lambda_start) = exp(a*(t_coal - s)) - exp(a*(t1 - s))
            // exp(a*(t_coal - s)) = e * a / (k * lambda_start) + exp(a*(t1 - s))
            // To use exp_m1/ln_1p for stability when terms are small:
            // exp(a*(t_coal - s)) - 1 = e * a / (k * lambda_start) + expm1(a*(t1 - s))
            let base_m1 = (a * (t1 - s)).exp_m1();
            let arg_m1 = e * a / (k * self.lambda_start) + base_m1;

            // If exp(a*(t_coal - s)) <= 0 (i.e. arg_m1 <= -1.0), it cannot coalesce in this epoch.
            if arg_m1 <= -1.0 {
                return None;
            }
            Some(s + arg_m1.ln_1p() / a)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Demography {
    pub epochs: Vec<Epoch>,
}

impl Demography {
    pub fn num_epochs(&self) -> usize {
        self.epochs.len()
    }

    /// End time of epoch i (start of next epoch, or infinity for the last).
    pub fn epoch_end(&self, i: usize) -> Time {
        if i + 1 < self.epochs.len() {
            self.epochs[i + 1].start_time
        } else {
            f64::INFINITY
        }
    }

    pub fn epoch_index_at(&self, t: Time) -> usize {
        match self
            .epochs
            .binary_search_by(|e| e.start_time.partial_cmp(&t).unwrap_or(Ordering::Less))
        {
            Ok(i) => i,
            // If binary_search_by returns Err(i), i is the index where t could be inserted to maintain sorted order.
            // Therefore, the epoch that covers t is at index i - 1.
            Err(i) => i - 1,
        }
    }

    pub fn constant(ne: f64) -> Result<Self, SMCPrimeError> {
        if ne <= 0.0 {
            return Err(SMCPrimeError::InvalidDemography(
                "Population size must be positive".into(),
            ));
        }
        Ok(Self {
            epochs: vec![Epoch {
                start_time: 0.0,
                lambda_start: 1.0 / ne,
                alpha: 0.0,
            }],
        })
    }

    pub fn piecewise_constant_epochs(epochs: &[(f64, f64)]) -> Result<Self, SMCPrimeError> {
        let triples: Vec<(f64, f64, f64)> = epochs.iter().map(|&(t, ne)| (t, ne, 0.0)).collect();
        Self::piecewise_exponential_epochs(&triples)
    }

    pub fn piecewise_exponential_epochs(epochs: &[(f64, f64, f64)]) -> Result<Self, SMCPrimeError> {
        if epochs.is_empty() {
            return Err(SMCPrimeError::InvalidDemography(
                "Demography must include at least one epoch".into(),
            ));
        }

        let (first_time, _, _) = epochs[0];
        if !first_time.is_finite() || first_time != 0.0 {
            return Err(SMCPrimeError::InvalidDemography(
                "First epoch must start at time 0".into(),
            ));
        }

        let mut parsed_epochs = Vec::with_capacity(epochs.len());
        let mut prev_time = f64::NEG_INFINITY;
        for (i, &(start_time, ne, alpha)) in epochs.iter().enumerate() {
            if !start_time.is_finite() {
                return Err(SMCPrimeError::InvalidDemography(format!(
                    "Epoch start time at index {i} must be finite"
                )));
            }
            if i > 0 && start_time <= prev_time {
                return Err(SMCPrimeError::InvalidDemography(
                    "Epoch start times must be strictly increasing".into(),
                ));
            }
            if !ne.is_finite() || ne <= 0.0 {
                return Err(SMCPrimeError::InvalidDemography(format!(
                    "Epoch size at index {i} must be a positive finite number"
                )));
            }
            if !alpha.is_finite() {
                return Err(SMCPrimeError::InvalidDemography(format!(
                    "Growth rate at index {i} must be finite"
                )));
            }
            parsed_epochs.push(Epoch {
                start_time,
                lambda_start: 1.0 / ne,
                alpha,
            });
            prev_time = start_time;
        }

        if parsed_epochs.last().expect("Not empty").alpha != 0.0 {
            return Err(SMCPrimeError::InvalidDemography(
                "Last epoch must have growth_rate = 0 (constant)".into(),
            ));
        }

        Ok(Self {
            epochs: parsed_epochs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Forward computation of the cumulative hazard (verified with sympy):
    ///
    /// \lambda(t) = \lambda_{0} e^{\alpha (t - s)}
    ///
    /// \alpha \neq 0: \frac{k \lambda_{0} \left(- e^{\alpha t_{1}} + e^{\alpha t_{2}}\right) e^{- \alpha s}}{\alpha}
    ///
    /// \alpha = 0: k \left(- \lambda_{0} t_{1} + \lambda_{0} t_{2}\right)
    ///
    /// This is the "obvious" formula that invert() must be the inverse of.
    fn forward_hazard(epoch: &Epoch, t1: f64, t2: f64, k: f64) -> f64 {
        let a = epoch.alpha;
        let s = epoch.start_time;
        let lam = epoch.lambda_start;
        if epoch.is_constant() {
            k * lam * (t2 - t1)
        } else {
            k * (lam / a) * ((a * (t2 - s)).exp() - (a * (t1 - s)).exp())
        }
    }

    /// Total hazard from t1 to ∞. Only finite when α < 0:
    ///
    /// - \frac{k \lambda_{0} e^{- \alpha \left(s - t_{1}\right)}}{\alpha}
    fn total_hazard(epoch: &Epoch, t1: f64, k: f64) -> f64 {
        if epoch.alpha >= 0.0 {
            return f64::INFINITY;
        }
        k * epoch.lambda_start / (-epoch.alpha) * (epoch.alpha * (t1 - epoch.start_time)).exp()
    }

    proptest! {
        #[test]
        fn invert_is_inverse_of_cumulative_hazard(
            ne in 1.0f64..1000.0,
            alpha in -0.05f64..0.05,
            t1_offset in 0.0f64..50.0,
            e in 0.01f64..10.0,
            k in 0.5f64..20.0,
        ) {
            let epoch = Epoch { start_time: 0.0, lambda_start: 1.0 / ne, alpha };
            let t1 = t1_offset;
            const EPSILON: f64 = 1e-12;

            match epoch.invert(t1, e, k) {
                Some(t_coal) => {
                    prop_assert!(t_coal >= t1);
                    let got = forward_hazard(&epoch, t1, t_coal, k);
                    let rel_err = (got - e).abs() / e;
                    prop_assert!(rel_err < EPSILON,
                        "e={e}, forward={got}, rel_err={rel_err}, alpha={alpha}");
                }
                None => {
                    let h_max = total_hazard(&epoch, t1, k);
                    prop_assert!(e > h_max,
                        "None but e={e} <= total_hazard={h_max}");
                }
            }
        }

        #[test]
        fn invert_is_monotone(
            ne in 1.0f64..1000.0,
            alpha in -0.02f64..0.02,
            e1 in 0.01f64..5.0,
            e2 in 0.01f64..5.0,
            k in 1.0f64..10.0,
        ) {
            let epoch = Epoch { start_time: 0.0, lambda_start: 1.0 / ne, alpha };
            if let (Some(t1), Some(t2)) = (epoch.invert(0.0, e1, k), epoch.invert(0.0, e2, k)) {
                if e1 < e2 {
                    prop_assert!(t1 <= t2, "e1={e1}<e2={e2} but t1={t1}>t2={t2}");
                }
            }
        }
    }
}
