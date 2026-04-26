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
    /// Cumulative hazard integral from t1 to t2 (unit rate multiplier).
    pub fn cumulative_hazard(&self, t1: Time, t2: Time) -> f64 {
        if self.alpha == 0.0 {
            self.lambda_start * (t2 - t1)
        } else {
            let a = self.alpha;
            let s = self.start_time;
            // Using expm1 for better numerical stability when a*(t-s) is small:
            // e^(x) - e^(y) = e^x - 1 - (e^y - 1) = expm1(x) - expm1(y)
            (self.lambda_start / a) * ((a * (t2 - s)).exp_m1() - (a * (t1 - s)).exp_m1())
        }
    }

    /// Invert the cumulative hazard: find t_coal > t1 such that k * H(t1, t_coal) = e.
    /// Returns None if coalescence cannot happen in this epoch (when the accumulated hazard is insufficient).
    pub fn invert(&self, t1: Time, e: f64, k: f64) -> Option<Time> {
        if self.alpha == 0.0 {
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
        match self.epochs.binary_search_by(|e| {
            e.start_time
                .partial_cmp(&t)
                .unwrap_or(Ordering::Less)
        }) {
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