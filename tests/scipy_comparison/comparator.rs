use crate::scipy_comparison::python_runner::SciPyResult;
use slsqp_rssl::{SlsqpMode, SlsqpResult};

#[derive(Debug, Clone)]
pub struct ComparisonReport {
    pub name: String,
    pub x_rust: Vec<f64>,
    pub x_scipy: Vec<f64>,
    pub x_diff: f64,
    pub fun_rust: f64,
    pub fun_scipy: f64,
    pub fun_diff: f64,
    pub max_cv_rust: f64,
    pub max_cv_scipy: f64,
    pub nit_rust: usize,
    pub nit_scipy: usize,
    pub status_rust: i32,
    pub status_scipy: i32,
    pub success_rust: bool,
    pub success_scipy: bool,
    pub success_match: bool,
    pub passed: bool,
}

impl ComparisonReport {
    pub fn print(&self) {
        println!("--------------------------------------------------");
        println!("Test Case: {}", self.name);
        println!("Status: {}", if self.passed { "PASSED" } else { "FAILED" });
        println!("  Success Match: {}", self.success_match);
        println!(
            "  Rust Status:  {} ({})",
            self.status_rust,
            if self.success_rust {
                "Success"
            } else {
                "Failure"
            }
        );
        println!(
            "  SciPy Status: {} ({})",
            self.status_scipy,
            if self.success_scipy {
                "Success"
            } else {
                "Failure"
            }
        );
        println!(
            "  Iterations:   Rust: {}, SciPy: {}",
            self.nit_rust, self.nit_scipy
        );
        println!(
            "  Objective:    Rust: {:.8e}, SciPy: {:.8e}, Diff: {:.8e}",
            self.fun_rust, self.fun_scipy, self.fun_diff
        );
        println!(
            "  Max CV:       Rust: {:.8e}, SciPy: {:.8e}",
            self.max_cv_rust, self.max_cv_scipy
        );
        println!("  X Difference: {:.8e}", self.x_diff);
        if !self.passed {
            println!("  Rust X:  {:?}", self.x_rust);
            println!("  SciPy X: {:?}", self.x_scipy);
        }
        println!("--------------------------------------------------");
    }
}

pub fn compare_results(
    name: &str,
    rust_res: &SlsqpResult,
    scipy_res: &SciPyResult,
    max_cv_rust: f64,
    x_tol: f64,
    fun_tol: f64,
) -> ComparisonReport {
    let success_rust = rust_res.status == SlsqpMode::Success as i32;
    let success_scipy = scipy_res.success;

    let mut x_diff: f64 = 0.0;
    for (a, b) in rust_res.x.iter().zip(scipy_res.x.iter()) {
        x_diff = x_diff.max((a - b).abs());
    }

    let fun_diff = (rust_res.fun - scipy_res.fun).abs();

    // Difference in iterations (ratio)
    let nit_diff_ratio = if scipy_res.nit > 0 {
        (rust_res.nit as f64 - scipy_res.nit as f64).abs() / scipy_res.nit as f64
    } else {
        rust_res.nit as f64
    };

    // Decision criteria:
    // 1. Success status must be consistent
    let success_match = success_rust == success_scipy;

    // 2. If both succeed, compare differences in x and fun
    let values_match = if success_rust {
        x_diff < x_tol && fun_diff < fun_tol
    } else {
        // If both fail, consider them consistent for now (status codes could be further checked)
        true
    };

    // 3. Iteration count is not a hard metric but is shown in the report
    let _nit_match = nit_diff_ratio < 0.5; // Allow 50% difference as a warning

    let passed = success_match && values_match;

    ComparisonReport {
        name: name.to_string(),
        x_rust: rust_res.x.clone(),
        x_scipy: scipy_res.x.clone(),
        x_diff,
        fun_rust: rust_res.fun,
        fun_scipy: scipy_res.fun,
        fun_diff,
        max_cv_rust,
        max_cv_scipy: scipy_res.max_cv,
        nit_rust: rust_res.nit,
        nit_scipy: scipy_res.nit,
        status_rust: rust_res.status as i32,
        status_scipy: scipy_res.status,
        success_rust,
        success_scipy,
        success_match,
        passed,
    }
}
