pub mod comparator;
pub mod python_runner;

pub use self::comparator::{ComparisonReport, compare_results};
pub use self::python_runner::run_scipy;
use slsqp_rssl::{Constraint, fmin_slsqp};
pub use slsqp_test_cases::{TestCase, TestConstraint};

pub fn run_comparison_test(
    test_case: TestCase,
    x_tol: f64,
    fun_tol: f64,
) -> Result<ComparisonReport, String> {
    run_comparison_test_with_acc(test_case, x_tol, fun_tol, None)
}

pub fn run_comparison_test_with_acc(
    test_case: TestCase,
    x_tol: f64,
    fun_tol: f64,
    override_acc: Option<f64>,
) -> Result<ComparisonReport, String> {
    // 1. Run Python version
    let python_problem = test_case.to_python_problem();
    let scipy_res = run_scipy(&python_problem)?;

    // 2. Run Rust version
    let mut rust_constraints = Vec::new();
    for tc in &test_case.constraints {
        match tc {
            TestConstraint::Eq { fun, .. } => {
                rust_constraints.push(Constraint::Eq(Box::new(*fun)));
            }
            TestConstraint::Ineq { fun, .. } => {
                rust_constraints.push(Constraint::Ineq(Box::new(*fun)));
            }
        }
    }

    let bounds = test_case.bounds.clone().unwrap_or_default();
    let objective_fn = test_case.objective_fn;

    let acc = override_acc.unwrap_or(test_case.tol);

    let rust_res = fmin_slsqp(
        &objective_fn,
        &test_case.x0,
        &bounds,
        rust_constraints,
        test_case.maxiter,
        acc,
        None,
    );

    // 3. Calculate Rust Max CV
    let mut max_cv_rust: f64 = 0.0;
    for tc in &test_case.constraints {
        match tc {
            TestConstraint::Eq { fun, .. } => {
                max_cv_rust = max_cv_rust.max(fun(&rust_res.x).abs());
            }
            TestConstraint::Ineq { fun, .. } => {
                max_cv_rust = max_cv_rust.max(0.0f64.max(-fun(&rust_res.x)));
            }
        }
    }
    if let Some(ref bounds) = test_case.bounds {
        for (i, &(l, u)) in bounds.iter().enumerate() {
            max_cv_rust = max_cv_rust.max(0.0f64.max(l - rust_res.x[i]));
            max_cv_rust = max_cv_rust.max(0.0f64.max(rust_res.x[i] - u));
        }
    }

    // 4. Compare results
    Ok(compare_results(
        &test_case.name,
        &rust_res,
        &scipy_res,
        max_cv_rust,
        x_tol,
        fun_tol,
    ))
}
