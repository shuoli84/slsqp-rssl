use slsqp_rssl::{Constraint, SlsqpMode, fmin_slsqp};

#[test]
fn test_fmin_slsqp_basic() {
    // Objective function: f(x) = x1^2 + x2^2
    let func = |x: &[f64]| x[0] * x[0] + x[1] * x[1];

    // Initial point
    let x0 = [2.0, 2.0];

    // Bounds: x1 >= 0, x2 >= 0
    let bounds = [(0.0, 10.0), (0.0, 10.0)];

    // Constraint: x1 + x2 - 2 = 0 (equality constraint)
    let constraints = vec![Constraint::Eq(Box::new(|x: &[f64]| x[0] + x[1] - 2.0))];

    let result = fmin_slsqp(func, &x0, &bounds, constraints, 100, 1e-6, None);

    println!("Status: {}", result.status);
    println!("Message: {}", result.message);
    println!("x: {:?}", result.x);
    println!("f: {}", result.fun);
    println!("Iterations: {}", result.nit);

    assert_eq!(result.status, SlsqpMode::Success as i32);
    assert!((result.x[0] - 1.0).abs() < 1e-4);
    assert!((result.x[1] - 1.0).abs() < 1e-4);
    assert!((result.fun - 2.0).abs() < 1e-4);
}

#[test]
fn test_fmin_slsqp_inequality_simple() {
    // Objective function: f(x) = x1^2 + x2^2
    let func = |x: &[f64]| x[0] * x[0] + x[1] * x[1];

    // Initial point
    let x0 = [2.0, 2.0];

    // Constraint: x1 + x2 >= 1.0
    let constraints = vec![Constraint::Ineq(Box::new(|x: &[f64]| x[0] + x[1] - 1.0))];

    let bounds = [];

    let result = fmin_slsqp(func, &x0, &bounds, constraints, 100, 1e-6, None);

    println!("Status: {}", result.status);
    println!("x: {:?}", result.x);
    println!("f: {}", result.fun);

    assert_eq!(result.status, SlsqpMode::Success as i32);
    // For f = x1^2 + x2^2, x1 + x2 >= 1, the optimal solution should be [0.5, 0.5]
    assert!((result.x[0] - 0.5).abs() < 1e-4);
    assert!((result.x[1] - 0.5).abs() < 1e-4);
    assert!((result.fun - 0.5).abs() < 1e-4);
}
