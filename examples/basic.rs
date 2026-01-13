use slsqp_rssl::{Constraint, fmin_slsqp};

fn main() -> std::io::Result<()> {
    // Rosenbrock function: f(x, y) = (a - x)^2 + b(y - x^2)^2
    // Minimum at (a, a^2). Usually a=1, b=100.
    let rosenbrock = |x: &[f64]| {
        let a = 1.0;
        let b = 100.0;
        (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
    };
    let constraints = vec![Constraint::Ineq(Box::new(|x| {
        2.0 - x[0].powi(2) - x[1].powi(2)
    }))];
    let x0 = &[-1.2, 1.0];
    let bounds = &[(-2.0, 2.0), (-2.0, 2.0)];

    println!("Starting optimization...");
    let res =
        fmin_slsqp(rosenbrock, x0, bounds, constraints, 100, 1e-6, None);

    println!("Optimization finished.");
    println!("Status: {}", res.message);
    println!("Final x: {:?}", res.x);
    println!("Objective: {:?}", res.fun);
    println!("Iterations: {}", res.nit);

    Ok(())
}
