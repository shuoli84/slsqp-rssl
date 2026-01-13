use criterion::{Criterion, criterion_group, criterion_main};
use slsqp_rssl::{Constraint, fmin_slsqp};
use slsqp_test_cases::{TestConstraint, create_test_cases};
use std::hint::black_box;

fn run_all_hs_cases() {
    let test_cases = create_test_cases();
    for test_case in test_cases {
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

        let _ = fmin_slsqp(
            &objective_fn,
            &test_case.x0,
            &bounds,
            rust_constraints,
            test_case.maxiter,
            test_case.tol,
            None,
        );
    }
}

fn bench_all_cases_combined(c: &mut Criterion) {
    c.bench_function("all_93_hs_cases_combined", |b| {
        b.iter(|| {
            black_box(run_all_hs_cases());
        })
    });
}

criterion_group!(benches, bench_all_cases_combined);
criterion_main!(benches);
