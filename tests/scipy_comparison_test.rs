mod scipy_comparison;

use scipy_comparison::run_comparison_test;
use slsqp_test_cases::create_test_cases;

#[test]
fn test_all_scipy_comparisons() {
    let test_cases = create_test_cases();

    let filter = std::env::var("TEST_CASE").ok();
    let mut run_count = 0;
    let mut failures = Vec::new();

    for test_case in test_cases {
        let name = test_case.name.clone();

        if let Some(ref f) = filter {
            if !name.contains(f) {
                continue;
            }
        }

        run_count += 1;
        println!("\n==================================================");
        println!("Running test case: {}", name);

        let (x_tol, fun_tol) = if let Some(t) = test_case.comparison_tol {
            (t, t)
        } else {
            (1e-4, 1e-4)
        };

        // 1. Run with Inexact Line Search (Default)
        println!("\n[LineSearchMode: Inexact]");
        match run_comparison_test(test_case.clone(), x_tol, fun_tol) {
            Ok(report) => {
                report.print();
                if !report.passed {
                    failures.push(format!("{} [Inexact] (results do not match)", name));
                }
            }
            Err(e) => {
                println!("Error running test {} [Inexact]: {}", name, e);
                failures.push(format!("{} [Inexact] (error: {})", name, e));
            }
        }

        println!("==================================================\n");
    }

    if let Some(ref f) = filter {
        if run_count == 0 {
            println!("Warning: No test cases matched filter '{}'", f);
        } else {
            println!("Ran {} test case(s) matching filter '{}'", run_count, f);
        }
    }

    if !failures.is_empty() {
        println!("\nFailures:");
        for failure in &failures {
            println!("  - {}", failure);
        }
        panic!("{} test(s) failed", failures.len());
    }
}
