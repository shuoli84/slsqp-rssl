use serde::{Deserialize, Serialize};
use slsqp_rssl::{nnls::{nnls, }, MatView};
use std::io::Write;
use std::process::{Command, Stdio};

#[derive(Serialize, Deserialize, Debug)]
struct NnlsProblem {
    a: Vec<f64>,
    m: usize,
    n: usize,
    b: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
struct NnlsResult {
    x: Vec<f64>,
    rnorm: f64,
}

#[allow(unused)]
struct ComparisonResult {
    rust_x: Vec<f64>,
    rust_rnorm: f64,
    rust_mode: i32,
    scipy_x: Vec<f64>,
    scipy_rnorm: f64,
    matches: bool,
    max_x_diff: f64,
    rnorm_diff: f64,
}

fn run_scipy_nnls(m: usize, n: usize, a: &[f64], b: &[f64]) -> NnlsResult {
    let problem = NnlsProblem {
        a: a.to_vec(),
        m,
        n,
        b: b.to_vec(),
    };

    let json_input = serde_json::to_string(&problem).unwrap();

    let mut child = Command::new("python3")
        .arg("tests/nnls_runner.py")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start python3");

    let mut stdin = child.stdin.take().expect("Failed to open stdin");
    stdin.write_all(json_input.as_bytes()).unwrap();
    drop(stdin);

    let output = child
        .wait_with_output()
        .expect("Failed to wait for python3");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "Python script failed: {}\nInput was: m={}, n={}, a={:?}, b={:?}",
            stderr, m, n, a, b
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout)
        .map_err(|e| {
            format!(
                "Failed to parse JSON: {}\nOutput was: {}\nInput: m={}, n={}",
                e, stdout, m, n
            )
        })
        .unwrap()
}

// Compute true residual norm: ||b - A*x||
fn compute_true_residual(m: usize, n: usize, a: &[f64], b: &[f64], x: &[f64]) -> f64 {
    let mut residual = vec![0.0; m];
    for i in 0..m {
        residual[i] = b[i];
        for j in 0..n {
            residual[i] -= a[j * m + i] * x[j];
        }
    }
    residual.iter().map(|&r| r * r).sum::<f64>().sqrt()
}

fn compare_nnls_detailed(
    m: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    tolerance: f64,
    test_name: &str,
) -> ComparisonResult {
    let mut x_rust = vec![0.0; n];
    let mut rnorm_rust = 0.0;
    let mut w_rust = vec![0.0; n];
    let mut z_rust = vec![0.0; m];
    let mut index_rust = vec![0; n];

    // Save original matrix for validation
    let a_original = a.to_vec();
    let b_original = b.to_vec();

    let mut a_data = a.to_vec();
    let mut a_rust = MatView::new(m, n, &mut a_data);
    let mut b_rust = b.to_vec();

    let mode_rust = match nnls(
        &mut a_rust,
        &mut b_rust,
        &mut x_rust,
        &mut rnorm_rust,
        &mut w_rust,
        &mut z_rust,
        &mut index_rust,
    ) {
        Ok(_) => 1,
        Err(err) => err as i32,
    };

    let scipy_res = run_scipy_nnls(m, n, &a_original, &b_original);

    // Compute difference
    let mut max_x_diff = 0.0;
    for i in 0..n {
        let diff = (x_rust[i] - scipy_res.x[i]).abs();
        if diff > max_x_diff {
            max_x_diff = diff;
        }
    }
    let rnorm_diff = (rnorm_rust - scipy_res.rnorm).abs();

    let matches = max_x_diff <= tolerance && rnorm_diff <= tolerance;

    if !matches {
        println!("\n========== Test Failure: {} ==========", test_name);
        println!("Dimensions: m={}, n={}", m, n);
        println!("Matrix A (column-major):");
        for i in 0..m {
            for j in 0..n {
                print!("{:12.6} ", a_original[j * m + i]);
            }
            println!();
        }
        println!("Vector b: {:?}", b_original);
        println!("\nRust Result:");
        println!("  x: {:?}", x_rust);
        println!("  rnorm: {:.12}", rnorm_rust);
        println!("  mode: {}", mode_rust);
        println!("\nSciPy Result:");
        println!("  x: {:?}", scipy_res.x);
        println!("  rnorm: {:.12}", scipy_res.rnorm);
        println!("\nDifference:");
        println!("  max |x_rust - x_scipy|: {:.12}", max_x_diff);
        println!("  |rnorm_rust - rnorm_scipy|: {:.12}", rnorm_diff);

        // Verify constraints
        println!("\nConstraint Verification:");
        for i in 0..n {
            if x_rust[i] < -1e-10 {
                println!(
                    "  Warning: Rust x[{}] = {} < 0 (violates non-negativity constraint)",
                    i, x_rust[i]
                );
            }
            if scipy_res.x[i] < -1e-10 {
                println!(
                    "  Warning: SciPy x[{}] = {} < 0 (violates non-negativity constraint)",
                    i, scipy_res.x[i]
                );
            }
        }

        // Compute true residual norm
        let rust_true_residual = compute_true_residual(m, n, &a_original, &b_original, &x_rust);
        let scipy_true_residual =
            compute_true_residual(m, n, &a_original, &b_original, &scipy_res.x);
        println!("  Computed true residual norm (||b - A*x||):");
        println!(
            "    Rust: {:.12} (reported: {:.12}, diff: {:.12})",
            rust_true_residual,
            rnorm_rust,
            (rust_true_residual - rnorm_rust).abs()
        );
        println!(
            "    SciPy: {:.12} (reported: {:.12}, diff: {:.12})",
            scipy_true_residual,
            scipy_res.rnorm,
            (scipy_true_residual - scipy_res.rnorm).abs()
        );

        // Check if residual calculation is correct
        if (rust_true_residual - rnorm_rust).abs() > 1e-6 {
            println!(
                "  ⚠️  Warning: Rust's reported residual norm does not match the true residual norm!"
            );
        }
        if (scipy_true_residual - scipy_res.rnorm).abs() > 1e-6 {
            println!(
                "  ⚠️  Warning: SciPy's reported residual norm does not match the true residual norm!"
            );
        }

        // Compare quality of solutions
        if rust_true_residual > scipy_true_residual + 1e-6 {
            println!(
                "  ⚠️  Warning: SciPy found a solution with smaller residual (diff: {:.12})",
                rust_true_residual - scipy_true_residual
            );
        } else if scipy_true_residual > rust_true_residual + 1e-6 {
            println!(
                "  ℹ️  Info: Rust found a solution with smaller residual (diff: {:.12})",
                scipy_true_residual - rust_true_residual
            );
        }
        println!("========================================\n");
    }

    ComparisonResult {
        rust_x: x_rust,
        rust_rnorm: rnorm_rust,
        rust_mode: mode_rust,
        scipy_x: scipy_res.x,
        scipy_rnorm: scipy_res.rnorm,
        matches,
        max_x_diff,
        rnorm_diff,
    }
}

// Test 1: Basic identity matrix
#[test]
fn test_01_identity_matrix() {
    let m = 3;
    let n = 3;
    let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let b = vec![1.0, 2.0, 3.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-8, "Identity Matrix");
    assert!(result.matches, "Identity Matrix test failed");
}

// Test 2: Single row matrix (m=1)
#[test]
fn test_02_single_row() {
    let m = 1;
    let n = 3;
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![6.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-8, "Single Row Matrix");
    assert!(result.matches, "Single Row Matrix test failed");
}

// Test 3: Single column matrix (n=1)
#[test]
fn test_03_single_column() {
    let m = 3;
    let n = 1;
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-8, "Single Column Matrix");
    assert!(result.matches, "Single Column Matrix test failed");
}

// Test 4: 1x1 matrix
#[test]
fn test_04_1x1_matrix() {
    let m = 1;
    let n = 1;
    let a = vec![2.0];
    let b = vec![4.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-8, "1x1 Matrix");
    assert!(result.matches, "1x1 Matrix test failed");
}

// Test 5: Overdetermined system (m > n)
#[test]
fn test_05_overdetermined() {
    let m = 5;
    let n = 2;
    let a = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-8, "Overdetermined System");
    assert!(result.matches, "Overdetermined System test failed");
}

// Test 6: Underdetermined system (m < n)
#[test]
fn test_06_underdetermined() {
    let m = 2;
    let n = 4;
    let a = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let b = vec![1.0, 2.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-8, "Underdetermined System");
    assert!(result.matches, "Underdetermined System test failed");
}

// Test 7: Zero matrix
#[test]
fn test_07_zero_matrix() {
    let m = 3;
    let n = 3;
    let a = vec![0.0; 9];
    let b = vec![1.0, 2.0, 3.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-8, "Zero Matrix");
    assert!(result.matches, "Zero Matrix test failed");
}

// Test 8: Zero vector b
#[test]
fn test_08_zero_b() {
    let m = 3;
    let n = 3;
    let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let b = vec![0.0; 3];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-8, "Zero Vector b");
    assert!(result.matches, "Zero Vector b test failed");
}

// Test 9: Linearly dependent columns
#[test]
fn test_09_linear_dependent_columns() {
    let m = 3;
    let n = 3;
    let a = vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let b = vec![1.0, 2.0, 3.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-8, "Linearly Dependent Columns");
    assert!(result.matches, "Linearly Dependent Columns test failed");
}

// Test 10: Ill-conditioned matrix (Hilbert matrix)
#[test]
fn test_10_ill_conditioned_hilbert() {
    let m = 4;
    let n = 4;
    // 4x4 Hilbert matrix
    let a = vec![
        1.0, 0.5, 0.333333, 0.25, 0.5, 0.333333, 0.25, 0.2, 0.333333, 0.25, 0.2, 0.166667, 0.25,
        0.2, 0.166667, 0.142857,
    ];
    let b = vec![1.0, 1.0, 1.0, 1.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-6, "Ill-conditioned Matrix (Hilbert)");
    assert!(result.matches, "Ill-conditioned Matrix test failed");
}

// Test 11: Large values
#[test]
fn test_11_large_values() {
    let m = 3;
    let n = 3;
    let a = vec![1e10, 0.0, 0.0, 0.0, 1e10, 0.0, 0.0, 0.0, 1e10];
    let b = vec![1e10, 2e10, 3e10];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-2, "Large Values");
    assert!(result.matches, "Large Values test failed");
}

// Test 12: Small values
#[test]
fn test_12_small_values() {
    let m = 3;
    let n = 3;
    let a = vec![1e-10, 0.0, 0.0, 0.0, 1e-10, 0.0, 0.0, 0.0, 1e-10];
    let b = vec![1e-10, 2e-10, 3e-10];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-8, "Small Values");
    assert!(result.matches, "Small Values test failed");
}

// Test 13: Negative vector b (requires zero solution)
#[test]
fn test_13_negative_b() {
    let m = 3;
    let n = 3;
    let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let b = vec![-1.0, -2.0, -3.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-8, "Negative Vector b");
    assert!(result.matches, "Negative Vector b test failed");
}

// Test 14: Mixed positive and negative matrix elements
#[test]
fn test_14_mixed_signs() {
    let m = 3;
    let n = 3;
    let a = vec![1.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 1.0];
    let b = vec![1.0, 2.0, 3.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-8, "Mixed Positive/Negative Elements");
    assert!(
        result.matches,
        "Mixed Positive/Negative Elements test failed"
    );
}

// Test 15: Case requiring column deletion
#[test]
fn test_15_column_deletion() {
    let m = 3;
    let n = 3;
    // Design a case where a column needs to be iteratively deleted
    let a = vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    let b = vec![1.0, -1.0, 1.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-8, "Column Deletion Case");
    assert!(result.matches, "Column Deletion Case test failed");
}

// Test 17: Boundary case - all elements are the same
#[test]
fn test_17_constant_matrix() {
    let m = 3;
    let n = 3;
    let a = vec![1.0; 9];
    let b = vec![1.0, 1.0, 1.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-8, "Constant Matrix");
    assert!(result.matches, "Constant Matrix test failed");
}

// Test 18: Sparse matrix
#[test]
fn test_18_sparse_matrix() {
    let m = 5;
    let n = 5;
    let mut a = vec![0.0; 25];
    a[0] = 1.0; // (0,0)
    a[6] = 1.0; // (1,1)
    a[12] = 1.0; // (2,2)
    a[18] = 1.0; // (3,3)
    a[24] = 1.0; // (4,4)
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-8, "Sparse Matrix");
    assert!(result.matches, "Sparse Matrix test failed");
}

// Test 19: Near-singular matrix
#[test]
fn test_19_near_singular() {
    let m = 3;
    let n = 3;
    let a = vec![1.0, 1.0, 1.0, 1.0, 1.000001, 1.0, 1.0, 1.0, 1.0];
    let b = vec![1.0, 2.0, 3.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-6, "Near-singular Matrix");

    // For near-singular matrices, the solution may not be unique.
    // If the residual norms match, it means both found an optimal solution.
    if !result.matches && result.rnorm_diff < 1e-10 {
        println!(
            "  ℹ️  Info: Solution for near-singular matrix is not unique, but residual norms match perfectly. Considered passed."
        );
        return;
    }
    assert!(result.matches, "Near-singular Matrix test failed");
}

// Test 20: Extreme aspect ratio
#[test]
fn test_20_extreme_aspect_ratio() {
    // Very wide
    let m = 2;
    let n = 10;
    let mut a = vec![0.0; m * n];
    for j in 0..n {
        a[j * m] = (j + 1) as f64;
        a[j * m + 1] = (j + 1) as f64 * 2.0;
    }
    let b = vec![10.0, 20.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-6, "Extreme Aspect Ratio (Wide)");
    assert!(result.matches, "Extreme Aspect Ratio (Wide) test failed");

    // Very tall
    let m = 10;
    let n = 2;
    let mut a = vec![0.0; m * n];
    for i in 0..m {
        a[i] = (i + 1) as f64;
        a[m + i] = (i + 1) as f64 * 2.0;
    }
    let b: Vec<f64> = (1..=m).map(|i| i as f64).collect();
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-6, "Extreme Aspect Ratio (Tall)");
    assert!(result.matches, "Extreme Aspect Ratio (Tall) test failed");
}

// Test 21: Numerical precision boundaries
#[test]
fn test_21_precision_boundary() {
    let m = 3;
    let n = 3;
    // Use values close to machine precision
    let eps = 1e-15;
    let a = vec![
        1.0 + eps,
        0.0,
        0.0,
        0.0,
        1.0 + eps,
        0.0,
        0.0,
        0.0,
        1.0 + eps,
    ];
    let b = vec![1.0, 2.0, 3.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-6, "Numerical Precision Boundary");
    assert!(result.matches, "Numerical Precision Boundary test failed");
}

// Test 22: Case requiring multiple iterations
#[test]
fn test_22_multiple_iterations() {
    let m = 5;
    let n = 5;
    // Design a matrix that requires multiple rounds of iteration
    let a = vec![
        1.0, 0.5, 0.3, 0.2, 0.1, 0.5, 1.0, 0.5, 0.3, 0.2, 0.3, 0.5, 1.0, 0.5, 0.3, 0.2, 0.3, 0.5,
        1.0, 0.5, 0.1, 0.2, 0.3, 0.5, 1.0,
    ];
    let b = vec![1.0, -1.0, 2.0, -2.0, 3.0];
    let result = compare_nnls_detailed(m, n, &a, &b, 1e-6, "Multiple Iterations");
    assert!(result.matches, "Multiple Iterations test failed");
}
