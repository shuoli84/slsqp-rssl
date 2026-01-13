# Slsqp-rssl (slsqp rust impl coded by shuoli)

A rust impl for the slsqp solver. I tried to make the impl the same as scipy's fortran impl, but the result won't be exact same. (E.g: different numberic errors caused by differnet math impl)

# Disclamer

This migration is a mix of "vibe" and "human" effort. I can't say which part is more important.. Without AI, it will be very hard for me to figure out and migrate all the FORTRAN logic. Without human effort, the code won't be rust native (it generates lots of noises, and sometimes goes into wrong direction etc. I spend a lot time to clean it up)

This impl passed scipy comparison for HS1-89 test cases. I didn't migrate other test cases caz Cursor failed to find the problem definition by somehow. The test-cases are in hs-test-suite crate and any help is welcomed.

One more thing, in the process of migrating, I understand how the solver works in high level, but still rusty in the underlying math. So any error found, please open a pr or at least provide a re-producable test case.

# Motivation

Recently, I started to adopt Vibe coding into my daily workflow and finished 3-4 small web (frontend) apps. The process is satisfying, and I wanted to see how far I can go. Then I picked the slsqp solver :). (I regreted several times in the migrating process).

Why slsqp solver, I always thought the solver is a magic, so wanted to learn more about it. And, I needed the solver in one of my project and couldn't find a proper rust impl, so used scipy's impl through PyO3. :)

# Features

- almost same impl as scipy
- wasm build (check wasm example)
- rust native, e.g: 0 based index vs fortran 1 based index

# Demo

[rust vs scipy](https://slsqp-vis.shuo23333.app/hs_all_cases_viz?case=HS7)

[wasm example](https://slsqp-wasm.shuo23333.app)

# Unsafe usage

I tried my best to avoid unsafe, but still one unsafe in code.

# Test & Benchmark

Running tests requires a python environment with scipy installed.

```bash
# run unit test and scipy comparison test
cargo test
```

```bash
# run benchmark test for all HS test cases
cargo bench
```

```bash
# run basic example rosenbrock
cargo run --example basic
```

# Example

```rust

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

```

# How to contribute

- Code review
- Open pr and fix bugs
- Star the project or follow me

# Social links

小红书: [shuo23333](https://www.xiaohongshu.com/user/profile/62027db800000000100081a7)

# License

MIT
