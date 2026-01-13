#!/usr/bin/env python3
import json
import sys

import numpy as np
from scipy.optimize import minimize


def solve():
    try:
        # Read JSON from standard input
        input_str = sys.stdin.read()
        if not input_str:
            return
        input_data = json.loads(input_str)

        obj_expr = input_data["objective"]
        x0 = np.array(input_data["x0"])
        bounds_raw = input_data.get("bounds")
        constraints_raw = input_data.get("constraints", [])
        tol = input_data.get("tol", 1e-6)
        maxiter = input_data.get("maxiter", 100)

        # Prepare eval environment
        eval_env = {
            "np": np,
            "x": None,  # Placeholder
            "exp": np.exp,
            "sqrt": np.sqrt,
            "log": np.log,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "inf": np.inf,
            "pi": np.pi,
            "abs": np.abs,
            "sum": sum,
            "range": range,
        }

        # Convert bounds
        bounds = None
        if bounds_raw:
            bounds = []
            for b in bounds_raw:
                l = b[0] if b[0] is not None else -np.inf
                u = b[1] if b[1] is not None else np.inf
                bounds.append((l, u))

        # Define objective function
        def objective(x):
            local_env = eval_env.copy()
            local_env["x"] = x
            val = eval(obj_expr, local_env)
            return float(val)

        # Define constraints
        cons = []
        for i, c in enumerate(constraints_raw):
            ctype = c["type"]
            expr = c["expr"]

            # Use closure to capture expr
            def make_con(e):
                def con(x):
                    val = eval(e, {**eval_env, "x": x})
                    if isinstance(val, (np.ndarray, list)):
                        return np.asfarray(val)
                    return float(val)

                return con

            cons.append({"type": ctype, "fun": make_con(expr)})

        # Iteration history recording: use increasing maxiter to get precise points
        history_with_f = []
        res = None

        # We try maxiter from 0 up to the requested maxiter
        # maxiter=0 usually returns the initial point x0
        for i in range(maxiter + 1):
            curr_res = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                tol=tol,
                options={"maxiter": i, "disp": False},
            )

            # Record the point if it's the first one or different from the last recorded one
            curr_x = curr_res.x.tolist()
            if not history_with_f or curr_x != history_with_f[-1]["x"]:
                history_with_f.append({"x": curr_x, "f": float(curr_res.fun)})

            res = curr_res
            # status 9 is "Iteration limit exceeded", which means it's still working
            # If status is something else (0: success, or other errors), it has finished
            if res.status != 9:
                break

        # Compute maximum constraint violation (Max CV) for the final result
        max_cv = 0.0

        # 1. Evaluate constraint terms
        for c in cons:
            val = c["fun"](res.x)
            if c["type"] == "eq":
                max_cv = max(max_cv, abs(val))
            else:  # 'ineq' means val >= 0
                max_cv = max(max_cv, max(0.0, -val))

        # 2. Evaluate bounds
        if bounds:
            for i, (l, u) in enumerate(bounds):
                if l is not None:
                    max_cv = max(max_cv, max(0.0, l - res.x[i]))
                if u is not None:
                    max_cv = max(max_cv, max(0.0, res.x[i] - u))

        # Prepare output result
        output = {
            "success": bool(res.success),
            "x": res.x.tolist(),
            "fun": float(res.fun),
            "nit": int(res.nit),
            "status": int(res.status),
            "message": str(res.message),
            "max_cv": float(max_cv),
            "history": history_with_f,
        }

        sys.stdout.write(json.dumps(output))
        sys.stdout.flush()

    except Exception as e:
        error_output = {"error": str(e)}
        sys.stdout.write(json.dumps(error_output))
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    solve()
