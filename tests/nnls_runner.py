#!/usr/bin/env python3
import sys
import json
import numpy as np
from scipy.optimize import nnls

def solve():
    try:
        input_str = sys.stdin.read()
        if not input_str:
            return
        input_data = json.loads(input_str)
        
        a = np.array(input_data['a']).reshape(input_data['m'], input_data['n'], order='F')
        b = np.array(input_data['b'])
        
        x, rnorm = nnls(a, b)
        
        output = {
            "x": x.tolist(),
            "rnorm": float(rnorm)
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

