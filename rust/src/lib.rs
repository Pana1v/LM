use ndarray::{Array1, Array2};

pub mod lib {
    use ndarray::{Array1, Array2};
    
    pub fn levenberg_marquardt(
    data: &Array2<f64>,
    initial_params: &Array1<f64>,
    max_iterations: usize,
    epsilon: f64,
    lambda_init: f64,
) -> (Array1<f64>, usize, f64) {
    let num_params = initial_params.len();
    let mut params = initial_params.clone();
    let mut error = f64::INFINITY;
    let mut iteration = 0;
    let mut lambda_val = lambda_init;

    while error > epsilon && iteration < max_iterations {
        let n = data.nrows();
        let mut J = Array2::<f64>::zeros((n, num_params));
        let mut r = Array1::<f64>::zeros(n);

        for i in 0..n {
            let x = data[[i, 0]];
            let y = data[[i, 1]];
            let f = (-params[0] * x).exp() * (params[1] * x).cos();

            r[i] = y - f;

            J[[i, 0]] = x * f;
            J[[i, 1]] = -x * (-params[0] * x).exp() * (params[1] * x).sin();
        }

        // Normal equations: (J^T * J + lambda * I) * delta = J^T * r
        let Jt = J.t();
        let A = Jt.dot(&J) + lambda_val * Array2::<f64>::eye(num_params);
        let b = Jt.dot(&r);

        // Solve linear system using Cholesky decomposition
        let delta = solve_ldlt(&A, &b);
        let new_params = &params + &delta;

        // Compute new error
        let mut new_r = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x = data[[i, 0]];
            let y = data[[i, 1]];
            let new_f = (-new_params[0] * x).exp() * (new_params[1] * x).cos();
            new_r[i] = y - new_f;
        }

        let new_error: f64 = new_r.mapv(|x| x * x).sum();

        if new_error < error {
            lambda_val /= 10.0;
            error = new_error;
            params = new_params;
        } else {
            lambda_val *= 10.0;
        }

        iteration += 1;
    }

    (params, iteration, error)
}

fn solve_ldlt(A: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = A.nrows();
    let mut L = Array2::<f64>::eye(n);
    let mut D = Array1::<f64>::zeros(n);

    // LDLT decomposition
    for j in 0..n {
        let mut sum_val = 0.0;
        for k in 0..j {
            sum_val += L[[j, k]] * L[[j, k]] * D[k];
        }
        D[j] = A[[j, j]] - sum_val;

        if D[j].abs() < 1e-10 {
            panic!("Matrix is singular or near-singular");
        }

        for i in (j + 1)..n {
            let mut sum_val = 0.0;
            for k in 0..j {
                sum_val += L[[i, k]] * L[[j, k]] * D[k];
            }
            L[[i, j]] = (A[[i, j]] - sum_val) / D[j];
        }
    }

    // Forward substitution: L * y = b
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum_val = 0.0;
        for k in 0..i {
            sum_val += L[[i, k]] * y[k];
        }
        y[i] = (b[i] - sum_val) / L[[i, i]];
    }

    // Diagonal solve: D * z = y
    let z: Array1<f64> = y.iter().zip(D.iter()).map(|(yi, di)| yi / di).collect();

    // Backward substitution: L^T * x = z
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum_val = 0.0;
        for k in (i + 1)..n {
            sum_val += L[[k, i]] * x[k];
        }
        x[i] = z[i] - sum_val;
    }

    x
}
}

