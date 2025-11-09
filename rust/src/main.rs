use levenberg_marquardt::lib::levenberg_marquardt;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader};
use std::time::Instant;

#[derive(Serialize, Deserialize)]
struct BenchmarkResults {
    execution_time_ms: f64,
    peak_memory_kb: f64,
    iterations: usize,
    final_error: f64,
    initial_params: Vec<f64>,
    final_params: Vec<f64>,
}

fn load_data_from_csv(filename: &str) -> Result<Array2<f64>, std::io::Error> {
    let file = fs::File::open(filename)?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 {
            continue; // Skip header
        }
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            let x: f64 = parts[0].trim().parse().expect("Failed to parse x");
            let y: f64 = parts[1].trim().parse().expect("Failed to parse y");
            data.push(vec![x, y]);
        }
    }

    let n = data.len();
    let mut array = Array2::<f64>::zeros((n, 2));
    for (i, row) in data.iter().enumerate() {
        array[[i, 0]] = row[0];
        array[[i, 1]] = row[1];
    }
    Ok(array)
}

fn generate_data() -> Array2<f64> {
    use rand::distributions::Distribution;
    use rand_distr::Normal;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let num_data_points = 100;
    let true_params = [0.5, 1.0];
    let noise_std = 0.1;
    let mut rng = StdRng::seed_from_u64(42);

    let normal = Normal::new(0.0, noise_std).unwrap();
    let mut array = Array2::<f64>::zeros((num_data_points, 2));

    for i in 0..num_data_points {
        let x = i as f64 / (num_data_points - 1) as f64;
        let y_true = (-true_params[0] * x).exp() * (true_params[1] * x).cos();
        let noise = normal.sample(&mut rng);
        array[[i, 0]] = x;
        array[[i, 1]] = y_true + noise;
    }

    array
}

fn main() {
    // Load or generate data
    let data = match load_data_from_csv("data/test_data.csv") {
        Ok(d) => d,
        Err(_) => {
            println!("Data file not found. Generating data...");
            generate_data()
        }
    };

    let initial_params = Array1::from_vec(vec![0.5, 1.0]);
    let max_iterations = 1000;
    let epsilon = 0.000001;
    let lambda_init = 0.01;

    // Measure execution time
    let start = Instant::now();
    let (final_params, iterations, final_error) = levenberg_marquardt(
        &data,
        &initial_params,
        max_iterations,
        epsilon,
        lambda_init,
    );
    let duration = start.elapsed();

    let execution_time_ms = duration.as_secs_f64() * 1000.0;

    // Memory usage - try to get actual peak memory on Windows
    let peak_memory_kb = get_peak_memory_kb() as f64;

    let results = BenchmarkResults {
        execution_time_ms,
        peak_memory_kb,
        iterations,
        final_error,
        initial_params: initial_params.to_vec(),
        final_params: final_params.to_vec(),
    };

    // Print results
    println!("Rust Results:");
    println!("Initial parameters: {:?}", results.initial_params);
    println!("Final parameters: {:?}", results.final_params);
    println!("Iterations: {}", results.iterations);
    println!("Execution time: {:.4} ms", results.execution_time_ms);
    println!("Peak memory: {:.2} KB", results.peak_memory_kb);
    println!("Final error: {:.10}", results.final_error);

    // Save to JSON
    let json = serde_json::to_string_pretty(&results).expect("Failed to serialize results");
    fs::write("results_rust.json", json).expect("Failed to write results file");
}

#[cfg(windows)]
fn get_peak_memory_kb() -> usize {
    use std::mem;
    use winapi::um::processthreadsapi::GetCurrentProcess;
    use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
    
    unsafe {
        let handle = GetCurrentProcess();
        let mut pmc: PROCESS_MEMORY_COUNTERS_EX = mem::zeroed();
        let size = mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        
        if GetProcessMemoryInfo(handle, &mut pmc as *mut _ as *mut _, size) != 0 {
            (pmc.PeakWorkingSetSize as usize) / 1024
        } else {
            // Fallback estimate
            estimate_memory_usage(100, 2)
        }
    }
}

#[cfg(not(windows))]
fn get_peak_memory_kb() -> usize {
    use libc::{getrusage, rusage, RUSAGE_SELF};
    
    unsafe {
        let mut usage: rusage = std::mem::zeroed();
        if getrusage(RUSAGE_SELF, &mut usage) == 0 {
            usage.ru_maxrss as usize
        } else {
            estimate_memory_usage(100, 2)
        }
    }
}

fn estimate_memory_usage(n: usize, num_params: usize) -> usize {
    // Rough estimate: data + Jacobian + matrices + vectors
    let data_size = n * 2 * 8; // 8 bytes per f64
    let jacobian_size = n * num_params * 8;
    let matrix_size = num_params * num_params * 8;
    let vector_size = num_params * 8;
    (data_size + jacobian_size + matrix_size * 3 + vector_size * 5) / 1024
}

