use serde::{Deserialize, Serialize};
use serde_json;
use slsqp_test_cases::PythonProblem;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SciPyResult {
    pub success: bool,
    pub x: Vec<f64>,
    pub fun: f64,
    pub nit: usize,
    pub status: i32,
    pub message: String,
    pub max_cv: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

fn get_cache_path(problem_json: &str) -> PathBuf {
    let mut hasher = DefaultHasher::new();
    problem_json.hash(&mut hasher);
    let hash = hasher.finish();

    let mut path = PathBuf::from("tests/scipy_cache");
    path.push(format!("scipy_{:x}.json", hash));
    path
}

pub fn run_scipy(problem: &PythonProblem) -> Result<SciPyResult, String> {
    let json_input = serde_json::to_string(problem).map_err(|e| e.to_string())?;

    // Check cache
    let cache_path = get_cache_path(&json_input);
    if cache_path.exists() {
        if let Ok(mut file) = fs::File::open(&cache_path) {
            let mut contents = String::new();
            if file.read_to_string(&mut contents).is_ok() {
                if let Ok(cached_res) = serde_json::from_str::<SciPyResult>(&contents) {
                    return Ok(cached_res);
                }
            }
        }
    }

    let mut child = Command::new("python3")
        .arg("tests/scipy_runner.py")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start python3: {}", e))?;

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    stdin
        .write_all(json_input.as_bytes())
        .map_err(|e| e.to_string())?;
    drop(stdin);

    let output = child.wait_with_output().map_err(|e| e.to_string())?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "Python script failed with status {}: {}",
            output.status, stderr
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let result: SciPyResult = serde_json::from_str(&stdout).map_err(|e| {
        format!(
            "Failed to parse Python output: {}\nOutput was: {}",
            e, stdout
        )
    })?;

    if let Some(err) = result.error {
        return Err(format!("Python script returned error: {}", err));
    }

    // Save to cache
    if let Ok(serialized) = serde_json::to_string_pretty(&result) {
        let _ = fs::create_dir_all("tests/scipy_cache");
        let _ = fs::write(&cache_path, serialized);
    }

    Ok(result)
}
