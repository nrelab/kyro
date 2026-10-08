use anyhow::{bail, Result};

/// Centralized, validated server configuration.
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub model_path: Option<String>,
    pub tokenizer_path: Option<String>,
    pub model_name: String,
    pub host: String,
    pub port: u16,
    /// Maximum allowed max_tokens per request.
    pub max_tokens_cap: usize,
    /// Maximum accepted prompt length in bytes.
    pub max_prompt_bytes: usize,
    /// Maximum number of messages per request.
    pub max_messages: usize,
    /// Per-request timeout for non-streaming completions.
    pub request_timeout_secs: u64,
    /// OpenTelemetry collector endpoint (e.g. http://localhost:4317).
    pub otlp_endpoint: Option<String>,
}

impl AppConfig {
    pub fn from_env_and_args() -> Result<Self> {
        let model_path =
            arg_value("--model-path").or_else(|| std::env::var("KYRO_MODEL_PATH").ok());
        let tokenizer_path =
            arg_value("--tokenizer-path").or_else(|| std::env::var("KYRO_TOKENIZER_PATH").ok());
        let model_name = arg_value("--model-name")
            .or_else(|| std::env::var("KYRO_MODEL_NAME").ok())
            .unwrap_or_else(|| "kyro".to_string());
        let host = arg_value("--host")
            .or_else(|| std::env::var("KYRO_HOST").ok())
            .unwrap_or_else(|| "0.0.0.0".to_string());
        let port = match arg_value("--port").or_else(|| std::env::var("KYRO_PORT").ok()) {
            Some(p) => p
                .parse::<u16>()
                .map_err(|_| anyhow::anyhow!("invalid port value: {:?}", p))?,
            None => 3000,
        };
        let otlp_endpoint =
            arg_value("--otlp-endpoint").or_else(|| std::env::var("KYRO_OTLP_ENDPOINT").ok());

        let cfg = Self {
            model_path,
            tokenizer_path,
            model_name,
            host,
            port,
            max_tokens_cap: parse_env_usize("KYRO_MAX_TOKENS_CAP", 4096)?,
            max_prompt_bytes: parse_env_usize("KYRO_MAX_PROMPT_BYTES", 64 * 1024)?,
            max_messages: parse_env_usize("KYRO_MAX_MESSAGES", 256)?,
            request_timeout_secs: parse_env_u64("KYRO_REQUEST_TIMEOUT_SECS", 600)?,
            otlp_endpoint,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_name.trim().is_empty() {
            bail!("model name must not be empty");
        }
        if self.max_tokens_cap == 0 {
            bail!("max_tokens_cap must be greater than zero");
        }
        if self.max_prompt_bytes == 0 {
            bail!("max_prompt_bytes must be greater than zero");
        }
        if self.max_messages == 0 {
            bail!("max_messages must be greater than zero");
        }
        if self.request_timeout_secs == 0 {
            bail!("request_timeout_secs must be greater than zero");
        }
        if let Some(p) = &self.model_path {
            if !std::path::Path::new(p).exists() {
                bail!("model path does not exist: {}", p);
            }
        }
        if let Some(t) = &self.tokenizer_path {
            if !std::path::Path::new(t).exists() {
                bail!("tokenizer path does not exist: {}", t);
            }
        }
        Ok(())
    }
}

fn parse_env_u64(name: &str, default: u64) -> Result<u64> {
    match std::env::var(name) {
        Ok(v) => v
            .parse::<u64>()
            .map_err(|_| anyhow::anyhow!("invalid {} value: {:?}", name, v)),
        Err(_) => Ok(default),
    }
}

fn parse_env_usize(name: &str, default: usize) -> Result<usize> {
    match std::env::var(name) {
        Ok(v) => v
            .parse::<usize>()
            .map_err(|_| anyhow::anyhow!("invalid {} value: {:?}", name, v)),
        Err(_) => Ok(default),
    }
}

fn arg_value(flag: &str) -> Option<String> {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if arg == flag {
            return args.next();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_model_name() {
        let mut cfg = AppConfig {
            model_path: None,
            tokenizer_path: None,
            model_name: "  ".into(),
            host: "0.0.0.0".into(),
            port: 3000,
            max_tokens_cap: 16,
            max_prompt_bytes: 1024,
            max_messages: 8,
            request_timeout_secs: 600,
            otlp_endpoint: None,
        };
        assert!(cfg.validate().is_err());
        cfg.model_name = "kyro".into();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn rejects_missing_model_path() {
        let cfg = AppConfig {
            model_path: Some("/no/such/path".into()),
            tokenizer_path: None,
            model_name: "kyro".into(),
            host: "0.0.0.0".into(),
            port: 3000,
            max_tokens_cap: 16,
            max_prompt_bytes: 1024,
            max_messages: 8,
            request_timeout_secs: 600,
            otlp_endpoint: None,
        };
        assert!(cfg.validate().is_err());
    }
}
