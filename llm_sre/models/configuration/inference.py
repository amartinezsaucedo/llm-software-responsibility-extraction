from __future__ import annotations


class InferenceConfigurationBuilder:
    _configuration = {
        "echo": True,
        "frequency_penalty": 0.0,
        "logprobs": None,
        "max_tokens": 512,
        "mirostat_mode": 0,
        "mirostat_tau": 5.0,
        "mirostat_eta": 0.1,
        "n_parts": -1,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.2,
        "stop": [],
        "stopping_criteria": None,
        "stream": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.0,
        "tfs_z": 1.0,
    }

    def set_max_tokens(self, max_tokens: int) -> InferenceConfigurationBuilder:
        self._configuration["max_tokens"] = max_tokens
        return self

    def set_repeat_penalty(self, repeat_penalty: float) -> InferenceConfigurationBuilder:
        self._configuration["repeat_penalty"] = repeat_penalty
        return self

    def set_temperature(self, temperature: float) -> InferenceConfigurationBuilder:
        self._configuration["temperature"] = temperature
        return self

    def build(self) -> dict:
        return self._configuration
