from __future__ import annotations


class ModelConfigurationBuilder:
    _configuration = {
        "embedding": False,
        "f16_kv": True,
        "last_n_tokens_size": 64,
        "logits_all": False,
        "lora_base": None,
        "lora_path": None,
        "low_vram": False,
        "main_gpu": 0,
        "model_path": "./model/gguf-model.bin",
        "mul_mat_q": True,
        "n_batch": 512,
        "n_ctx": 4096,
        "n_gpu_layers": 2,
        "n_threads": None,
        "numa": False,
        "rope_freq_base": 10000.0,
        "rope_freq_scale": 1.0,
        "use_mmap": False,
        "use_mlock": True,
        "verbose": True,
        "vocab_only": False,
    }

    def set_model_path(self, model_path: str) -> ModelConfigurationBuilder:
        self._configuration["model_path"] = model_path
        return self

    def build(self) -> dict:
        return self._configuration
