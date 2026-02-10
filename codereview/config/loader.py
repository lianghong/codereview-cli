"""Configuration loader for model and provider settings.

This module provides the ConfigLoader class that:
- Loads configuration from models.yaml
- Expands environment variables (${VAR_NAME} syntax)
- Parses provider configurations (Bedrock, Azure OpenAI, NVIDIA)
- Resolves model names and aliases to their configurations
- Manages scanning configuration for file discovery

The loader is typically accessed via the get_config_loader() singleton in __init__.py.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from codereview.config.models import (
    AzureOpenAIConfig,
    BedrockConfig,
    GoogleGenAIConfig,
    InferenceParams,
    ModelConfig,
    NVIDIAConfig,
    PricingConfig,
    ProviderConfig,
    ScanningConfig,
)


class ConfigLoader:
    """Loads and manages model configuration from YAML."""

    def __init__(self, config_path: Path | None = None):
        """Initialize loader with config file path.

        Args:
            config_path: Path to models.yaml (defaults to codereview/config/models.yaml)
        """
        if config_path is None:
            config_path = Path(__file__).parent / "models.yaml"

        self.config_path = config_path
        self._raw_config: dict[str, Any] = {}
        self._models_by_id: dict[
            str, tuple[str, ModelConfig]
        ] = {}  # {id: (provider, config)}
        self._providers: dict[str, ProviderConfig] = {}
        self._scanning_config: ScanningConfig = ScanningConfig()

        self._load_config()

    def _load_config(self) -> None:
        """Load and parse YAML configuration."""
        try:
            with open(self.config_path, encoding="utf-8") as f:
                raw_yaml = f.read()

            # Parse YAML first, then expand env vars in parsed values.
            # This prevents API keys containing YAML-special characters
            # (like : # { }) from corrupting the YAML structure.
            self._raw_config = yaml.safe_load(raw_yaml) or {}
            self._expand_env_vars_in_dict(self._raw_config)

            # Parse providers and models
            self._parse_providers()

            # Parse scanning config
            self._parse_scanning_config()
        except FileNotFoundError:
            raise ValueError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except PermissionError:
            raise ValueError(
                f"Permission denied reading configuration: {self.config_path}"
            )

    def _expand_env_var_string(self, text: str) -> str:
        """Expand ${VAR_NAME} placeholders in a single string value.

        Args:
            text: String that may contain ${VAR} placeholders

        Returns:
            String with variables expanded
        """

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            value = os.environ.get(var_name, "")
            if not value:
                logging.warning("Environment variable not set: %s", var_name)
                return ""
            return value

        return re.sub(r"\$\{([A-Z_]+)\}", replacer, text)

    def _expand_env_vars_in_dict(self, data: Any) -> None:
        """Recursively expand ${VAR_NAME} in parsed YAML dict values in-place.

        This operates on the already-parsed YAML structure, so env var values
        containing YAML-special characters (: # { }) are handled safely.

        Args:
            data: Parsed YAML data (dict, list, or scalar)
        """
        if isinstance(data, dict):
            for key in data:
                value = data[key]
                if isinstance(value, str) and "${" in value:
                    data[key] = self._expand_env_var_string(value)
                elif isinstance(value, (dict, list)):
                    self._expand_env_vars_in_dict(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str) and "${" in item:
                    data[i] = self._expand_env_var_string(item)
                elif isinstance(item, (dict, list)):
                    self._expand_env_vars_in_dict(item)

    def _register_model(
        self, provider: str, model_config: "ModelConfig", name: str
    ) -> None:
        """Register a model ID or alias, warning on conflicts.

        Args:
            provider: Provider name (bedrock, azure_openai, nvidia)
            model_config: Model configuration
            name: Model ID or alias to register
        """
        if name in self._models_by_id:
            existing_provider, existing_config = self._models_by_id[name]
            if existing_provider != provider:
                logging.warning(
                    f"Model name conflict: '{name}' exists in both "
                    f"'{existing_provider}' ({existing_config.name}) and "
                    f"'{provider}' ({model_config.name}). "
                    f"Using '{existing_provider}'. To use '{provider}', "
                    f"choose a unique ID or alias."
                )
                return  # Keep first registration
        self._models_by_id[name] = (provider, model_config)

    def _parse_providers(self) -> None:
        """Parse provider configurations and models."""
        providers_section = self._raw_config.get("providers", {})

        # Parse Bedrock provider
        if "bedrock" in providers_section:
            bedrock_data = providers_section["bedrock"]
            bedrock_config = BedrockConfig(
                region=bedrock_data.get("region", "us-west-2")
            )
            self._providers["bedrock"] = bedrock_config

            # Parse Bedrock models
            for model_data in bedrock_data.get("models", []):
                model_config = self._parse_model_config(model_data)
                self._register_model("bedrock", model_config, model_config.id)

                # Register aliases
                for alias in model_config.aliases:
                    self._register_model("bedrock", model_config, alias)

        # Parse Azure OpenAI provider
        if "azure_openai" in providers_section:
            azure_data = providers_section["azure_openai"]

            # Always register Azure models for display (--list-models)
            for model_data in azure_data.get("models", []):
                model_config = self._parse_model_config(model_data)
                self._register_model("azure_openai", model_config, model_config.id)

                # Register aliases
                for alias in model_config.aliases:
                    self._register_model("azure_openai", model_config, alias)

            # Only register provider config if credentials are present
            # (required for actual API calls, not for listing models)
            endpoint = azure_data.get("endpoint", "")
            api_key = azure_data.get("api_key", "")
            api_version = azure_data.get("api_version", "")

            if endpoint and api_key and api_version:
                try:
                    azure_config = AzureOpenAIConfig(
                        endpoint=endpoint,
                        api_key=api_key,
                        api_version=api_version,
                    )
                    self._providers["azure_openai"] = azure_config
                except (KeyError, ValueError, TypeError, ValidationError) as e:
                    logging.info(
                        f"Azure OpenAI provider not configured: {e}. "
                        "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and "
                        "AZURE_OPENAI_API_VERSION environment variables to enable Azure models."
                    )

        # Parse NVIDIA provider
        if "nvidia" in providers_section:
            nvidia_data = providers_section["nvidia"]

            # Always register NVIDIA models for display (--list-models)
            for model_data in nvidia_data.get("models", []):
                model_config = self._parse_model_config(model_data)
                self._register_model("nvidia", model_config, model_config.id)

                # Register aliases
                for alias in model_config.aliases:
                    self._register_model("nvidia", model_config, alias)

            # Only register provider config if API key is present
            # (required for actual API calls, not for listing models)
            api_key = nvidia_data.get("api_key", "")

            if api_key:
                try:
                    nvidia_config = NVIDIAConfig(
                        api_key=api_key,
                        base_url=nvidia_data.get("base_url"),
                    )
                    self._providers["nvidia"] = nvidia_config
                except (KeyError, ValueError, TypeError, ValidationError) as e:
                    logging.info(
                        f"NVIDIA provider not configured: {e}. "
                        "Set NVIDIA_API_KEY environment variable to enable NVIDIA models."
                    )

        # Parse Google Generative AI provider
        if "google_genai" in providers_section:
            google_data = providers_section["google_genai"]

            # Always register Google GenAI models for display (--list-models)
            for model_data in google_data.get("models", []):
                model_config = self._parse_model_config(model_data)
                self._register_model("google_genai", model_config, model_config.id)

                # Register aliases
                for alias in model_config.aliases:
                    self._register_model("google_genai", model_config, alias)

            # Only register provider config if API key is present
            # (required for actual API calls, not for listing models)
            api_key = google_data.get("api_key", "")

            if api_key:
                try:
                    google_config = GoogleGenAIConfig(
                        api_key=api_key,
                        request_timeout=google_data.get("request_timeout", 300),
                    )
                    self._providers["google_genai"] = google_config
                except (KeyError, ValueError, TypeError, ValidationError) as e:
                    logging.info(
                        f"Google GenAI provider not configured: {e}. "
                        "Set GOOGLE_API_KEY environment variable to enable Google models."
                    )

    def _parse_scanning_config(self) -> None:
        """Parse file scanning configuration."""
        scanning_data = self._raw_config.get("scanning", {})
        if scanning_data:
            self._scanning_config = ScanningConfig(
                max_file_size_kb=scanning_data.get("max_file_size_kb", 500),
                warn_file_size_kb=scanning_data.get("warn_file_size_kb", 100),
                exclude_patterns=scanning_data.get("exclude_patterns", []),
                exclude_extensions=scanning_data.get("exclude_extensions", []),
            )

    @property
    def scanning_config(self) -> ScanningConfig:
        """Get file scanning configuration."""
        return self._scanning_config

    def _parse_model_config(self, model_data: dict[str, Any]) -> ModelConfig:
        """Parse model configuration from YAML data.

        Args:
            model_data: Raw model data from YAML

        Returns:
            Validated ModelConfig
        """
        # Parse pricing
        pricing_data = model_data["pricing"]
        pricing = PricingConfig(
            input_per_million=pricing_data["input_per_million"],
            output_per_million=pricing_data["output_per_million"],
        )

        # Parse inference params
        inference_params = None
        if "inference_params" in model_data:
            params_data = model_data["inference_params"]
            inference_params = InferenceParams(
                temperature=params_data.get("default_temperature"),
                top_p=params_data.get("default_top_p"),
                top_k=params_data.get("default_top_k"),
                max_output_tokens=params_data.get("max_output_tokens"),
                enable_thinking=params_data.get("enable_thinking"),
                clear_thinking=params_data.get("clear_thinking"),
            )

        # Create ModelConfig
        return ModelConfig(
            id=model_data["id"],
            name=model_data["name"],
            aliases=model_data.get("aliases", []),
            pricing=pricing,
            inference_params=inference_params,
            full_id=model_data.get("full_id"),
            deployment_name=model_data.get("deployment_name"),
            use_responses_api=model_data.get("use_responses_api"),
            supports_tool_use=model_data.get("supports_tool_use", True),
        )

    def resolve_model(self, name: str) -> tuple[str, ModelConfig]:
        """Resolve model name to provider and config.

        Args:
            name: Model ID or alias

        Returns:
            Tuple of (provider_name, model_config)

        Raises:
            ValueError: If model name not found
        """
        if name not in self._models_by_id:
            available = sorted(self._models_by_id.keys())
            raise ValueError(
                f"Unknown model: {name}. Available models: {', '.join(available)}"
            )

        return self._models_by_id[name]

    def get_provider_config(self, provider_name: str) -> ProviderConfig:
        """Get provider configuration.

        Args:
            provider_name: Provider name (bedrock, azure_openai)

        Returns:
            Provider configuration

        Raises:
            ValueError: If provider not found
        """
        if provider_name not in self._providers:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {', '.join(self._providers.keys())}"
            )

        return self._providers[provider_name]

    def list_models(self) -> dict[str, list[ModelConfig]]:
        """List all models grouped by provider.

        Returns:
            Dictionary mapping provider name to list of ModelConfig
        """
        result: dict[str, list[ModelConfig]] = {}

        # Collect unique model configs per provider
        seen_ids: set[tuple[str, str]] = set()  # (provider, model_id)

        for model_id, (provider, model_config) in self._models_by_id.items():
            # Only include primary IDs, not aliases
            if model_id == model_config.id:
                key = (provider, model_id)
                if key not in seen_ids:
                    seen_ids.add(key)
                    if provider not in result:
                        result[provider] = []
                    result[provider].append(model_config)

        return result

    def get_model_aliases(self) -> dict[str, str]:
        """Get all model aliases mapped to their primary IDs.

        Returns:
            Dictionary mapping alias names to primary model IDs
        """
        aliases: dict[str, str] = {}
        for model_id, (_, model_config) in self._models_by_id.items():
            if model_id != model_config.id:
                # This is an alias
                aliases[model_id] = model_config.id
            else:
                # Primary ID also serves as an alias to itself
                aliases[model_id] = model_id
        return aliases
