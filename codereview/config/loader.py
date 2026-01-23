import os
import re
from pathlib import Path
from typing import Any

import yaml

from codereview.config.models import (
    AzureOpenAIConfig,
    BedrockConfig,
    InferenceParams,
    ModelConfig,
    PricingConfig,
    ProviderConfig,
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
        self._models_by_id: dict[str, tuple[str, ModelConfig]] = (
            {}
        )  # {id: (provider, config)}
        self._providers: dict[str, ProviderConfig] = {}

        self._load_config()

    def _load_config(self) -> None:
        """Load and parse YAML configuration."""
        # Read YAML file
        with open(self.config_path) as f:
            raw_yaml = f.read()

        # Expand environment variables
        expanded_yaml = self._expand_env_vars(raw_yaml)

        # Parse YAML
        self._raw_config = yaml.safe_load(expanded_yaml)

        # Parse providers and models
        self._parse_providers()

    def _expand_env_vars(self, text: str) -> str:
        """Expand ${VAR_NAME} to environment variable values.

        Args:
            text: YAML text with ${VAR} placeholders

        Returns:
            Text with variables expanded
        """

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            value = os.environ.get(var_name, "")
            if not value:
                # For required vars like API keys, this is OK during loading
                # Validation will catch missing required values
                return ""
            return value

        return re.sub(r"\$\{([A-Z_]+)\}", replacer, text)

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
                model_id = model_config.id
                self._models_by_id[model_id] = ("bedrock", model_config)

                # Register aliases
                for alias in model_config.aliases:
                    self._models_by_id[alias] = ("bedrock", model_config)

        # Parse Azure OpenAI provider
        if "azure_openai" in providers_section:
            azure_data = providers_section["azure_openai"]

            # Only register Azure provider if credentials are present
            # This allows users to have Azure models in config without using them
            endpoint = azure_data.get("endpoint", "")
            api_key = azure_data.get("api_key", "")

            if endpoint and api_key:
                try:
                    azure_config = AzureOpenAIConfig(
                        endpoint=endpoint,
                        api_key=api_key,
                        api_version=azure_data["api_version"],
                    )
                    self._providers["azure_openai"] = azure_config

                    # Parse Azure models
                    for model_data in azure_data.get("models", []):
                        model_config = self._parse_model_config(model_data)
                        model_id = model_config.id
                        self._models_by_id[model_id] = ("azure_openai", model_config)

                        # Register aliases
                        for alias in model_config.aliases:
                            self._models_by_id[alias] = ("azure_openai", model_config)
                except Exception:
                    # Skip Azure provider if configuration is invalid
                    # Users can still use Bedrock models
                    pass

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
            available = sorted(set(k for k in self._models_by_id.keys()))
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
