"""Tests for credential validation functionality."""

from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

from codereview.providers.base import ValidationResult


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test basic ValidationResult creation."""
        result = ValidationResult(valid=True, provider="Test Provider")

        assert result.valid is True
        assert result.provider == "Test Provider"
        assert result.checks == []
        assert result.errors == []
        assert result.warnings == []
        assert result.suggestions == []

    def test_add_passing_check(self):
        """Test adding a passing check."""
        result = ValidationResult(valid=True, provider="Test")
        result.add_check("Test Check", True, "Check passed")

        assert len(result.checks) == 1
        assert result.checks[0] == ("Test Check", True, "Check passed")
        assert len(result.errors) == 0  # Passing checks don't add errors

    def test_add_failing_check(self):
        """Test adding a failing check adds to errors."""
        result = ValidationResult(valid=True, provider="Test")
        result.add_check("Test Check", False, "Check failed")

        assert len(result.checks) == 1
        assert result.checks[0] == ("Test Check", False, "Check failed")
        assert len(result.errors) == 1
        assert result.errors[0] == "Check failed"

    def test_add_warning(self):
        """Test adding a warning."""
        result = ValidationResult(valid=True, provider="Test")
        result.add_warning("This is a warning")

        assert len(result.warnings) == 1
        assert result.warnings[0] == "This is a warning"

    def test_add_suggestion(self):
        """Test adding a suggestion."""
        result = ValidationResult(valid=True, provider="Test")
        result.add_suggestion("Try this fix")

        assert len(result.suggestions) == 1
        assert result.suggestions[0] == "Try this fix"


class TestBedrockValidation:
    """Tests for BedrockProvider validation."""

    def test_bedrock_validation_no_credentials(self):
        """Test validation fails when no AWS credentials."""
        from codereview.config.models import BedrockConfig, ModelConfig, PricingConfig
        from codereview.providers.bedrock import BedrockProvider

        with patch("codereview.providers.bedrock.ChatBedrockConverse"):
            with patch("boto3.Session") as mock_session:
                # Simulate no credentials
                mock_session.return_value.get_credentials.return_value = None

                model_config = ModelConfig(
                    id="test",
                    name="Test Model",
                    full_id="test.model.v1",
                    pricing=PricingConfig(
                        input_per_million=1.0, output_per_million=5.0
                    ),
                )
                provider_config = BedrockConfig(region="us-east-1")

                provider = BedrockProvider(model_config, provider_config)
                result = provider.validate_credentials()

                assert result.valid is False
                assert result.provider == "AWS Bedrock"
                assert any("No AWS credentials found" in e for e in result.errors)
                assert len(result.suggestions) > 0

    def test_bedrock_validation_valid_credentials(self):
        """Test validation passes with valid credentials."""
        from codereview.config.models import BedrockConfig, ModelConfig, PricingConfig
        from codereview.providers.bedrock import BedrockProvider

        with patch("codereview.providers.bedrock.ChatBedrockConverse"):
            with patch("boto3.Session") as mock_session:
                # Simulate valid credentials
                mock_creds = Mock()
                mock_session.return_value.get_credentials.return_value = mock_creds

                # Mock STS
                mock_sts = Mock()
                mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
                mock_session.return_value.client.return_value = mock_sts

                model_config = ModelConfig(
                    id="test",
                    name="Test Model",
                    full_id="test.model.v1",
                    pricing=PricingConfig(
                        input_per_million=1.0, output_per_million=5.0
                    ),
                )
                provider_config = BedrockConfig(region="us-east-1")

                provider = BedrockProvider(model_config, provider_config)
                result = provider.validate_credentials()

                assert result.valid is True
                assert result.provider == "AWS Bedrock"
                assert any("AWS Credentials" in c[0] and c[1] for c in result.checks)
                assert any("AWS Identity" in c[0] and c[1] for c in result.checks)

    def test_bedrock_validation_expired_token(self):
        """Test validation handles expired token."""
        from codereview.config.models import BedrockConfig, ModelConfig, PricingConfig
        from codereview.providers.bedrock import BedrockProvider

        with patch("codereview.providers.bedrock.ChatBedrockConverse"):
            with patch("boto3.Session") as mock_session:
                mock_creds = Mock()
                mock_session.return_value.get_credentials.return_value = mock_creds

                # Mock STS to raise ExpiredToken
                mock_sts = Mock()
                mock_sts.get_caller_identity.side_effect = ClientError(
                    {"Error": {"Code": "ExpiredToken", "Message": "Token expired"}},
                    "GetCallerIdentity",
                )
                mock_session.return_value.client.return_value = mock_sts

                model_config = ModelConfig(
                    id="test",
                    name="Test Model",
                    full_id="test.model.v1",
                    pricing=PricingConfig(
                        input_per_million=1.0, output_per_million=5.0
                    ),
                )
                provider_config = BedrockConfig(region="us-east-1")

                provider = BedrockProvider(model_config, provider_config)
                result = provider.validate_credentials()

                assert result.valid is False
                assert any("ExpiredToken" in e for e in result.errors)
                assert any("expired" in s.lower() for s in result.suggestions)


class TestAzureValidation:
    """Tests for AzureOpenAIProvider validation."""

    def test_azure_validation_placeholder_api_key(self):
        """Test validation fails when API key is a placeholder."""
        from codereview.config.models import (
            AzureOpenAIConfig,
            ModelConfig,
            PricingConfig,
        )
        from codereview.providers.azure_openai import AzureOpenAIProvider

        with patch("codereview.providers.azure_openai.AzureChatOpenAI"):
            model_config = ModelConfig(
                id="test",
                name="Test Model",
                deployment_name="test-deployment",
                pricing=PricingConfig(input_per_million=1.0, output_per_million=5.0),
            )
            provider_config = AzureOpenAIConfig(
                endpoint="https://test.openai.azure.com",
                api_key="your-api-key-here",  # Placeholder key
                api_version="2024-01-01",
            )

            provider = AzureOpenAIProvider(model_config, provider_config)
            result = provider.validate_credentials()

            assert result.valid is False
            assert result.provider == "Azure OpenAI"
            assert any("api key" in e.lower() for e in result.errors)

    def test_azure_validation_placeholder_endpoint(self):
        """Test validation fails when endpoint is a placeholder."""
        from codereview.config.models import (
            AzureOpenAIConfig,
            ModelConfig,
            PricingConfig,
        )
        from codereview.providers.azure_openai import AzureOpenAIProvider

        with patch("codereview.providers.azure_openai.AzureChatOpenAI"):
            model_config = ModelConfig(
                id="test",
                name="Test Model",
                deployment_name="test-deployment",
                pricing=PricingConfig(input_per_million=1.0, output_per_million=5.0),
            )
            provider_config = AzureOpenAIConfig(
                endpoint="https://your-resource.openai.azure.com",  # Placeholder
                api_key="test-key-12345678901234567890",
                api_version="2024-01-01",
            )

            provider = AzureOpenAIProvider(model_config, provider_config)
            result = provider.validate_credentials()

            assert result.valid is False
            assert any("endpoint" in e.lower() for e in result.errors)

    def test_azure_validation_valid_config(self):
        """Test validation passes with valid config."""
        import os

        from codereview.config.models import (
            AzureOpenAIConfig,
            ModelConfig,
            PricingConfig,
        )
        from codereview.providers.azure_openai import AzureOpenAIProvider

        # Skip connection test
        os.environ["CODEREVIEW_SKIP_CONNECTION_TEST"] = "1"

        try:
            with patch("codereview.providers.azure_openai.AzureChatOpenAI"):
                model_config = ModelConfig(
                    id="test",
                    name="Test Model",
                    deployment_name="test-deployment",
                    pricing=PricingConfig(
                        input_per_million=1.0, output_per_million=5.0
                    ),
                )
                provider_config = AzureOpenAIConfig(
                    endpoint="https://test.openai.azure.com",
                    api_key="test-key-12345678901234567890",
                    api_version="2024-01-01",
                )

                provider = AzureOpenAIProvider(model_config, provider_config)
                result = provider.validate_credentials()

                assert result.valid is True
                assert result.provider == "Azure OpenAI"
                assert any("API Key" in c[0] and c[1] for c in result.checks)
                assert any("Endpoint" in c[0] and c[1] for c in result.checks)
                assert any("Deployment" in c[0] and c[1] for c in result.checks)
        finally:
            os.environ.pop("CODEREVIEW_SKIP_CONNECTION_TEST", None)

    def test_azure_validation_short_api_key_warning(self):
        """Test validation warns when API key seems too short."""
        import os

        from codereview.config.models import (
            AzureOpenAIConfig,
            ModelConfig,
            PricingConfig,
        )
        from codereview.providers.azure_openai import AzureOpenAIProvider

        # Skip connection test
        os.environ["CODEREVIEW_SKIP_CONNECTION_TEST"] = "1"

        try:
            with patch("codereview.providers.azure_openai.AzureChatOpenAI"):
                model_config = ModelConfig(
                    id="test",
                    name="Test Model",
                    deployment_name="test-deployment",
                    pricing=PricingConfig(
                        input_per_million=1.0, output_per_million=5.0
                    ),
                )
                provider_config = AzureOpenAIConfig(
                    endpoint="https://test.openai.azure.com",
                    api_key="short-key",  # Valid but suspiciously short
                    api_version="2024-01-01",
                )

                provider = AzureOpenAIProvider(model_config, provider_config)
                result = provider.validate_credentials()

                # Should still be valid but with warning
                assert result.valid is True
                assert any("short" in w.lower() for w in result.warnings)
        finally:
            os.environ.pop("CODEREVIEW_SKIP_CONNECTION_TEST", None)
