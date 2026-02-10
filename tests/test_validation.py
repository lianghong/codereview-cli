"""Tests for credential validation functionality."""

import os
from unittest.mock import Mock
from unittest.mock import patch

from botocore.exceptions import ClientError
from codereview.providers.base import ValidationResult
import httpx


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
        from codereview.config.models import BedrockConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
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
        from codereview.config.models import BedrockConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
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
        from codereview.config.models import BedrockConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
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

    def test_bedrock_validation_invalid_client_token(self):
        """Test validation handles InvalidClientTokenId STS error."""
        from codereview.config.models import BedrockConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
        from codereview.providers.bedrock import BedrockProvider

        with patch("codereview.providers.bedrock.ChatBedrockConverse"):
            with patch("boto3.Session") as mock_session:
                mock_creds = Mock()
                mock_session.return_value.get_credentials.return_value = mock_creds

                # Mock STS to raise InvalidClientTokenId
                mock_sts = Mock()
                mock_sts.get_caller_identity.side_effect = ClientError(
                    {
                        "Error": {
                            "Code": "InvalidClientTokenId",
                            "Message": "The security token included in the request is invalid",
                        }
                    },
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
                assert any("InvalidClientTokenId" in e for e in result.errors)
                assert any("access key" in s.lower() for s in result.suggestions)

    def test_bedrock_validation_model_found(self):
        """Test validation passes when model is found in Bedrock list."""
        from codereview.config.models import BedrockConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
        from codereview.providers.bedrock import BedrockProvider

        with patch("codereview.providers.bedrock.ChatBedrockConverse"):
            with patch("boto3.Session") as mock_session:
                mock_creds = Mock()
                mock_session.return_value.get_credentials.return_value = mock_creds

                # Mock STS success
                mock_sts = Mock()
                mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

                # Mock Bedrock list_foundation_models returning our model
                mock_bedrock = Mock()
                mock_bedrock.list_foundation_models.return_value = {
                    "modelSummaries": [
                        {"modelId": "anthropic.claude-v1"},
                        {"modelId": "test.model.v1"},
                    ]
                }

                # Return different clients based on service name
                def client_factory(service, **kwargs):
                    if service == "sts":
                        return mock_sts
                    if service == "bedrock":
                        return mock_bedrock
                    return Mock()

                mock_session.return_value.client.side_effect = client_factory

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
                assert any("Model Access" in c[0] and c[1] for c in result.checks)

    def test_bedrock_validation_model_not_found(self):
        """Test validation warns when model is not in Bedrock list."""
        from codereview.config.models import BedrockConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
        from codereview.providers.bedrock import BedrockProvider

        with patch("codereview.providers.bedrock.ChatBedrockConverse"):
            with patch("boto3.Session") as mock_session:
                mock_creds = Mock()
                mock_session.return_value.get_credentials.return_value = mock_creds

                mock_sts = Mock()
                mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

                # Mock Bedrock returning empty list (model not found)
                mock_bedrock = Mock()
                mock_bedrock.list_foundation_models.return_value = {
                    "modelSummaries": [
                        {"modelId": "anthropic.claude-v1"},
                    ]
                }

                def client_factory(service, **kwargs):
                    if service == "sts":
                        return mock_sts
                    if service == "bedrock":
                        return mock_bedrock
                    return Mock()

                mock_session.return_value.client.side_effect = client_factory

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

                # Should still be valid but with warning
                assert result.valid is True
                assert any("confirm" in w.lower() for w in result.warnings)
                assert any("enabled" in s.lower() for s in result.suggestions)

    def test_bedrock_validation_cross_region_model_id(self):
        """Test validation strips global. prefix for cross-region model matching."""
        from codereview.config.models import BedrockConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
        from codereview.providers.bedrock import BedrockProvider

        with patch("codereview.providers.bedrock.ChatBedrockConverse"):
            with patch("boto3.Session") as mock_session:
                mock_creds = Mock()
                mock_session.return_value.get_credentials.return_value = mock_creds

                mock_sts = Mock()
                mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

                # Model list has base ID (without global. prefix)
                mock_bedrock = Mock()
                mock_bedrock.list_foundation_models.return_value = {
                    "modelSummaries": [
                        {"modelId": "anthropic.claude-opus-4-6-v1"},
                    ]
                }

                def client_factory(service, **kwargs):
                    if service == "sts":
                        return mock_sts
                    if service == "bedrock":
                        return mock_bedrock
                    return Mock()

                mock_session.return_value.client.side_effect = client_factory

                # full_id uses global. prefix for cross-region inference
                model_config = ModelConfig(
                    id="opus",
                    name="Claude Opus 4.6",
                    full_id="global.anthropic.claude-opus-4-6-v1",
                    pricing=PricingConfig(
                        input_per_million=5.0, output_per_million=25.0
                    ),
                )
                provider_config = BedrockConfig(region="us-east-1")

                provider = BedrockProvider(model_config, provider_config)
                result = provider.validate_credentials()

                assert result.valid is True
                assert any("Model Access" in c[0] and c[1] for c in result.checks)

    def test_bedrock_validation_access_denied_model_check(self):
        """Test validation warns (not fails) when AccessDeniedException on list models."""
        from codereview.config.models import BedrockConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
        from codereview.providers.bedrock import BedrockProvider

        with patch("codereview.providers.bedrock.ChatBedrockConverse"):
            with patch("boto3.Session") as mock_session:
                mock_creds = Mock()
                mock_session.return_value.get_credentials.return_value = mock_creds

                mock_sts = Mock()
                mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

                # Mock Bedrock raising AccessDeniedException
                mock_bedrock = Mock()
                mock_bedrock.list_foundation_models.side_effect = ClientError(
                    {
                        "Error": {
                            "Code": "AccessDeniedException",
                            "Message": "User is not authorized",
                        }
                    },
                    "ListFoundationModels",
                )

                def client_factory(service, **kwargs):
                    if service == "sts":
                        return mock_sts
                    if service == "bedrock":
                        return mock_bedrock
                    return Mock()

                mock_session.return_value.client.side_effect = client_factory

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

                # AccessDeniedException on list_foundation_models is a warning, not failure
                assert result.valid is True
                assert any("Cannot list" in w for w in result.warnings)
                assert any("ListFoundationModels" in s for s in result.suggestions)

    def test_bedrock_validation_generic_credential_error(self):
        """Test validation handles generic exception during credential check."""
        from codereview.config.models import BedrockConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
        from codereview.providers.bedrock import BedrockProvider

        with patch("codereview.providers.bedrock.ChatBedrockConverse"):
            with patch("boto3.Session") as mock_session:
                # Simulate generic exception during credential lookup
                mock_session.return_value.get_credentials.side_effect = RuntimeError(
                    "Unexpected credential error"
                )

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
                assert any("Error checking credentials" in e for e in result.errors)


class TestAzureValidation:
    """Tests for AzureOpenAIProvider validation."""

    def test_azure_validation_placeholder_api_key(self):
        """Test validation fails when API key is a placeholder."""
        from codereview.config.models import AzureOpenAIConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
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
        from codereview.config.models import AzureOpenAIConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
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

        from codereview.config.models import AzureOpenAIConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
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

        from codereview.config.models import AzureOpenAIConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
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

    def test_azure_validation_invalid_url_format(self):
        """Test validation fails when endpoint URL has invalid format (no scheme/netloc)."""
        from codereview.config.models import AzureOpenAIConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
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
                api_key="test-key-12345678901234567890",
                api_version="2024-01-01",
            )

            provider = AzureOpenAIProvider(model_config, provider_config)

            # Override frozen field using object.__setattr__ to bypass Pydantic
            object.__setattr__(provider.provider_config, "endpoint", "not-a-url")

            result = provider.validate_credentials()

            assert result.valid is False
            assert any("Invalid endpoint URL" in e for e in result.errors)

    def test_azure_validation_http_endpoint_warning(self):
        """Test validation warns when endpoint uses HTTP instead of HTTPS."""
        from codereview.config.models import AzureOpenAIConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
        from codereview.providers.azure_openai import AzureOpenAIProvider

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

                # Override frozen field to HTTP endpoint
                object.__setattr__(
                    provider.provider_config,
                    "endpoint",
                    "http://test.openai.azure.com",
                )

                result = provider.validate_credentials()

                assert result.valid is True
                assert any("HTTPS" in w for w in result.warnings)
        finally:
            os.environ.pop("CODEREVIEW_SKIP_CONNECTION_TEST", None)

    def test_azure_validation_missing_deployment(self):
        """Test validation fails when deployment name is not set."""
        from codereview.config.models import AzureOpenAIConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
        from codereview.providers.azure_openai import AzureOpenAIProvider

        with patch("codereview.providers.azure_openai.AzureChatOpenAI"):
            model_config = ModelConfig(
                id="test",
                name="Test Model",
                # No deployment_name
                pricing=PricingConfig(input_per_million=1.0, output_per_million=5.0),
            )
            provider_config = AzureOpenAIConfig(
                endpoint="https://test.openai.azure.com",
                api_key="test-key-12345678901234567890",
                api_version="2024-01-01",
            )

            provider = AzureOpenAIProvider(model_config, provider_config)
            result = provider.validate_credentials()

            assert result.valid is False
            assert any("Deployment" in c[0] and not c[1] for c in result.checks)
            assert any("deployment_name" in s for s in result.suggestions)

    def test_azure_validation_connection_test_success(self):
        """Test connection test passes when endpoint returns 200."""
        from codereview.config.models import AzureOpenAIConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
        from codereview.providers.azure_openai import AzureOpenAIProvider

        # Ensure connection test is NOT skipped
        os.environ.pop("CODEREVIEW_SKIP_CONNECTION_TEST", None)

        with patch("codereview.providers.azure_openai.AzureChatOpenAI"):
            model_config = ModelConfig(
                id="test",
                name="Test Model",
                deployment_name="test-deployment",
                pricing=PricingConfig(input_per_million=1.0, output_per_million=5.0),
            )
            provider_config = AzureOpenAIConfig(
                endpoint="https://test.openai.azure.com",
                api_key="test-key-12345678901234567890",
                api_version="2024-01-01",
            )

            provider = AzureOpenAIProvider(model_config, provider_config)

            # Mock httpx to return 200
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            mock_client.get.return_value = mock_response

            with patch("httpx.Client", return_value=mock_client):
                result = provider.validate_credentials()

            assert result.valid is True
            assert any(
                "Connection" in c[0] and c[1] and "authenticated" in c[2]
                for c in result.checks
            )

    def test_azure_validation_connection_test_auth_error(self):
        """Test connection test still passes when endpoint returns 401."""
        from codereview.config.models import AzureOpenAIConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
        from codereview.providers.azure_openai import AzureOpenAIProvider

        os.environ.pop("CODEREVIEW_SKIP_CONNECTION_TEST", None)

        with patch("codereview.providers.azure_openai.AzureChatOpenAI"):
            model_config = ModelConfig(
                id="test",
                name="Test Model",
                deployment_name="test-deployment",
                pricing=PricingConfig(input_per_million=1.0, output_per_million=5.0),
            )
            provider_config = AzureOpenAIConfig(
                endpoint="https://test.openai.azure.com",
                api_key="test-key-12345678901234567890",
                api_version="2024-01-01",
            )

            provider = AzureOpenAIProvider(model_config, provider_config)

            # Mock httpx to return 401
            mock_response = Mock()
            mock_response.status_code = 401
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            mock_client.get.return_value = mock_response

            with patch("httpx.Client", return_value=mock_client):
                result = provider.validate_credentials()

            assert result.valid is True
            assert any(
                "Connection" in c[0] and c[1] and "reachable" in c[2]
                for c in result.checks
            )

    def test_azure_validation_connection_test_exception(self):
        """Test connection test failure produces warning, not failure."""
        from codereview.config.models import AzureOpenAIConfig
        from codereview.config.models import ModelConfig
        from codereview.config.models import PricingConfig
        from codereview.providers.azure_openai import AzureOpenAIProvider

        os.environ.pop("CODEREVIEW_SKIP_CONNECTION_TEST", None)

        with patch("codereview.providers.azure_openai.AzureChatOpenAI"):
            model_config = ModelConfig(
                id="test",
                name="Test Model",
                deployment_name="test-deployment",
                pricing=PricingConfig(input_per_million=1.0, output_per_million=5.0),
            )
            provider_config = AzureOpenAIConfig(
                endpoint="https://test.openai.azure.com",
                api_key="test-key-12345678901234567890",
                api_version="2024-01-01",
            )

            provider = AzureOpenAIProvider(model_config, provider_config)

            # Mock httpx to raise a connection error
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")

            with patch("httpx.Client", return_value=mock_client):
                result = provider.validate_credentials()

            # Should still be valid but with warning
            assert result.valid is True
            assert any("Connection test failed" in w for w in result.warnings)
            assert any("network connectivity" in s.lower() for s in result.suggestions)


class TestNVIDIAValidation:
    """Tests for NVIDIAProvider validation."""

    def test_nvidia_validation_missing_model_id(self):
        """Test validation fails when model full_id is not set."""
        from codereview.config.models import ModelConfig
        from codereview.config.models import NVIDIAConfig
        from codereview.config.models import PricingConfig
        from codereview.providers.nvidia import NVIDIAProvider

        os.environ["CODEREVIEW_SKIP_CONNECTION_TEST"] = "1"

        try:
            with patch("codereview.providers.nvidia.ChatNVIDIA"):
                model_config = ModelConfig(
                    id="test",
                    name="Test Model",
                    full_id="vendor/test-model",
                    pricing=PricingConfig(
                        input_per_million=0.0, output_per_million=0.0
                    ),
                )
                provider_config = NVIDIAConfig(api_key="nvapi-test-key-12345")

                provider = NVIDIAProvider(model_config, provider_config)

                # Override full_id to None after construction
                object.__setattr__(provider.model_config, "full_id", None)

                result = provider.validate_credentials()

                assert result.valid is False
                assert any("Model ID" in c[0] and not c[1] for c in result.checks)
                assert any("full_id" in s for s in result.suggestions)
        finally:
            os.environ.pop("CODEREVIEW_SKIP_CONNECTION_TEST", None)

    def test_nvidia_validation_connection_timeout(self):
        """Test validation warns on connection timeout."""
        from codereview.config.models import ModelConfig
        from codereview.config.models import NVIDIAConfig
        from codereview.config.models import PricingConfig
        from codereview.providers.nvidia import NVIDIAProvider

        os.environ.pop("CODEREVIEW_SKIP_CONNECTION_TEST", None)

        with patch("codereview.providers.nvidia.ChatNVIDIA"):
            model_config = ModelConfig(
                id="test",
                name="Test Model",
                full_id="vendor/test-model",
                pricing=PricingConfig(input_per_million=0.0, output_per_million=0.0),
            )
            provider_config = NVIDIAConfig(api_key="nvapi-test-key-12345")

            provider = NVIDIAProvider(model_config, provider_config)

            # Mock httpx to raise TimeoutException
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            mock_client.get.side_effect = httpx.TimeoutException("Request timed out")

            with patch("httpx.Client", return_value=mock_client):
                result = provider.validate_credentials()

            assert result.valid is True
            assert any("timed out" in w.lower() for w in result.warnings)

    def test_nvidia_validation_custom_base_url(self):
        """Test validation shows custom base URL in checks."""
        from codereview.config.models import ModelConfig
        from codereview.config.models import NVIDIAConfig
        from codereview.config.models import PricingConfig
        from codereview.providers.nvidia import NVIDIAProvider

        os.environ["CODEREVIEW_SKIP_CONNECTION_TEST"] = "1"

        try:
            with patch("codereview.providers.nvidia.ChatNVIDIA"):
                model_config = ModelConfig(
                    id="test",
                    name="Test Model",
                    full_id="vendor/test-model",
                    pricing=PricingConfig(
                        input_per_million=0.0, output_per_million=0.0
                    ),
                )
                provider_config = NVIDIAConfig(
                    api_key="nvapi-test-key-12345",
                    base_url="https://my-nim.example.com/v1",
                )

                provider = NVIDIAProvider(model_config, provider_config)
                result = provider.validate_credentials()

                assert result.valid is True
                assert any(
                    "Base URL" in c[0] and "custom endpoint" in c[2]
                    for c in result.checks
                )
        finally:
            os.environ.pop("CODEREVIEW_SKIP_CONNECTION_TEST", None)

    def test_nvidia_validation_connection_other_status(self):
        """Test validation handles non-200 non-auth status codes."""
        from codereview.config.models import ModelConfig
        from codereview.config.models import NVIDIAConfig
        from codereview.config.models import PricingConfig
        from codereview.providers.nvidia import NVIDIAProvider

        os.environ.pop("CODEREVIEW_SKIP_CONNECTION_TEST", None)

        with patch("codereview.providers.nvidia.ChatNVIDIA"):
            model_config = ModelConfig(
                id="test",
                name="Test Model",
                full_id="vendor/test-model",
                pricing=PricingConfig(input_per_million=0.0, output_per_million=0.0),
            )
            provider_config = NVIDIAConfig(api_key="nvapi-test-key-12345")

            provider = NVIDIAProvider(model_config, provider_config)

            # Mock httpx to return 503
            mock_response = Mock()
            mock_response.status_code = 503
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            mock_client.get.return_value = mock_response

            with patch("httpx.Client", return_value=mock_client):
                result = provider.validate_credentials()

            assert result.valid is True
            assert any(
                "Connection" in c[0] and c[1] and "503" in c[2] for c in result.checks
            )


class TestCLIValidation:
    """Tests for CLI --validate error paths."""

    def test_validate_unknown_model(self):
        """Test --validate with ValueError from factory exits with code 1."""
        from click.testing import CliRunner
        from codereview.cli import main

        runner = CliRunner()

        # Mock factory to raise ValueError (simulates model resolution failure)
        with patch("codereview.cli.ProviderFactory") as mock_factory_cls:
            mock_factory_cls.return_value.create_provider.side_effect = ValueError(
                "Unknown model: nonexistent-model"
            )
            result = runner.invoke(main, ["--validate", "-m", "opus"])

            assert result.exit_code == 1
            assert "Error" in result.output

    def test_validate_unexpected_error(self):
        """Test --validate handles generic Exception with exit code 1."""
        from click.testing import CliRunner
        from codereview.cli import main

        runner = CliRunner()

        with patch("codereview.cli.ProviderFactory") as mock_factory_cls:
            mock_factory_cls.return_value.create_provider.side_effect = RuntimeError(
                "Unexpected test error"
            )
            result = runner.invoke(main, ["--validate", "-m", "opus"])

            assert result.exit_code == 1
            assert "Unexpected" in result.output or "error" in result.output.lower()
