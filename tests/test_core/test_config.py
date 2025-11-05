"""Tests for GenerationConfig."""

import pytest
from batch_router.core.config import GenerationConfig


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_config_creation_with_all_params(self):
        """Test creating GenerationConfig with all parameters."""
        config = GenerationConfig(
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            top_k=40,
            stop_sequences=["END"],
            presence_penalty=0.5,
            frequency_penalty=0.3
        )
        assert config.temperature == 0.7
        assert config.max_tokens == 100
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.stop_sequences == ["END"]
        assert config.presence_penalty == 0.5
        assert config.frequency_penalty == 0.3

    def test_config_defaults(self):
        """Test GenerationConfig with default values."""
        config = GenerationConfig()
        assert config.temperature is None
        assert config.max_tokens is None
        assert config.top_p is None
        assert config.top_k is None
        assert config.stop_sequences is None
        assert config.presence_penalty is None
        assert config.frequency_penalty is None

    def test_temperature_validation(self):
        """Test temperature parameter validation."""
        # Valid values
        GenerationConfig(temperature=0.0)
        GenerationConfig(temperature=1.0)
        GenerationConfig(temperature=2.0)

        # Invalid values
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            GenerationConfig(temperature=-0.1)

        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            GenerationConfig(temperature=2.1)

    def test_max_tokens_validation(self):
        """Test max_tokens parameter validation."""
        # Valid values
        GenerationConfig(max_tokens=1)
        GenerationConfig(max_tokens=1000)

        # Invalid values
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            GenerationConfig(max_tokens=0)

        with pytest.raises(ValueError, match="max_tokens must be positive"):
            GenerationConfig(max_tokens=-1)

    def test_top_p_validation(self):
        """Test top_p parameter validation."""
        # Valid values
        GenerationConfig(top_p=0.0)
        GenerationConfig(top_p=0.5)
        GenerationConfig(top_p=1.0)

        # Invalid values
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            GenerationConfig(top_p=-0.1)

        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            GenerationConfig(top_p=1.1)

    def test_top_k_validation(self):
        """Test top_k parameter validation."""
        # Valid values
        GenerationConfig(top_k=1)
        GenerationConfig(top_k=100)

        # Invalid values
        with pytest.raises(ValueError, match="top_k must be positive"):
            GenerationConfig(top_k=0)

        with pytest.raises(ValueError, match="top_k must be positive"):
            GenerationConfig(top_k=-1)

    def test_presence_penalty_validation(self):
        """Test presence_penalty parameter validation."""
        # Valid values
        GenerationConfig(presence_penalty=-2.0)
        GenerationConfig(presence_penalty=0.0)
        GenerationConfig(presence_penalty=2.0)

        # Invalid values
        with pytest.raises(ValueError, match="presence_penalty must be between -2 and 2"):
            GenerationConfig(presence_penalty=-2.1)

        with pytest.raises(ValueError, match="presence_penalty must be between -2 and 2"):
            GenerationConfig(presence_penalty=2.1)

    def test_frequency_penalty_validation(self):
        """Test frequency_penalty parameter validation."""
        # Valid values
        GenerationConfig(frequency_penalty=-2.0)
        GenerationConfig(frequency_penalty=0.0)
        GenerationConfig(frequency_penalty=2.0)

        # Invalid values
        with pytest.raises(ValueError, match="frequency_penalty must be between -2 and 2"):
            GenerationConfig(frequency_penalty=-2.1)

        with pytest.raises(ValueError, match="frequency_penalty must be between -2 and 2"):
            GenerationConfig(frequency_penalty=2.1)

    def test_to_dict(self):
        """Test converting GenerationConfig to dictionary."""
        config = GenerationConfig(
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )
        config_dict = config.to_dict()

        assert config_dict == {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9
        }

    def test_to_dict_excludes_none(self):
        """Test that to_dict excludes None values."""
        config = GenerationConfig(temperature=0.7)
        config_dict = config.to_dict()

        assert "temperature" in config_dict
        assert "max_tokens" not in config_dict
        assert "top_p" not in config_dict
        assert "top_k" not in config_dict

    def test_to_dict_empty(self):
        """Test to_dict with all None values."""
        config = GenerationConfig()
        config_dict = config.to_dict()

        assert config_dict == {}
