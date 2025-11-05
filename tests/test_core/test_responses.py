"""Tests for response structures."""

import pytest
from batch_router.core.responses import RequestCounts, BatchStatusResponse, UnifiedResult
from batch_router.core.enums import BatchStatus, ResultStatus


class TestRequestCounts:
    """Tests for RequestCounts dataclass."""

    def test_request_counts_creation(self):
        """Test creating RequestCounts."""
        counts = RequestCounts(
            total=100,
            processing=20,
            succeeded=70,
            errored=8,
            cancelled=1,
            expired=1
        )
        assert counts.total == 100
        assert counts.processing == 20
        assert counts.succeeded == 70
        assert counts.errored == 8
        assert counts.cancelled == 1
        assert counts.expired == 1

    def test_request_counts_defaults(self):
        """Test RequestCounts with default values."""
        counts = RequestCounts(total=100)
        assert counts.total == 100
        assert counts.processing == 0
        assert counts.succeeded == 0
        assert counts.errored == 0
        assert counts.cancelled == 0
        assert counts.expired == 0

    def test_is_complete_true(self):
        """Test is_complete returns True when no processing requests."""
        counts = RequestCounts(total=100, processing=0, succeeded=100)
        assert counts.is_complete() is True

    def test_is_complete_false(self):
        """Test is_complete returns False when requests are processing."""
        counts = RequestCounts(total=100, processing=50, succeeded=50)
        assert counts.is_complete() is False

    def test_success_rate(self):
        """Test success_rate calculation."""
        counts = RequestCounts(total=100, succeeded=75, errored=25)
        assert counts.success_rate() == 75.0

    def test_success_rate_zero_total(self):
        """Test success_rate with zero total."""
        counts = RequestCounts(total=0)
        assert counts.success_rate() == 0.0

    def test_success_rate_all_succeeded(self):
        """Test success_rate with all succeeded."""
        counts = RequestCounts(total=100, succeeded=100)
        assert counts.success_rate() == 100.0

    def test_success_rate_none_succeeded(self):
        """Test success_rate with none succeeded."""
        counts = RequestCounts(total=100, errored=100)
        assert counts.success_rate() == 0.0


class TestBatchStatusResponse:
    """Tests for BatchStatusResponse dataclass."""

    def test_batch_status_response_creation(self):
        """Test creating BatchStatusResponse."""
        counts = RequestCounts(total=100, succeeded=100)
        response = BatchStatusResponse(
            batch_id="batch_123",
            provider="openai",
            status=BatchStatus.COMPLETED,
            request_counts=counts,
            created_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T01:00:00Z"
        )
        assert response.batch_id == "batch_123"
        assert response.provider == "openai"
        assert response.status == BatchStatus.COMPLETED
        assert response.request_counts.total == 100
        assert response.created_at == "2024-01-01T00:00:00Z"
        assert response.completed_at == "2024-01-01T01:00:00Z"

    def test_batch_status_response_defaults(self):
        """Test BatchStatusResponse with default values."""
        counts = RequestCounts(total=100)
        response = BatchStatusResponse(
            batch_id="batch_123",
            provider="openai",
            status=BatchStatus.IN_PROGRESS,
            request_counts=counts,
            created_at="2024-01-01T00:00:00Z"
        )
        assert response.completed_at is None
        assert response.expires_at is None
        assert response.provider_data == {}

    def test_is_complete_completed(self):
        """Test is_complete for COMPLETED status."""
        counts = RequestCounts(total=100, succeeded=100)
        response = BatchStatusResponse(
            batch_id="batch_123",
            provider="openai",
            status=BatchStatus.COMPLETED,
            request_counts=counts,
            created_at="2024-01-01T00:00:00Z"
        )
        assert response.is_complete() is True

    def test_is_complete_failed(self):
        """Test is_complete for FAILED status."""
        counts = RequestCounts(total=100, errored=100)
        response = BatchStatusResponse(
            batch_id="batch_123",
            provider="openai",
            status=BatchStatus.FAILED,
            request_counts=counts,
            created_at="2024-01-01T00:00:00Z"
        )
        assert response.is_complete() is True

    def test_is_complete_cancelled(self):
        """Test is_complete for CANCELLED status."""
        counts = RequestCounts(total=100, cancelled=100)
        response = BatchStatusResponse(
            batch_id="batch_123",
            provider="openai",
            status=BatchStatus.CANCELLED,
            request_counts=counts,
            created_at="2024-01-01T00:00:00Z"
        )
        assert response.is_complete() is True

    def test_is_complete_expired(self):
        """Test is_complete for EXPIRED status."""
        counts = RequestCounts(total=100, expired=100)
        response = BatchStatusResponse(
            batch_id="batch_123",
            provider="openai",
            status=BatchStatus.EXPIRED,
            request_counts=counts,
            created_at="2024-01-01T00:00:00Z"
        )
        assert response.is_complete() is True

    def test_is_complete_in_progress(self):
        """Test is_complete for IN_PROGRESS status."""
        counts = RequestCounts(total=100, processing=50, succeeded=50)
        response = BatchStatusResponse(
            batch_id="batch_123",
            provider="openai",
            status=BatchStatus.IN_PROGRESS,
            request_counts=counts,
            created_at="2024-01-01T00:00:00Z"
        )
        assert response.is_complete() is False


class TestUnifiedResult:
    """Tests for UnifiedResult dataclass."""

    def test_unified_result_success(self):
        """Test creating successful UnifiedResult."""
        result = UnifiedResult(
            custom_id="req1",
            status=ResultStatus.SUCCEEDED,
            response={"choices": [{"message": {"content": "Hello!"}}]}
        )
        assert result.custom_id == "req1"
        assert result.status == ResultStatus.SUCCEEDED
        assert result.response is not None
        assert result.error is None

    def test_unified_result_error(self):
        """Test creating errored UnifiedResult."""
        result = UnifiedResult(
            custom_id="req1",
            status=ResultStatus.ERRORED,
            error={"code": "invalid_request", "message": "Bad input"}
        )
        assert result.custom_id == "req1"
        assert result.status == ResultStatus.ERRORED
        assert result.response is None
        assert result.error is not None
        assert result.error["code"] == "invalid_request"

    def test_unified_result_defaults(self):
        """Test UnifiedResult with default values."""
        result = UnifiedResult(
            custom_id="req1",
            status=ResultStatus.SUCCEEDED
        )
        assert result.response is None
        assert result.error is None
        assert result.provider_data == {}

    def test_get_text_response_openai_format(self):
        """Test get_text_response with OpenAI format."""
        result = UnifiedResult(
            custom_id="req1",
            status=ResultStatus.SUCCEEDED,
            response={
                "choices": [
                    {"message": {"content": "Hello from OpenAI!"}}
                ]
            }
        )
        assert result.get_text_response() == "Hello from OpenAI!"

    def test_get_text_response_anthropic_format(self):
        """Test get_text_response with Anthropic format."""
        result = UnifiedResult(
            custom_id="req1",
            status=ResultStatus.SUCCEEDED,
            response={
                "content": [
                    {"text": "Hello from Anthropic!"}
                ]
            }
        )
        assert result.get_text_response() == "Hello from Anthropic!"

    def test_get_text_response_google_format(self):
        """Test get_text_response with Google format."""
        result = UnifiedResult(
            custom_id="req1",
            status=ResultStatus.SUCCEEDED,
            response={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Hello from Google!"}
                            ]
                        }
                    }
                ]
            }
        )
        assert result.get_text_response() == "Hello from Google!"

    def test_get_text_response_direct_text(self):
        """Test get_text_response with direct text field."""
        result = UnifiedResult(
            custom_id="req1",
            status=ResultStatus.SUCCEEDED,
            response={"text": "Direct text response"}
        )
        assert result.get_text_response() == "Direct text response"

    def test_get_text_response_error_status(self):
        """Test get_text_response returns None for error status."""
        result = UnifiedResult(
            custom_id="req1",
            status=ResultStatus.ERRORED,
            error={"message": "Error occurred"}
        )
        assert result.get_text_response() is None

    def test_get_text_response_no_response(self):
        """Test get_text_response returns None when no response."""
        result = UnifiedResult(
            custom_id="req1",
            status=ResultStatus.SUCCEEDED,
            response=None
        )
        assert result.get_text_response() is None

    def test_get_text_response_invalid_format(self):
        """Test get_text_response returns None for invalid format."""
        result = UnifiedResult(
            custom_id="req1",
            status=ResultStatus.SUCCEEDED,
            response={"unknown": "format"}
        )
        assert result.get_text_response() is None
