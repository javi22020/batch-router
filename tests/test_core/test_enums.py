"""Tests for core enums."""

import pytest
from batch_router.core.enums import BatchStatus, ResultStatus


class TestBatchStatus:
    """Tests for BatchStatus enum."""

    def test_batch_status_values(self):
        """Test all BatchStatus enum values exist."""
        assert BatchStatus.VALIDATING.value == "validating"
        assert BatchStatus.IN_PROGRESS.value == "in_progress"
        assert BatchStatus.COMPLETED.value == "completed"
        assert BatchStatus.FAILED.value == "failed"
        assert BatchStatus.CANCELLED.value == "cancelled"
        assert BatchStatus.EXPIRED.value == "expired"

    def test_batch_status_membership(self):
        """Test BatchStatus enum membership."""
        assert BatchStatus.VALIDATING in BatchStatus
        assert BatchStatus.COMPLETED in BatchStatus

    def test_batch_status_count(self):
        """Test correct number of BatchStatus values."""
        assert len(BatchStatus) == 6


class TestResultStatus:
    """Tests for ResultStatus enum."""

    def test_result_status_values(self):
        """Test all ResultStatus enum values exist."""
        assert ResultStatus.SUCCEEDED.value == "succeeded"
        assert ResultStatus.ERRORED.value == "errored"
        assert ResultStatus.CANCELLED.value == "cancelled"
        assert ResultStatus.EXPIRED.value == "expired"

    def test_result_status_membership(self):
        """Test ResultStatus enum membership."""
        assert ResultStatus.SUCCEEDED in ResultStatus
        assert ResultStatus.ERRORED in ResultStatus

    def test_result_status_count(self):
        """Test correct number of ResultStatus values."""
        assert len(ResultStatus) == 4
