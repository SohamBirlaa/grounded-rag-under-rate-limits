"""The Relay contracts package.

Exports typed schemas, error hierarchy, and CSV contract definitions.
"""

from relay.contracts.errors import (
    CircuitOpenError,
    CitationVerificationError,
    DailyBudgetExhaustedError,
    DeadlineShedError,
    EpistemicRefusalError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QueueFullError,
    QueueTimeoutError,
    RateLimitError,
    RelayError,
    RequestDeadlineExceededError,
    RetrievalError,
    ToolTimeoutError,
    ValidationError,
)
from relay.contracts.schemas import (
    CSV_MANDATORY_COLUMNS,
    AnswerGradeType,
    CitationRecord,
    CSVRowRecord,
    LatencyBreakdown,
    PaperIdType,
    ProviderSourceType,
    QueueTransitionEvent,
    RelayRequest,
    RelayResponse,
    RequestStatusType,
    RetrievalSettingType,
    ScenarioMetricsRecord,
    ScenarioType,
    StageSpanRecord,
    Table3RetrievalSettingLabel,
    map_label_to_retrieval_setting,
    map_retrieval_setting_to_label,
)

__all__ = [
    # Enums & Types
    "ScenarioType",
    "ProviderSourceType",
    "RetrievalSettingType",
    "Table3RetrievalSettingLabel",
    "PaperIdType",
    "RequestStatusType",
    "AnswerGradeType",
    "CSV_MANDATORY_COLUMNS",
    # Mappings
    "map_retrieval_setting_to_label",
    "map_label_to_retrieval_setting",
    # Schemas
    "RelayRequest",
    "CitationRecord",
    "LatencyBreakdown",
    "RelayResponse",
    "CSVRowRecord",
    "StageSpanRecord",
    "QueueTransitionEvent",
    "ScenarioMetricsRecord",
    # Errors
    "RelayError",
    "ValidationError",
    "RequestDeadlineExceededError",
    "QueueFullError",
    "QueueTimeoutError",
    "DeadlineShedError",
    "CircuitOpenError",
    "ProviderError",
    "ProviderUnavailableError",
    "ProviderTimeoutError",
    "ProviderAuthenticationError",
    "RateLimitError",
    "DailyBudgetExhaustedError",
    "RetrievalError",
    "ToolTimeoutError",
    "CitationVerificationError",
    "EpistemicRefusalError",
]

