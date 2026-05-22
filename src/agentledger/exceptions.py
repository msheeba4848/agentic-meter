"""Custom exceptions for agentledger."""
from typing import Optional


class AgentLedgerError(Exception):
    """Base exception for all agentledger errors."""


class BudgetExceeded(AgentLedgerError):
    """Raised when a tracked operation has exceeded the configured budget.

    Attributes:
        spent: Total amount spent so far, in USD.
        budget: The configured budget, in USD.
    """

    def __init__(
        self, spent: float, budget: float, message: Optional[str] = None
    ):
        self.spent = spent
        self.budget = budget
        msg = (
            message
            or f"Budget exceeded: ${spent:.4f} spent, budget was ${budget:.4f}"
        )
        super().__init__(msg)
