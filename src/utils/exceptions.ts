export class BudgetExceededError extends Error {
  constructor(
    public spent: number,
    public budget: number,
    message?: string,
  ) {
    super(
      message ?? `Budget exceeded: spent $${spent.toFixed(6)} of $${budget.toFixed(6)} budget`,
    );
    this.name = "BudgetExceededError";
  }
}

export class TimeoutExceededError extends Error {
  constructor(
    public elapsed: number,
    public timeout: number,
    public partialAnswer: string | null = null,
    message?: string,
  ) {
    super(message ?? `Timeout exceeded: ${elapsed.toFixed(1)}s of ${timeout.toFixed(1)}s limit`);
    this.name = "TimeoutExceededError";
  }
}

export class TokenLimitExceededError extends Error {
  constructor(
    public tokensUsed: number,
    public tokenLimit: number,
    public partialAnswer: string | null = null,
    message?: string,
  ) {
    super(
      message ??
        `Token limit exceeded: ${tokensUsed.toLocaleString()} of ${tokenLimit.toLocaleString()} tokens`,
    );
    this.name = "TokenLimitExceededError";
  }
}

export class ErrorThresholdExceededError extends Error {
  constructor(
    public errorCount: number,
    public threshold: number,
    public lastError: string | null = null,
    public partialAnswer: string | null = null,
    message?: string,
  ) {
    super(
      message ??
        `Error threshold exceeded: ${errorCount} consecutive errors (limit: ${threshold})`,
    );
    this.name = "ErrorThresholdExceededError";
  }
}

export class CancellationError extends Error {
  constructor(public partialAnswer: string | null = null, message?: string) {
    super(message ?? "Execution cancelled by user");
    this.name = "CancellationError";
  }
}
