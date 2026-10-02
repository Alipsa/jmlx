package se.alipsa.jmlx.models;

/**
 * A submission was rejected because the scheduler is at capacity right now. Retrying later can
 * succeed, unlike an {@link IllegalArgumentException} for a request that can never be served.
 *
 * <p>Not "queue full": a cancelled request frees its queue slot but keeps its admission permit
 * until its completion has been dispatched, so a slow completion dispatcher can reject a submission
 * while the waiting queue is empty.
 */
public final class BatchAdmissionRejectedException extends IllegalStateException {
  private static final long serialVersionUID = 1L;

  BatchAdmissionRejectedException(String message) {
    super(message);
  }
}
