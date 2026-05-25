const ACTIVE_JOB_KEY = 'fashion_vision_active_job'

export function getActiveJobId(): string | null {
  try {
    return localStorage.getItem(ACTIVE_JOB_KEY)
  } catch {
    return null
  }
}

export function setActiveJobId(jobId: string): void {
  try {
    localStorage.setItem(ACTIVE_JOB_KEY, jobId)
  } catch {
    // ignore storage errors
  }
}

export function clearActiveJobId(): void {
  try {
    localStorage.removeItem(ACTIVE_JOB_KEY)
  } catch {
    // ignore storage errors
  }
}
