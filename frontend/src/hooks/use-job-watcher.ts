/* ------------------------------------------------------------------ */
/*  Auto-refresh hook — watches generation jobs and triggers           */
/*  React Query cache invalidation when jobs complete.                 */
/*                                                                     */
/*  Also populates the Zustand activeJobs store for TopBar display.    */
/* ------------------------------------------------------------------ */
import { useEffect, useRef, useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useJobs } from './use-api';
import { useAppStore } from '@/store';
import { useToast } from '@/hooks/use-toast';
import type { GenerationProgress } from '@/types';

interface JobSnapshot {
  id: string;
  status: string;
  progress_pct: number;
  current_message: string | null;
  qa_pair_count: number | null;
}

/**
 * Watches generation jobs for a project and auto-refreshes related
 * queries when jobs transition to completed/failed/cancelled.
 *
 * Also:
 * - Populates the Zustand `activeJobs` store (used by TopBar)
 * - Shows toast notifications on completion
 * - Uses adaptive polling (faster during active generation)
 */
export function useJobCompletionWatcher(projectId: string | undefined) {
  const qc = useQueryClient();
  const { toast } = useToast();
  const updateJobProgress = useAppStore((s) => s.updateJobProgress);
  const removeJob = useAppStore((s) => s.removeJob);
  const prevJobsRef = useRef<Map<string, JobSnapshot>>(new Map());

  const { data: jobs } = useJobs(projectId ?? '');

  // Track whether there are any active jobs for adaptive polling
  const hasActiveJobs = jobs?.some(
    (j) => j.status === 'in_progress' || j.status === 'queued'
  ) ?? false;

  // Set adaptive polling interval on the jobs query
  useEffect(() => {
    if (!projectId) return;

    // When jobs are active, poll faster (2s); otherwise standard (5s)
    const interval = hasActiveJobs ? 2000 : 5000;
    qc.setQueryDefaults(['jobs', projectId], {
      refetchInterval: interval,
    });
  }, [projectId, hasActiveJobs, qc]);

  // Detect job state transitions and trigger appropriate invalidations
  useEffect(() => {
    if (!jobs || !projectId) return;

    const prev = prevJobsRef.current;
    const terminalStates = new Set(['completed', 'failed', 'cancelled']);
    let needsRefresh = false;

    for (const job of jobs) {
      const prevJob = prev.get(job.id);
      const wasActive = prevJob && !terminalStates.has(prevJob.status);
      const isNowTerminal = terminalStates.has(job.status);

      // Update Zustand store for TopBar
      if (job.status === 'in_progress' || job.status === 'queued') {
        updateJobProgress({
          job_id: job.id,
          status: job.status,
          progress_pct: job.progress_pct ?? 0,
          current_message: job.current_message ?? '',
        } as GenerationProgress);
      }

      // Detect transition to terminal state
      if (wasActive && isNowTerminal) {
        needsRefresh = true;

        // Update Zustand (will auto-remove after 5s)
        updateJobProgress({
          job_id: job.id,
          status: job.status,
          progress_pct: job.progress_pct ?? 100,
          current_message: job.current_message ?? '',
        } as GenerationProgress);

        // Show toast notification
        if (job.status === 'completed') {
          const count = job.qa_pair_count;
          toast({
            title: 'Generation complete!',
            description: count
              ? `${count} new Q&A pairs have been added.`
              : 'Your dataset has been updated.',
          });
        } else if (job.status === 'failed') {
          toast({
            title: 'Generation failed',
            description: job.error_message || 'An error occurred during generation.',
            variant: 'destructive',
          });
        } else if (job.status === 'cancelled') {
          toast({
            title: 'Generation cancelled',
            description: 'The generation job was cancelled.',
          });
        }
      }
    }

    // Also detect when jobs disappear from active (for edge cases)
    for (const [prevId, prevJob] of prev) {
      if (!terminalStates.has(prevJob.status)) {
        const currentJob = jobs.find(j => j.id === prevId);
        if (currentJob && terminalStates.has(currentJob.status)) {
          needsRefresh = true;
        }
      }
    }

    // Trigger comprehensive cache invalidation
    if (needsRefresh) {
      // Invalidate all project-related queries
      qc.invalidateQueries({ queryKey: ['qa-pairs', projectId] });
      qc.invalidateQueries({ queryKey: ['projects', projectId] });
      qc.invalidateQueries({ queryKey: ['projects'] });
      qc.invalidateQueries({ queryKey: ['sources', projectId] });
    }

    // Update snapshot
    const nextSnapshot = new Map<string, JobSnapshot>();
    for (const job of jobs) {
      nextSnapshot.set(job.id, {
        id: job.id,
        status: job.status,
        progress_pct: job.progress_pct ?? 0,
        current_message: job.current_message,
        qa_pair_count: job.qa_pair_count ?? null,
      });
    }
    prevJobsRef.current = nextSnapshot;
  }, [jobs, projectId, qc, toast, updateJobProgress]);

  return { hasActiveJobs };
}

/**
 * Hook to refetch data when navigating to a page.
 * Ensures fresh data on mount without requiring manual refresh.
 */
export function useRefreshOnMount(projectId: string | undefined, queryKeys: string[][]) {
  const qc = useQueryClient();

  useEffect(() => {
    if (!projectId) return;
    // Invalidate specified queries on mount to get fresh data
    for (const key of queryKeys) {
      qc.invalidateQueries({ queryKey: key });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);
}
