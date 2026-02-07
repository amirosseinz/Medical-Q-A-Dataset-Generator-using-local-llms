/* ------------------------------------------------------------------ */
/*  Global UI store – Zustand                                         */
/* ------------------------------------------------------------------ */
import { create } from 'zustand';
import type { GenerationProgress } from '@/types';

interface AppState {
  // Sidebar
  sidebarOpen: boolean;
  toggleSidebar: () => void;

  // Active generation tracking
  activeJobs: Map<string, GenerationProgress>;
  updateJobProgress: (progress: GenerationProgress) => void;
  removeJob: (jobId: string) => void;

  // Toast queue is handled by shadcn, but we track notification count
  unreadNotifications: number;
  incrementNotifications: () => void;
  clearNotifications: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  sidebarOpen: true,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

  activeJobs: new Map(),
  updateJobProgress: (progress) =>
    set((s) => {
      const next = new Map(s.activeJobs);
      next.set(progress.job_id, progress);
      // Remove completed / failed / cancelled
      if (['completed', 'failed', 'cancelled'].includes(progress.status)) {
        setTimeout(() => {
          useAppStore.getState().removeJob(progress.job_id);
        }, 5000);
      }
      return { activeJobs: next };
    }),
  removeJob: (jobId) =>
    set((s) => {
      const next = new Map(s.activeJobs);
      next.delete(jobId);
      return { activeJobs: next };
    }),

  unreadNotifications: 0,
  incrementNotifications: () =>
    set((s) => ({ unreadNotifications: s.unreadNotifications + 1 })),
  clearNotifications: () => set({ unreadNotifications: 0 }),
}));
