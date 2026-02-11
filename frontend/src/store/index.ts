/* ------------------------------------------------------------------ */
/*  Global UI store – Zustand                                         */
/* ------------------------------------------------------------------ */
import { create } from 'zustand';
import type { GenerationProgress } from '@/types';

/* ── Theme helpers ─────────────────────────────────────────────────── */
export type Theme = 'light' | 'dark' | 'system';

function getStoredTheme(): Theme {
  try {
    const stored = localStorage.getItem('theme');
    if (stored === 'light' || stored === 'dark' || stored === 'system') return stored;
  } catch { /* SSR / incognito safe */ }
  return 'system';
}

function resolveTheme(theme: Theme): 'light' | 'dark' {
  if (theme !== 'system') return theme;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

/** Apply the resolved theme class to <html> with smooth transition */
export function applyThemeToDOM(theme: Theme) {
  const resolved = resolveTheme(theme);
  const root = document.documentElement;
  // Add transition class for smooth color change
  root.classList.add('theme-transition');
  root.classList.toggle('dark', resolved === 'dark');
  // Remove transition class after animation completes
  setTimeout(() => root.classList.remove('theme-transition'), 350);
}

interface AppState {
  // Theme
  theme: Theme;
  setTheme: (t: Theme) => void;

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
  theme: getStoredTheme(),
  setTheme: (t) => {
    try { localStorage.setItem('theme', t); } catch { /* noop */ }
    applyThemeToDOM(t);
    set({ theme: t });
  },
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
