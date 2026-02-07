/* ------------------------------------------------------------------ */
/*  HTTP API client – all backend calls go through here               */
/* ------------------------------------------------------------------ */
import type {
  Project,
  ProjectCreate,
  ProjectUpdate,
  Source,
  QAPair,
  QAPairUpdate,
  QAPairStats,
  GenerationConfig,
  GenerationJob,
  ExportRequest,
  OllamaStatus,
  PaginatedResponse,
  ValidationStatus,
} from '@/types';

const BASE = '/api/v1';

// ---- helpers ----
async function request<T>(
  url: string,
  options?: RequestInit,
): Promise<T> {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${res.status}`);
  }
  // Handle 204 No Content
  if (res.status === 204) return undefined as unknown as T;
  return res.json();
}

// ---- Projects ----
export const projectsApi = {
  list: async (): Promise<Project[]> => {
    const res = await request<PaginatedResponse<Project>>(`${BASE}/projects?page_size=100`);
    return res.items;
  },
  get: (id: string) => request<Project>(`${BASE}/projects/${id}`),
  create: (data: ProjectCreate) =>
    request<Project>(`${BASE}/projects`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  update: (id: string, data: ProjectUpdate) =>
    request<Project>(`${BASE}/projects/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),
  delete: (id: string) =>
    request<void>(`${BASE}/projects/${id}`, { method: 'DELETE' }),
};

// ---- Sources ----
export const sourcesApi = {
  list: (projectId: string) =>
    request<Source[]>(`${BASE}/projects/${projectId}/sources`),
  upload: async (projectId: string, files: File[]): Promise<Source[]> => {
    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));
    const res = await fetch(`${BASE}/projects/${projectId}/sources/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail ?? 'Upload failed');
    }
    const data = await res.json();
    // Backend returns { uploaded: Source[], errors: string[] }
    return data.uploaded ?? data;
  },
  delete: (projectId: string, sourceId: string) =>
    request<void>(`${BASE}/sources/${sourceId}`, {
      method: 'DELETE',
    }),
};

// ---- Generation ----
export const generationApi = {
  start: (projectId: string, config: Partial<GenerationConfig>) =>
    request<{ job_id: string; celery_task_id: string }>(
      `${BASE}/projects/${projectId}/generate`,
      { method: 'POST', body: JSON.stringify(config) },
    ),
  cancel: (projectId: string, jobId: string) =>
    request<void>(`${BASE}/jobs/${jobId}`, {
      method: 'DELETE',
    }),
  getProgress: (projectId: string, jobId: string) =>
    request<GenerationJob>(`${BASE}/jobs/${jobId}/progress`),
  listJobs: (projectId: string) =>
    request<GenerationJob[]>(`${BASE}/projects/${projectId}/jobs`),
};

// ---- QA Pairs (routes match backend: /qa-pairs/{id} for mutate, /projects/{id}/qa-pairs for list) ----
export const qaPairsApi = {
  list: (
    projectId: string,
    params: {
      page?: number;
      page_size?: number;
      validation_status?: ValidationStatus | 'all';
      source_type?: string;
      min_quality_score?: number;
      search?: string;
      sort_by?: string;
      sort_dir?: 'asc' | 'desc';
    } = {},
  ) => {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== '' && v !== 'all')
        searchParams.set(k, String(v));
    });
    return request<PaginatedResponse<QAPair>>(
      `${BASE}/projects/${projectId}/qa-pairs?${searchParams}`,
    );
  },
  stats: (projectId: string) =>
    request<QAPairStats>(`${BASE}/projects/${projectId}/qa-pairs/stats`),
  update: (projectId: string, pairId: string, data: QAPairUpdate) =>
    request<QAPair>(`${BASE}/qa-pairs/${pairId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  batchUpdate: (
    projectId: string,
    ids: string[],
    data: { validation_status: ValidationStatus },
  ) =>
    request<{ updated: number }>(
      `${BASE}/qa-pairs/batch-update`,
      { method: 'POST', body: JSON.stringify({ ids, validation_status: data.validation_status }) },
    ),
  delete: (projectId: string, pairId: string) =>
    request<void>(`${BASE}/qa-pairs/${pairId}`, {
      method: 'DELETE',
    }),
};

// ---- Export ----
export const exportApi = {
  download: async (projectId: string, config: ExportRequest) => {
    // Step 1: POST export config to generate the file
    const exportResult = await request<{
      message: string;
      filename?: string;
      download_url?: string;
      files?: Record<string, string>;
      download_base?: string;
      total_pairs: number;
    }>(`${BASE}/projects/${projectId}/export`, {
      method: 'POST',
      body: JSON.stringify(config),
    });

    // Step 2: Download the generated file(s)
    if (exportResult.download_url) {
      // Single file export
      const res = await fetch(exportResult.download_url);
      if (!res.ok) throw new Error('Failed to download export file');
      const blob = await res.blob();
      const filename = exportResult.filename ?? `export.${config.format}`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } else if (exportResult.files && exportResult.download_base) {
      // Split export — download each file
      for (const [, fname] of Object.entries(exportResult.files)) {
        const res = await fetch(`${exportResult.download_base}/${fname}`);
        if (!res.ok) continue;
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fname;
        a.click();
        URL.revokeObjectURL(url);
      }
    } else {
      throw new Error('Export succeeded but no download URL returned');
    }
  },
};

// ---- Ollama ----
export const ollamaApi = {
  status: () => request<OllamaStatus>(`${BASE}/ollama/status`),
  models: () => request<OllamaStatus>(`${BASE}/ollama/models`),
};
