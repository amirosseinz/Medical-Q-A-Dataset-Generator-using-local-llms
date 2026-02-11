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
  EnhancedAnalytics,
  GenerationConfig,
  GenerationJob,
  GenerationProvider,
  ExportRequest,
  OllamaStatus,
  PaginatedResponse,
  ValidationStatus,
  LLMReviewResponse,
  LLMProvider,
  LLMProviderConfig,
  LLMProviderCreate,
  LLMProviderUpdate,
  LLMProviderTestResult,
  ReviewStartRequest,
  ReviewSession,
  FactCheckResult,
  CostEstimate,
  AutoApproveWorkflowRequest,
  AcceptSuggestionResult,
} from '@/types';

const BASE = '/api/v1';

// ---- helpers ----
async function request<T>(
  url: string,
  options?: RequestInit,
): Promise<T> {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json; charset=utf-8', ...options?.headers },
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
      method: 'PUT',
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
  providers: () =>
    request<GenerationProvider[]>(`${BASE}/generation/providers`),
  validateKey: (provider: string) =>
    request<{ provider: string; has_key: boolean; message: string }>(
      `${BASE}/generation/validate-key/${encodeURIComponent(provider)}`,
    ),
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
      source_document?: string;
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
  analytics: (projectId: string) =>
    request<EnhancedAnalytics>(`${BASE}/projects/${projectId}/qa-pairs/analytics`),
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

// ---- LLM Review ----
export const reviewApi = {
  providers: () => request<LLMProvider[]>(`${BASE}/review/providers`),
  // Legacy sync review (small batches)
  review: (
    projectId: string,
    data: {
      qa_pair_ids: string[];
      provider: string;
      api_key?: string;
      model?: string;
      ollama_url?: string;
    },
  ) =>
    request<LLMReviewResponse>(`${BASE}/projects/${projectId}/review`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  // Session-based async review (recommended for large batches)
  startSession: (projectId: string, data: ReviewStartRequest) =>
    request<ReviewSession>(`${BASE}/projects/${projectId}/review/start`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  getSession: (sessionId: string) =>
    request<ReviewSession>(`${BASE}/review/sessions/${sessionId}`),
  cancelSession: (sessionId: string) =>
    request<ReviewSession>(`${BASE}/review/sessions/${sessionId}/cancel`, {
      method: 'POST',
    }),
  resumeSession: (sessionId: string) =>
    request<ReviewSession>(`${BASE}/review/sessions/${sessionId}/resume`, {
      method: 'POST',
    }),
  estimateCost: (pairCount: number, provider: string, model?: string) =>
    request<CostEstimate>(
      `${BASE}/review/estimate-cost?pair_count=${pairCount}&provider=${provider}${model ? `&model=${model}` : ''}`,
    ),
  autoApprove: (projectId: string, minOverall: number = 7.0) =>
    request<{ approved: number; rejected: number; message: string }>(
      `${BASE}/projects/${projectId}/review/auto-approve?min_overall=${minOverall}`,
      { method: 'POST' },
    ),
  factCheck: (
    projectId: string,
    data: {
      qa_pair_id: string;
      provider: string;
      api_key_id?: string;
      api_key?: string;
      model?: string;
      ollama_url?: string;
    },
  ) =>
    request<FactCheckResult>(`${BASE}/projects/${projectId}/fact-check`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  autoApproveWorkflow: (projectId: string, data: AutoApproveWorkflowRequest) =>
    request<ReviewSession>(`${BASE}/projects/${projectId}/review/auto-approve-workflow`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  acceptSuggestion: (pairId: string) =>
    request<AcceptSuggestionResult>(`${BASE}/qa-pairs/${pairId}/accept-suggestion`, {
      method: 'POST',
    }),
  revertSuggestion: (pairId: string) =>
    request<AcceptSuggestionResult>(`${BASE}/qa-pairs/${pairId}/revert-suggestion`, {
      method: 'POST',
    }),
};

// ---- LLM Providers (API Key Management) ----
export const llmProvidersApi = {
  list: () =>
    request<LLMProviderConfig[]>(`${BASE}/settings/llm-providers`),
  create: (data: LLMProviderCreate) =>
    request<LLMProviderConfig>(`${BASE}/settings/llm-providers`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  update: (id: string, data: LLMProviderUpdate) =>
    request<LLMProviderConfig>(`${BASE}/settings/llm-providers/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),
  delete: (id: string) =>
    request<void>(`${BASE}/settings/llm-providers/${id}`, {
      method: 'DELETE',
    }),
  test: (id: string) =>
    request<LLMProviderTestResult>(`${BASE}/settings/llm-providers/${id}/test`, {
      method: 'POST',
    }),
  refreshModels: (id: string, force: boolean = false) =>
    request<LLMProviderTestResult>(
      `${BASE}/settings/llm-providers/${id}/refresh-models?force=${force}`,
      { method: 'POST' },
    ),
  getModels: (id: string) =>
    request<{ models: string[]; model_details: Record<string, unknown>; fetched_at: string | null; cached: boolean }>(
      `${BASE}/settings/llm-providers/${id}/models`,
    ),
};
