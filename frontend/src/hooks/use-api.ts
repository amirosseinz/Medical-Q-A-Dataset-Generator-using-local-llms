/* ------------------------------------------------------------------ */
/*  TanStack Query hooks wrapping the API client                      */
/* ------------------------------------------------------------------ */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { projectsApi, sourcesApi, generationApi, qaPairsApi, ollamaApi, reviewApi, llmProvidersApi } from '@/lib/api';
import type { ProjectCreate, ProjectUpdate, QAPairUpdate, ValidationStatus, LLMProviderCreate, LLMProviderUpdate, ReviewStartRequest, AutoApproveWorkflowRequest } from '@/types';
import { toast } from '@/hooks/use-toast';

// ---- Projects ----
export function useProjects() {
  return useQuery({
    queryKey: ['projects'],
    queryFn: projectsApi.list,
    refetchInterval: 15000, // refresh every 15s on dashboard
  });
}

export function useProject(id: string) {
  return useQuery({
    queryKey: ['projects', id],
    queryFn: () => projectsApi.get(id),
    enabled: !!id,
  });
}

export function useCreateProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: ProjectCreate) => projectsApi.create(data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects'] });
      toast({ title: 'Project created' });
    },
    onError: (err: Error) => {
      toast({ title: 'Failed to create project', description: err.message, variant: 'destructive' });
    },
  });
}

export function useUpdateProject(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: ProjectUpdate) => projectsApi.update(id, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects'] });
      qc.invalidateQueries({ queryKey: ['projects', id] });
    },
  });
}

export function useDeleteProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => projectsApi.delete(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects'] });
      toast({ title: 'Project deleted' });
    },
  });
}

// ---- Sources ----
export function useSources(projectId: string) {
  return useQuery({
    queryKey: ['sources', projectId],
    queryFn: () => sourcesApi.list(projectId),
    enabled: !!projectId,
  });
}

export function useUploadSources(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (files: File[]) => sourcesApi.upload(projectId, files),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['sources', projectId] });
      toast({ title: 'Files uploaded successfully' });
    },
    onError: (err: Error) => {
      toast({ title: 'Upload failed', description: err.message, variant: 'destructive' });
    },
  });
}

export function useDeleteSource(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sourceId: string) => sourcesApi.delete(projectId, sourceId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['sources', projectId] });
    },
  });
}

// ---- Generation ----
export function useStartGeneration(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (config: Record<string, unknown>) => generationApi.start(projectId, config),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jobs', projectId] });
      toast({ title: 'Generation started' });
    },
    onError: (err: Error) => {
      toast({ title: 'Failed to start generation', description: err.message, variant: 'destructive' });
    },
  });
}

export function useCancelGeneration(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => generationApi.cancel(projectId, jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jobs', projectId] });
      toast({ title: 'Generation cancelled' });
    },
  });
}

export function useJobs(projectId: string) {
  return useQuery({
    queryKey: ['jobs', projectId],
    queryFn: () => generationApi.listJobs(projectId),
    enabled: !!projectId,
    refetchInterval: 5000,
  });
}

export function useJobProgress(projectId: string, jobId: string) {
  return useQuery({
    queryKey: ['jobs', projectId, jobId],
    queryFn: () => generationApi.getProgress(projectId, jobId),
    enabled: !!projectId && !!jobId,
    refetchInterval: 2000,
  });
}

// ---- QA Pairs ----
export function useQAPairs(
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
) {
  return useQuery({
    queryKey: ['qa-pairs', projectId, params],
    queryFn: () => qaPairsApi.list(projectId, params),
    enabled: !!projectId,
  });
}

export function useQAPairStats(projectId: string) {
  return useQuery({
    queryKey: ['qa-pairs', projectId, 'stats'],
    queryFn: () => qaPairsApi.stats(projectId),
    enabled: !!projectId,
  });
}

export function useEnhancedAnalytics(projectId: string) {
  return useQuery({
    queryKey: ['qa-pairs', projectId, 'analytics'],
    queryFn: () => qaPairsApi.analytics(projectId),
    enabled: !!projectId,
  });
}

export function useUpdateQAPair(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ pairId, data }: { pairId: string; data: QAPairUpdate }) =>
      qaPairsApi.update(projectId, pairId, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['qa-pairs', projectId] });
    },
  });
}

export function useBatchUpdateQAPairs(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ ids, data }: { ids: string[]; data: { validation_status: ValidationStatus } }) =>
      qaPairsApi.batchUpdate(projectId, ids, data),
    onSuccess: (result) => {
      qc.invalidateQueries({ queryKey: ['qa-pairs', projectId] });
      qc.invalidateQueries({ queryKey: ['projects', projectId] });
      qc.invalidateQueries({ queryKey: ['projects'] });
      toast({ title: `Updated ${result.updated} Q&A pairs` });
    },
  });
}

export function useDeleteQAPair(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (pairId: string) => qaPairsApi.delete(projectId, pairId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['qa-pairs', projectId] });
      qc.invalidateQueries({ queryKey: ['projects', projectId] });
      qc.invalidateQueries({ queryKey: ['projects'] });
    },
  });
}

// ---- Ollama ----
export function useOllamaStatus() {
  return useQuery({
    queryKey: ['ollama', 'status'],
    queryFn: ollamaApi.status,
    refetchInterval: 30_000,
    retry: false,
  });
}

// ---- Generation Providers ----
export function useGenerationProviders() {
  return useQuery({
    queryKey: ['generation', 'providers'],
    queryFn: generationApi.providers,
    staleTime: 60_000,
  });
}

// ---- LLM Review ----
export function useReviewProviders() {
  return useQuery({
    queryKey: ['review', 'providers'],
    queryFn: reviewApi.providers,
  });
}

export function useReviewQAPairs(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: {
      qa_pair_ids: string[];
      provider: string;
      api_key?: string;
      model?: string;
      ollama_url?: string;
    }) => reviewApi.review(projectId, data),
    onSuccess: (result) => {
      qc.invalidateQueries({ queryKey: ['qa-pairs', projectId] });
      toast({
        title: `Review complete`,
        description: `${result.total_reviewed} pairs reviewed. Avg score: ${result.avg_overall?.toFixed(1) ?? '—'}/10`,
      });
    },
    onError: (err: Error) => {
      toast({ title: 'Review failed', description: err.message, variant: 'destructive' });
    },
  });
}

export function useAutoApprove(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (minOverall: number) => reviewApi.autoApprove(projectId, minOverall),
    onSuccess: (result) => {
      qc.invalidateQueries({ queryKey: ['qa-pairs', projectId] });
      qc.invalidateQueries({ queryKey: ['projects', projectId] });
      qc.invalidateQueries({ queryKey: ['projects'] });
      toast({ title: result.message });
    },
  });
}

// ---- LLM Providers (API Key Management) ----
export function useLLMProviders() {
  return useQuery({
    queryKey: ['llm-providers'],
    queryFn: llmProvidersApi.list,
  });
}

export function useCreateLLMProvider() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: LLMProviderCreate) => llmProvidersApi.create(data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llm-providers'] });
      qc.invalidateQueries({ queryKey: ['review', 'providers'] });
      qc.invalidateQueries({ queryKey: ['generation', 'providers'] });
      toast({ title: 'API key saved' });
    },
    onError: (err: Error) => {
      toast({ title: 'Failed to save API key', description: err.message, variant: 'destructive' });
    },
  });
}

export function useUpdateLLMProvider() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: LLMProviderUpdate }) =>
      llmProvidersApi.update(id, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llm-providers'] });
      qc.invalidateQueries({ queryKey: ['review', 'providers'] });
      qc.invalidateQueries({ queryKey: ['generation', 'providers'] });
      toast({ title: 'Provider updated' });
    },
  });
}

export function useDeleteLLMProvider() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => llmProvidersApi.delete(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['llm-providers'] });
      qc.invalidateQueries({ queryKey: ['review', 'providers'] });
      qc.invalidateQueries({ queryKey: ['generation', 'providers'] });
      toast({ title: 'API key deleted' });
    },
  });
}

export function useTestLLMProvider() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => llmProvidersApi.test(id),
    onSuccess: (result) => {
      qc.invalidateQueries({ queryKey: ['llm-providers'] });
      qc.invalidateQueries({ queryKey: ['review', 'providers'] });
      qc.invalidateQueries({ queryKey: ['generation', 'providers'] });
      toast({
        title: result.success ? 'Connection successful' : 'Connection failed',
        description: result.message,
        variant: result.success ? 'default' : 'destructive',
      });
    },
  });
}

export function useRefreshModels() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, force }: { id: string; force?: boolean }) =>
      llmProvidersApi.refreshModels(id, force),
    onSuccess: (result) => {
      qc.invalidateQueries({ queryKey: ['llm-providers'] });
      qc.invalidateQueries({ queryKey: ['review', 'providers'] });
      qc.invalidateQueries({ queryKey: ['generation', 'providers'] });
      toast({
        title: result.success ? 'Models refreshed' : 'Refresh failed',
        description: result.message,
        variant: result.success ? 'default' : 'destructive',
      });
    },
    onError: (err: Error) => {
      toast({ title: 'Failed to refresh models', description: err.message, variant: 'destructive' });
    },
  });
}

// ---- Review Sessions ----
export function useStartReviewSession(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: ReviewStartRequest) => reviewApi.startSession(projectId, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['qa-pairs', projectId] });
    },
    onError: (err: Error) => {
      toast({ title: 'Failed to start review', description: err.message, variant: 'destructive' });
    },
  });
}

export function useReviewSession(sessionId: string | null) {
  return useQuery({
    queryKey: ['review-session', sessionId],
    queryFn: () => reviewApi.getSession(sessionId!),
    enabled: !!sessionId,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 2000;
      if (data.status === 'in_progress' || data.status === 'pending') return 2000;
      return false;
    },
  });
}

export function useCancelReviewSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) => reviewApi.cancelSession(sessionId),
    onSuccess: (_, sessionId) => {
      qc.invalidateQueries({ queryKey: ['review-session', sessionId] });
      toast({ title: 'Review session cancelled' });
    },
  });
}

export function useFactCheck(projectId: string) {
  return useMutation({
    mutationFn: (data: {
      qa_pair_id: string;
      provider: string;
      api_key_id?: string;
      api_key?: string;
      model?: string;
      ollama_url?: string;
    }) => reviewApi.factCheck(projectId, data),
    onError: (err: Error) => {
      toast({ title: 'Fact check failed', description: err.message, variant: 'destructive' });
    },
  });
}

export function useAutoApproveWorkflow(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: AutoApproveWorkflowRequest) => reviewApi.autoApproveWorkflow(projectId, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['qa-pairs', projectId] });
      qc.invalidateQueries({ queryKey: ['projects', projectId] });
      qc.invalidateQueries({ queryKey: ['projects'] });
    },
    onError: (err: Error) => {
      toast({ title: 'Auto-approve workflow failed', description: err.message, variant: 'destructive' });
    },
  });
}

export function useAcceptSuggestion(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (pairId: string) => reviewApi.acceptSuggestion(pairId),
    onSuccess: (result) => {
      qc.invalidateQueries({ queryKey: ['qa-pairs', projectId] });
      toast({ title: 'Suggestion applied', description: result.message });
    },
    onError: (err: Error) => {
      toast({ title: 'Failed to apply suggestion', description: err.message, variant: 'destructive' });
    },
  });
}

export function useRevertSuggestion(projectId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (pairId: string) => reviewApi.revertSuggestion(pairId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['qa-pairs', projectId] });
      toast({ title: 'Suggestion reverted' });
    },
    onError: (err: Error) => {
      toast({ title: 'Failed to revert suggestion', description: err.message, variant: 'destructive' });
    },
  });
}
