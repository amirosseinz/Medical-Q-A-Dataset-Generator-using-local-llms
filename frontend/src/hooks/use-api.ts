/* ------------------------------------------------------------------ */
/*  TanStack Query hooks wrapping the API client                      */
/* ------------------------------------------------------------------ */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { projectsApi, sourcesApi, generationApi, qaPairsApi, ollamaApi } from '@/lib/api';
import type { ProjectCreate, ProjectUpdate, QAPairUpdate, ValidationStatus } from '@/types';
import { toast } from '@/hooks/use-toast';

// ---- Projects ----
export function useProjects() {
  return useQuery({
    queryKey: ['projects'],
    queryFn: projectsApi.list,
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
