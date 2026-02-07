/* ------------------------------------------------------------------ */
/*  Shared TypeScript types mirroring backend Pydantic schemas        */
/* ------------------------------------------------------------------ */

// ---- Enums (must match backend app/schemas/common.py) ----
export type ProjectStatus = 'draft' | 'active' | 'archived';
export type JobStatus = 'queued' | 'in_progress' | 'completed' | 'failed' | 'cancelled';
export type SourceType = 'medquad' | 'pdf_ollama' | 'pubmed_ollama';
export type ValidationStatus = 'pending' | 'approved' | 'rejected';
export type FileType = 'pdf' | 'xml' | 'docx' | 'pubmed';
export type ExportFormat = 'csv' | 'json' | 'jsonl' | 'alpaca' | 'openai' | 'parquet';
export type ChunkingStrategy = 'word_count' | 'paragraph' | 'section';
export type DifficultyLevel = 'beginner' | 'intermediate' | 'advanced';
export type QuestionType = 'factual' | 'reasoning' | 'comparison' | 'application';

// ---- Project ----
export interface Project {
  id: string;
  name: string;
  domain: string;
  description: string;
  status: ProjectStatus;
  config: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  total_sources: number;
  total_qa_pairs: number;
  total_approved: number;
  avg_quality_score: number | null;
}

export interface ProjectCreate {
  name: string;
  domain: string;
  description?: string;
  config?: Record<string, unknown>;
}

export interface ProjectUpdate {
  name?: string;
  domain?: string;
  description?: string;
  status?: ProjectStatus;
  config?: Record<string, unknown>;
}

// ---- Source ----
export interface Source {
  id: string;
  project_id: string;
  filename: string;
  file_type: FileType;
  size_bytes: number | null;
  processing_status: string;
  created_at: string;
}

// ---- Chunk ----
export interface Chunk {
  id: string;
  source_id: string;
  project_id: string;
  chunk_index: number;
  content: string;
  word_count: number;
}

// ---- QA Pair (matches backend QAPairResponse) ----
export interface QAPair {
  id: string;
  project_id: string;
  chunk_id: string | null;
  question: string;
  answer: string;
  source_type: string;
  model_used: string | null;
  prompt_template: string | null;
  quality_score: number | null;
  validation_status: ValidationStatus;
  human_edited: boolean;
  metadata_json: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface QAPairUpdate {
  question?: string;
  answer?: string;
  validation_status?: ValidationStatus;
}

export interface QAPairStats {
  total: number;
  approved: number;
  pending: number;
  rejected: number;
  avg_quality_score: number | null;
  by_source_type: Record<string, number>;
  by_model: Record<string, number>;
}

// ---- Generation ----
export interface GenerationConfig {
  medical_terms: string;
  email: string;
  ollama_model: string;
  temperature: number;
  target_pairs: number;
  chunk_size: number;
  chunk_overlap: number;
  use_pubmed: boolean;
  pubmed_retmax: number;
  min_quality_score: number;
  difficulty_levels: DifficultyLevel[];
  question_types: QuestionType[];
}

export interface GenerationJob {
  id: string;
  project_id: string;
  celery_task_id: string | null;
  status: JobStatus;
  progress_pct: number;
  current_message: string | null;
  error_message: string | null;
  config: Record<string, unknown> | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

// ---- Progress (WebSocket) ----
export interface GenerationProgress {
  job_id: string;
  status: JobStatus;
  progress_pct: number;
  current_message: string | null;
  error_message: string | null;
}

// ---- Export (matches backend ExportRequest) ----
export interface ExportRequest {
  format: string;
  validation_statuses?: ValidationStatus[];
  min_quality_score?: number;
  train_split: number;
  val_split: number;
  test_split: number;
  include_metadata?: boolean;
}

// ---- Ollama ----
export interface OllamaModel {
  name: string;
  size: number;
  modified_at: string;
}

export interface OllamaStatus {
  connected: boolean;
  url: string;
  models: OllamaModel[];
}

// ---- Pagination (matches backend PaginatedResponse) ----
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// ---- Quality Check ----
export interface QualityCheck {
  id: string;
  qa_pair_id: string;
  check_type: string;
  passed: boolean;
  score: number;
  details: string;
}
