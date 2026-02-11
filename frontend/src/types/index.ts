/* ------------------------------------------------------------------ */
/*  Shared TypeScript types mirroring backend Pydantic schemas        */
/* ------------------------------------------------------------------ */

// ---- Enums (must match backend app/schemas/common.py) ----
export type ProjectStatus = 'draft' | 'active' | 'archived';
export type JobStatus = 'queued' | 'in_progress' | 'completed' | 'failed' | 'cancelled';
export type SourceType = string; // Dynamic: medquad | pdf_ollama | rag_openrouter | etc.
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
  source_document: string | null;
  source_metadata: Record<string, unknown> | null;
  model_used: string | null;
  provider: string | null;
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
  by_source_document: Record<string, number>;
}

export interface FileAnalytics {
  filename: string;
  source_type: string;
  pair_count: number;
  avg_quality: number | null;
  approved: number;
  rejected: number;
  pending: number;
}

export interface EnhancedAnalytics {
  by_file: FileAnalytics[];
  quality_histogram: { range: string; count: number }[];
  generation_timeline: { date: string; count: number }[];
}

// ---- Generation ----
export interface GenerationConfig {
  medical_terms: string;
  email: string;
  provider: string;          // ollama | openai | anthropic | gemini | openrouter
  api_key_id?: string;       // stored key ID for cloud providers
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

export interface GenerationProvider {
  name: string;
  models: string[];
  requires_api_key: boolean;
  has_stored_key: boolean;
  stored_key_id: string | null;
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
  generation_number: number | null;
  qa_pair_count: number | null;
  output_files: Record<string, unknown> | null;
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
  generation_job_id?: string;
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

// ---- LLM Review ----
export interface LLMReviewResult {
  qa_pair_id: string;
  accuracy: number;
  completeness: number;
  clarity: number;
  relevance: number;
  overall: number;
  recommendation: 'approve' | 'revise' | 'reject';
  feedback: string;
  error?: string | null;
}

export interface LLMReviewResponse {
  results: LLMReviewResult[];
  total_reviewed: number;
  avg_overall: number | null;
}

export interface LLMProvider {
  name: string;
  models: string[];
  requires_api_key: boolean;
  has_stored_key?: boolean;
  models_source?: 'hardcoded' | 'fetched' | 'none';
  models_fetched_at?: string | null;
}

// ---- Stored LLM Provider (API Key Management) ----
export interface LLMProviderConfig {
  id: string;
  provider_name: string;
  display_name: string | null;
  organization_id: string | null;
  masked_key: string;
  is_valid: boolean;
  enabled: boolean;
  is_default: boolean;
  available_models: string[] | null;
  model_details: Record<string, Record<string, unknown>> | null;
  models_fetched_at: string | null;
  rate_limits: Record<string, unknown> | null;
  last_tested_at: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface LLMProviderCreate {
  provider_name: string;
  api_key: string;
  organization_id?: string;
  display_name?: string;
  is_default?: boolean;
}

export interface LLMProviderUpdate {
  api_key?: string;
  organization_id?: string;
  display_name?: string;
  is_default?: boolean;
  enabled?: boolean;
}

export interface LLMProviderTestResult {
  success: boolean;
  message: string;
  available_models: string[];
  error: string | null;
}

// ---- Review Sessions ----
export interface ReviewStartRequest {
  qa_pair_ids: string[];
  provider: string;
  api_key_id?: string;
  api_key?: string;
  model?: string;
  ollama_url?: string;
  speed?: 'normal' | 'fast' | 'slow';
}

export interface ReviewSession {
  id: string;
  project_id: string;
  provider: string;
  model: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'cancelled';
  total_pairs: number;
  completed_pairs: number;
  failed_pairs: number;
  approved_count: number;
  revise_count: number;
  rejected_count: number;
  avg_overall_score: number | null;
  total_cost_usd: number;
  current_message: string | null;
  error_message: string | null;
  results: LLMReviewResult[] | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

// ---- Fact Check ----
export interface FactCheckResult {
  qa_pair_id: string;
  factual_accuracy: number;
  analysis: string[];
  suggested_answer: string | null;
  confidence: number;
  cost_usd: number;
  error: string | null;
}

export interface CostEstimate {
  estimated_cost_usd: number;
  pair_count: number;
  model: string;
  provider: string;
}

// ---- Auto-Approve Workflow ----
export interface AutoApproveWorkflowRequest {
  qa_pair_ids: string[];
  provider: string;
  api_key_id?: string;
  api_key?: string;
  model?: string;
  ollama_url?: string;
  speed?: string;
  threshold: number;
  auto_accept_suggestions: boolean;
  suggestion_threshold_min?: number;
  suggestion_threshold_max?: number;
}

export interface AcceptSuggestionResult {
  qa_pair_id: string;
  old_answer: string;
  new_answer: string;
  applied: boolean;
  message: string;
}
