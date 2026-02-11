import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Search,
  CheckCircle2,
  XCircle,
  Clock,
  Edit3,
  Save,
  X,
  ChevronLeft,
  ChevronRight,
  Filter,
  FileText,
  Bot,
  Loader2,
  ShieldCheck,
  Zap,
  Undo2,
  Settings2,
  Sparkles,
} from 'lucide-react';
import {
  useProject,
  useQAPairs,
  useQAPairStats,
  useUpdateQAPair,
  useBatchUpdateQAPairs,
  useReviewProviders,
  useReviewQAPairs,
  useAutoApprove,
  useLLMProviders,
  useStartReviewSession,
  useReviewSession,
  useCancelReviewSession,
  useFactCheck,
  useAutoApproveWorkflow,
  useAcceptSuggestion,
  useRevertSuggestion,
} from '@/hooks/use-api';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { formatNumber } from '@/lib/utils';
import type { QAPair, ValidationStatus, LLMReviewResult } from '@/types';

const PAGE_SIZE = 20;

export default function ReviewPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();

  const { data: project } = useProject(projectId!);
  const { data: stats } = useQAPairStats(projectId!);

  // Filter state
  const [page, setPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<ValidationStatus | 'all'>('all');
  const [sourceFilter, setSourceFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState('created_at');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  const { data: pairsData, isLoading } = useQAPairs(projectId!, {
    page,
    page_size: PAGE_SIZE,
    validation_status: statusFilter,
    source_type: sourceFilter === 'all' ? undefined : sourceFilter,
    search: searchQuery || undefined,
    sort_by: sortBy,
    sort_dir: sortDir,
  });

  const updatePair = useUpdateQAPair(projectId!);
  const batchUpdate = useBatchUpdateQAPairs(projectId!);
  const { data: providers } = useReviewProviders();
  const { data: storedKeys } = useLLMProviders();
  const reviewMutation = useReviewQAPairs(projectId!);
  const autoApproveMutation = useAutoApprove(projectId!);
  const startSession = useStartReviewSession(projectId!);
  const cancelSession = useCancelReviewSession();
  const factCheckMutation = useFactCheck(projectId!);
  const autoApproveWorkflow = useAutoApproveWorkflow(projectId!);
  const acceptSuggestion = useAcceptSuggestion(projectId!);
  const revertSuggestion = useRevertSuggestion(projectId!);

  // Selection
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editQuestion, setEditQuestion] = useState('');
  const [editAnswer, setEditAnswer] = useState('');

  // LLM Review state
  const [reviewOpen, setReviewOpen] = useState(false);
  const [reviewProvider, setReviewProvider] = useState('ollama');
  const [reviewModel, setReviewModel] = useState('');
  const [reviewApiKey, setReviewApiKey] = useState('');
  const [reviewResults, setReviewResults] = useState<Record<string, LLMReviewResult>>({});

  // Session-based review state
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [progressOpen, setProgressOpen] = useState(false);
  const { data: sessionData } = useReviewSession(activeSessionId);

  // Fact check state
  const [factCheckResults, setFactCheckResults] = useState<Record<string, { factual_accuracy: number; analysis: string[]; suggested_answer: string | null }>>({});
  const [factCheckingId, setFactCheckingId] = useState<string | null>(null);

  // Auto-approve workflow state
  const [workflowOpen, setWorkflowOpen] = useState(false);
  const [workflowSessionId, setWorkflowSessionId] = useState<string | null>(null);
  const [workflowProgressOpen, setWorkflowProgressOpen] = useState(false);
  const [approvalThreshold, setApprovalThreshold] = useState(7.0);
  const [autoAcceptSuggestions, setAutoAcceptSuggestions] = useState(false);
  const { data: workflowSessionData } = useReviewSession(workflowSessionId);

  // Sticky action bar scroll tracking
  const actionBarRef = useRef<HTMLDivElement>(null);
  const actionBarPlaceholderRef = useRef<HTMLDivElement>(null);
  const [isSticky, setIsSticky] = useState(false);

  const handleScroll = useCallback(() => {
    if (!actionBarPlaceholderRef.current) return;
    const rect = actionBarPlaceholderRef.current.getBoundingClientRect();
    setIsSticky(rect.top < 0);
  }, []);

  useEffect(() => {
    if (selected.size > 0) {
      window.addEventListener('scroll', handleScroll, { passive: true });
      handleScroll(); // initial check
      return () => window.removeEventListener('scroll', handleScroll);
    } else {
      setIsSticky(false);
    }
  }, [selected.size, handleScroll]);

  // Check if selected provider has a stored key
  const selectedProviderInfo = providers?.find((p) => p.name === reviewProvider);
  const hasStoredKey = selectedProviderInfo?.has_stored_key ?? false;
  const storedKeyForProvider = storedKeys?.find(
    (k) => k.provider_name === reviewProvider && k.enabled && k.is_default
  );

  // Whether any review provider is configured (for fact-check button enabling)
  const hasAnyReviewProvider = reviewProvider === 'ollama' || hasStoredKey || !!reviewApiKey;
  const reviewModelLabel = reviewModel || selectedProviderInfo?.models?.[0] || '';
  const reviewProviderLabel = reviewProvider.charAt(0).toUpperCase() + reviewProvider.slice(1);

  // When session completes, update review results
  useEffect(() => {
    if (sessionData?.status === 'completed' && sessionData.results) {
      const newResults = { ...reviewResults };
      for (const r of sessionData.results) {
        newResults[r.qa_pair_id] = r as LLMReviewResult;
      }
      setReviewResults(newResults);
    }
  }, [sessionData?.status, sessionData?.results]);

  const toggleSelect = (id: string) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelected(next);
  };

  const selectAll = () => {
    if (!pairsData) return;
    if (selected.size === pairsData.items.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(pairsData.items.map((p) => p.id)));
    }
  };

  const startEdit = (pair: QAPair) => {
    setEditingId(pair.id);
    setEditQuestion(pair.question);
    setEditAnswer(pair.answer);
  };

  const saveEdit = () => {
    if (!editingId) return;
    updatePair.mutate(
      { pairId: editingId, data: { question: editQuestion, answer: editAnswer } },
      { onSuccess: () => setEditingId(null) },
    );
  };

  const handleBatchAction = (status: ValidationStatus) => {
    if (selected.size === 0) return;
    batchUpdate.mutate(
      { ids: Array.from(selected), data: { validation_status: status } },
      { onSuccess: () => setSelected(new Set()) },
    );
  };

  const handleLLMReview = () => {
    if (selected.size === 0) return;
    const needsKey = selectedProviderInfo?.requires_api_key;
    if (needsKey && !hasStoredKey && !reviewApiKey) return;

    // Use session-based review for larger batches
    if (selected.size > 5) {
      startSession.mutate(
        {
          qa_pair_ids: Array.from(selected),
          provider: reviewProvider,
          api_key_id: storedKeyForProvider?.id,
          api_key: (!hasStoredKey && reviewApiKey) ? reviewApiKey : undefined,
          model: reviewModel || undefined,
        },
        {
          onSuccess: (session) => {
            setActiveSessionId(session.id);
            setReviewOpen(false);
            setProgressOpen(true);
          },
        },
      );
    } else {
      // Small batch — use sync endpoint
      reviewMutation.mutate(
        {
          qa_pair_ids: Array.from(selected),
          provider: reviewProvider,
          api_key: hasStoredKey ? undefined : (reviewApiKey || undefined),
          model: reviewModel || undefined,
        },
        {
          onSuccess: (response) => {
            const newResults = { ...reviewResults };
            for (const r of response.results) {
              newResults[r.qa_pair_id] = r;
            }
            setReviewResults(newResults);
            setReviewOpen(false);
          },
        },
      );
    }
  };

  const handleFactCheck = (pairId: string) => {
    if (!hasAnyReviewProvider) return;
    setFactCheckingId(pairId);
    factCheckMutation.mutate(
      {
        qa_pair_id: pairId,
        provider: reviewProvider || 'ollama',
        api_key_id: storedKeyForProvider?.id,
        api_key: (!hasStoredKey && reviewApiKey) ? reviewApiKey : undefined,
        model: reviewModel || undefined,
      },
      {
        onSuccess: (result) => {
          setFactCheckResults((prev) => ({
            ...prev,
            [pairId]: {
              factual_accuracy: result.factual_accuracy,
              analysis: result.analysis,
              suggested_answer: result.suggested_answer,
            },
          }));
          setFactCheckingId(null);
        },
        onError: () => setFactCheckingId(null),
      },
    );
  };

  const handleAcceptSuggestion = (pairId: string) => {
    acceptSuggestion.mutate(pairId, {
      onSuccess: (result) => {
        // Clear fact-check results since answer changed
        setFactCheckResults((prev) => {
          const next = { ...prev };
          delete next[pairId];
          return next;
        });
      },
    });
  };

  const handleRevertSuggestion = (pairId: string) => {
    revertSuggestion.mutate(pairId, {
      onSuccess: () => {
        setFactCheckResults((prev) => {
          const next = { ...prev };
          delete next[pairId];
          return next;
        });
      },
    });
  };

  const handleStartAutoApproveWorkflow = () => {
    if (selected.size === 0) return;
    autoApproveWorkflow.mutate(
      {
        qa_pair_ids: Array.from(selected),
        provider: reviewProvider,
        api_key_id: storedKeyForProvider?.id,
        api_key: (!hasStoredKey && reviewApiKey) ? reviewApiKey : undefined,
        model: reviewModel || undefined,
        threshold: approvalThreshold,
        auto_accept_suggestions: autoAcceptSuggestions,
        suggestion_threshold_min: 6.0,
        suggestion_threshold_max: approvalThreshold - 0.1,
      },
      {
        onSuccess: (session) => {
          setWorkflowSessionId(session.id);
          setWorkflowOpen(false);
          setWorkflowProgressOpen(true);
        },
      },
    );
  };

  const statusIcon = (status: ValidationStatus) => {
    switch (status) {
      case 'approved':
        return <CheckCircle2 className="h-4 w-4 text-success-500" />;
      case 'rejected':
        return <XCircle className="h-4 w-4 text-destructive" />;
      default:
        return <Clock className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const sessionProgress = sessionData
    ? sessionData.total_pairs > 0
      ? Math.round((sessionData.completed_pairs / sessionData.total_pairs) * 100)
      : 0
    : 0;

  const workflowProgress = workflowSessionData
    ? workflowSessionData.total_pairs > 0
      ? Math.round((workflowSessionData.completed_pairs / workflowSessionData.total_pairs) * 100)
      : 0
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" onClick={() => navigate(`/projects/${projectId}`)}>
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <div className="flex-1">
          <h1 className="text-2xl font-bold tracking-tight">
            Review Q&A Pairs
          </h1>
          <p className="text-sm text-muted-foreground">
            {project?.name} — {formatNumber(stats?.total ?? 0)} total pairs
          </p>
        </div>
        <div className="flex items-center gap-2">
          {reviewModelLabel && (
            <span className="text-xs text-muted-foreground bg-muted px-2 py-1 rounded">
              <Bot className="inline h-3 w-3 mr-1" />
              {reviewProviderLabel}: {reviewModelLabel}
            </span>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={() => autoApproveMutation.mutate(7.0)}
            disabled={autoApproveMutation.isPending}
          >
            <Zap className="mr-1 h-4 w-4" />
            Quick Auto-Approve
          </Button>
        </div>
      </div>

      {/* Stats summary */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { label: 'Total', value: stats?.total ?? 0, color: 'text-foreground' },
          { label: 'Approved', value: stats?.approved ?? 0, color: 'text-success-600' },
          { label: 'Rejected', value: stats?.rejected ?? 0, color: 'text-destructive' },
          { label: 'Pending', value: stats?.pending ?? 0, color: 'text-muted-foreground' },
        ].map((s) => (
          <Card key={s.label}>
            <CardContent className="py-3 px-4">
              <p className="text-xs text-muted-foreground">{s.label}</p>
              <p className={`text-xl font-bold ${s.color}`}>{formatNumber(s.value)}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="flex flex-wrap items-center gap-3 py-3">
          <Filter className="h-4 w-4 text-muted-foreground" />
          <div className="relative flex-1 max-w-xs">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search Q&A pairs…"
              className="pl-9"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setPage(1);
              }}
            />
          </div>
          <Select
            value={statusFilter}
            onValueChange={(v) => {
              setStatusFilter(v as ValidationStatus | 'all');
              setPage(1);
            }}
          >
            <SelectTrigger className="w-40">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Statuses</SelectItem>
              <SelectItem value="pending">Pending</SelectItem>
              <SelectItem value="approved">Approved</SelectItem>
              <SelectItem value="rejected">Rejected</SelectItem>
            </SelectContent>
          </Select>
          <Select
            value={sourceFilter}
            onValueChange={(v) => {
              setSourceFilter(v);
              setPage(1);
            }}
          >
            <SelectTrigger className="w-36">
              <SelectValue placeholder="Source" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Sources</SelectItem>
              <SelectItem value="medquad">MedQuAD</SelectItem>
              <SelectItem value="pdf_ollama">PDF + Ollama</SelectItem>
              <SelectItem value="pubmed_ollama">PubMed + Ollama</SelectItem>
            </SelectContent>
          </Select>
          <Select value={sortBy} onValueChange={setSortBy}>
            <SelectTrigger className="w-40">
              <SelectValue placeholder="Sort by" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="created_at">Date</SelectItem>
              <SelectItem value="quality_score">Quality</SelectItem>
              <SelectItem value="validation_status">Status</SelectItem>
            </SelectContent>
          </Select>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))}
          >
            {sortDir === 'asc' ? '↑' : '↓'}
          </Button>
        </CardContent>
      </Card>

      {/* Batch actions — sticky when scrolled */}
      <div ref={actionBarPlaceholderRef}>
        {selected.size > 0 && (
          <div
            ref={actionBarRef}
            className={`
              flex items-center gap-3 rounded-lg border bg-background p-4
              transition-shadow duration-150 ease-in-out
              ${isSticky
                ? 'fixed top-0 left-0 right-0 z-[100] rounded-none border-x-0 border-t-0 border-b shadow-[0_4px_6px_rgba(0,0,0,0.1)]'
                : 'bg-muted/50'
              }
            `}
          >
            <div className={`flex items-center gap-3 w-full ${isSticky ? 'max-w-5xl mx-auto px-2' : ''}`}>
              <span className="text-sm font-semibold whitespace-nowrap">
                {selected.size} selected
              </span>
              <Separator orientation="vertical" className="h-5" />
              <Button
                size="sm"
                className="bg-emerald-500 hover:bg-emerald-600 text-white h-10"
                onClick={() => handleBatchAction('approved')}
              >
                <CheckCircle2 className="mr-1.5 h-4 w-4" />
                Approve
              </Button>
              <Button
                size="sm"
                className="bg-red-500 hover:bg-red-600 text-white h-10"
                onClick={() => handleBatchAction('rejected')}
              >
                <XCircle className="mr-1.5 h-4 w-4" />
                Reject
              </Button>
              <Button
                size="sm"
                className="bg-purple-500 hover:bg-purple-600 text-white h-10"
                onClick={() => setWorkflowOpen(true)}
                disabled={!hasAnyReviewProvider}
                title={!hasAnyReviewProvider ? 'Configure an LLM Review provider first' : 'Auto review, fact-check, and approve'}
              >
                <Sparkles className="mr-1.5 h-4 w-4" />
                Auto-Approve
              </Button>
              <Button
                size="sm"
                className="bg-blue-500 hover:bg-blue-600 text-white h-10"
                onClick={() => setReviewOpen(true)}
              >
                <Bot className="mr-1.5 h-4 w-4" />
                LLM Review
              </Button>
              <Button
                size="sm"
                variant="ghost"
                className="text-gray-500 h-10 ml-auto"
                onClick={() => setSelected(new Set())}
              >
                Clear
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Q&A list */}
      <div className="space-y-3">
        {pairsData && pairsData.items.length > 0 && (
          <div className="flex items-center gap-2 px-1">
            <Checkbox
              checked={selected.size === pairsData.items.length && selected.size > 0}
              onCheckedChange={selectAll}
            />
            <span className="text-xs text-muted-foreground">Select all on page</span>
          </div>
        )}

        {isLoading &&
          [1, 2, 3].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="space-y-3 py-4">
                <div className="h-4 w-3/4 rounded bg-muted" />
                <div className="h-3 w-full rounded bg-muted" />
                <div className="h-3 w-2/3 rounded bg-muted" />
              </CardContent>
            </Card>
          ))}

        {pairsData?.items.map((pair) => (
          <Card
            key={pair.id}
            className={`transition-shadow ${selected.has(pair.id) ? 'ring-2 ring-primary/30' : ''}`}
          >
            <CardContent className="py-4">
              <div className="flex items-start gap-3">
                <Checkbox
                  checked={selected.has(pair.id)}
                  onCheckedChange={() => toggleSelect(pair.id)}
                  className="mt-1"
                />
                <div className="flex-1 space-y-3">
                  {editingId === pair.id ? (
                    <div className="space-y-3">
                      <div>
                        <label className="text-xs font-medium text-muted-foreground">Question</label>
                        <Textarea
                          value={editQuestion}
                          onChange={(e) => setEditQuestion(e.target.value)}
                          className="mt-1"
                          rows={2}
                        />
                      </div>
                      <div>
                        <label className="text-xs font-medium text-muted-foreground">Answer</label>
                        <Textarea
                          value={editAnswer}
                          onChange={(e) => setEditAnswer(e.target.value)}
                          className="mt-1"
                          rows={4}
                        />
                      </div>
                      <div className="flex gap-2">
                        <Button size="sm" onClick={saveEdit}>
                          <Save className="mr-1 h-3.5 w-3.5" />
                          Save
                        </Button>
                        <Button size="sm" variant="ghost" onClick={() => setEditingId(null)}>
                          <X className="mr-1 h-3.5 w-3.5" />
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="flex items-start justify-between">
                        <p className="text-sm font-medium leading-snug">
                          Q: {pair.question}
                        </p>
                        <div className="flex items-center gap-1 shrink-0 ml-4">
                          {statusIcon(pair.validation_status)}
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-7 w-7"
                            onClick={() => startEdit(pair)}
                          >
                            <Edit3 className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        A: {pair.answer}
                      </p>
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge variant="outline" className="text-xs">
                          {pair.source_type}
                        </Badge>
                        {pair.source_document && (
                          <span className="inline-flex items-center gap-1 text-xs text-muted-foreground max-w-[200px] truncate" title={pair.source_document}>
                            <FileText className="h-3 w-3 shrink-0" />
                            {pair.source_document}
                          </span>
                        )}
                        {pair.quality_score != null && (
                          <Badge
                            variant={pair.quality_score >= 0.7 ? 'success' : pair.quality_score >= 0.4 ? 'warning' : 'destructive'}
                            className="text-xs"
                          >
                            {(pair.quality_score * 100).toFixed(0)}%
                          </Badge>
                        )}
                        {pair.human_edited && (
                          <Badge variant="info" className="text-xs">
                            Edited
                          </Badge>
                        )}
                        {pair.model_used && (
                          <span className="text-xs text-muted-foreground">{pair.model_used}</span>
                        )}
                      </div>

                      {/* LLM Review inline scores */}
                      {(reviewResults[pair.id] || (pair.metadata_json as Record<string, unknown>)?.llm_review) && (() => {
                        const review = reviewResults[pair.id] || (pair.metadata_json as Record<string, unknown>)?.llm_review as Record<string, unknown> | undefined;
                        if (!review) return null;
                        const overall = Number(review.overall ?? 0);
                        const rec = String(review.recommendation ?? 'revise');
                        return (
                          <div className="mt-2 rounded-md border bg-muted/30 p-2 space-y-1">
                            <div className="flex items-center gap-2 text-xs">
                              <Bot className="h-3.5 w-3.5 text-primary" />
                              <span className="font-medium">LLM Review</span>
                              <Badge
                                variant={rec === 'approve' ? 'success' : rec === 'reject' ? 'destructive' : 'warning'}
                                className="text-xs"
                              >
                                {rec}
                              </Badge>
                              <span className="ml-auto font-mono">{overall.toFixed(1)}/10</span>
                            </div>
                            <div className="grid grid-cols-4 gap-1 text-xs text-muted-foreground">
                              <span>Accuracy: {Number(review.accuracy ?? 0)}</span>
                              <span>Complete: {Number(review.completeness ?? 0)}</span>
                              <span>Clarity: {Number(review.clarity ?? 0)}</span>
                              <span>Relevance: {Number(review.relevance ?? 0)}</span>
                            </div>
                            {review.feedback && (
                              <p className="text-xs text-muted-foreground italic">
                                {String(review.feedback)}
                              </p>
                            )}
                          </div>
                        );
                      })()}

                      {/* Fact Check results */}
                      {factCheckResults[pair.id] && (() => {
                        const fc = factCheckResults[pair.id];
                        const score = Math.min(10, Math.max(0, fc.factual_accuracy));
                        const label =
                          score >= 9 ? '✓ Highly Accurate' :
                          score >= 7 ? '✓ Accurate' :
                          score >= 5 ? '⚠ Partially Accurate' :
                          '✗ Inaccurate';
                        const variant =
                          score >= 7 ? 'success' as const :
                          score >= 5 ? 'warning' as const :
                          'destructive' as const;
                        return (
                          <div className="mt-2 rounded-md border border-blue-200 bg-blue-50 dark:bg-blue-950/20 dark:border-blue-800 p-2 space-y-1">
                            <div className="flex items-center gap-2 text-xs">
                              <ShieldCheck className="h-3.5 w-3.5 text-blue-600" />
                              <span className="font-medium">Fact Check</span>
                              <Badge variant={variant} className="text-xs">
                                {score.toFixed(1)} / 10 — {label}
                              </Badge>
                            </div>
                            {fc.suggested_answer && (
                              <div className="text-xs space-y-1.5">
                                <div>
                                  <span className="font-medium">Suggested:</span>{' '}
                                  <span className="text-muted-foreground">{fc.suggested_answer}</span>
                                </div>
                                <div className="flex gap-1">
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    className="h-6 text-xs border-green-300 text-green-700 hover:bg-green-50 dark:border-green-700 dark:text-green-400 dark:hover:bg-green-950/30"
                                    onClick={() => handleAcceptSuggestion(pair.id)}
                                    disabled={acceptSuggestion.isPending}
                                  >
                                    {acceptSuggestion.isPending ? (
                                      <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                                    ) : (
                                      <CheckCircle2 className="mr-1 h-3 w-3" />
                                    )}
                                    Accept Suggestion
                                  </Button>
                                </div>
                              </div>
                            )}
                            {/* Undo: show if suggestion was previously applied */}
                            {!!(pair.metadata_json as Record<string, unknown>)?.suggestion_applied && !fc.suggested_answer && (
                              <div className="flex items-center gap-1 text-xs">
                                <Badge variant="info" className="text-xs">Suggestion Applied</Badge>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-5 text-xs px-1"
                                  onClick={() => handleRevertSuggestion(pair.id)}
                                  disabled={revertSuggestion.isPending}
                                >
                                  <Undo2 className="mr-1 h-3 w-3" />
                                  Undo
                                </Button>
                              </div>
                            )}
                            {fc.analysis && fc.analysis.length > 0 && (
                              <ul className="text-xs text-muted-foreground list-disc list-inside">
                                {fc.analysis.map((a, i) => (
                                  <li key={i}>{a}</li>
                                ))}
                              </ul>
                            )}
                          </div>
                        );
                      })()}
                    </>
                  )}
                </div>
              </div>

              {/* Quick actions row */}
              {editingId !== pair.id && (
                <div className="mt-3 flex gap-1 pl-9">
                  {pair.validation_status !== 'approved' && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 text-xs"
                      onClick={() =>
                        updatePair.mutate({ pairId: pair.id, data: { validation_status: 'approved' } })
                      }
                    >
                      <CheckCircle2 className="mr-1 h-3 w-3 text-success-500" />
                      Approve
                    </Button>
                  )}
                  {pair.validation_status !== 'rejected' && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 text-xs"
                      onClick={() =>
                        updatePair.mutate({ pairId: pair.id, data: { validation_status: 'rejected' } })
                      }
                    >
                      <XCircle className="mr-1 h-3 w-3 text-destructive" />
                      Reject
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 text-xs"
                    onClick={() => handleFactCheck(pair.id)}
                    disabled={factCheckingId === pair.id || !hasAnyReviewProvider}
                    title={!hasAnyReviewProvider ? 'Configure an LLM Review provider first' : `Fact-check using ${reviewProviderLabel}${reviewModelLabel ? ': ' + reviewModelLabel : ''}`}
                  >
                    {factCheckingId === pair.id ? (
                      <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                    ) : (
                      <ShieldCheck className="mr-1 h-3 w-3 text-blue-600" />
                    )}
                    Fact Check
                    {reviewModelLabel && (
                      <span className="ml-1 text-muted-foreground font-normal">({reviewModelLabel})</span>
                    )}
                  </Button>
                  {!!(pair.metadata_json as Record<string, unknown>)?.suggestion_applied && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 text-xs text-orange-600"
                      onClick={() => handleRevertSuggestion(pair.id)}
                      disabled={revertSuggestion.isPending}
                    >
                      <Undo2 className="mr-1 h-3 w-3" />
                      Undo Suggestion
                    </Button>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Pagination */}
      {pairsData && pairsData.total_pages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Page {pairsData.page} of {pairsData.total_pages} ({formatNumber(pairsData.total)} total)
          </p>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={page <= 1}
              onClick={() => setPage((p) => p - 1)}
            >
              <ChevronLeft className="mr-1 h-4 w-4" />
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={page >= pairsData.total_pages}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
              <ChevronRight className="ml-1 h-4 w-4" />
            </Button>
          </div>
        </div>
      )}

      {/* LLM Review Dialog */}
      <Dialog open={reviewOpen} onOpenChange={setReviewOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>LLM Review</DialogTitle>
            <DialogDescription>
              Review {selected.size} selected Q&A pair{selected.size !== 1 ? 's' : ''} using an external LLM.
              {selected.size > 5 && ' Session-based review will be used for progress tracking.'}
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label>Provider</Label>
              <Select value={reviewProvider} onValueChange={(v) => { setReviewProvider(v); setReviewModel(''); }}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {providers?.map((p) => (
                    <SelectItem key={p.name} value={p.name}>
                      {p.name.charAt(0).toUpperCase() + p.name.slice(1)}
                      {!p.requires_api_key && ' (Local)'}
                      {p.has_stored_key && ' ✓'}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {selectedProviderInfo?.requires_api_key && (
              <div className="grid gap-2">
                {hasStoredKey ? (
                  <div className="flex items-center gap-2 rounded-md bg-green-50 dark:bg-green-950/20 p-2 text-sm">
                    <CheckCircle2 className="h-4 w-4 text-green-600" />
                    <span>Using stored API key{storedKeyForProvider?.display_name ? ` (${storedKeyForProvider.display_name})` : ''}</span>
                  </div>
                ) : (
                  <>
                    <Label>API Key</Label>
                    <Input
                      type="password"
                      placeholder="sk-... or your API key"
                      value={reviewApiKey}
                      onChange={(e) => setReviewApiKey(e.target.value)}
                    />
                    <p className="text-xs text-muted-foreground">
                      Tip: Save API keys in Settings to avoid re-entering them.
                    </p>
                  </>
                )}
              </div>
            )}

            <div className="grid gap-2">
              <Label>Model</Label>
              <Select value={reviewModel} onValueChange={setReviewModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select model (optional)" />
                </SelectTrigger>
                <SelectContent>
                  {(providers?.find((p) => p.name === reviewProvider)?.models || []).map((m) => (
                    <SelectItem key={m} value={m}>{m}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setReviewOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleLLMReview}
              disabled={
                reviewMutation.isPending || startSession.isPending ||
                (selectedProviderInfo?.requires_api_key && !hasStoredKey && !reviewApiKey)
              }
            >
              {(reviewMutation.isPending || startSession.isPending) ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Starting…
                </>
              ) : (
                <>
                  <Bot className="mr-2 h-4 w-4" />
                  Review {selected.size} pair{selected.size !== 1 ? 's' : ''}
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Review Progress Dialog */}
      <Dialog open={progressOpen} onOpenChange={(open) => {
        if (!open && sessionData && ['completed', 'failed', 'cancelled'].includes(sessionData.status)) {
          setProgressOpen(false);
          setActiveSessionId(null);
        }
      }}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Review Progress</DialogTitle>
            <DialogDescription>
              {sessionData?.status === 'in_progress' && 'Reviewing Q&A pairs with rate limit handling…'}
              {sessionData?.status === 'pending' && 'Starting review session…'}
              {sessionData?.status === 'completed' && 'Review completed!'}
              {sessionData?.status === 'failed' && 'Review session failed.'}
              {sessionData?.status === 'cancelled' && 'Review session cancelled.'}
            </DialogDescription>
          </DialogHeader>

          {sessionData && (
            <div className="space-y-4 py-2">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>
                    {sessionData.completed_pairs} / {sessionData.total_pairs} pairs
                  </span>
                  <span>{sessionProgress}%</span>
                </div>
                <Progress value={sessionProgress} />
              </div>

              {sessionData.current_message && (
                <p className="text-xs text-muted-foreground bg-muted/50 rounded p-2">
                  {sessionData.current_message}
                </p>
              )}

              {sessionData.error_message && (
                <p className="text-xs text-destructive bg-destructive/10 rounded p-2">
                  {sessionData.error_message}
                </p>
              )}

              <div className="grid grid-cols-3 gap-2 text-center">
                <div className="rounded border p-2">
                  <p className="text-lg font-bold text-green-600">{sessionData.approved_count}</p>
                  <p className="text-xs text-muted-foreground">Approved</p>
                </div>
                <div className="rounded border p-2">
                  <p className="text-lg font-bold text-yellow-600">{sessionData.revise_count}</p>
                  <p className="text-xs text-muted-foreground">Revise</p>
                </div>
                <div className="rounded border p-2">
                  <p className="text-lg font-bold text-red-600">{sessionData.rejected_count}</p>
                  <p className="text-xs text-muted-foreground">Rejected</p>
                </div>
              </div>

              {sessionData.avg_overall_score != null && (
                <p className="text-sm text-center">
                  Average Score: <span className="font-bold">{sessionData.avg_overall_score.toFixed(1)}</span>/10
                </p>
              )}

              {sessionData.total_cost_usd > 0 && (
                <p className="text-xs text-muted-foreground text-center">
                  Estimated cost: ${sessionData.total_cost_usd.toFixed(4)}
                </p>
              )}
            </div>
          )}

          <DialogFooter>
            {sessionData?.status === 'in_progress' && (
              <Button
                variant="destructive"
                size="sm"
                onClick={() => activeSessionId && cancelSession.mutate(activeSessionId)}
                disabled={cancelSession.isPending}
              >
                Cancel Review
              </Button>
            )}
            {(sessionData?.status === 'completed' || sessionData?.status === 'failed' || sessionData?.status === 'cancelled') && (
              <Button
                onClick={() => {
                  setProgressOpen(false);
                  setActiveSessionId(null);
                }}
              >
                Close
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Auto-Approve Workflow Config Dialog */}
      <Dialog open={workflowOpen} onOpenChange={setWorkflowOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              <Sparkles className="inline h-5 w-5 mr-1 text-purple-500" />
              Auto-Approve Workflow
            </DialogTitle>
            <DialogDescription>
              Automatically review, fact-check, and approve {selected.size} selected pair{selected.size !== 1 ? 's' : ''}.
              Pairs scoring ≥ threshold are auto-approved; others remain pending.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            {/* Provider & model info */}
            <div className="rounded-md bg-muted/50 p-3 text-sm space-y-1">
              <div className="flex items-center gap-2">
                <Bot className="h-4 w-4 text-primary" />
                <span className="font-medium">Using: {reviewProviderLabel}</span>
                {reviewModelLabel && <Badge variant="outline" className="text-xs">{reviewModelLabel}</Badge>}
              </div>
              <p className="text-xs text-muted-foreground">
                Configure provider/model in the LLM Review dialog.
              </p>
            </div>

            {/* Threshold slider */}
            <div className="grid gap-2">
              <Label>
                Approval Threshold: {approvalThreshold.toFixed(1)} / 10
              </Label>
              <Slider
                value={[approvalThreshold]}
                onValueChange={([v]) => setApprovalThreshold(v)}
                min={6.0}
                max={10.0}
                step={0.5}
              />
              <p className="text-xs text-muted-foreground">
                Pairs with fact-check score ≥ {approvalThreshold.toFixed(1)} are auto-approved
              </p>
            </div>

            {/* Auto-accept suggestions toggle */}
            <div className="flex items-center justify-between rounded-md border p-3">
              <div>
                <Label className="text-sm">Auto-accept borderline suggestions</Label>
                <p className="text-xs text-muted-foreground mt-0.5">
                  If score is 6.0—{(approvalThreshold - 0.1).toFixed(1)} and a suggestion is available,
                  apply it and re-evaluate
                </p>
              </div>
              <Switch
                checked={autoAcceptSuggestions}
                onCheckedChange={setAutoAcceptSuggestions}
              />
            </div>

            {/* Workflow steps preview */}
            <div className="rounded-md border p-3 space-y-2">
              <p className="text-xs font-medium">Workflow per pair:</p>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Badge variant="outline" className="text-xs">1</Badge>
                <span>LLM Review (quality scoring)</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Badge variant="outline" className="text-xs">2</Badge>
                <span>Fact Check (accuracy scoring)</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Badge variant="outline" className="text-xs">3</Badge>
                <span>Auto-approve if score ≥ {approvalThreshold.toFixed(1)}</span>
              </div>
              {autoAcceptSuggestions && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Badge variant="outline" className="text-xs">3b</Badge>
                  <span>Accept suggestion + re-evaluate if borderline</span>
                </div>
              )}
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setWorkflowOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleStartAutoApproveWorkflow}
              disabled={autoApproveWorkflow.isPending || !hasAnyReviewProvider}
              className="bg-purple-600 hover:bg-purple-700"
            >
              {autoApproveWorkflow.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Starting…
                </>
              ) : (
                <>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Start Workflow
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Auto-Approve Workflow Progress Dialog */}
      <Dialog open={workflowProgressOpen} onOpenChange={(open) => {
        if (!open && workflowSessionData && ['completed', 'failed', 'cancelled'].includes(workflowSessionData.status)) {
          setWorkflowProgressOpen(false);
          setWorkflowSessionId(null);
        }
      }}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              <Sparkles className="inline h-5 w-5 mr-1 text-purple-500" />
              Auto-Approve Progress
            </DialogTitle>
            <DialogDescription>
              {workflowSessionData?.status === 'in_progress' && 'Running automated review → fact-check → approve…'}
              {workflowSessionData?.status === 'pending' && 'Starting auto-approve workflow…'}
              {workflowSessionData?.status === 'completed' && 'Workflow completed!'}
              {workflowSessionData?.status === 'failed' && 'Workflow failed.'}
              {workflowSessionData?.status === 'cancelled' && 'Workflow cancelled.'}
            </DialogDescription>
          </DialogHeader>

          {workflowSessionData && (
            <div className="space-y-4 py-2">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>
                    {workflowSessionData.completed_pairs} / {workflowSessionData.total_pairs} pairs
                  </span>
                  <span>{workflowProgress}%</span>
                </div>
                <Progress value={workflowProgress} />
              </div>

              {workflowSessionData.current_message && (
                <p className="text-xs text-muted-foreground bg-muted/50 rounded p-2">
                  {workflowSessionData.current_message}
                </p>
              )}

              {workflowSessionData.error_message && (
                <p className="text-xs text-destructive bg-destructive/10 rounded p-2">
                  {workflowSessionData.error_message}
                </p>
              )}

              {/* Results summary */}
              {workflowSessionData.results && workflowSessionData.results.length > 0 && (() => {
                const results = workflowSessionData.results as unknown as Array<Record<string, unknown>>;
                const approved = results.filter(r => r.decision === 'approved').length;
                const pending = results.filter(r => r.decision === 'pending').length;
                const errors = results.filter(r => r.decision === 'error').length;
                const suggestionsApplied = results.filter(r => (r as Record<string, unknown>).re_evaluation).length;
                return (
                  <div className="space-y-2">
                    <div className="grid grid-cols-3 gap-2 text-center">
                      <div className="rounded border border-green-200 bg-green-50 dark:bg-green-950/20 p-2">
                        <p className="text-lg font-bold text-green-600">{approved}</p>
                        <p className="text-xs text-muted-foreground">Auto-Approved</p>
                      </div>
                      <div className="rounded border border-yellow-200 bg-yellow-50 dark:bg-yellow-950/20 p-2">
                        <p className="text-lg font-bold text-yellow-600">{pending}</p>
                        <p className="text-xs text-muted-foreground">Needs Review</p>
                      </div>
                      <div className="rounded border p-2">
                        <p className="text-lg font-bold text-red-600">{errors}</p>
                        <p className="text-xs text-muted-foreground">Errors</p>
                      </div>
                    </div>
                    {suggestionsApplied > 0 && (
                      <p className="text-xs text-center text-muted-foreground">
                        {suggestionsApplied} suggestion{suggestionsApplied !== 1 ? 's' : ''} auto-applied
                      </p>
                    )}
                  </div>
                );
              })()}

              {workflowSessionData.avg_overall_score != null && (
                <p className="text-sm text-center">
                  Avg Review Score: <span className="font-bold">{workflowSessionData.avg_overall_score.toFixed(1)}</span>/10
                </p>
              )}

              {workflowSessionData.total_cost_usd > 0 && (
                <p className="text-xs text-muted-foreground text-center">
                  Estimated cost: ${workflowSessionData.total_cost_usd.toFixed(4)}
                </p>
              )}
            </div>
          )}

          <DialogFooter>
            {(workflowSessionData?.status === 'in_progress' || workflowSessionData?.status === 'pending') && (
              <Button
                variant="destructive"
                size="sm"
                onClick={() => workflowSessionId && cancelSession.mutate(workflowSessionId)}
                disabled={cancelSession.isPending}
              >
                Cancel Workflow
              </Button>
            )}
            {(workflowSessionData?.status === 'completed' || workflowSessionData?.status === 'failed' || workflowSessionData?.status === 'cancelled') && (
              <Button
                onClick={() => {
                  setWorkflowProgressOpen(false);
                  setWorkflowSessionId(null);
                }}
              >
                Close
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}