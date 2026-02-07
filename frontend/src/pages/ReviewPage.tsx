import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Search,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Clock,
  Edit3,
  Save,
  X,
  ChevronLeft,
  ChevronRight,
  Filter,
} from 'lucide-react';
import {
  useProject,
  useQAPairs,
  useQAPairStats,
  useUpdateQAPair,
  useBatchUpdateQAPairs,
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
import { formatNumber } from '@/lib/utils';
import type { QAPair, ValidationStatus } from '@/types';

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

  // Selection
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editQuestion, setEditQuestion] = useState('');
  const [editAnswer, setEditAnswer] = useState('');

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
          <Select
            value={sortBy}
            onValueChange={(v) => setSortBy(v)}
          >
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

      {/* Batch actions */}
      {selected.size > 0 && (
        <div className="flex items-center gap-3 rounded-lg border bg-muted/50 p-3">
          <span className="text-sm font-medium">
            {selected.size} selected
          </span>
          <Separator orientation="vertical" className="h-5" />
          <Button size="sm" variant="outline" onClick={() => handleBatchAction('approved')}>
            <CheckCircle2 className="mr-1 h-3.5 w-3.5 text-success-500" />
            Approve
          </Button>
          <Button size="sm" variant="outline" onClick={() => handleBatchAction('rejected')}>
            <XCircle className="mr-1 h-3.5 w-3.5 text-destructive" />
            Reject
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setSelected(new Set())}>
            Clear
          </Button>
        </div>
      )}

      {/* Q&A list */}
      <div className="space-y-3">
        {/* Select all */}
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
                    /* Edit mode */
                    <div className="space-y-3">
                      <div>
                        <label className="text-xs font-medium text-muted-foreground">
                          Question
                        </label>
                        <Textarea
                          value={editQuestion}
                          onChange={(e) => setEditQuestion(e.target.value)}
                          className="mt-1"
                          rows={2}
                        />
                      </div>
                      <div>
                        <label className="text-xs font-medium text-muted-foreground">
                          Answer
                        </label>
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
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => setEditingId(null)}
                        >
                          <X className="mr-1 h-3.5 w-3.5" />
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : (
                    /* View mode */
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
                          <span className="text-xs text-muted-foreground">
                            {pair.model_used}
                          </span>
                        )}
                      </div>
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
                        updatePair.mutate({
                          pairId: pair.id,
                          data: { validation_status: 'approved' },
                        })
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
                        updatePair.mutate({
                          pairId: pair.id,
                          data: { validation_status: 'rejected' },
                        })
                      }
                    >
                      <XCircle className="mr-1 h-3 w-3 text-destructive" />
                      Reject
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
    </div>
  );
}
