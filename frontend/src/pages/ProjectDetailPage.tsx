import { useState, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import {
  ArrowLeft,
  Upload,
  Play,
  X,
  FileText,
  Globe,
  Cpu,
  Download,
  Eye,
  Trash2,
  Loader2,
  Settings2,
} from 'lucide-react';
import {
  useProject,
  useSources,
  useUploadSources,
  useDeleteSource,
  useStartGeneration,
  useJobs,
  useCancelGeneration,
  useQAPairStats,
  useOllamaStatus,
} from '@/hooks/use-api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Separator } from '@/components/ui/separator';
import { formatNumber, formatDate } from '@/lib/utils';
import { exportApi } from '@/lib/api';
import type { ExportFormat, GenerationJob } from '@/types';

export default function ProjectDetailPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();

  const { data: project, isLoading: projectLoading } = useProject(projectId!);
  const { data: sources } = useSources(projectId!);
  const { data: jobs } = useJobs(projectId!);
  const { data: stats } = useQAPairStats(projectId!);
  const { data: ollama } = useOllamaStatus();

  const uploadSources = useUploadSources(projectId!);
  const deleteSource = useDeleteSource(projectId!);
  const startGeneration = useStartGeneration(projectId!);
  const cancelGeneration = useCancelGeneration(projectId!);

  // Generation config state
  const [configOpen, setConfigOpen] = useState(false);
  const [model, setModel] = useState('');
  const [medicalTerms, setMedicalTerms] = useState('');
  const [pubmedEmail, setPubmedEmail] = useState('');
  const [targetPairs, setTargetPairs] = useState(1000);
  const [difficulty, setDifficulty] = useState('intermediate');
  const [chunkSize, setChunkSize] = useState(500);
  const [includePubmed, setIncludePubmed] = useState(false);
  const [pubmedMax, setPubmedMax] = useState(1000);
  const [temperature, setTemperature] = useState(0.7);
  const [minQuality, setMinQuality] = useState(0.6);

  // Export state
  const [exportFormat, setExportFormat] = useState<ExportFormat>('csv');
  const [exporting, setExporting] = useState(false);

  // File drop
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        uploadSources.mutate(acceptedFiles);
      }
    },
    [uploadSources],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/xml': ['.xml'],
      'application/xml': ['.xml'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    multiple: true,
  });

  const handleStartGeneration = () => {
    startGeneration.mutate({
      ollama_model: model || (ollama?.models?.[0]?.name ?? 'llama3'),
      medical_terms: medicalTerms || project?.domain || 'medical conditions',
      email: pubmedEmail,
      target_pairs: targetPairs,
      difficulty_levels: difficulty === 'mixed'
        ? ['beginner', 'intermediate', 'advanced']
        : [difficulty],
      chunk_size: chunkSize,
      use_pubmed: includePubmed,
      pubmed_retmax: pubmedMax,
      temperature,
      min_quality_score: minQuality,
    });
    setConfigOpen(false);
  };

  const handleExport = async () => {
    if (!projectId) return;
    setExporting(true);
    try {
      await exportApi.download(projectId, {
        format: exportFormat,
        validation_statuses: undefined,
        min_quality_score: 0,
        train_split: 0.8,
        val_split: 0.1,
        test_split: 0.1,
        include_metadata: false,
      });
    } finally {
      setExporting(false);
    }
  };

  const activeJob = jobs?.find(
    (j) => j.status === 'in_progress' || j.status === 'queued',
  );

  if (projectLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!project) {
    return (
      <div className="py-20 text-center">
        <p className="text-muted-foreground">Project not found.</p>
        <Button variant="link" onClick={() => navigate('/dashboard')}>
          Back to dashboard
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" onClick={() => navigate('/dashboard')}>
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <div className="flex-1">
          <h1 className="text-2xl font-bold tracking-tight">{project.name}</h1>
          <p className="text-sm text-muted-foreground">{project.domain}</p>
        </div>
        <div className="flex items-center gap-2">
          <Link to={`/projects/${projectId}/review`}>
            <Button variant="outline">
              <Eye className="mr-2 h-4 w-4" />
              Review Q&A Pairs
            </Button>
          </Link>
          <Button
            onClick={() => setConfigOpen(true)}
            disabled={!!activeJob || !ollama?.connected}
          >
            <Play className="mr-2 h-4 w-4" />
            Generate
          </Button>
        </div>
      </div>

      {/* Stats cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Sources</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{sources?.length ?? 0}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Q&A Pairs</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatNumber(stats?.total ?? 0)}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Approved</CardTitle>
            <Badge variant="success" className="text-xs">{stats?.approved ?? 0}</Badge>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-success-600">
              {stats?.total ? ((stats.approved / stats.total) * 100).toFixed(0) : 0}%
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Quality</CardTitle>
            <Globe className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {stats?.avg_quality_score != null
                ? `${(stats.avg_quality_score * 100).toFixed(0)}%`
                : '—'}
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="sources" className="space-y-4">
        <TabsList>
          <TabsTrigger value="sources">Sources</TabsTrigger>
          <TabsTrigger value="jobs">Generation Jobs</TabsTrigger>
          <TabsTrigger value="export">Export</TabsTrigger>
        </TabsList>

        {/* ---- Sources Tab ---- */}
        <TabsContent value="sources" className="space-y-4">
          {/* Drop zone */}
          <div
            {...getRootProps()}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors ${
              isDragActive
                ? 'border-primary bg-primary/5'
                : 'border-border hover:border-primary/50'
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="h-10 w-10 text-muted-foreground/50" />
            <p className="mt-2 text-sm font-medium">
              {isDragActive
                ? 'Drop files here…'
                : 'Drag & drop files, or click to browse'}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              PDF, XML (MedQuAD), DOCX — max 50 MB per file
            </p>
            {uploadSources.isPending && (
              <div className="mt-3 flex items-center gap-2 text-sm text-primary">
                <Loader2 className="h-4 w-4 animate-spin" />
                Uploading…
              </div>
            )}
          </div>

          {/* File list */}
          {sources && sources.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Uploaded Sources</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {sources.map((src) => (
                  <div
                    key={src.id}
                    className="flex items-center justify-between rounded-md border p-3"
                  >
                    <div className="flex items-center gap-3">
                      <FileText className="h-5 w-5 text-muted-foreground" />
                      <div>
                        <p className="text-sm font-medium">{src.filename}</p>
                        <p className="text-xs text-muted-foreground">
                          {src.file_type.toUpperCase()} ·{' '}
                          {src.size_bytes != null
                            ? `${(src.size_bytes / 1024).toFixed(0)} KB`
                            : 'Unknown size'}
                        </p>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => deleteSource.mutate(src.id)}
                    >
                      <Trash2 className="h-4 w-4 text-muted-foreground" />
                    </Button>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* ---- Jobs Tab ---- */}
        <TabsContent value="jobs" className="space-y-4">
          {activeJob && (
            <Card className="border-primary/30 bg-primary/5">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">
                    Active Generation
                  </CardTitle>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => cancelGeneration.mutate(activeJob.id)}
                  >
                    <X className="mr-1 h-3 w-3" />
                    Cancel
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">
                    {activeJob.current_message || 'Starting…'}
                  </span>
                  <span className="font-medium">{activeJob.progress_pct}%</span>
                </div>
                <Progress value={activeJob.progress_pct} />
              </CardContent>
            </Card>
          )}

          {jobs && jobs.length > 0 ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Job History</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {jobs
                  .filter((j) => j.id !== activeJob?.id)
                  .map((job) => (
                    <JobRow key={job.id} job={job} />
                  ))}
              </CardContent>
            </Card>
          ) : (
            <Card className="flex flex-col items-center py-12">
              <Cpu className="h-10 w-10 text-muted-foreground/40" />
              <p className="mt-3 text-sm text-muted-foreground">
                No generation jobs yet. Upload sources and click Generate.
              </p>
            </Card>
          )}
        </TabsContent>

        {/* ---- Export Tab ---- */}
        <TabsContent value="export" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Export Dataset</CardTitle>
              <CardDescription>
                Download your Q&A pairs in various formats
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-2">
                <Label>Format</Label>
                <Select
                  value={exportFormat}
                  onValueChange={(v) => setExportFormat(v as ExportFormat)}
                >
                  <SelectTrigger className="w-60">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="csv">CSV</SelectItem>
                    <SelectItem value="json">JSON</SelectItem>
                    <SelectItem value="jsonl">JSONL</SelectItem>
                    <SelectItem value="alpaca">Alpaca Format</SelectItem>
                    <SelectItem value="openai">OpenAI Format</SelectItem>
                    <SelectItem value="parquet">Parquet</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button
                onClick={handleExport}
                disabled={exporting || !stats?.total}
              >
                {exporting ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Download className="mr-2 h-4 w-4" />
                )}
                Export {formatNumber(stats?.total ?? 0)} pairs
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Generation Config Dialog */}
      <Dialog open={configOpen} onOpenChange={setConfigOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Generation Configuration</DialogTitle>
            <DialogDescription>
              Configure how Q&A pairs will be generated
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4 max-h-[60vh] overflow-y-auto pr-2">
            {/* Medical Terms */}
            <div className="grid gap-2">
              <Label>Medical Terms / Keywords</Label>
              <Input
                placeholder={`e.g. ${project?.domain || 'heart failure, diabetes, hypertension'}`}
                value={medicalTerms}
                onChange={(e) => setMedicalTerms(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                Comma-separated. Leave empty to use project domain.
              </p>
            </div>

            {/* Model */}
            <div className="grid gap-2">
              <Label>Ollama Model</Label>
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {ollama?.models?.map((m) => (
                    <SelectItem key={m.name} value={m.name}>
                      {m.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Target pairs */}
            <div className="grid gap-2">
              <Label>Target Q&A Pairs: {targetPairs}</Label>
              <Slider
                value={[targetPairs]}
                onValueChange={([v]) => setTargetPairs(v)}
                min={10}
                max={5000}
                step={10}
              />
            </div>

            {/* Difficulty */}
            <div className="grid gap-2">
              <Label>Difficulty</Label>
              <Select value={difficulty} onValueChange={setDifficulty}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="mixed">Mixed</SelectItem>
                  <SelectItem value="beginner">Beginner</SelectItem>
                  <SelectItem value="intermediate">Intermediate</SelectItem>
                  <SelectItem value="advanced">Advanced</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Chunk size */}
            <div className="grid gap-2">
              <Label>Chunk Size (words): {chunkSize}</Label>
              <Slider
                value={[chunkSize]}
                onValueChange={([v]) => setChunkSize(v)}
                min={100}
                max={2000}
                step={50}
              />
            </div>

            {/* Temperature */}
            <div className="grid gap-2">
              <Label>Temperature: {temperature.toFixed(1)}</Label>
              <Slider
                value={[temperature]}
                onValueChange={([v]) => setTemperature(v)}
                min={0}
                max={1.5}
                step={0.1}
              />
            </div>

            {/* Min quality */}
            <div className="grid gap-2">
              <Label>Min Quality Score: {(minQuality * 100).toFixed(0)}%</Label>
              <Slider
                value={[minQuality]}
                onValueChange={([v]) => setMinQuality(v)}
                min={0}
                max={1}
                step={0.05}
              />
            </div>

            <Separator />

            {/* PubMed */}
            <div className="flex items-center justify-between">
              <Label htmlFor="pubmed-toggle">Include PubMed Articles</Label>
              <Switch
                id="pubmed-toggle"
                checked={includePubmed}
                onCheckedChange={setIncludePubmed}
              />
            </div>
            {includePubmed && (
              <>
                <div className="grid gap-2">
                  <Label>PubMed Email (required by NCBI)</Label>
                  <Input
                    placeholder="your-email@example.com"
                    value={pubmedEmail}
                    onChange={(e) => setPubmedEmail(e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    NCBI requires an email address for PubMed API access.
                  </p>
                </div>
                <div className="grid gap-2">
                  <Label>Max PubMed Articles: {pubmedMax}</Label>
                  <Slider
                    value={[pubmedMax]}
                    onValueChange={([v]) => setPubmedMax(v)}
                    min={10}
                    max={5000}
                    step={10}
                  />
                </div>
              </>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfigOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleStartGeneration} disabled={startGeneration.isPending}>
              <Settings2 className="mr-2 h-4 w-4" />
              {startGeneration.isPending ? 'Starting…' : 'Start Generation'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function JobRow({ job }: { job: GenerationJob }) {
  const statusColor: Record<string, string> = {
    completed: 'success',
    failed: 'destructive',
    cancelled: 'warning',
    queued: 'secondary',
    in_progress: 'info',
  };

  const statusLabel: Record<string, string> = {
    completed: 'Completed',
    failed: 'Failed',
    cancelled: 'Cancelled',
    queued: 'Queued',
    in_progress: 'In Progress',
  };

  return (
    <div className="flex items-center justify-between rounded-md border p-3">
      <div>
        <div className="flex items-center gap-2">
          <Badge variant={statusColor[job.status] as 'success' | 'destructive' | 'warning' | 'secondary' | 'info'}>
            {statusLabel[job.status] ?? job.status}
          </Badge>
          {job.current_message && (
            <span className="text-sm text-muted-foreground">
              {job.current_message}
            </span>
          )}
        </div>
        {job.error_message && (
          <p className="mt-1 text-xs text-destructive">{job.error_message}</p>
        )}
      </div>
      <span className="text-xs text-muted-foreground">
        {job.completed_at ? formatDate(job.completed_at) : formatDate(job.created_at)}
      </span>
    </div>
  );
}
