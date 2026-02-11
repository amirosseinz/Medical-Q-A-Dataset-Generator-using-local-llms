import { useState, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { useQuery } from '@tanstack/react-query';
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
  AlertTriangle,
  CheckCircle2,
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
  useGenerationProviders,
} from '@/hooks/use-api';
import { useJobCompletionWatcher } from '@/hooks/use-job-watcher';
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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { InfoTooltip } from '@/components/ui/info-tooltip';
import { formatNumber, formatDate } from '@/lib/utils';
import { exportApi, generationApi } from '@/lib/api';
import type { ExportFormat, GenerationJob } from '@/types';

export default function ProjectDetailPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();

  const { data: project, isLoading: projectLoading } = useProject(projectId!);
  const { data: sources } = useSources(projectId!);
  const { data: jobs } = useJobs(projectId!);
  const { data: stats } = useQAPairStats(projectId!);
  const { data: ollama } = useOllamaStatus();
  const { data: genProviders } = useGenerationProviders();

  // Auto-refresh: watches jobs and invalidates queries on completion
  useJobCompletionWatcher(projectId);

  const uploadSources = useUploadSources(projectId!);
  const deleteSource = useDeleteSource(projectId!);
  const startGeneration = useStartGeneration(projectId!);
  const cancelGeneration = useCancelGeneration(projectId!);

  // Generation config state
  const [configOpen, setConfigOpen] = useState(false);
  const [genProvider, setGenProvider] = useState('ollama');
  const [model, setModel] = useState('');
  const [medicalTerms, setMedicalTerms] = useState('');
  const [pubmedEmail, setPubmedEmail] = useState('');
  const [targetPairs, setTargetPairs] = useState(50);
  const [difficulty, setDifficulty] = useState('intermediate');
  const [chunkSize, setChunkSize] = useState(500);
  const [includePubmed, setIncludePubmed] = useState(false);
  const [pubmedMax, setPubmedMax] = useState(1000);
  const [temperature, setTemperature] = useState(0.7);
  const [minQuality, setMinQuality] = useState(0.4);
  const [maxWorkers, setMaxWorkers] = useState(5);
  const [pdfChunkLimit, setPdfChunkLimit] = useState(0); // 0 = unlimited
  const [pubmedChunkLimit, setPubmedChunkLimit] = useState(0); // 0 = unlimited

  // API key validation for cloud providers
  const isCloudProvider = genProvider !== 'ollama';
  const { data: keyValidation, isLoading: keyValidating } = useQuery({
    queryKey: ['generation', 'validate-key', genProvider],
    queryFn: () => generationApi.validateKey(genProvider),
    enabled: isCloudProvider && configOpen,
    staleTime: 30_000,
  });
  const keyMissing = isCloudProvider && keyValidation && !keyValidation.has_key;

  // Export state
  const [exportFormat, setExportFormat] = useState<ExportFormat>('csv');
  const [exporting, setExporting] = useState(false);
  const [exportingJobId, setExportingJobId] = useState<string | null>(null);
  const [trainSplit, setTrainSplit] = useState(80);
  const [valSplit, setValSplit] = useState(10);
  const [testSplit, setTestSplit] = useState(10);

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
    const selectedProvider = genProviders?.find(p => p.name === genProvider);
    startGeneration.mutate({
      provider: genProvider,
      api_key_id: selectedProvider?.stored_key_id ?? undefined,
      ollama_model: model || (genProvider === 'ollama'
        ? (ollama?.models?.[0]?.name ?? 'llama3')
        : (selectedProvider?.models?.[0] ?? '')),
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
      max_workers: maxWorkers,
      pdf_chunk_limit: pdfChunkLimit > 0 ? pdfChunkLimit : undefined,
      pubmed_chunk_limit: pubmedChunkLimit > 0 ? pubmedChunkLimit : undefined,
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
        train_split: trainSplit / 100,
        val_split: valSplit / 100,
        test_split: testSplit / 100,
        include_metadata: false,
      });
    } finally {
      setExporting(false);
    }
  };

  const handleExportGeneration = async (jobId: string, format: ExportFormat = 'csv') => {
    if (!projectId) return;
    setExportingJobId(jobId);
    try {
      await exportApi.download(projectId, {
        format,
        generation_job_id: jobId,
        validation_statuses: undefined,
        min_quality_score: 0,
        train_split: trainSplit / 100,
        val_split: valSplit / 100,
        test_split: testSplit / 100,
        include_metadata: false,
      });
    } finally {
      setExportingJobId(null);
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
            disabled={!!activeJob || (!ollama?.connected && !genProviders?.some(p => p.has_stored_key))}
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
                    <JobRow
                      key={job.id}
                      job={job}
                      onExport={(format) => handleExportGeneration(job.id, format)}
                      isExporting={exportingJobId === job.id}
                    />
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

              {/* Dataset Split Configuration */}
              <Separator />
              <div className="space-y-3">
                <Label className="text-sm font-medium">Train / Validation / Test Split</Label>
                <div className="flex items-center gap-2 flex-wrap">
                  {[
                    { label: '80/10/10', t: 80, v: 10, te: 10 },
                    { label: '70/15/15', t: 70, v: 15, te: 15 },
                    { label: '90/5/5', t: 90, v: 5, te: 5 },
                    { label: '100/0/0', t: 100, v: 0, te: 0 },
                  ].map((preset) => (
                    <Button
                      key={preset.label}
                      variant={trainSplit === preset.t && valSplit === preset.v && testSplit === preset.te ? 'default' : 'outline'}
                      size="sm"
                      className="text-xs h-7"
                      onClick={() => { setTrainSplit(preset.t); setValSplit(preset.v); setTestSplit(preset.te); }}
                    >
                      {preset.label}
                    </Button>
                  ))}
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">Train %</Label>
                    <Input
                      type="number"
                      min={0}
                      max={100}
                      value={trainSplit}
                      onChange={(e) => setTrainSplit(Math.max(0, Math.min(100, Number(e.target.value))))}
                      className="h-8 text-sm"
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">Validation %</Label>
                    <Input
                      type="number"
                      min={0}
                      max={100}
                      value={valSplit}
                      onChange={(e) => setValSplit(Math.max(0, Math.min(100, Number(e.target.value))))}
                      className="h-8 text-sm"
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">Test %</Label>
                    <Input
                      type="number"
                      min={0}
                      max={100}
                      value={testSplit}
                      onChange={(e) => setTestSplit(Math.max(0, Math.min(100, Number(e.target.value))))}
                      className="h-8 text-sm"
                    />
                  </div>
                </div>
                {trainSplit + valSplit + testSplit !== 100 && (
                  <p className="text-xs text-destructive">
                    Splits must sum to 100% (currently {trainSplit + valSplit + testSplit}%)
                  </p>
                )}
                {stats?.total != null && stats.total > 0 && trainSplit + valSplit + testSplit === 100 && (
                  <p className="text-xs text-muted-foreground">
                    Preview: {Math.round(stats.total * trainSplit / 100)} train
                    {valSplit > 0 && <> · {Math.round(stats.total * valSplit / 100)} val</>}
                    {testSplit > 0 && <> · {Math.round(stats.total * testSplit / 100)} test</>}
                  </p>
                )}
              </div>

              <Button
                onClick={handleExport}
                disabled={exporting || !stats?.total || trainSplit + valSplit + testSplit !== 100}
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
            {/* LLM Provider */}
            <div className="grid gap-2">
              <Label>
                LLM Provider
                <InfoTooltip text="Choose the LLM provider for Q&A generation. Cloud providers require a stored API key (configure in Settings)." />
              </Label>
              <Select value={genProvider} onValueChange={(v) => { setGenProvider(v); setModel(''); }}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {genProviders?.map((p) => (
                    <SelectItem key={p.name} value={p.name} disabled={p.requires_api_key && !p.has_stored_key}>
                      <span className="flex items-center gap-1.5">
                        <span className="capitalize">{p.name}</span>
                        {p.requires_api_key && p.has_stored_key && (
                          <CheckCircle2 className="h-3 w-3 text-green-500" />
                        )}
                        {p.requires_api_key && !p.has_stored_key && (
                          <span className="text-xs text-muted-foreground">(no API key)</span>
                        )}
                      </span>
                    </SelectItem>
                  )) ?? (
                    <SelectItem value="ollama">Ollama (Local)</SelectItem>
                  )}
                </SelectContent>
              </Select>
              {isCloudProvider && keyValidating && (
                <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Checking API key…
                </p>
              )}
              {keyMissing && (
                <p className="flex items-center gap-1.5 text-xs text-destructive">
                  <AlertTriangle className="h-3 w-3" />
                  No API key configured for {genProvider}.{' '}
                  <Link to="/settings" className="underline font-medium">
                    Add key in Settings
                  </Link>
                </p>
              )}
              {isCloudProvider && keyValidation?.has_key && !keyValidating && (
                <p className="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-400">
                  <CheckCircle2 className="h-3 w-3" />
                  API key configured
                </p>
              )}
            </div>

            {/* Medical Terms */}
            <div className="grid gap-2">
              <Label>
                Medical Terms / Keywords
                <InfoTooltip text="Comma-separated medical keywords used to search PubMed and guide Q&A generation. Leave empty to auto-use the project domain." />
              </Label>
              <Input
                placeholder={`e.g. ${project?.domain || 'heart failure, diabetes, hypertension'}`}
                value={medicalTerms}
                onChange={(e) => setMedicalTerms(e.target.value)}
              />
            </div>

            {/* Model */}
            <div className="grid gap-2">
              <Label>
                {genProvider === 'ollama' ? 'Ollama Model' : `${genProvider.charAt(0).toUpperCase() + genProvider.slice(1)} Model`}
                <InfoTooltip text="The LLM model used to generate Q&A pairs. Larger models produce higher quality but are slower." />
              </Label>
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {genProvider === 'ollama' ? (
                    ollama?.models?.map((m) => (
                      <SelectItem key={m.name} value={m.name}>
                        {m.name}
                      </SelectItem>
                    ))
                  ) : (
                    genProviders?.find(p => p.name === genProvider)?.models?.map((m) => (
                      <SelectItem key={m} value={m}>
                        {m}
                      </SelectItem>
                    ))
                  )}
                </SelectContent>
              </Select>
            </div>

            {/* Target pairs */}
            <div className="grid gap-2">
              <Label>
                Target Q&A Pairs: {targetPairs}
                <InfoTooltip text="Total number of Q&A pairs to generate. The system will attempt to reach this target by processing chunks from all sources." />
              </Label>
              <Slider
                value={[targetPairs]}
                onValueChange={([v]) => setTargetPairs(v)}
                min={10}
                max={5000}
                step={10}
              />
            </div>

            {/* Max workers */}
            <div className="grid gap-2">
              <Label>
                Concurrent Workers: {maxWorkers}
                <InfoTooltip text="Number of parallel requests sent to Ollama. Higher = faster but uses more RAM/GPU. Reduce if you experience OOM errors." />
              </Label>
              <Slider
                value={[maxWorkers]}
                onValueChange={([v]) => setMaxWorkers(v)}
                min={1}
                max={10}
                step={1}
              />
            </div>

            {/* Difficulty */}
            <div className="grid gap-2">
              <Label>
                Difficulty
                <InfoTooltip text="Controls the complexity of generated questions. 'Mixed' produces a balanced spread of beginner, intermediate, and advanced questions." />
              </Label>
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
              <Label>
                Chunk Size (words): {chunkSize}
                <InfoTooltip text="Number of words per text chunk. Larger chunks give more context but may dilute focus. 300-500 works well for most medical texts." />
              </Label>
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
              <Label>
                Temperature: {temperature.toFixed(1)}
                <InfoTooltip text="Controls randomness in generation. Lower (0.3-0.5) = more focused/factual. Higher (0.7-1.0) = more creative/diverse answers." />
              </Label>
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
              <Label>
                Min Quality Score: {(minQuality * 100).toFixed(0)}%
                <InfoTooltip text="Minimum quality threshold for generated pairs. Pairs scoring below this are discarded. Higher = fewer but better quality pairs." />
              </Label>
              <Slider
                value={[minQuality]}
                onValueChange={([v]) => setMinQuality(v)}
                min={0}
                max={1}
                step={0.05}
              />
            </div>

            <Separator />

            {/* Per-source chunk limits */}
            <div className="grid gap-2">
              <Label>
                PDF Chunk Limit: {pdfChunkLimit === 0 ? 'Unlimited' : pdfChunkLimit}
                <InfoTooltip text="Maximum number of chunks to use from uploaded PDF/DOCX files. Set to 0 for no limit. Useful when you want to cap PDF contributions." />
              </Label>
              <Slider
                value={[pdfChunkLimit]}
                onValueChange={([v]) => setPdfChunkLimit(v)}
                min={0}
                max={500}
                step={10}
              />
            </div>

            <Separator />

            {/* PubMed */}
            <div className="flex items-center justify-between">
              <Label htmlFor="pubmed-toggle">
                Include PubMed Articles
                <InfoTooltip text="Fetch and process articles from PubMed/NCBI to supplement your uploaded documents. Requires an email address for NCBI API." />
              </Label>
              <Switch
                id="pubmed-toggle"
                checked={includePubmed}
                onCheckedChange={setIncludePubmed}
              />
            </div>
            {includePubmed && (
              <>
                <div className="grid gap-2">
                  <Label>
                    PubMed Email (required by NCBI)
                    <InfoTooltip text="NCBI requires an email address for PubMed API access. They use it to contact you if your usage patterns are unusual." />
                  </Label>
                  <Input
                    placeholder="your-email@example.com"
                    value={pubmedEmail}
                    onChange={(e) => setPubmedEmail(e.target.value)}
                  />
                </div>
                <div className="grid gap-2">
                  <Label>
                    Max PubMed Articles: {pubmedMax}
                    <InfoTooltip text="Maximum number of PubMed articles to fetch. More articles = more diverse content but longer processing time." />
                  </Label>
                  <Slider
                    value={[pubmedMax]}
                    onValueChange={([v]) => setPubmedMax(v)}
                    min={10}
                    max={5000}
                    step={10}
                  />
                </div>
                <div className="grid gap-2">
                  <Label>
                    PubMed Chunk Limit: {pubmedChunkLimit === 0 ? 'Unlimited' : pubmedChunkLimit}
                    <InfoTooltip text="Maximum number of chunks to use from PubMed articles. Set to 0 for no limit. Useful to ensure PDF sources aren't overwhelmed." />
                  </Label>
                  <Slider
                    value={[pubmedChunkLimit]}
                    onValueChange={([v]) => setPubmedChunkLimit(v)}
                    min={0}
                    max={500}
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
            <Button onClick={handleStartGeneration} disabled={startGeneration.isPending || !!keyMissing}>
              <Settings2 className="mr-2 h-4 w-4" />
              {startGeneration.isPending ? 'Starting…' : keyMissing ? 'API Key Required' : 'Start Generation'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function JobRow({ job, onExport, isExporting }: { job: GenerationJob; onExport?: (format: ExportFormat) => void; isExporting?: boolean }) {
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

  const canExport = job.status === 'completed' && job.qa_pair_count != null && job.qa_pair_count > 0;

  return (
    <div className="flex items-center justify-between rounded-md border p-3">
      <div>
        <div className="flex items-center gap-2">
          {job.generation_number != null && (
            <span className="text-xs font-mono bg-muted px-1.5 py-0.5 rounded">
              Gen #{job.generation_number}
            </span>
          )}
          <Badge variant={statusColor[job.status] as 'success' | 'destructive' | 'warning' | 'secondary' | 'info'}>
            {statusLabel[job.status] ?? job.status}
          </Badge>
          {job.qa_pair_count != null && job.qa_pair_count > 0 && (
            <span className="text-xs text-muted-foreground">
              {formatNumber(job.qa_pair_count)} pairs
            </span>
          )}
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
      <div className="flex items-center gap-2">
        {canExport && onExport && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" disabled={isExporting}>
                {isExporting ? (
                  <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                ) : (
                  <Download className="mr-1 h-3 w-3" />
                )}
                Export
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => onExport('csv')}>CSV</DropdownMenuItem>
              <DropdownMenuItem onClick={() => onExport('json')}>JSON</DropdownMenuItem>
              <DropdownMenuItem onClick={() => onExport('jsonl')}>JSONL</DropdownMenuItem>
              <DropdownMenuItem onClick={() => onExport('alpaca')}>Alpaca Format</DropdownMenuItem>
              <DropdownMenuItem onClick={() => onExport('openai')}>OpenAI Format</DropdownMenuItem>
              <DropdownMenuItem onClick={() => onExport('parquet')}>Parquet</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
        <span className="text-xs text-muted-foreground">
          {job.completed_at ? formatDate(job.completed_at) : formatDate(job.created_at)}
        </span>
      </div>
    </div>
  );
}
