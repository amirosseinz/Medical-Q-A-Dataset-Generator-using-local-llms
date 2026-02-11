import { useState } from 'react';
import { useOllamaStatus, useReviewProviders, useLLMProviders, useCreateLLMProvider, useUpdateLLMProvider, useDeleteLLMProvider, useTestLLMProvider, useRefreshModels } from '@/hooks/use-api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Wifi, WifiOff, HardDrive, Bot, Key, Info, Plus, Trash2, TestTube, Loader2, CheckCircle, XCircle, Pencil, RefreshCw } from 'lucide-react';
import type { LLMProviderConfig } from '@/types';

export default function SettingsPage() {
  const { data: ollama, isLoading } = useOllamaStatus();
  const { data: providers } = useReviewProviders();
  const { data: storedProviders, isLoading: loadingStored } = useLLMProviders();
  const createProvider = useCreateLLMProvider();
  const updateProvider = useUpdateLLMProvider();
  const deleteProvider = useDeleteLLMProvider();
  const testProvider = useTestLLMProvider();
  const refreshModels = useRefreshModels();

  const [showAddDialog, setShowAddDialog] = useState(false);
  const [editingProvider, setEditingProvider] = useState<LLMProviderConfig | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [refreshingId, setRefreshingId] = useState<string | null>(null);

  // Helper: format "time ago" string
  const timeAgo = (dateStr: string | null): string => {
    if (!dateStr) return 'Never';
    const diff = Date.now() - new Date(dateStr).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'Just now';
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  };

  // Add form state
  const [formProvider, setFormProvider] = useState('openai');
  const [formApiKey, setFormApiKey] = useState('');
  const [formDisplayName, setFormDisplayName] = useState('');
  const [formOrgId, setFormOrgId] = useState('');
  const [formIsDefault, setFormIsDefault] = useState(true);

  const resetForm = () => {
    setFormProvider('openai');
    setFormApiKey('');
    setFormDisplayName('');
    setFormOrgId('');
    setFormIsDefault(true);
  };

  const handleAdd = () => {
    createProvider.mutate(
      {
        provider_name: formProvider,
        api_key: formApiKey,
        display_name: formDisplayName || undefined,
        organization_id: formOrgId || undefined,
        is_default: formIsDefault,
      },
      {
        onSuccess: () => {
          setShowAddDialog(false);
          resetForm();
        },
      },
    );
  };

  const handleUpdate = () => {
    if (!editingProvider) return;
    updateProvider.mutate(
      {
        id: editingProvider.id,
        data: {
          api_key: formApiKey || undefined,
          display_name: formDisplayName || undefined,
          organization_id: formOrgId || undefined,
          is_default: formIsDefault,
        },
      },
      {
        onSuccess: () => {
          setEditingProvider(null);
          resetForm();
        },
      },
    );
  };

  const openEdit = (p: LLMProviderConfig) => {
    setEditingProvider(p);
    setFormProvider(p.provider_name);
    setFormApiKey('');
    setFormDisplayName(p.display_name || '');
    setFormOrgId(p.organization_id || '');
    setFormIsDefault(p.is_default);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">
          System configuration and connection status
        </p>
      </div>

      {/* Ollama Status */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Ollama Connection</CardTitle>
          <CardDescription>
            Local LLM server used for Q&A generation
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-3">
            {isLoading ? (
              <Badge variant="secondary">Checking…</Badge>
            ) : ollama?.connected ? (
              <Badge variant="success" className="gap-1">
                <Wifi className="h-3 w-3" />
                Connected
              </Badge>
            ) : (
              <Badge variant="destructive" className="gap-1">
                <WifiOff className="h-3 w-3" />
                Disconnected
              </Badge>
            )}
            <span className="text-sm text-muted-foreground">
              {ollama?.url ?? 'http://localhost:11434'}
            </span>
          </div>

          {ollama?.connected && ollama.models && ollama.models.length > 0 && (
            <div>
              <p className="mb-2 text-sm font-medium">Available Models</p>
              <div className="grid gap-2 sm:grid-cols-2">
                {ollama.models.map((m) => (
                  <div
                    key={m.name}
                    className="flex items-center gap-2 rounded-md border p-3"
                  >
                    <HardDrive className="h-4 w-4 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium">{m.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {(m.size / 1e9).toFixed(1)} GB
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {ollama?.connected && (!ollama.models || ollama.models.length === 0) && (
            <p className="text-sm text-muted-foreground">
              No models installed. Run{' '}
              <code className="rounded bg-muted px-1 py-0.5 text-xs">
                ollama pull llama3
              </code>{' '}
              to install a model.
            </p>
          )}

          {!ollama?.connected && !isLoading && (
            <div className="rounded-md bg-destructive/10 p-3 text-sm">
              <p className="font-medium text-destructive">Cannot connect to Ollama</p>
              <p className="mt-1 text-muted-foreground">
                Make sure Ollama is running on your machine. Start it with{' '}
                <code className="rounded bg-muted px-1 py-0.5 text-xs">ollama serve</code>
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* LLM API Key Management */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base flex items-center gap-2">
                <Key className="h-4 w-4" />
                LLM API Keys
              </CardTitle>
              <CardDescription>
                Store API keys for LLM review providers. Keys are encrypted at rest.
              </CardDescription>
            </div>
            <Button size="sm" onClick={() => { resetForm(); setShowAddDialog(true); }}>
              <Plus className="h-4 w-4 mr-1" />
              Add Key
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {loadingStored ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading stored keys…
            </div>
          ) : storedProviders && storedProviders.length > 0 ? (
            storedProviders.map((sp) => (
              <div
                key={sp.id}
                className="rounded-md border p-3 space-y-2"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3 min-w-0">
                    <Key className="h-4 w-4 text-muted-foreground shrink-0" />
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-medium capitalize">{sp.provider_name}</p>
                        {sp.display_name && (
                          <span className="text-xs text-muted-foreground">({sp.display_name})</span>
                        )}
                        {sp.is_default && <Badge variant="secondary" className="text-[10px] px-1.5 py-0">Default</Badge>}
                      </div>
                      <p className="text-xs text-muted-foreground font-mono">{sp.masked_key}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    {sp.is_valid ? (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    ) : (
                      <XCircle className="h-4 w-4 text-destructive" />
                    )}
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={() => testProvider.mutate(sp.id)}
                      disabled={testProvider.isPending}
                    >
                      {testProvider.isPending ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <TestTube className="h-4 w-4" />
                      )}
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={() => openEdit(sp)}
                    >
                      <Pencil className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-destructive"
                      onClick={() => setDeleteConfirm(sp.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                {/* Models info + Refresh */}
                <div className="flex items-center justify-between pl-7">
                  <div className="text-xs text-muted-foreground">
                    {sp.available_models && sp.available_models.length > 0 ? (
                      <>
                        <span className="font-medium text-foreground">{sp.available_models.length}</span> models
                        {sp.models_fetched_at && (
                          <> &middot; fetched {timeAgo(sp.models_fetched_at)}</>
                        )}
                      </>
                    ) : (
                      <span>No models fetched yet</span>
                    )}
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 text-xs gap-1"
                    onClick={() => {
                      setRefreshingId(sp.id);
                      refreshModels.mutate(
                        { id: sp.id, force: true },
                        { onSettled: () => setRefreshingId(null) },
                      );
                    }}
                    disabled={refreshingId === sp.id || refreshModels.isPending}
                  >
                    {refreshingId === sp.id ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <RefreshCw className="h-3 w-3" />
                    )}
                    Refresh Models
                  </Button>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-6 text-sm text-muted-foreground">
              <Key className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No API keys stored yet</p>
              <p className="text-xs mt-1">Add an API key to use cloud LLM providers for Q&A review</p>
            </div>
          )}

          <div className="flex items-start gap-2 rounded-md bg-muted/50 p-3 text-xs text-muted-foreground">
            <Info className="h-3.5 w-3.5 mt-0.5 shrink-0" />
            <p>
              API keys are encrypted using Fernet symmetric encryption before storage.
              Stored keys are automatically used on the Review page — no need to re-enter them.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* LLM Review Providers */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Bot className="h-4 w-4" />
            Available Review Providers
          </CardTitle>
          <CardDescription>
            LLM APIs supported for automated Q&A quality review
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {providers?.map((provider) => (
            <div
              key={provider.name}
              className="flex items-center justify-between rounded-md border p-3"
            >
              <div className="flex items-center gap-3">
                {provider.requires_api_key ? (
                  <Key className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <HardDrive className="h-4 w-4 text-muted-foreground" />
                )}
                <div>
                  <p className="text-sm font-medium capitalize">{provider.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {provider.models_source === 'fetched' ? (
                      <>
                        <span className="font-medium text-foreground">{provider.models.length}</span> model{provider.models.length !== 1 ? 's' : ''} available
                        {provider.models_fetched_at && (
                          <> &middot; updated {timeAgo(provider.models_fetched_at)}</>
                        )}
                      </>
                    ) : provider.has_stored_key ? (
                      'Models not yet fetched — use Refresh Models above'
                    ) : provider.requires_api_key ? (
                      'Add an API key to discover available models'
                    ) : provider.models.length > 0 ? (
                      `${provider.models.length} model${provider.models.length !== 1 ? 's' : ''} installed`
                    ) : (
                      'No models available'
                    )}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {provider.has_stored_key && (
                  <Badge variant="success" className="text-xs">Key Stored ✓</Badge>
                )}
                {provider.models_source === 'fetched' && (
                  <Badge variant="outline" className="text-xs">Live</Badge>
                )}
                {provider.requires_api_key && !provider.has_stored_key ? (
                  <Badge variant="secondary">API Key Required</Badge>
                ) : !provider.requires_api_key ? (
                  <Badge variant="success">Local</Badge>
                ) : null}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* About */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">About</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>
            <span className="font-medium text-foreground">Medical Q&A Dataset Generator</span>{' '}
            v2.1.0
          </p>
          <p>
            A production-ready tool for generating high-quality medical question-answer
            datasets from PDFs, MedQuAD XML files, and PubMed articles using local LLMs
            via Ollama.
          </p>
          <p>
            Features: Rate-limited LLM review, encrypted API key storage, multi-generation
            tracking, fact-checking, and session-based review with progress.
          </p>
          <p>
            Built with FastAPI, React, Celery, Redis, and SQLite.
          </p>
        </CardContent>
      </Card>

      {/* Add API Key Dialog */}
      <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add API Key</DialogTitle>
            <DialogDescription>
              Store an API key for an LLM provider. It will be encrypted at rest.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <Label>Provider</Label>
              <Select value={formProvider} onValueChange={setFormProvider}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="openai">OpenAI</SelectItem>
                  <SelectItem value="anthropic">Anthropic</SelectItem>
                  <SelectItem value="google">Google (Gemini)</SelectItem>
                  <SelectItem value="openrouter">OpenRouter</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>API Key</Label>
              <Input
                type="password"
                placeholder={formProvider === 'openai' ? 'sk-...' : formProvider === 'anthropic' ? 'sk-ant-...' : formProvider === 'openrouter' ? 'sk-or-v1-...' : 'AIza...'}
                value={formApiKey}
                onChange={(e) => setFormApiKey(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Display Name (optional)</Label>
              <Input
                placeholder="e.g. Personal, Work, Research"
                value={formDisplayName}
                onChange={(e) => setFormDisplayName(e.target.value)}
              />
            </div>
            {formProvider === 'openai' && (
              <div className="space-y-2">
                <Label>Organization ID (optional)</Label>
                <Input
                  placeholder="org-..."
                  value={formOrgId}
                  onChange={(e) => setFormOrgId(e.target.value)}
                />
              </div>
            )}
            {formProvider === 'openrouter' && (
              <div className="flex items-start gap-2 rounded-md bg-blue-50 dark:bg-blue-950/20 p-3 text-xs text-muted-foreground">
                <Info className="h-3.5 w-3.5 mt-0.5 shrink-0 text-blue-500" />
                <p>
                  OpenRouter provides access to 100+ models from multiple providers with a single API key.
                  Get your key at{' '}
                  <a href="https://openrouter.ai/keys" target="_blank" rel="noopener noreferrer" className="underline text-blue-600">openrouter.ai/keys</a>
                </p>
              </div>
            )}
            <div className="flex items-center gap-2">
              <Switch checked={formIsDefault} onCheckedChange={setFormIsDefault} />
              <Label>Set as default for this provider</Label>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowAddDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleAdd} disabled={!formApiKey || createProvider.isPending}>
              {createProvider.isPending && <Loader2 className="h-4 w-4 animate-spin mr-1" />}
              Save Key
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit API Key Dialog */}
      <Dialog open={!!editingProvider} onOpenChange={(open) => !open && setEditingProvider(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit API Key</DialogTitle>
            <DialogDescription>
              Update settings for your {editingProvider?.provider_name} API key.
              Leave the API key field empty to keep the current key.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <Label>New API Key (leave empty to keep current)</Label>
              <Input
                type="password"
                placeholder="Enter new key or leave empty"
                value={formApiKey}
                onChange={(e) => setFormApiKey(e.target.value)}
              />
              {editingProvider && (
                <p className="text-xs text-muted-foreground">
                  Current: {editingProvider.masked_key}
                </p>
              )}
            </div>
            <div className="space-y-2">
              <Label>Display Name</Label>
              <Input
                placeholder="e.g. Personal, Work"
                value={formDisplayName}
                onChange={(e) => setFormDisplayName(e.target.value)}
              />
            </div>
            <div className="flex items-center gap-2">
              <Switch checked={formIsDefault} onCheckedChange={setFormIsDefault} />
              <Label>Set as default for this provider</Label>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditingProvider(null)}>
              Cancel
            </Button>
            <Button onClick={handleUpdate} disabled={updateProvider.isPending}>
              {updateProvider.isPending && <Loader2 className="h-4 w-4 animate-spin mr-1" />}
              Update
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteConfirm} onOpenChange={(open: boolean) => !open && setDeleteConfirm(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete API Key?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently remove this API key from storage. You'll need to add it again to use this provider.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (deleteConfirm) {
                  deleteProvider.mutate(deleteConfirm);
                  setDeleteConfirm(null);
                }
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
