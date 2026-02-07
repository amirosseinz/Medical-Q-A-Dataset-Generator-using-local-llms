import { useOllamaStatus } from '@/hooks/use-api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Wifi, WifiOff, HardDrive } from 'lucide-react';

export default function SettingsPage() {
  const { data: ollama, isLoading } = useOllamaStatus();

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

      {/* About */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">About</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>
            <span className="font-medium text-foreground">Medical Q&A Dataset Generator</span>{' '}
            v2.0.0
          </p>
          <p>
            A production-ready tool for generating high-quality medical question-answer
            datasets from PDFs, MedQuAD XML files, and PubMed articles using local LLMs
            via Ollama.
          </p>
          <p>
            Built with FastAPI, React, Celery, Redis, and SQLite.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
