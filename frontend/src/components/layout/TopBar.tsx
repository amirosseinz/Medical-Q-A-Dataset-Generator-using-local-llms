import { useQuery } from '@tanstack/react-query';
import { Activity, Wifi, WifiOff } from 'lucide-react';
import { ollamaApi } from '@/lib/api';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { useAppStore } from '@/store';

export default function TopBar() {
  const { activeJobs } = useAppStore();

  const { data: ollamaStatus } = useQuery({
    queryKey: ['ollama', 'status'],
    queryFn: ollamaApi.status,
    refetchInterval: 30_000,
    retry: false,
  });

  const runningJobs = Array.from(activeJobs.values()).filter(
    (j) => j.status === 'in_progress',
  );

  return (
    <TooltipProvider>
      <header className="flex h-14 items-center justify-between border-b bg-card px-6">
        {/* Left — breadcrumb area (populated by pages) */}
        <div id="topbar-left" className="flex items-center gap-2" />

        {/* Right — status indicators */}
        <div className="flex items-center gap-4">
          {/* Active jobs indicator */}
          {runningJobs.length > 0 && (
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-1.5 text-sm text-warning-600">
                  <Activity className="h-4 w-4 animate-pulse-subtle" />
                  <span className="font-medium">
                    {runningJobs.length} job{runningJobs.length > 1 ? 's' : ''} running
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                {runningJobs.map((j) => (
                  <div key={j.job_id} className="text-xs">
                    {j.current_message} — {j.progress_pct}%
                  </div>
                ))}
              </TooltipContent>
            </Tooltip>
          )}

          {/* Ollama connection status */}
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge
                variant={ollamaStatus?.connected ? 'success' : 'destructive'}
                className="cursor-default gap-1"
              >
                {ollamaStatus?.connected ? (
                  <Wifi className="h-3 w-3" />
                ) : (
                  <WifiOff className="h-3 w-3" />
                )}
                Ollama
              </Badge>
            </TooltipTrigger>
            <TooltipContent>
              {ollamaStatus?.connected
                ? `Connected — ${ollamaStatus.models?.length ?? 0} model(s) available`
                : 'Disconnected — check Ollama is running'}
            </TooltipContent>
          </Tooltip>
        </div>
      </header>
    </TooltipProvider>
  );
}
