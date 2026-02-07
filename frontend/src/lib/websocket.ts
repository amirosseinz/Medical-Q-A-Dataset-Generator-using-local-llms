/* ------------------------------------------------------------------ */
/*  WebSocket manager for real-time generation progress               */
/* ------------------------------------------------------------------ */
import type { GenerationProgress } from '@/types';

type ProgressCallback = (progress: GenerationProgress) => void;

class WebSocketManager {
  private ws: WebSocket | null = null;
  private listeners = new Map<string, Set<ProgressCallback>>();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 30000;
  private jobId: string | null = null;

  connect(jobId: string) {
    // Disconnect previous connection if any
    if (this.ws) {
      this.disconnect();
    }
    this.jobId = jobId;
    this.reconnectDelay = 1000;
    this._connect();
  }

  private _connect() {
    if (!this.jobId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const url = `${protocol}//${host}/api/v1/ws/jobs/${this.jobId}`;

    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      console.log('[WS] Connected');
      this.reconnectDelay = 1000;
    };

    this.ws.onmessage = (event) => {
      try {
        const data: GenerationProgress = JSON.parse(event.data);
        const jobListeners = this.listeners.get(data.job_id);
        if (jobListeners) {
          jobListeners.forEach((cb) => cb(data));
        }
        // Also notify global listeners
        const globalListeners = this.listeners.get('*');
        if (globalListeners) {
          globalListeners.forEach((cb) => cb(data));
        }
      } catch {
        // ignore non-JSON messages
      }
    };

    this.ws.onclose = () => {
      console.log('[WS] Disconnected, reconnecting...');
      this._scheduleReconnect();
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  private _scheduleReconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => {
      this._connect();
      this.reconnectDelay = Math.min(
        this.reconnectDelay * 2,
        this.maxReconnectDelay,
      );
    }, this.reconnectDelay);
  }

  subscribe(jobId: string, callback: ProgressCallback): () => void {
    if (!this.listeners.has(jobId)) {
      this.listeners.set(jobId, new Set());
    }
    this.listeners.get(jobId)!.add(callback);

    // Return unsubscribe function
    return () => {
      this.listeners.get(jobId)?.delete(callback);
      if (this.listeners.get(jobId)?.size === 0) {
        this.listeners.delete(jobId);
      }
    };
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
    this.jobId = null;
    this.listeners.clear();
  }
}

export const wsManager = new WebSocketManager();
