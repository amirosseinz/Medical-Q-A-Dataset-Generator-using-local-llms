import { useProjects, useQAPairStats, useEnhancedAnalytics } from '@/hooks/use-api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { formatNumber } from '@/lib/utils';
import { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  AreaChart,
  Area,
} from 'recharts';

const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#0ea5e9', '#8b5cf6'];

export default function AnalyticsPage() {
  const { data: projects } = useProjects();
  const [selectedProjectId, setSelectedProjectId] = useState<string>('');

  // Use the first project if none selected
  const projectId = selectedProjectId || projects?.[0]?.id || '';
  const { data: stats } = useQAPairStats(projectId);
  const { data: analytics } = useEnhancedAnalytics(projectId);

  const sourceData = stats?.by_source_type
    ? Object.entries(stats.by_source_type).map(([name, value]) => ({
        name: name.toUpperCase().replace('_', ' + '),
        value,
      }))
    : [];

  // Source documents: specific files/articles that generated Q&A pairs
  const sourceDocData = stats?.by_source_document
    ? Object.entries(stats.by_source_document)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 10)
        .map(([name, count]) => ({
          name: name.length > 40 ? name.slice(0, 37) + '…' : name,
          fullName: name,
          count,
        }))
    : [];

  const modelData = stats?.by_model
    ? Object.entries(stats.by_model).map(([name, count]) => ({
        name: name || 'Unknown',
        count,
      }))
    : [];

  const statusData = stats
    ? [
        { name: 'Approved', value: stats.approved, color: '#10b981' },
        { name: 'Rejected', value: stats.rejected, color: '#ef4444' },
        { name: 'Pending', value: stats.pending, color: '#64748b' },
      ].filter((d) => d.value > 0)
    : [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Analytics</h1>
          <p className="text-muted-foreground">
            Quality metrics and dataset statistics
          </p>
        </div>
        {projects && projects.length > 0 && (
          <Select value={projectId} onValueChange={setSelectedProjectId}>
            <SelectTrigger className="w-64">
              <SelectValue placeholder="Select project" />
            </SelectTrigger>
            <SelectContent>
              {projects.map((p) => (
                <SelectItem key={p.id} value={p.id}>
                  {p.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>

      {/* Summary cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total Pairs</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{formatNumber(stats?.total ?? 0)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Approval Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-success-600">
              {stats?.total ? `${((stats.approved / stats.total) * 100).toFixed(1)}%` : '—'}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Avg Quality</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">
              {stats?.avg_quality_score != null
                ? `${(stats.avg_quality_score * 100).toFixed(1)}%`
                : '—'}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Rejection Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-destructive">
              {stats?.total ? `${((stats.rejected / stats.total) * 100).toFixed(1)}%` : '—'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Validation Status Pie */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Validation Status</CardTitle>
          </CardHeader>
          <CardContent>
            {statusData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={statusData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                    nameKey="name"
                    label={({ name, percent }) =>
                      `${name} ${(percent * 100).toFixed(0)}%`
                    }
                  >
                    {statusData.map((entry, index) => (
                      <Cell key={index} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[280px] items-center justify-center text-muted-foreground">
                No data available
              </div>
            )}
          </CardContent>
        </Card>

        {/* Source Distribution Pie */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Source Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            {sourceData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={sourceData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                    nameKey="name"
                    label={({ name, percent }) =>
                      `${name} ${(percent * 100).toFixed(0)}%`
                    }
                  >
                    {sourceData.map((_, index) => (
                      <Cell key={index} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[280px] items-center justify-center text-muted-foreground">
                No data available
              </div>
            )}
          </CardContent>
        </Card>

        {/* Model Distribution Bar */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-base">Model Usage</CardTitle>
          </CardHeader>
          <CardContent>
            {modelData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={modelData}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                  <XAxis dataKey="name" className="text-xs" />
                  <YAxis className="text-xs" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '0.5rem',
                    }}
                  />
                  <Bar dataKey="count" fill="#2563eb" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                No data available
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Top Source Documents */}
      {sourceDocData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Top Source Documents</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={Math.max(200, sourceDocData.length * 36)}>
              <BarChart data={sourceDocData} layout="vertical" margin={{ left: 20, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis type="number" className="text-xs" />
                <YAxis type="category" dataKey="name" className="text-xs" width={200} tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '0.5rem',
                  }}
                  formatter={(value: number) => [`${value} pairs`, 'Q&A Pairs']}
                />
                <Bar dataKey="count" fill="#0ea5e9" radius={[0, 4, 4, 0]} name="Q&A Pairs" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Quality Histogram + Timeline */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Quality Distribution Histogram */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Quality Score Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            {analytics?.quality_histogram && analytics.quality_histogram.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={analytics.quality_histogram}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                  <XAxis dataKey="range" className="text-xs" />
                  <YAxis className="text-xs" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '0.5rem',
                    }}
                  />
                  <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} name="Pairs" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[280px] items-center justify-center text-muted-foreground">
                No data available
              </div>
            )}
          </CardContent>
        </Card>

        {/* Generation Timeline */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Generation Timeline</CardTitle>
          </CardHeader>
          <CardContent>
            {analytics?.generation_timeline && analytics.generation_timeline.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={analytics.generation_timeline}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                  <XAxis dataKey="date" className="text-xs" />
                  <YAxis className="text-xs" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '0.5rem',
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="count"
                    stroke="#10b981"
                    fill="#10b98120"
                    name="Pairs Generated"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[280px] items-center justify-center text-muted-foreground">
                No data available
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Per-File Breakdown Table */}
      {analytics?.by_file && analytics.by_file.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Per-File Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>File</TableHead>
                  <TableHead>Source</TableHead>
                  <TableHead className="text-right">Pairs</TableHead>
                  <TableHead className="text-right">Avg Quality</TableHead>
                  <TableHead className="text-right">Approved</TableHead>
                  <TableHead className="text-right">Rejected</TableHead>
                  <TableHead className="text-right">Pending</TableHead>
                  <TableHead className="w-32">Approval</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {analytics.by_file.map((file) => {
                  const total = file.approved + file.rejected + file.pending;
                  const approvalPct = total > 0 ? (file.approved / total) * 100 : 0;
                  return (
                    <TableRow key={file.filename}>
                      <TableCell className="font-medium text-sm max-w-[200px] truncate">
                        {file.filename}
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline" className="text-xs">
                          {file.source_type}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">{formatNumber(file.pair_count)}</TableCell>
                      <TableCell className="text-right">
                        {file.avg_quality != null ? (
                          <span
                            className={
                              file.avg_quality >= 0.7
                                ? 'text-success-600'
                                : file.avg_quality >= 0.4
                                ? 'text-warning-600'
                                : 'text-destructive'
                            }
                          >
                            {(file.avg_quality * 100).toFixed(0)}%
                          </span>
                        ) : (
                          '—'
                        )}
                      </TableCell>
                      <TableCell className="text-right text-success-600">{file.approved}</TableCell>
                      <TableCell className="text-right text-destructive">{file.rejected}</TableCell>
                      <TableCell className="text-right text-muted-foreground">{file.pending}</TableCell>
                      <TableCell>
                        <Progress value={approvalPct} className="h-2" />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
