import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, FolderOpen, Archive, Trash2, MoreVertical, AlertTriangle } from 'lucide-react';
import { useProjects, useCreateProject, useDeleteProject } from '@/hooks/use-api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { formatDate, formatNumber } from '@/lib/utils';
import type { Project } from '@/types';

export default function DashboardPage() {
  const navigate = useNavigate();
  const { data: projects, isLoading } = useProjects();
  const createProject = useCreateProject();
  const deleteProject = useDeleteProject();

  const [createOpen, setCreateOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<Project | null>(null);
  const [name, setName] = useState('');
  const [domain, setDomain] = useState('');
  const [description, setDescription] = useState('');

  const handleCreate = () => {
    if (!name.trim()) return;
    createProject.mutate(
      { name: name.trim(), domain: domain.trim() || 'medical', description: description.trim() },
      {
        onSuccess: (project) => {
          setCreateOpen(false);
          setName('');
          setDomain('');
          setDescription('');
          navigate(`/projects/${project.id}`);
        },
      },
    );
  };

  const activeProjects = projects?.filter((p) => p.status === 'draft' || p.status === 'active') ?? [];
  const archivedProjects = projects?.filter((p) => p.status === 'archived') ?? [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground">
            Manage your medical Q&A dataset projects
          </p>
        </div>
        <Dialog open={createOpen} onOpenChange={setCreateOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              New Project
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New Project</DialogTitle>
              <DialogDescription>
                Set up a new dataset generation project.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="name">Project Name</Label>
                <Input
                  id="name"
                  placeholder="e.g. Cardiology QA Dataset"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="domain">Medical Domain</Label>
                <Input
                  id="domain"
                  placeholder="e.g. cardiology, oncology, general"
                  value={domain}
                  onChange={(e) => setDomain(e.target.value)}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="description">Description (optional)</Label>
                <Input
                  id="description"
                  placeholder="Brief description of the project"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setCreateOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreate} disabled={!name.trim() || createProject.isPending}>
                {createProject.isPending ? 'Creating…' : 'Create Project'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Loading state */}
      {isLoading && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader className="space-y-2">
                <div className="h-5 w-48 rounded bg-muted" />
                <div className="h-4 w-32 rounded bg-muted" />
              </CardHeader>
              <CardContent>
                <div className="h-4 w-full rounded bg-muted" />
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Empty state */}
      {!isLoading && projects?.length === 0 && (
        <Card className="flex flex-col items-center justify-center py-16">
          <FolderOpen className="h-12 w-12 text-muted-foreground/50" />
          <h3 className="mt-4 text-lg font-semibold">No projects yet</h3>
          <p className="mt-1 text-sm text-muted-foreground">
            Create your first project to start generating medical Q&A datasets.
          </p>
          <Button className="mt-6" onClick={() => setCreateOpen(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Create Project
          </Button>
        </Card>
      )}

      {/* Active projects */}
      {activeProjects.length > 0 && (
        <div>
          <h2 className="mb-3 text-lg font-semibold">Active Projects</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {activeProjects.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                onClick={() => navigate(`/projects/${project.id}`)}
                onDelete={() => setDeleteTarget(project)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Archived projects */}
      {archivedProjects.length > 0 && (
        <div>
          <h2 className="mb-3 text-lg font-semibold text-muted-foreground">
            <Archive className="mr-2 inline h-4 w-4" />
            Archived ({archivedProjects.length})
          </h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {archivedProjects.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                onClick={() => navigate(`/projects/${project.id}`)}
                onDelete={() => setDeleteTarget(project)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Delete confirmation dialog */}
      <AlertDialog open={!!deleteTarget} onOpenChange={(open) => !open && setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-destructive" />
              Delete Project
            </AlertDialogTitle>
            <AlertDialogDescription className="space-y-2">
              <span className="block">
                Are you sure you want to delete{' '}
                <span className="font-semibold text-foreground">{deleteTarget?.name}</span>?
              </span>
              <span className="block">
                This will permanently remove{' '}
                <span className="font-semibold text-foreground">
                  {formatNumber(deleteTarget?.total_qa_pairs ?? 0)} Q&A pairs
                </span>
                , all uploaded sources, FAISS indices, and exported files. Any running generation
                jobs will be cancelled.
              </span>
              <span className="block text-destructive font-medium">This action cannot be undone.</span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              disabled={deleteProject.isPending}
              onClick={() => {
                if (deleteTarget) {
                  deleteProject.mutate(deleteTarget.id, {
                    onSuccess: () => setDeleteTarget(null),
                  });
                }
              }}
            >
              {deleteProject.isPending ? 'Deleting\u2026' : 'Delete Project'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

function ProjectCard({
  project,
  onClick,
  onDelete,
}: {
  project: Project;
  onClick: () => void;
  onDelete: () => void;
}) {
  return (
    <Card
      className="cursor-pointer transition-shadow hover:shadow-md"
      onClick={onClick}
    >
      <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-2">
        <div className="space-y-1">
          <CardTitle className="text-base">{project.name}</CardTitle>
          <CardDescription>{project.domain}</CardDescription>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <MoreVertical className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem
              className="text-destructive"
              onClick={(e) => {
                e.stopPropagation();
                onDelete();
              }}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <div>
            <span className="font-medium text-foreground">
              {formatNumber(project.total_qa_pairs)}
            </span>{' '}
            Q&A pairs
          </div>
          <div>
            <span className="font-medium text-foreground">
              {project.total_sources}
            </span>{' '}
            sources
          </div>
        </div>
        {project.avg_quality_score != null && (
          <div className="mt-2">
            <Badge variant={project.avg_quality_score >= 0.7 ? 'success' : 'warning'}>
              Quality: {(project.avg_quality_score * 100).toFixed(0)}%
            </Badge>
          </div>
        )}
        <p className="mt-2 text-xs text-muted-foreground">
          Updated {formatDate(project.updated_at)}
        </p>
      </CardContent>
    </Card>
  );
}
