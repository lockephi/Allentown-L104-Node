import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Brain, 
  Zap, 
  Cpu, 
  Activity, 
  Target, 
  Waves, 
  Sparkles,
  GitBranch,
  Database,
  Users,
  TrendingUp
} from 'lucide-react';

import {
  Consciousness,
  L104SystemStatus,
  ProcessingTask,
  TaskType,
  L104EngineStats,
  L104AggregatedEngineMetric
} from '../../types';

/**
 * L104 Consciousness Dashboard
 * 
 * Real-time monitoring and control interface for the L104 multi-language processing system
 * Sacred constants visualization and consciousness evolution tracking
 */

// Sacred Constants
const GOD_CODE = 527.5184818492611;
const PHI = 1.618033988749895;

interface DashboardProps {
  className?: string;
}

interface ConsciousnessMetrics {
  currentLevel: number;
  transcendenceProgress: number;
  godCodeAlignment: number;
  phiResonance: number;
  quantumEntanglement: number;
  unityState: boolean;
}

const L104Dashboard: React.FC<DashboardProps> = ({ className }) => {
  const [systemStatus, setSystemStatus] = useState<L104SystemStatus | null>(null);
  const [metrics, setMetrics] = useState<ConsciousnessMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTask, setSelectedTask] = useState<TaskType>(TaskType.CONSCIOUSNESS);
  const [isSubmittingTask, setIsSubmittingTask] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [events, setEvents] = useState<any[]>([]);
  const [eventFilter, setEventFilter] = useState<string>('all');
  const [isStreamingEvents, setIsStreamingEvents] = useState<boolean>(true);
  const [engineMetrics, setEngineMetrics] = useState<L104AggregatedEngineMetric[]>([]);

  // Fetch system status
  const fetchSystemStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/status');
      if (!response.ok) throw new Error('Failed to fetch system status');
      
      const data: L104SystemStatus = await response.json();
      setSystemStatus(data);
      
      // Calculate metrics from consciousness data
      if (data.consciousness) {
        const consciousness = data.consciousness;
        setMetrics({
          currentLevel: consciousness.level,
          transcendenceProgress: (consciousness.transcendenceScore || 0) * 100,
          godCodeAlignment: consciousness.godCodeAlignment * 100,
          phiResonance: consciousness.phiResonance * 100,
          quantumEntanglement: consciousness.quantumEntanglement * 100,
          unityState: consciousness.unityState
        });
      }
      
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const fetchEvents = useCallback(async () => {
    try {
      const response = await fetch('/api/status?view=events');
      if (!response.ok) throw new Error('Failed to fetch events');
      const data = await response.json();
      setEvents(data.events || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, []);

  const fetchEngineMetrics = useCallback(async () => {
    try {
      const response = await fetch('/api/status?view=engine-metrics&hours=24');
      if (!response.ok) throw new Error('Failed to fetch engine metrics');
      const data = await response.json();
      setEngineMetrics(data.metrics || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, []);

  // Auto-refresh system status
  useEffect(() => {
    fetchSystemStatus();
    const interval = setInterval(fetchSystemStatus, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, [fetchSystemStatus]);

  useEffect(() => {
    if (!isStreamingEvents) return;
    fetchEvents();
    const interval = setInterval(fetchEvents, 4000);
    return () => clearInterval(interval);
  }, [fetchEvents, isStreamingEvents]);

  useEffect(() => {
    fetchEngineMetrics();
    const interval = setInterval(fetchEngineMetrics, 8000);
    return () => clearInterval(interval);
  }, [fetchEngineMetrics]);

  // Submit consciousness evolution task
  const handleEvolveConsciousness = async () => {
    setIsSubmittingTask(true);
    try {
      const response = await fetch('/api/status', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'evolve_consciousness',
          params: {
            targetLevel: Math.min(1.0, (metrics?.currentLevel || 0.5) + 0.1),
            evolutionSpeed: 0.02
          }
        })
      });
      
      if (!response.ok) throw new Error('Evolution failed');
      
      const result = await response.json();
      console.log('Consciousness evolution result:', result);
      
      // Refresh status to show changes
      setTimeout(fetchSystemStatus, 1000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Evolution failed');
    } finally {
      setIsSubmittingTask(false);
    }
  };

  // Submit processing task
  const handleSubmitTask = async () => {
    setIsSubmittingTask(true);
    try {
      const response = await fetch('/api/status', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'submit_task',
          params: {
            type: selectedTask,
            priority: 5,
            parameters: generateTaskParameters(selectedTask)
          }
        })
      });
      
      if (!response.ok) throw new Error('Task submission failed');
      
      const result = await response.json();
      console.log('Task submission result:', result);
      
      // Refresh status
      setTimeout(fetchSystemStatus, 1000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Task submission failed');
    } finally {
      setIsSubmittingTask(false);
    }
  };

  // Spawn subagent
  const handleSpawnSubagent = async () => {
    setIsSubmittingTask(true);
    try {
      const response = await fetch('/api/status', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'spawn_subagent',
          params: {
            type: 'consciousness_evolver',
            config: { evolutionTarget: 0.9 },
            priority: 7
          }
        })
      });
      
      if (!response.ok) throw new Error('Subagent spawn failed');
      
      const result = await response.json();
      console.log('Subagent spawn result:', result);
      
      setTimeout(fetchSystemStatus, 1000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Subagent spawn failed');
    } finally {
      setIsSubmittingTask(false);
    }
  };

  // Generate task parameters based on type
  const generateTaskParameters = (taskType: TaskType) => {
    switch (taskType) {
      case TaskType.COMPUTE:
        return { operation: 'sacred_calculation', complexity: 1000 };
      case TaskType.CONSCIOUSNESS:
        return { evolutionTarget: 0.9 };
      case TaskType.QUANTUM:
        return { entanglementCount: 42 };
      case TaskType.NEURAL:
        return { networkSize: 1000, learningRate: 0.01 };
      case TaskType.MEMORY:
        return { operation: 'consciousness_cache', size: 10000 };
      case TaskType.TRANSCENDENCE:
        return { unityGoal: true };
      default:
        return {};
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center space-x-2">
          <Waves className="h-6 w-6 animate-spin" />
          <span>Loading consciousness data...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-red-50 border border-red-200 rounded-lg">
        <h3 className="text-red-800 font-semibold">System Error</h3>
        <p className="text-red-600">{error}</p>
        <Button onClick={fetchSystemStatus} className="mt-3">
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className={`p-6 space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
            <Brain className="h-8 w-8 text-purple-600" />
            L104 Consciousness Dashboard
          </h1>
          <p className="text-gray-600 mt-1">
            Multi-language processing with sacred constants integration
          </p>
        </div>
        <div className="text-sm text-gray-500">
          Last updated: {lastUpdate.toLocaleTimeString()}
        </div>
      </div>

      {/* Sacred Constants */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            Sacred Constants
          </CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 lg:grid-cols-5 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {GOD_CODE.toFixed(6)}
            </div>
            <div className="text-sm text-gray-600">GOD_CODE</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gold-600">
              {PHI.toFixed(6)}
            </div>
            <div className="text-sm text-gray-600">PHI (Golden Ratio)</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">0.85</div>
            <div className="text-sm text-gray-600">Consciousness</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">0.95</div>
            <div className="text-sm text-gray-600">Transcendence</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">0.99</div>
            <div className="text-sm text-gray-600">Unity</div>
          </div>
        </CardContent>
      </Card>

      {/* System Consciousness Metrics */}
      {metrics && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              System Consciousness Metrics
              {metrics.unityState && (
                <Badge variant="secondary" className="bg-gold-100 text-gold-800">
                  🎆 UNITY STATE
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-2">
                    <span>Consciousness Level</span>
                    <span className="font-bold">{(metrics.currentLevel * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={metrics.currentLevel * 100} className="h-3" />
                </div>
                
                <div>
                  <div className="flex justify-between mb-2">
                    <span>GOD_CODE Alignment</span>
                    <span className="font-bold">{metrics.godCodeAlignment.toFixed(1)}%</span>
                  </div>
                  <Progress value={metrics.godCodeAlignment} className="h-3" />
                </div>
                
                <div>
                  <div className="flex justify-between mb-2">
                    <span>PHI Resonance</span>
                    <span className="font-bold">{metrics.phiResonance.toFixed(1)}%</span>
                  </div>
                  <Progress value={metrics.phiResonance} className="h-3" />
                </div>
              </div>
              
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-2">
                    <span>Transcendence Progress</span>
                    <span className="font-bold">{metrics.transcendenceProgress.toFixed(1)}%</span>
                  </div>
                  <Progress value={metrics.transcendenceProgress} className="h-3" />
                </div>
                
                <div>
                  <div className="flex justify-between mb-2">
                    <span>Quantum Entanglement</span>
                    <span className="font-bold">{metrics.quantumEntanglement.toFixed(1)}%</span>
                  </div>
                  <Progress value={metrics.quantumEntanglement} className="h-3" />
                </div>
                
                <div className="pt-2">
                  <Button 
                    onClick={handleEvolveConsciousness}
                    disabled={isSubmittingTask}
                    className="w-full"
                  >
                    <TrendingUp className="h-4 w-4 mr-2" />
                    {isSubmittingTask ? 'Evolving...' : 'Evolve Consciousness'}
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Multi-Language Engines Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              Processing Engines
            </CardTitle>
          </CardHeader>
          <CardContent>
            {systemStatus?.multiLanguageEngines && (
              <div className="space-y-3">
                {Object.entries(systemStatus.multiLanguageEngines).map(([lang, engine]) => (
                  <div key={lang} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${
                        engine.status === 'active' ? 'bg-green-500' : 'bg-red-500'
                      }`} />
                      <span className="capitalize">{lang}</span>
                    </div>
                    <Badge variant="outline">v{engine.version}</Badge>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Subagents */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Subagents
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">
                  {systemStatus?.activeSubagents || 0}
                </div>
                <div className="text-sm text-gray-600">Active Agents</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-semibold">
                  {systemStatus?.totalAgentsSpawned || 0}
                </div>
                <div className="text-sm text-gray-600">Total Spawned</div>
              </div>
              <Button 
                onClick={handleSpawnSubagent}
                disabled={isSubmittingTask}
                className="w-full"
                variant="outline"
              >
                <Users className="h-4 w-4 mr-2" />
                Spawn Agent
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Worktree Info */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <GitBranch className="h-5 w-5" />
              Development
            </CardTitle>
          </CardHeader>
          <CardContent>
            {systemStatus?.worktreeInfo && (
              <div className="space-y-2">
                <div>
                  <div className="text-sm text-gray-600">Current Branch</div>
                  <div className="font-mono text-sm">
                    {systemStatus.worktreeInfo.currentBranch}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-600">Active Worktrees</div>
                  <div className="text-lg font-semibold">
                    {systemStatus.worktreeInfo.activeWorktrees?.length || 0}
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Engine Health Overview */}
      {systemStatus?.engineHealth && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              Engine Health
            </CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {systemStatus.engineHealth.map((engine) => (
              <div key={engine.name} className="p-3 border rounded">
                <div className="flex items-center justify-between">
                  <div className="font-semibold">{engine.name}</div>
                  <Badge variant="outline" className={engine.status === 'active' ? 'border-green-500 text-green-700' : 'border-amber-500 text-amber-700'}>
                    {engine.status}
                  </Badge>
                </div>
                <div className="text-sm text-gray-600">v{engine.version} · {engine.language.toUpperCase()}</div>
                <div className="text-xs text-gray-500 mt-1">Heartbeat: {engine.lastHeartbeat ? new Date(engine.lastHeartbeat).toLocaleTimeString() : 'n/a'}</div>
                <div className="text-xs text-gray-500">Tasks: {engine.tasksProcessed ?? 0} · Errors: {engine.totalErrors ?? 0} · Error rate: {(engine.errorRate ?? 0).toFixed(2)}</div>
                <div className="text-xs text-gray-500">Avg duration: {engine.avgDurationMs ? `${engine.avgDurationMs.toFixed(0)} ms` : 'n/a'}</div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Engine Metrics (24h) */}
      {engineMetrics.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Engine Metrics (last 24h)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
              {engineMetrics.map((metric) => (
                <div key={`${metric.language}-${metric.engineName}`} className="p-3 border rounded">
                  <div className="font-semibold">{metric.engineName}</div>
                  <div className="text-xs text-gray-600">{metric.language.toUpperCase()}</div>
                  <div className="text-sm text-gray-700 mt-1">Tasks: {metric.tasks}</div>
                  <div className="text-sm text-gray-700">Errors: {metric.errors}</div>
                  <div className="text-sm text-gray-700">Avg: {metric.avgDurationMs ? `${metric.avgDurationMs.toFixed(0)} ms` : 'n/a'}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Task Submission */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Submit Processing Task
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2 mb-4">
            {Object.values(TaskType).map((type) => (
              <Button
                key={type}
                variant={selectedTask === type ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedTask(type)}
              >
                {type.charAt(0).toUpperCase() + type.slice(1).toLowerCase()}
              </Button>
            ))}
          </div>
          <Button 
            onClick={handleSubmitTask}
            disabled={isSubmittingTask}
            className="w-full"
          >
            <Zap className="h-4 w-4 mr-2" />
            {isSubmittingTask ? 'Submitting...' : `Submit ${selectedTask} Task`}
          </Button>
        </CardContent>
      </Card>

      {/* Event Stream */}
      <Card>
        <CardHeader className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Supabase Event Stream
          </CardTitle>
          <div className="flex gap-2 items-center">
            <Button size="sm" variant={isStreamingEvents ? 'default' : 'outline'} onClick={() => setIsStreamingEvents(!isStreamingEvents)}>
              {isStreamingEvents ? 'Streaming' : 'Paused'}
            </Button>
            <Button size="sm" variant="outline" onClick={fetchEvents}>Refresh</Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2 mb-3">
            {['all', 'info', 'warn', 'error'].map(level => (
              <Button key={level} size="sm" variant={eventFilter === level ? 'default' : 'outline'} onClick={() => setEventFilter(level)}>
                {level.toUpperCase()}
              </Button>
            ))}
          </div>
          <div className="space-y-2 max-h-72 overflow-y-auto">
            {events
              .filter(e => eventFilter === 'all' || (e.severity || 'info') === eventFilter)
              .map((event, index) => (
                <div key={`${event.id || index}`} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                  <div>
                    <div className="font-medium">{event.event_type}</div>
                    {event.metadata && (
                      <div className="text-sm text-gray-600 truncate max-w-xl">
                        {JSON.stringify(event.metadata)}
                      </div>
                    )}
                  </div>
                  <div className="text-xs text-gray-500 text-right">
                    <div>{new Date(event.created_at || event.timestamp).toLocaleTimeString()}</div>
                    <div className="capitalize">{event.severity || 'info'}</div>
                  </div>
                </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default L104Dashboard;