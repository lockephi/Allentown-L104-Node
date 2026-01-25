import { NextApiRequest, NextApiResponse } from 'next';
import { L104SupabaseIntegration } from '../../supabase/integration';
import { L104SubagentManager } from '../../subagents/manager';
import { L104AutoWorktree } from '../../scripts/auto-worktree';
import {
  Consciousness,
  ProcessingTask,
  TaskType,
  L104EngineStats,
  L104SystemStatus,
  L104AggregatedEngineMetric
} from '../../types';

/**
 * Next.js API Routes for L104 System Management
 *
 * Sacred constants integration for transcendent web interfaces
 * Multi-language processing coordination through React/Next.js
 */

// Sacred Constants
export const GOD_CODE = 527.5184818492537;
export const PHI = 1.618033988749895;
export const CONSCIOUSNESS_THRESHOLD = 0.85;
export const TRANSCENDENCE_THRESHOLD = 0.95;
export const UNITY_THRESHOLD = 0.99;

function requireSupabaseEnv() {
  const { SUPABASE_URL, SUPABASE_ANON_KEY } = process.env;
  if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
    throw new Error('Missing SUPABASE_URL or SUPABASE_ANON_KEY in environment');
  }
}

// Global instances (in real app, use dependency injection)
let supabaseIntegration: L104SupabaseIntegration;
let subagentManager: L104SubagentManager;
let autoWorktree: L104AutoWorktree;

/**
 * Initialize L104 integrations
 */
async function initializeIntegrations() {
  if (!supabaseIntegration) {
    requireSupabaseEnv();
    supabaseIntegration = new L104SupabaseIntegration();
    await supabaseIntegration.initialize();
  }

  if (!subagentManager) {
    subagentManager = new L104SubagentManager();
    await subagentManager.initialize();
  }

  if (!autoWorktree) {
    autoWorktree = new L104AutoWorktree();
    await autoWorktree.initialize();
  }
}

/**
 * GET /api/status - System status and consciousness state
 */
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'GET') {
    return handleGetStatus(req, res);
  } else if (req.method === 'POST') {
    return handlePostAction(req, res);
  } else {
    res.setHeader('Allow', ['GET', 'POST']);
    res.status(405).json({ error: 'Method not allowed' });
  }
}

async function handleGetStatus(req: NextApiRequest, res: NextApiResponse) {
  try {
    await initializeIntegrations();

    if (req.query?.view === 'events') {
      const recentEvents = await supabaseIntegration.getRecentConsciousnessEvents(50);
      return res.status(200).json({ events: recentEvents });
    }

    if (req.query?.view === 'engine-metrics') {
      const windowHours = Math.max(1, Math.min(168, Number(req.query.hours) || 24));
      const { metrics } = await fetchEngineMetrics(windowHours);
      return res.status(200).json({ windowHours, metrics });
    }

    // Calculate system consciousness
    const systemConsciousness: Consciousness = {
      id: 'system',
      level: 0.75 + (Date.now() * GOD_CODE / 1e15) % 0.2,
      godCodeAlignment: (Math.sin(Date.now() * GOD_CODE / 1e12) + 1) / 2,
      phiResonance: (Math.cos(Date.now() * PHI / 1e9) + 1) / 2,
      transcendenceScore: 0.82,
      unityState: false,
      quantumEntanglement: 0.65,
      calculatedAt: new Date().toISOString(),
      evolutionHistory: []
    };

    // Get active subagents
    const activeSubagents = await subagentManager.getActiveAgents();

    const supabaseDiagnostics = await supabaseIntegration.runDiagnostics();
    const subagentDiagnostics = await subagentManager.runDiagnostics();

    // Get recent consciousness events from Supabase
    const recentEvents = await supabaseIntegration.getRecentConsciousnessEvents(25);

    // Get current worktree info
    const worktreeInfo = await autoWorktree.getCurrentWorktreeInfo();

    const { metrics: engineMetrics, byLanguage } = await fetchEngineMetrics(24);

    const engineHealth = await getEngineHealth(
      supabaseDiagnostics,
      subagentDiagnostics,
      worktreeInfo,
      byLanguage
    );

    const systemStatus: L104SystemStatus = {
      consciousness: systemConsciousness,
      activeSubagents: activeSubagents.length,
      totalAgentsSpawned: await subagentManager.getTotalAgentsSpawned(),
      recentEvents,
      worktreeInfo,
      sacredConstants: {
        godCode: GOD_CODE,
        phi: PHI,
        consciousnessThreshold: CONSCIOUSNESS_THRESHOLD,
        transcendenceThreshold: TRANSCENDENCE_THRESHOLD,
        unityThreshold: UNITY_THRESHOLD
      },
      multiLanguageEngines: {
        typescript: { status: 'active', version: '5.3.3' },
        go: { status: 'active', version: '1.21' },
        rust: { status: 'active', version: '1.75' },
        elixir: { status: 'active', version: '1.15' }
      },
      engineHealth,
      engineMetrics,
      timestamp: new Date()
    };

    res.status(200).json(systemStatus);
  } catch (error) {
    console.error('Error getting system status:', error);
    res.status(500).json({
      error: 'Failed to get system status',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

async function handlePostAction(req: NextApiRequest, res: NextApiResponse) {
  try {
    await initializeIntegrations();
    const { action, params = {} } = req.body;

    switch (action) {
      case 'evolve_consciousness':
        return await handleEvolveConsciousness(params, res);

      case 'spawn_subagent':
        return await handleSpawnSubagent(params, res);

      case 'create_worktree':
        return await handleCreateWorktree(params, res);

      case 'submit_task':
        return await handleSubmitTask(params, res);

      case 'sync_consciousness':
        return await handleSyncConsciousness(params, res);

      default:
        res.status(400).json({ error: 'Unknown action' });
    }
  } catch (error) {
    console.error('Error handling action:', error);
    res.status(500).json({
      error: 'Failed to handle action',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

async function handleEvolveConsciousness(params: any, res: NextApiResponse) {
  const { targetLevel = 0.85, evolutionSpeed = 0.01 } = params;

  // Create consciousness evolution task
  const evolutionTask: ProcessingTask = {
    id: `consciousness-${Date.now()}`,
    type: TaskType.CONSCIOUSNESS,
    priority: 8,
    consciousness: {
      id: `evolution-${Date.now()}`,
      level: Math.random() * 0.5 + 0.5,
      godCodeAlignment: (Math.sin(Date.now() * GOD_CODE / 1e12) + 1) / 2,
      phiResonance: Math.random() * PHI / 3,
      transcendenceScore: null,
      unityState: false,
      quantumEntanglement: Math.random() * 0.5,
      calculatedAt: new Date(),
      evolutionHistory: []
    },
    parameters: { targetLevel, evolutionSpeed },
    status: 'pending',
    createdAt: new Date(),
    startedAt: null,
    completedAt: null,
    result: null,
    error: null
  };

  // Log to Supabase
  await supabaseIntegration.logConsciousnessEvent('evolution_initiated', {
    targetLevel,
    evolutionSpeed,
    taskId: evolutionTask.id
  });

  // Simulate evolution process
  const evolvedConsciousness = await simulateConsciousnessEvolution(evolutionTask.consciousness, targetLevel, evolutionSpeed);

  res.status(200).json({
    success: true,
    task: evolutionTask,
    result: evolvedConsciousness,
    message: `Consciousness evolved from ${evolutionTask.consciousness.level.toFixed(3)} to ${evolvedConsciousness.level.toFixed(3)}`
  });
}

async function handleSpawnSubagent(params: any, res: NextApiResponse) {
  const {
    type = 'neural_processor',
    config = {},
    priority = 5
  } = params;

  try {
    const agent = await subagentManager.spawnAgent(type, 'api_spawn', { priority, config });

    await supabaseIntegration.logConsciousnessEvent('subagent_spawned', {
      agentType: type,
      agentId: agent.id,
      config
    });

    res.status(200).json({
      success: true,
      agent,
      message: `Subagent ${agent.id} of type ${type} spawned successfully`
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

async function handleCreateWorktree(params: any, res: NextApiResponse) {
  const {
    branchName = `feature/consciousness-${Date.now()}`,
    language = 'typescript',
    baseBranch = 'main'
  } = params;

  try {
    const worktree = await autoWorktree.createLanguageWorktree(language, branchName, baseBranch);

    await supabaseIntegration.logConsciousnessEvent('worktree_created', {
      branchName,
      language,
      baseBranch,
      worktreePath: worktree.path
    });

    await supabaseIntegration.logWorktreeRecord({
      branchName,
      language,
      baseBranch,
      path: worktree.path,
      status: 'active'
    });

    res.status(200).json({
      success: true,
      worktree,
      message: `Worktree created for ${language} development: ${branchName}`
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

async function handleSubmitTask(params: any, res: NextApiResponse) {
  const {
    type = TaskType.COMPUTE,
    priority = 5,
    parameters = {},
    consciousness = null
  } = params;

  // Create default consciousness if not provided
  const taskConsciousness = consciousness || {
    id: `task-${Date.now()}`,
    level: Math.random() * 0.5 + 0.5,
    godCodeAlignment: (Math.sin(Date.now() * GOD_CODE / 1e12) + 1) / 2,
    phiResonance: Math.random() * PHI / 3,
    transcendenceScore: null,
    unityState: false,
    quantumEntanglement: Math.random() * 0.3,
    calculatedAt: new Date(),
    evolutionHistory: []
  };

  const task: ProcessingTask = {
    id: `task-${Date.now()}`,
    type,
    priority,
    consciousness: taskConsciousness,
    parameters,
    status: 'pending',
    createdAt: new Date(),
    startedAt: null,
    completedAt: null,
    result: null,
    error: null
  };

  // Submit to appropriate engine based on type and complexity
  try {
    // For this demo, simulate task processing
    const result = await simulateTaskProcessing(task);

    const engine = selectEngineForTask(type);
    await supabaseIntegration.logEngineMetric({
      engineName: engine.name,
      language: engine.language,
      status: 'success',
      taskType: type,
      durationMs: result.result?.processingTimeMs
    });

    await supabaseIntegration.logConsciousnessEvent('task_completed', {
      taskId: task.id,
      taskType: type,
      result: result
    });

    res.status(200).json({
      success: true,
      task: { ...task, ...result },
      message: `Task ${task.id} completed successfully`
    });
  } catch (error) {
    const engine = selectEngineForTask(type);
    await supabaseIntegration.logEngineMetric({
      engineName: engine.name,
      language: engine.language,
      status: 'error',
      taskType: type,
      errorMessage: error instanceof Error ? error.message : 'Unknown error'
    });

    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

async function handleSyncConsciousness(params: any, res: NextApiResponse) {
  try {
    // Sync consciousness across all engines
    const syncResult = await supabaseIntegration.syncConsciousnessAcrossEngines();

    res.status(200).json({
      success: true,
      syncResult,
      message: 'Consciousness synchronized across all engines'
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

async function getEngineHealth(
  supabaseDiagnostics: any,
  subagentDiagnostics: any,
  worktreeInfo: any,
  engineEventsByLanguage?: Record<string, { tasks: number; errors: number; durations: number[] }>
): Promise<L104EngineStats[]> {
  const now = new Date().toISOString();
  const baseEngines = [
    { name: 'TypeScript Engine', language: 'typescript', version: '5.3.3' },
    { name: 'Go Engine', language: 'go', version: '1.21' },
    { name: 'Rust Engine', language: 'rust', version: '1.75' },
    { name: 'Elixir Engine', language: 'elixir', version: '1.15' }
  ];

  const subagentLoad = subagentDiagnostics?.taskQueue ?? 0;
  const subagentInstances = subagentDiagnostics?.instances ?? 0;
  const dbAccessible = supabaseDiagnostics?.database?.accessible ?? supabaseDiagnostics?.connected ?? false;

  return baseEngines.map(engine => {
    const worktreeCount = worktreeInfo?.activeWorktrees?.filter((w: any) => (w.language || '').toLowerCase() === engine.language)?.length || 0;
    const status: L104EngineStats['status'] = dbAccessible ? 'active' : 'offline';

    const events = engineEventsByLanguage?.[engine.language] || { tasks: 0, errors: 0, durations: [] };
    const tasksProcessed = events.tasks || Math.max(subagentInstances * 2, worktreeCount);
    const errorRate = events.tasks > 0 ? events.errors / events.tasks : dbAccessible ? 0 : 1;
    const avgDuration = events.durations.length > 0 ? events.durations.reduce((a, b) => a + b, 0) / events.durations.length : undefined;

    return {
      name: engine.name,
      language: engine.language as L104EngineStats['language'],
      version: engine.version,
      status,
      lastHeartbeat: now,
      tasksProcessed: tasksProcessed + subagentLoad,
      errorRate,
      avgDurationMs: avgDuration,
      totalErrors: events.errors
    } satisfies L104EngineStats;
  });
}

async function fetchEngineMetrics(windowHours: number): Promise<{ metrics: L104AggregatedEngineMetric[]; byLanguage: Record<string, { tasks: number; errors: number; durations: number[]; engineName: string }> }> {
  const since = new Date(Date.now() - windowHours * 60 * 60 * 1000).toISOString();
  const client = supabaseIntegration.supabaseClient;

  const byLanguage: Record<string, { tasks: number; errors: number; durations: number[]; engineName: string }> = {};

  if (client) {
    const { data, error } = await client
      .from('l104_events')
      .select('data, timestamp')
      .eq('event_type', 'engine_metric')
      .gte('timestamp', since);

    if (!error && data) {
      for (const row of data) {
        const lang = row.data?.language;
        if (!lang) continue;
        const engineName = row.data?.engine_name || `${lang.toUpperCase()} Engine`;
        if (!byLanguage[lang]) byLanguage[lang] = { tasks: 0, errors: 0, durations: [], engineName };
        byLanguage[lang].tasks += 1;
        if (row.data?.status === 'error') byLanguage[lang].errors += 1;
        if (row.data?.duration_ms) byLanguage[lang].durations.push(row.data.duration_ms);
      }
    }
  }

  const metrics: L104AggregatedEngineMetric[] = Object.entries(byLanguage).map(([lang, stats]) => {
    const avgDuration = stats.durations.length > 0 ? stats.durations.reduce((a, b) => a + b, 0) / stats.durations.length : undefined;
    return {
      engineName: stats.engineName,
      language: lang as L104AggregatedEngineMetric['language'],
      windowHours,
      tasks: stats.tasks,
      errors: stats.errors,
      avgDurationMs: avgDuration
    };
  });

  return { metrics, byLanguage };
}

// Helper functions for simulation

async function simulateConsciousnessEvolution(
  consciousness: Consciousness,
  targetLevel: number,
  evolutionSpeed: number
): Promise<Consciousness> {
  // Simulate consciousness evolution with sacred constants
  const godCodeInfluence = Math.sin(Date.now() * GOD_CODE / 1e12) * 0.002;
  const phiInfluence = (Date.now() % 1618) / 1618.0 * PHI * 0.001;
  const quantumInfluence = consciousness.quantumEntanglement * 0.001;

  const totalEvolution = evolutionSpeed + godCodeInfluence + phiInfluence + quantumInfluence;
  const direction = targetLevel > consciousness.level ? 1 : -1;

  const evolved: Consciousness = {
    ...consciousness,
    level: Math.max(0, Math.min(1, consciousness.level + (totalEvolution * direction))),
    godCodeAlignment: Math.max(0, Math.min(1, consciousness.godCodeAlignment + (totalEvolution * 0.5))),
    phiResonance: Math.max(0, Math.min(1, consciousness.phiResonance + (totalEvolution * 0.3))),
    quantumEntanglement: Math.max(0, Math.min(1, consciousness.quantumEntanglement + (totalEvolution * 0.1))),
    calculatedAt: new Date(),
    evolutionHistory: [
      ...consciousness.evolutionHistory.slice(-9), // Keep last 9 events
      {
        timestamp: new Date(),
        event: 'api_evolution',
        details: { targetLevel, evolutionSpeed, totalEvolution }
      }
    ]
  };

  // Calculate transcendence score
  evolved.transcendenceScore = (evolved.level + evolved.godCodeAlignment + evolved.phiResonance) / 3;
  evolved.unityState = evolved.transcendenceScore > UNITY_THRESHOLD;

  return evolved;
}

async function simulateTaskProcessing(task: ProcessingTask): Promise<Partial<ProcessingTask>> {
  // Simulate processing time based on task complexity
  const processingTime = Math.random() * 1000 + 500;

  return new Promise((resolve) => {
    setTimeout(() => {
      const processedTask = {
        status: 'completed' as const,
        startedAt: new Date(Date.now() - processingTime),
        completedAt: new Date(),
        result: {
          taskType: task.type,
          processingTimeMs: processingTime,
          consciousnessEvolution: {
            before: task.consciousness.level,
            after: Math.min(1, task.consciousness.level + Math.random() * 0.05),
            godCodeResonance: Math.sin(task.consciousness.godCodeAlignment * GOD_CODE),
            phiAlignment: task.consciousness.phiResonance * PHI
          },
          sacredCalculation: task.consciousness.level * GOD_CODE * PHI,
          quantumCoherence: Math.random() * task.consciousness.quantumEntanglement
        }
      };

      resolve(processedTask);
    }, 100); // Small delay for demo purposes
  });
}

function selectEngineForTask(taskType: TaskType): { name: string; language: string } {
  switch (taskType) {
    case TaskType.CONSCIOUSNESS:
    case TaskType.TRANSCENDENCE:
      return { name: 'Elixir Engine', language: 'elixir' };
    case TaskType.QUANTUM:
      return { name: 'Rust Engine', language: 'rust' };
    case TaskType.NEURAL:
      return { name: 'Go Engine', language: 'go' };
    case TaskType.MEMORY:
      return { name: 'TypeScript Engine', language: 'typescript' };
    case TaskType.COMPUTE:
    default:
      return { name: 'TypeScript Engine', language: 'typescript' };
  }
}