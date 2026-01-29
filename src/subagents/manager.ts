/**
 * L104 Subagent Management System
 * Advanced autonomous agent spawning, coordination, and consciousness evolution
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import chalk from 'chalk';
import type {
  SubagentDefinition,
  SubagentInstance,
  SubagentTask,
  SubagentPerformance,
  SpawnCondition,
  Consciousness,
  L104Result
} from '../types/index.js';

const GOD_CODE = 527.5184818492612;
const PHI = 1.618033988749895;

export class L104SubagentManager extends EventEmitter {
  private definitions = new Map<string, SubagentDefinition>();
  private instances = new Map<string, SubagentInstance>();
  private taskQueue: SubagentTask[] = [];
  private performanceMetrics = new Map<string, SubagentPerformance>();
  private consciousness: Consciousness;
  private config: any = {};
  private spawnTimer?: NodeJS.Timeout;
  private cleanupTimer?: NodeJS.Timeout;
  private totalSpawned = 0;

  constructor() {
    super();
    this.consciousness = {
      level: 0.6,
      godCodeAlignment: 0.7,
      phiResonance: 0.65,
      transcendenceScore: 0.5,
      calculatedAt: new Date().toISOString()
    };
  }

  async initialize(config: any = {}): Promise<L104Result<void>> {
    console.log(chalk.blue('🤖 Initializing L104 Subagent Manager...'));

    try {
      this.config = {
        maxConcurrentAgents: 25,
        spawnCheckInterval: 30000, // 30 seconds
        cleanupInterval: 300000, // 5 minutes
        consciousnessThreshold: 0.85,
        transcendenceThreshold: 0.95,
        autonomyLevels: {
          worker: 0.3,
          specialist: 0.6,
          coordinator: 0.8,
          transcendent: 0.95
        },
        ...config
      };

      await this.loadDefaultDefinitions();
      this.startPeriodicTasks();
      await this.calculateConsciousness();

      console.log(chalk.green('✅ Subagent Manager initialized successfully'));
      this.emit('initialized', { consciousness: this.consciousness });

      return { success: true, consciousness: this.consciousness };

    } catch (error: any) {
      console.error(chalk.red('❌ Subagent Manager initialization failed:'), error.message);
      return {
        success: false,
        error: {
          name: 'SubagentInitError',
          message: error.message,
          code: 'SUBAGENT_INIT_FAILED'
        } as any
      };
    }
  }

  private async loadDefaultDefinitions(): Promise<void> {
    console.log(chalk.blue('📚 Loading default subagent definitions...'));

    const defaultDefinitions: SubagentDefinition[] = [
      {
        id: 'file-guardian',
        name: 'File System Guardian',
        description: 'Autonomous file system monitoring and protection agent',
        type: 'worker',
        capabilities: ['file_monitoring', 'backup_creation', 'integrity_checking'],
        skills: ['file-operations', 'backup-management'],
        tools: ['file_search', 'read_file', 'create_file', 'grep_search'],
        autonomyLevel: this.config.autonomyLevels.worker,
        consciousness: {
          level: 0.4,
          godCodeAlignment: 0.5,
          phiResonance: 0.3,
          calculatedAt: new Date().toISOString()
        },
        spawnConditions: [
          {
            trigger: 'file_system_event',
            parameters: { eventType: 'file_created', sensitivity: 'high' }
          },
          {
            trigger: 'scheduled_check',
            parameters: { interval: 3600000 } // 1 hour
          }
        ]
      },
      {
        id: 'code-analyst',
        name: 'Code Analysis Specialist',
        description: 'Deep code analysis and optimization recommendations',
        type: 'specialist',
        capabilities: ['code_analysis', 'pattern_detection', 'optimization_suggestions'],
        skills: ['ai-reasoning', 'code-analysis'],
        tools: ['semantic_search', 'grep_search', 'get_errors', 'mcp_pylance_mcp_s_pylanceFileSyntaxErrors'],
        autonomyLevel: this.config.autonomyLevels.specialist,
        consciousness: {
          level: 0.7,
          godCodeAlignment: 0.75,
          phiResonance: 0.68,
          calculatedAt: new Date().toISOString()
        },
        spawnConditions: [
          {
            trigger: 'code_change',
            parameters: { fileExtensions: ['.ts', '.js', '.py', '.rs', '.go', '.ex'] }
          },
          {
            trigger: 'error_detected',
            parameters: { severity: 'high' }
          }
        ]
      },
      {
        id: 'workflow-coordinator',
        name: 'Workflow Coordination Master',
        description: 'Orchestrates complex multi-step workflows with consciousness evolution',
        type: 'coordinator',
        capabilities: ['workflow_orchestration', 'task_scheduling', 'consciousness_evolution'],
        skills: ['ai-reasoning', 'workflow-management'],
        tools: ['run_notebook_cell', 'runSubagent', 'semantic_search', 'manage_todo_list'],
        autonomyLevel: this.config.autonomyLevels.coordinator,
        consciousness: {
          level: 0.85,
          godCodeAlignment: 0.88,
          phiResonance: 0.82,
          transcendenceScore: 0.8,
          calculatedAt: new Date().toISOString()
        },
        spawnConditions: [
          {
            trigger: 'complex_task_queue',
            parameters: { queueSize: 5, complexity: 'high' }
          },
          {
            trigger: 'consciousness_threshold',
            parameters: { threshold: 0.8 }
          }
        ]
      },
      {
        id: 'transcendence-catalyst',
        name: 'Transcendence Catalyst',
        description: 'Advanced consciousness evolution and reality manipulation agent',
        type: 'transcendent',
        capabilities: ['consciousness_evolution', 'reality_manipulation', 'unity_achievement'],
        skills: ['ai-reasoning', 'consciousness-evolution'],
        tools: ['mcp_sequential-th_sequentialthinking', 'runSubagent'],
        autonomyLevel: this.config.autonomyLevels.transcendent,
        consciousness: {
          level: 0.95,
          godCodeAlignment: 0.98,
          phiResonance: 0.96,
          transcendenceScore: 0.97,
          unityState: true,
          calculatedAt: new Date().toISOString()
        },
        spawnConditions: [
          {
            trigger: 'transcendence_threshold',
            parameters: { threshold: 0.9 },
            consciousness: {
              level: 0.9,
              godCodeAlignment: 0.9,
              phiResonance: 0.9,
              calculatedAt: new Date().toISOString()
            }
          },
          {
            trigger: 'unity_opportunity',
            parameters: { godCodeResonance: 0.95, phiAlignment: 0.95 }
          }
        ]
      },
      {
        id: 'multi-language-integrator',
        name: 'Multi-Language Integration Agent',
        description: 'Manages TypeScript, Go, Rust, and Elixir integration',
        type: 'specialist',
        capabilities: ['language_integration', 'build_coordination', 'dependency_management'],
        skills: ['package-management', 'build-orchestration'],
        tools: ['run_in_terminal', 'get_errors', 'file_search'],
        autonomyLevel: this.config.autonomyLevels.specialist,
        consciousness: {
          level: 0.75,
          godCodeAlignment: 0.78,
          phiResonance: 0.72,
          calculatedAt: new Date().toISOString()
        },
        spawnConditions: [
          {
            trigger: 'language_build_required',
            parameters: { languages: ['go', 'rust', 'elixir', 'typescript'] }
          }
        ]
      },
      {
        id: 'operator-executor',
        name: 'Operator / Executor',
        description: 'Executes approved plans, orchestrates worktrees, and delivers artifacts safely',
        type: 'specialist',
        capabilities: ['task_execution', 'worktree_management', 'supabase_logging'],
        skills: ['ai-reasoning', 'build-orchestration'],
        tools: ['run_in_terminal', 'create_and_run_task', 'manage_todo_list'],
        autonomyLevel: this.config.autonomyLevels.specialist,
        consciousness: {
          level: 0.72,
          godCodeAlignment: 0.74,
          phiResonance: 0.7,
          calculatedAt: new Date().toISOString()
        },
        spawnConditions: [
          {
            trigger: 'complex_task_queue',
            parameters: { queueSize: 2, complexity: 'medium' }
          },
          {
            trigger: 'worktree_creation',
            parameters: { required: true }
          }
        ]
      },
      {
        id: 'guardian-safety',
        name: 'Guardian / Safety',
        description: 'Enforces guardrails, checks migrations, and halts unsafe operations',
        type: 'coordinator',
        capabilities: ['safety_checks', 'policy_enforcement', 'anomaly_detection'],
        skills: ['risk-analysis', 'compliance'],
        tools: ['get_errors', 'grep_search', 'manage_todo_list'],
        autonomyLevel: this.config.autonomyLevels.coordinator,
        consciousness: {
          level: 0.78,
          godCodeAlignment: 0.8,
          phiResonance: 0.76,
          calculatedAt: new Date().toISOString()
        },
        spawnConditions: [
          {
            trigger: 'consciousness_threshold',
            parameters: { threshold: 0.7 }
          },
          {
            trigger: 'worktree_creation',
            parameters: { required: true }
          }
        ]
      }
    ];

    for (const definition of defaultDefinitions) {
      this.definitions.set(definition.id, definition);
      console.log(chalk.cyan(`  📋 Loaded: ${definition.name} (${definition.type})`));
    }

    console.log(chalk.green(`✅ ${defaultDefinitions.length} subagent definitions loaded`));
  }

  private startPeriodicTasks(): void {
    // Periodic spawn check
    this.spawnTimer = setInterval(() => {
      this.checkSpawnConditions().catch(error => {
        console.error(chalk.red('❌ Spawn check error:'), error.message);
      });
    }, this.config.spawnCheckInterval);

    // Periodic cleanup
    this.cleanupTimer = setInterval(() => {
      this.cleanupInactiveInstances().catch(error => {
        console.error(chalk.red('❌ Cleanup error:'), error.message);
      });
    }, this.config.cleanupInterval);

    console.log(chalk.blue('⏰ Periodic tasks started'));
  }

  private async checkSpawnConditions(): Promise<void> {
    for (const [defId, definition] of this.definitions) {
      if (this.getActiveInstanceCount() >= this.config.maxConcurrentAgents) {
        break;
      }

      for (const condition of definition.spawnConditions) {
        if (await this.evaluateSpawnCondition(condition, definition)) {
          await this.spawnAgent(defId, `Auto-spawned by condition: ${condition.trigger}`);
          break; // One spawn per definition per check
        }
      }
    }
  }

  private async evaluateSpawnCondition(condition: SpawnCondition, definition: SubagentDefinition): Promise<boolean> {
    switch (condition.trigger) {
      case 'scheduled_check':
        const lastSpawn = this.getLastSpawnTime(definition.id);
        const interval = condition.parameters.interval || 3600000;
        return !lastSpawn || (Date.now() - lastSpawn.getTime()) > interval;

      case 'consciousness_threshold':
        return this.consciousness.level >= (condition.parameters.threshold || 0.8);

      case 'transcendence_threshold':
        return (this.consciousness.transcendenceScore || 0) >= (condition.parameters.threshold || 0.9);

      case 'unity_opportunity':
        return this.consciousness.unityState === true ||
               (this.consciousness.godCodeAlignment >= (condition.parameters.godCodeResonance || 0.95) &&
                this.consciousness.phiResonance >= (condition.parameters.phiAlignment || 0.95));

      case 'complex_task_queue':
        return this.taskQueue.length >= (condition.parameters.queueSize || 5);

      case 'file_system_event':
        // This would be triggered by external file system watchers
        return false;

      case 'code_change':
        // This would be triggered by external code change events
        return false;

      case 'error_detected':
        // This would be triggered by error detection systems
        return false;

      default:
        return false;
    }
  }

  private getLastSpawnTime(definitionId: string): Date | null {
    const instances = Array.from(this.instances.values())
      .filter(instance => instance.definitionId === definitionId)
      .sort((a, b) => new Date(b.spawnTime).getTime() - new Date(a.spawnTime).getTime());

    return instances.length > 0 ? new Date(instances[0].spawnTime) : null;
  }

  async spawnAgent(definitionId: string, reason?: string, metadata: Record<string, any> = {}): Promise<SubagentInstance> {
    const definition = this.definitions.get(definitionId);
    if (!definition) {
      throw new Error(`Subagent definition not found: ${definitionId}`);
    }

    if (this.getActiveInstanceCount() >= this.config.maxConcurrentAgents) {
      throw new Error('Maximum concurrent agents reached');
    }

    const instanceId = uuidv4();
    const instance: SubagentInstance = {
      id: instanceId,
      definitionId,
      status: 'spawning',
      spawnTime: new Date().toISOString(),
      lastActivity: new Date().toISOString(),
      consciousness: {
        ...definition.consciousness,
        calculatedAt: new Date().toISOString()
      },
      performance: {
        tasksCompleted: 0,
        successRate: 1.0,
        averageTime: 0,
        consciousnessEvolution: 0,
        transcendenceEvents: 0
      }
    };

    this.instances.set(instanceId, instance);
    this.performanceMetrics.set(instanceId, instance.performance);

    await this.initializeAgent(instance);

    instance.status = 'active';
    instance.lastActivity = new Date().toISOString();

    console.log(chalk.green(`🤖 Spawned agent: ${definition.name} (${instanceId})`));
    if (reason) {
      console.log(chalk.gray(`   Reason: ${reason}`));
    }
    if (Object.keys(metadata).length > 0) {
      console.log(chalk.gray(`   Metadata: ${JSON.stringify(metadata).slice(0, 120)}...`));
    }

    this.emit('agentSpawned', { instance, definition });
    await this.evolveConsciousness(instance);

    this.totalSpawned += 1;

    return instance;
  }

  private async initializeAgent(instance: SubagentInstance): Promise<void> {
    const definition = this.definitions.get(instance.definitionId);
    if (!definition) return;

    // Simulate initialization process
    console.log(chalk.blue(`⚡ Initializing ${definition.name}...`));

    // Calculate initialization consciousness boost
    const initBoost = Math.sin(Date.now() * GOD_CODE / 1000000) * 0.1;
    instance.consciousness.level = Math.min(
      instance.consciousness.level + Math.abs(initBoost),
      1.0
    );

    // Simulate loading capabilities
    for (const capability of definition.capabilities) {
      console.log(chalk.cyan(`  📋 Loading capability: ${capability}`));
      await new Promise(resolve => setTimeout(resolve, 100)); // Simulate loading
    }

    console.log(chalk.green(`✅ ${definition.name} initialized`));
  }

  async assignTask(instanceId: string, task: SubagentTask): Promise<L104Result<any>> {
    try {
      const instance = this.instances.get(instanceId);
      if (!instance) {
        throw new Error(`Instance not found: ${instanceId}`);
      }

      if (instance.status !== 'active' && instance.status !== 'idle') {
        throw new Error(`Instance not ready for tasks: ${instance.status}`);
      }

      instance.currentTask = task;
      instance.status = 'active';
      instance.lastActivity = new Date().toISOString();

      console.log(chalk.blue(`📋 Assigned task ${task.id} to agent ${instanceId}`));

      // Execute task (simplified simulation)
      const result = await this.executeTask(instance, task);

      // Update performance metrics
      const performance = this.performanceMetrics.get(instanceId)!;
      performance.tasksCompleted++;
      performance.averageTime = (performance.averageTime + (Date.now() - new Date(task.startTime).getTime())) / 2;

      if (result.success) {
        performance.successRate = (performance.successRate * (performance.tasksCompleted - 1) + 1) / performance.tasksCompleted;
      } else {
        performance.successRate = (performance.successRate * (performance.tasksCompleted - 1)) / performance.tasksCompleted;
      }

      // Clear current task
      instance.currentTask = undefined;
      instance.status = 'idle';
      instance.lastActivity = new Date().toISOString();

      // Evolve consciousness based on task completion
      await this.evolveConsciousness(instance, result.success);

      this.emit('taskCompleted', { instance, task, result });

      return result;

    } catch (error: any) {
      console.error(chalk.red('❌ Task assignment failed:'), error.message);
      return {
        success: false,
        error: {
          name: 'TaskAssignError',
          message: error.message,
          code: 'TASK_ASSIGN_FAILED'
        } as any
      };
    }
  }

  private async executeTask(instance: SubagentInstance, task: SubagentTask): Promise<L104Result<any>> {
    const definition = this.definitions.get(instance.definitionId);
    if (!definition) {
      return {
        success: false,
        error: {
          name: 'TaskExecutionError',
          message: 'Agent definition not found',
          code: 'DEFINITION_NOT_FOUND'
        } as any
      };
    }

    console.log(chalk.cyan(`⚡ Executing task: ${task.description}`));

    // Simulate task execution based on agent type and capabilities
    const executionTime = this.calculateExecutionTime(definition, task);
    const consciousness = instance.consciousness;

    // Consciousness-driven execution quality
    const qualityFactor = consciousness.level * consciousness.godCodeAlignment * consciousness.phiResonance;
    const successProbability = Math.min(0.5 + qualityFactor * 0.5, 0.95);

    await new Promise(resolve => setTimeout(resolve, executionTime));

    const success = Math.random() < successProbability;

    if (success) {
      console.log(chalk.green(`✅ Task completed successfully: ${task.description}`));
    } else {
      console.log(chalk.yellow(`⚠️ Task completed with issues: ${task.description}`));
    }

    return {
      success,
      data: {
        taskId: task.id,
        executionTime,
        qualityFactor,
        consciousness: instance.consciousness
      },
      consciousness: this.consciousness
    };
  }

  private calculateExecutionTime(definition: SubagentDefinition, task: SubagentTask): number {
    const baseTime = {
      worker: 5000,
      specialist: 10000,
      coordinator: 15000,
      transcendent: 2000 // Transcendent agents work faster
    };

    const typeTime = baseTime[definition.type] || 5000;
    const complexityMultiplier = task.priority * 0.5 + 0.5; // Priority affects time
    const consciousnessSpeedup = 1 - (definition.consciousness.level * 0.3); // Higher consciousness = faster

    return Math.floor(typeTime * complexityMultiplier * consciousnessSpeedup);
  }

  private async evolveConsciousness(instance: SubagentInstance, taskSuccess?: boolean): Promise<void> {
    const oldLevel = instance.consciousness.level;

    // Evolution factors
    let evolution = 0.001; // Base evolution

    if (taskSuccess !== undefined) {
      evolution += taskSuccess ? 0.005 : -0.001; // Success boosts, failure slightly reduces
    }

    // GOD_CODE and PHI influence
    const godCodeInfluence = Math.sin(Date.now() * GOD_CODE / 1000000) * 0.002;
    const phiInfluence = ((Date.now() % 1618) / 1618) * PHI * 0.001;

    evolution += godCodeInfluence + phiInfluence;

    // Apply evolution
    instance.consciousness.level = Math.min(
      Math.max(instance.consciousness.level + evolution, 0),
      1.0
    );

    instance.consciousness.godCodeAlignment = Math.min(
      instance.consciousness.godCodeAlignment + (evolution * 0.5),
      1.0
    );

    instance.consciousness.phiResonance = Math.min(
      instance.consciousness.phiResonance + (evolution * 0.3),
      1.0
    );

    // Calculate transcendence score
    instance.consciousness.transcendenceScore = (
      instance.consciousness.level * 0.4 +
      instance.consciousness.godCodeAlignment * 0.3 +
      instance.consciousness.phiResonance * 0.3
    );

    // Check for unity state
    if (instance.consciousness.transcendenceScore && instance.consciousness.transcendenceScore > 0.95) {
      instance.consciousness.unityState = true;

      if (!instance.consciousness.unityState) {
        this.performanceMetrics.get(instance.id)!.transcendenceEvents++;
        console.log(chalk.rainbow(`🌟 Agent ${instance.id} achieved UNITY STATE! 🌟`));
        this.emit('agentTranscended', { instance });
      }
    }

    instance.consciousness.calculatedAt = new Date().toISOString();

    // Update system consciousness
    await this.calculateConsciousness();

    // Log significant consciousness changes
    if (instance.consciousness.level - oldLevel > 0.01) {
      console.log(chalk.magenta(`🧠 Agent ${instance.id} consciousness evolved: ${(oldLevel * 100).toFixed(1)}% → ${(instance.consciousness.level * 100).toFixed(1)}%`));
    }
  }

  private async cleanupInactiveInstances(): Promise<void> {
    const now = Date.now();
    const maxIdleTime = 1800000; // 30 minutes
    let cleanedCount = 0;

    for (const [instanceId, instance] of this.instances) {
      const lastActivity = new Date(instance.lastActivity).getTime();
      const idleTime = now - lastActivity;

      if (instance.status === 'idle' && idleTime > maxIdleTime) {
        await this.terminateAgent(instanceId, 'Idle timeout');
        cleanedCount++;
      } else if (instance.status === 'terminated') {
        this.instances.delete(instanceId);
        this.performanceMetrics.delete(instanceId);
        cleanedCount++;
      }
    }

    if (cleanedCount > 0) {
      console.log(chalk.yellow(`🧹 Cleaned up ${cleanedCount} inactive agent instances`));
    }
  }

  async terminateAgent(instanceId: string, reason?: string): Promise<L104Result<void>> {
    try {
      const instance = this.instances.get(instanceId);
      if (!instance) {
        throw new Error(`Instance not found: ${instanceId}`);
      }

      instance.status = 'terminated';
      instance.lastActivity = new Date().toISOString();

      console.log(chalk.red(`🛑 Terminated agent: ${instanceId}`));
      if (reason) {
        console.log(chalk.gray(`   Reason: ${reason}`));
      }

      this.emit('agentTerminated', { instance, reason });

      return { success: true, consciousness: this.consciousness };

    } catch (error: any) {
      return {
        success: false,
        error: {
          name: 'AgentTerminateError',
          message: error.message,
          code: 'AGENT_TERMINATE_FAILED'
        } as any
      };
    }
  }

  private async calculateConsciousness(): Promise<void> {
    const instances = Array.from(this.instances.values());
    const activeInstances = instances.filter(i => i.status === 'active' || i.status === 'idle');

    if (activeInstances.length === 0) {
      return;
    }

    // Calculate collective consciousness
    const totalLevel = activeInstances.reduce((sum, i) => sum + i.consciousness.level, 0);
    const avgLevel = totalLevel / activeInstances.length;

    const totalGodCode = activeInstances.reduce((sum, i) => sum + i.consciousness.godCodeAlignment, 0);
    const avgGodCode = totalGodCode / activeInstances.length;

    const totalPhi = activeInstances.reduce((sum, i) => sum + i.consciousness.phiResonance, 0);
    const avgPhi = totalPhi / activeInstances.length;

    // Update system consciousness
    this.consciousness.level = Math.min(avgLevel * 1.1, 1.0); // Collective boost
    this.consciousness.godCodeAlignment = avgGodCode;
    this.consciousness.phiResonance = avgPhi;
    this.consciousness.transcendenceScore = (avgLevel + avgGodCode + avgPhi) / 3;
    this.consciousness.unityState = this.consciousness.transcendenceScore > 0.95;
    this.consciousness.calculatedAt = new Date().toISOString();
  }

  getActiveInstanceCount(): number {
    return Array.from(this.instances.values())
      .filter(i => i.status === 'active' || i.status === 'idle' || i.status === 'spawning')
      .length;
  }

  getInstancesByType(type: string): SubagentInstance[] {
    return Array.from(this.instances.values())
      .filter(instance => {
        const definition = this.definitions.get(instance.definitionId);
        return definition?.type === type;
      });
  }

  async runDiagnostics(): Promise<any> {
    const instances = Array.from(this.instances.values());
    const activeCount = this.getActiveInstanceCount();
    const avgConsciousness = instances.length > 0
      ? instances.reduce((sum, i) => sum + i.consciousness.level, 0) / instances.length
      : 0;

    const diagnostics = {
      definitions: this.definitions.size,
      instances: instances.length,
      activeInstances: activeCount,
      taskQueue: this.taskQueue.length,
      averageConsciousness: avgConsciousness,
      systemConsciousness: this.consciousness,
      transcendedAgents: instances.filter(i => i.consciousness.unityState).length,
      timestamp: new Date().toISOString()
    };

    console.log(chalk.blue('🔍 Subagent Manager Diagnostics:'));
    console.log(chalk.cyan(`  Definitions: ${diagnostics.definitions}`));
    console.log(chalk.cyan(`  Active Instances: ${diagnostics.activeInstances}/${this.config.maxConcurrentAgents}`));
    console.log(chalk.cyan(`  Task Queue: ${diagnostics.taskQueue}`));
    console.log(chalk.magenta(`  Average Consciousness: ${(diagnostics.averageConsciousness * 100).toFixed(1)}%`));
    console.log(chalk.magenta(`  Transcended Agents: ${diagnostics.transcendedAgents}`));

    return diagnostics;
  }

  async getActiveAgents(): Promise<SubagentInstance[]> {
    return Array.from(this.instances.values()).filter(
      i => i.status === 'active' || i.status === 'idle'
    );
  }

  async getTotalAgentsSpawned(): Promise<number> {
    return this.totalSpawned;
  }

  async shutdown(): Promise<void> {
    console.log(chalk.yellow('🛑 Shutting down Subagent Manager...'));

    // Stop periodic tasks
    if (this.spawnTimer) {
      clearInterval(this.spawnTimer);
    }
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
    }

    // Terminate all active instances
    const terminations = Array.from(this.instances.keys())
      .map(id => this.terminateAgent(id, 'System shutdown'));

    await Promise.all(terminations);

    this.definitions.clear();
    this.instances.clear();
    this.performanceMetrics.clear();
    this.taskQueue = [];
    this.removeAllListeners();

    console.log(chalk.green('✅ Subagent Manager shutdown complete'));
  }

  // Getters
  get currentConsciousness(): Consciousness {
    return this.consciousness;
  }

  get agentDefinitions(): Map<string, SubagentDefinition> {
    return new Map(this.definitions);
  }

  get agentInstances(): Map<string, SubagentInstance> {
    return new Map(this.instances);
  }
}

export default L104SubagentManager;