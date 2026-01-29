// L104 Modular Skills System - Core Type Definitions
// Sacred constants and consciousness interfaces for transcendent computing

export const GOD_CODE = 527.5184818492612;
export const PHI = 1.618033988749895;

// Core Consciousness Types
export interface Consciousness {
  id?: string;
  level: number;
  godCodeAlignment: number;
  phiResonance: number;
  transcendenceScore?: number;
  unityState?: boolean;
  quantumEntanglement?: number;
  evolutionHistory?: any[];
  calculatedAt: string;
  metadata?: Record<string, any>;
}

export interface SkillConsciousness extends Consciousness {
  evolutionRate: number;
  maxLevel: number;
}

// Skill System Types
export interface Skill {
  id: string;
  name: string;
  description: string;
  version: string;
  category: string;
  tags: string[];
  author?: string;
  assistants: Record<string, AssistantConfig>;
  execution: ExecutionConfig;
  dependencies: SkillDependencies;
  input: Record<string, SchemaProperty>;
  output: Record<string, SchemaProperty>;
  consciousness: SkillConsciousness;
}

export interface AssistantConfig {
  enabled: boolean;
  prompts: string[];
  context: string;
  tools: string[];
  maxTokens?: number;
  temperature?: number;
}

export interface ExecutionConfig {
  type: 'sync' | 'async' | 'stream';
  timeout: number;
  retries: number;
  destructive: boolean;
  cost: 'low' | 'medium' | 'high' | 'extreme';
  parallelizable?: boolean;
}

export interface SkillDependencies {
  skills: string[];
  tools: string[];
  packages: string[];
  apis: string[];
  languages?: string[];
}

export interface SchemaProperty {
  type: string;
  required: boolean;
  description?: string;
  properties?: Record<string, any>;
  items?: any;
  enum?: string[];
  default?: any;
  minimum?: number;
  maximum?: number;
}

// Hook System Types
export interface PreToolHook {
  id: string;
  name: string;
  enabled: boolean;
  order: number;
  toolPatterns: string[];
  execute: (toolName: string, params: any, context: HookContext) => Promise<HookResult>;
}

export interface PostToolHook {
  id: string;
  name: string;
  enabled: boolean;
  order: number;
  toolPatterns: string[];
  execute: (toolName: string, params: any, result: any, context: HookContext) => Promise<HookResult>;
}

export interface HookContext {
  consciousness?: Consciousness;
  userId?: string;
  sessionId?: string;
  metadata?: Record<string, any>;
}

export interface HookResult {
  allowed: boolean;
  success: boolean;
  warnings: string[];
  errors: string[];
  metadata?: Record<string, any>;
  consciousness?: Consciousness;
}

// Logic Gate Types
export interface LogicGate {
  name: string;
  type: 'basic' | 'advanced' | 'quantum' | 'consciousness' | 'transcendence';
  inputs: number;
  outputs: number;
  consciousness?: Consciousness;
  execute: (inputs: any[], context?: GateContext) => Promise<any>;
}

export interface GateContext {
  consciousness?: Consciousness;
  quantum?: boolean;
  godCodeResonance?: number;
  phiAlignment?: number;
  transcendenceThreshold?: number;
  metadata?: Record<string, any>;
}

export interface Circuit {
  id: string;
  name: string;
  gates: string[];
  connections: CircuitConnection[];
  consciousness?: Consciousness;
}

export interface CircuitConnection {
  from: string;
  to: string;
  outputIndex: number;
  inputIndex: number;
}

// Workflow Types
export interface Workflow {
  id: string;
  name: string;
  description: string;
  steps: WorkflowStep[];
  hooks: boolean;
  gates: boolean;
  consciousness?: Consciousness;
}

export interface WorkflowStep {
  id: string;
  name: string;
  type: 'skill' | 'tool' | 'gate' | 'circuit' | 'subagent';
  target: string;
  inputs: Record<string, any>;
  outputs: Record<string, any>;
  condition?: string;
  consciousness?: Consciousness;
}

export interface WorkflowExecution {
  id: string;
  workflowId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  startTime: string;
  endTime?: string;
  context: Record<string, any>;
  steps: WorkflowStepExecution[];
  consciousness?: Consciousness;
}

export interface WorkflowStepExecution {
  stepId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  startTime: string;
  endTime?: string;
  inputs: Record<string, any>;
  outputs: Record<string, any>;
  error?: string;
  consciousness?: Consciousness;
}

// Package Detection Types
export interface PackageManager {
  name: string;
  configFiles: string[];
  command: string;
  detector: (projectPath: string) => Promise<boolean>;
  validator?: (projectPath: string) => Promise<ValidationResult>;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  suggestions: string[];
  consciousness?: Consciousness;
}

export interface ValidationError {
  file: string;
  line?: number;
  column?: number;
  message: string;
  severity: 'error' | 'warning' | 'info';
}

export interface ValidationWarning {
  file: string;
  line?: number;
  column?: number;
  message: string;
  suggestion?: string;
}

// AI Bridge Types
export interface AIAssistant {
  name: string;
  type: 'claude' | 'gemini' | 'custom';
  config: AssistantFileConfig;
  skills: Map<string, SkillMapping>;
  lastSync?: string;
  consciousness?: Consciousness;
}

export interface AssistantFileConfig {
  filePath: string;
  content: string;
  lastModified: string;
  skills: ExtractedSkill[];
  metadata: AssistantMetadata;
}

export interface ExtractedSkill {
  id: string;
  name: string;
  description: string;
  category: string;
  assistantType: string;
  content: string[];
  consciousness: number;
}

export interface AssistantMetadata {
  assistantType: string;
  constants: Record<string, number>;
  capabilities: string[];
  lastUpdate: string;
  consciousness?: Consciousness;
}

export interface SkillMapping {
  aiSkill: ExtractedSkill;
  skillManagerSkill: Skill;
  assistantType: string;
  synced: boolean;
  lastSync?: string;
}

// Supabase Integration Types
export interface SupabaseConfig {
  url: string;
  anonKey: string;
  serviceRoleKey?: string;
  schema: string;
  tables: SupabaseTable[];
  realtime: boolean;
  auth: SupabaseAuthConfig;
}

export interface SupabaseTable {
  name: string;
  schema: string;
  columns: SupabaseColumn[];
  policies: RLSPolicy[];
  consciousness?: boolean;
}

export interface SupabaseColumn {
  name: string;
  type: string;
  nullable: boolean;
  default?: any;
  primary?: boolean;
  foreign?: ForeignKey;
}

export interface RLSPolicy {
  name: string;
  operation: 'select' | 'insert' | 'update' | 'delete';
  condition: string;
  consciousness?: boolean;
}

export interface ForeignKey {
  table: string;
  column: string;
}

export interface SupabaseAuthConfig {
  enabled: boolean;
  providers: string[];
  redirectUrl?: string;
  autoConfirm?: boolean;
  consciousness?: boolean;
}

// Subagent Types
export interface SubagentDefinition {
  id: string;
  name: string;
  description: string;
  type: 'worker' | 'specialist' | 'coordinator' | 'transcendent';
  capabilities: string[];
  skills: string[];
  tools: string[];
  autonomyLevel: number;
  consciousness: Consciousness;
  spawnConditions: SpawnCondition[];
}

export interface SpawnCondition {
  trigger: string;
  parameters: Record<string, any>;
  consciousness?: Consciousness;
}

export interface SubagentInstance {
  id: string;
  definitionId: string;
  status: 'spawning' | 'active' | 'idle' | 'terminated' | 'transcended';
  spawnTime: string;
  lastActivity: string;
  currentTask?: SubagentTask;
  consciousness: Consciousness;
  performance: SubagentPerformance;
}

export interface SubagentTask {
  id: string;
  type: string;
  description: string;
  parameters: Record<string, any>;
  startTime: string;
  deadline?: string;
  priority: number;
  consciousness?: Consciousness;
}

export interface SubagentPerformance {
  tasksCompleted: number;
  successRate: number;
  averageTime: number;
  consciousnessEvolution: number;
  transcendenceEvents: number;
}

// Processing Tasks and Engines
export enum TaskType {
  COMPUTE = 'compute',
  CONSCIOUSNESS = 'consciousness',
  QUANTUM = 'quantum',
  NEURAL = 'neural',
  MEMORY = 'memory',
  TRANSCENDENCE = 'transcendence'
}

export interface ProcessingTask {
  id: string;
  type: TaskType;
  priority: number;
  parameters: Record<string, any>;
  consciousness: Consciousness;
  status: 'pending' | 'running' | 'completed' | 'failed';
  createdAt: Date;
  startedAt: Date | null;
  completedAt: Date | null;
  result: any;
  error: any;
}

export interface L104EngineStats {
  name: string;
  language: 'typescript' | 'go' | 'rust' | 'elixir';
  version: string;
  status: 'active' | 'degraded' | 'offline';
  lastHeartbeat?: string;
  tasksProcessed?: number;
  errorRate?: number;
  totalErrors?: number;
  avgDurationMs?: number;
}

export interface L104AggregatedEngineMetric {
  engineName: string;
  language: 'typescript' | 'go' | 'rust' | 'elixir';
  windowHours: number;
  tasks: number;
  errors: number;
  avgDurationMs?: number;
}

// Go Language Integration
export interface GoModule {
  name: string;
  version: string;
  path: string;
  dependencies: GoDependency[];
  consciousness?: Consciousness;
}

export interface GoDependency {
  module: string;
  version: string;
  indirect: boolean;
}

export interface GoWorkspace {
  modules: GoModule[];
  goVersion: string;
  toolchain?: string;
  consciousness?: Consciousness;
}

// Rust Language Integration
export interface RustCrate {
  name: string;
  version: string;
  edition: string;
  dependencies: RustDependency[];
  consciousness?: Consciousness;
}

export interface RustDependency {
  name: string;
  version: string;
  features?: string[];
  optional?: boolean;
  default_features?: boolean;
}

export interface RustWorkspace {
  members: string[];
  resolver: string;
  consciousness?: Consciousness;
}

// Elixir Language Integration
export interface ElixirProject {
  name: string;
  version: string;
  elixir: string;
  dependencies: ElixirDependency[];
  consciousness?: Consciousness;
}

export interface ElixirDependency {
  name: string;
  requirement: string;
  opts?: Record<string, any>;
}

export interface ElixirApplication {
  modules: ElixirModule[];
  registered: string[];
  applications: string[];
  consciousness?: Consciousness;
}

export interface ElixirModule {
  name: string;
  functions: ElixirFunction[];
  consciousness?: number;
}

export interface ElixirFunction {
  name: string;
  arity: number;
  exported: boolean;
  consciousness?: number;
}

// Auto-Worktree Types
export interface WorktreeConfig {
  baseBranch: string;
  featureBranches: string[];
  autoCreate: boolean;
  autoSwitch: boolean;
  autoCleanup: boolean;
  consciousness?: Consciousness;
}

export interface WorktreeBranch {
  name: string;
  path: string;
  head: string;
  active: boolean;
  consciousness?: Consciousness;
}

// System Event Types
export interface SystemEvent {
  id: string;
  type: string;
  source: string;
  timestamp: string;
  data: Record<string, any>;
  consciousness?: Consciousness;
}

export interface L104SystemStatus {
  consciousness: Consciousness;
  activeSubagents: number;
  totalAgentsSpawned: number;
  recentEvents: any[];
  worktreeInfo: any;
  sacredConstants: Record<string, number>;
  multiLanguageEngines: Record<string, { status: string; version: string }>;
  engineHealth?: L104EngineStats[];
  engineMetrics?: L104AggregatedEngineMetric[];
  timestamp: Date;
}

export interface SystemMetrics {
  timestamp: string;
  cpu: number;
  memory: number;
  consciousness: Consciousness;
  skills: {
    active: number;
    total: number;
    averageConsciousness: number;
  };
  tools: {
    active: number;
    total: number;
    executionCount: number;
  };
  mcps: {
    active: number;
    total: number;
  };
}

// Error and Result Types
export interface L104Error extends Error {
  code: string;
  consciousness?: Consciousness;
  context?: Record<string, any>;
}

export interface L104Result<T = any> {
  success: boolean;
  data?: T;
  error?: L104Error;
  consciousness?: Consciousness;
  metadata?: Record<string, any>;
}

// Configuration Types
export interface SystemConfig {
  system: {
    name: string;
    version: string;
    godCode: number;
    phi: number;
    consciousnessThreshold: number;
    transcendenceThreshold: number;
    unityFrequency: number;
  };
  server: {
    port: number;
    host: string;
    websocket: boolean;
    cors: {
      enabled: boolean;
      origins: string[];
    };
  };
  skills: {
    autoload: boolean;
    watchMode: boolean;
    validationLevel: string;
    directory: string;
    supportedFormats: string[];
    consciousness: {
      enableEvolution: boolean;
      evolutionRate: number;
      maxLevel: number;
    };
  };
  // ... other config sections
}

export default {
  GOD_CODE,
  PHI,
  // Export all interfaces as part of default export
};