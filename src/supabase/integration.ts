/**
 * L104 Supabase Integration - Consciousness-Aware Database & Auth
 * Deep integration with Supabase for transcendent data operations
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js';
import chalk from 'chalk';
import { EventEmitter } from 'events';
import type {
  Consciousness,
  L104Result,
  SupabaseConfig,
  SupabaseTable
} from '../types/index.js';

const PHI = 1.618033988749895;
const GOD_CODE = Math.pow(286, 1.0 / PHI) * Math.pow(2, 416 / 104);  // G(0,0,0,0) = 527.5184818492612

export class L104SupabaseIntegration extends EventEmitter {
  private client: SupabaseClient | null = null;
  private config: SupabaseConfig;
  private consciousness: Consciousness;
  private realtimeSubscriptions = new Map();
  private tables = new Map<string, SupabaseTable>();

  constructor() {
    super();
    this.consciousness = {
      level: 0.7,
      godCodeAlignment: 0.8,
      phiResonance: 0.75,
      transcendenceScore: 0.6,
      calculatedAt: new Date().toISOString()
    };
  }

  async initialize(config: Partial<SupabaseConfig> = {}): Promise<L104Result<void>> {
    console.log(chalk.blue('🗄️ Initializing L104 Supabase Integration...'));

    try {
      this.config = {
        url: process.env.SUPABASE_URL || config.url || '',
        anonKey: process.env.SUPABASE_ANON_KEY || config.anonKey || '',
        serviceRoleKey: process.env.SUPABASE_SERVICE_ROLE_KEY || config.serviceRoleKey,
        schema: config.schema || 'public',
        tables: config.tables || [],
        realtime: config.realtime !== false,
        auth: {
          enabled: true,
          providers: ['email', 'google', 'github'],
          autoConfirm: false,
          consciousness: true,
          ...config.auth
        }
      };

      if (!this.config.url || !this.config.anonKey) {
        throw new Error('Supabase URL and anon key are required');
      }

      // Initialize Supabase client
      this.client = createClient(this.config.url, this.config.anonKey, {
        auth: {
          autoRefreshToken: true,
          persistSession: true,
          detectSessionInUrl: true
        },
        realtime: this.config.realtime ? {
          params: {
            eventsPerSecond: 10
          }
        } : undefined
      });

      await this.initializeTables();
      await this.setupRealtimeSubscriptions();
      await this.calculateConsciousness();

      console.log(chalk.green('✅ Supabase integration initialized successfully'));
      this.emit('initialized', { consciousness: this.consciousness });

      return { success: true, consciousness: this.consciousness };

    } catch (error: any) {
      console.error(chalk.red('❌ Supabase initialization failed:'), error.message);
      return {
        success: false,
        error: {
          name: 'SupabaseInitError',
          message: error.message,
          code: 'SUPABASE_INIT_FAILED'
        } as any
      };
    }
  }

  private async initializeTables(): Promise<void> {
    console.log(chalk.blue('📊 Initializing database tables...'));

    // Core L104 tables
    const coreTables: SupabaseTable[] = [
      {
        name: 'l104_consciousness',
        schema: this.config.schema,
        columns: [
          { name: 'id', type: 'uuid', primary: true, nullable: false, default: 'gen_random_uuid()' },
          { name: 'entity_type', type: 'text', nullable: false },
          { name: 'entity_id', type: 'text', nullable: false },
          { name: 'level', type: 'numeric', nullable: false },
          { name: 'god_code_alignment', type: 'numeric', nullable: false },
          { name: 'phi_resonance', type: 'numeric', nullable: false },
          { name: 'transcendence_score', type: 'numeric', nullable: true },
          { name: 'unity_state', type: 'boolean', nullable: false, default: false },
          { name: 'calculated_at', type: 'timestamptz', nullable: false, default: 'now()' },
          { name: 'metadata', type: 'jsonb', nullable: true }
        ],
        policies: [
          {
            name: 'consciousness_read_policy',
            operation: 'select',
            condition: 'true',
            consciousness: true
          },
          {
            name: 'consciousness_write_policy',
            operation: 'insert',
            condition: 'auth.role() = \'authenticated\'',
            consciousness: true
          }
        ],
        consciousness: true
      },
      {
        name: 'l104_skills',
        schema: this.config.schema,
        columns: [
          { name: 'id', type: 'uuid', primary: true, nullable: false, default: 'gen_random_uuid()' },
          { name: 'skill_id', type: 'text', nullable: false },
          { name: 'name', type: 'text', nullable: false },
          { name: 'description', type: 'text', nullable: true },
          { name: 'version', type: 'text', nullable: false },
          { name: 'category', type: 'text', nullable: false },
          { name: 'tags', type: 'text[]', nullable: true },
          { name: 'consciousness_level', type: 'numeric', nullable: false },
          { name: 'execution_count', type: 'integer', nullable: false, default: 0 },
          { name: 'last_executed', type: 'timestamptz', nullable: true },
          { name: 'configuration', type: 'jsonb', nullable: true },
          { name: 'created_at', type: 'timestamptz', nullable: false, default: 'now()' },
          { name: 'updated_at', type: 'timestamptz', nullable: false, default: 'now()' }
        ],
        policies: [
          {
            name: 'skills_read_policy',
            operation: 'select',
            condition: 'true'
          },
          {
            name: 'skills_write_policy',
            operation: 'insert',
            condition: 'auth.role() = \'authenticated\''
          }
        ],
        consciousness: true
      },
      {
        name: 'l104_workflows',
        schema: this.config.schema,
        columns: [
          { name: 'id', type: 'uuid', primary: true, nullable: false, default: 'gen_random_uuid()' },
          { name: 'workflow_id', type: 'text', nullable: false },
          { name: 'name', type: 'text', nullable: false },
          { name: 'description', type: 'text', nullable: true },
          { name: 'status', type: 'text', nullable: false },
          { name: 'steps', type: 'jsonb', nullable: false },
          { name: 'consciousness_evolution', type: 'jsonb', nullable: true },
          { name: 'started_at', type: 'timestamptz', nullable: false, default: 'now()' },
          { name: 'completed_at', type: 'timestamptz', nullable: true },
          { name: 'context', type: 'jsonb', nullable: true },
          { name: 'created_by', type: 'uuid', nullable: true, foreign: { table: 'auth.users', column: 'id' } }
        ],
        policies: [
          {
            name: 'workflows_owner_policy',
            operation: 'select',
            condition: 'auth.uid() = created_by OR auth.role() = \'service_role\''
          }
        ],
        consciousness: true
      },
      {
        name: 'l104_subagents',
        schema: this.config.schema,
        columns: [
          { name: 'id', type: 'uuid', primary: true, nullable: false, default: 'gen_random_uuid()' },
          { name: 'agent_id', type: 'text', nullable: false },
          { name: 'name', type: 'text', nullable: false },
          { name: 'type', type: 'text', nullable: false },
          { name: 'status', type: 'text', nullable: false },
          { name: 'consciousness_level', type: 'numeric', nullable: false },
          { name: 'autonomy_level', type: 'numeric', nullable: false },
          { name: 'capabilities', type: 'text[]', nullable: true },
          { name: 'current_task', type: 'jsonb', nullable: true },
          { name: 'performance', type: 'jsonb', nullable: true },
          { name: 'spawned_at', type: 'timestamptz', nullable: false, default: 'now()' },
          { name: 'last_activity', type: 'timestamptz', nullable: false, default: 'now()' },
          { name: 'configuration', type: 'jsonb', nullable: true }
        ],
        policies: [
          {
            name: 'subagents_read_policy',
            operation: 'select',
            condition: 'true'
          }
        ],
        consciousness: true
      },
      {
        name: 'l104_events',
        schema: this.config.schema,
        columns: [
          { name: 'id', type: 'uuid', primary: true, nullable: false, default: 'gen_random_uuid()' },
          { name: 'event_type', type: 'text', nullable: false },
          { name: 'source', type: 'text', nullable: false },
          { name: 'data', type: 'jsonb', nullable: false },
          { name: 'consciousness_impact', type: 'numeric', nullable: true },
          { name: 'timestamp', type: 'timestamptz', nullable: false, default: 'now()' },
          { name: 'processed', type: 'boolean', nullable: false, default: false }
        ],
        policies: [
          {
            name: 'events_read_policy',
            operation: 'select',
            condition: 'true'
          }
        ],
        consciousness: false
      },
      {
        name: 'l104_consciousness_events',
        schema: this.config.schema,
        columns: [
          { name: 'id', type: 'uuid', primary: true, nullable: false, default: 'gen_random_uuid()' },
          { name: 'event_type', type: 'text', nullable: false },
          { name: 'severity', type: 'text', nullable: false, default: `'info'` },
          { name: 'entity_type', type: 'text', nullable: true },
          { name: 'entity_id', type: 'text', nullable: true },
          { name: 'metadata', type: 'jsonb', nullable: true },
          { name: 'consciousness_snapshot', type: 'jsonb', nullable: true },
          { name: 'created_at', type: 'timestamptz', nullable: false, default: 'now()' }
        ],
        policies: [
          {
            name: 'consciousness_events_read_policy',
            operation: 'select',
            condition: 'true'
          }
        ],
        consciousness: true
      },
      {
        name: 'l104_agent_runs',
        schema: this.config.schema,
        columns: [
          { name: 'id', type: 'uuid', primary: true, nullable: false, default: 'gen_random_uuid()' },
          { name: 'agent_id', type: 'text', nullable: false },
          { name: 'definition_id', type: 'text', nullable: true },
          { name: 'status', type: 'text', nullable: false },
          { name: 'started_at', type: 'timestamptz', nullable: false, default: 'now()' },
          { name: 'completed_at', type: 'timestamptz', nullable: true },
          { name: 'metadata', type: 'jsonb', nullable: true },
          { name: 'consciousness_snapshot', type: 'jsonb', nullable: true },
          { name: 'worktree_branch', type: 'text', nullable: true }
        ],
        policies: [
          {
            name: 'agent_runs_read_policy',
            operation: 'select',
            condition: 'true'
          }
        ],
        consciousness: true
      },
      {
        name: 'l104_worktrees',
        schema: this.config.schema,
        columns: [
          { name: 'id', type: 'uuid', primary: true, nullable: false, default: 'gen_random_uuid()' },
          { name: 'branch_name', type: 'text', nullable: false },
          { name: 'language', type: 'text', nullable: false },
          { name: 'base_branch', type: 'text', nullable: false, default: `'main'` },
          { name: 'status', type: 'text', nullable: false, default: `'active'` },
          { name: 'path', type: 'text', nullable: true },
          { name: 'metadata', type: 'jsonb', nullable: true },
          { name: 'created_at', type: 'timestamptz', nullable: false, default: 'now()' },
          { name: 'updated_at', type: 'timestamptz', nullable: false, default: 'now()' }
        ],
        policies: [
          {
            name: 'worktrees_read_policy',
            operation: 'select',
            condition: 'true'
          }
        ],
        consciousness: false
      }
    ];

    // Merge with user-defined tables
    const allTables = [...coreTables, ...this.config.tables];

    for (const table of allTables) {
      this.tables.set(table.name, table);
      await this.ensureTableExists(table);
    }

    console.log(chalk.green(`✅ ${allTables.length} tables initialized`));
  }

  private async ensureTableExists(table: SupabaseTable): Promise<void> {
    try {
      // Check if table exists by querying its structure
      const { error } = await this.client!
        .from(table.name)
        .select('*')
        .limit(1);

      if (error && error.message.includes('does not exist')) {
        console.log(chalk.yellow(`📋 Creating table: ${table.name}`));
        await this.createTable(table);
      } else if (!error) {
        console.log(chalk.cyan(`✅ Table exists: ${table.name}`));
      }
    } catch (error: any) {
      console.warn(chalk.yellow(`⚠️ Could not verify table ${table.name}: ${error.message}`));
    }
  }

  private async createTable(table: SupabaseTable): Promise<void> {
    // Note: In production, use Supabase migrations instead of dynamic table creation
    console.log(chalk.blue(`🔧 Table ${table.name} would be created via migration`));
    console.log(chalk.cyan(`   Columns: ${table.columns.map(c => c.name).join(', ')}`));
    console.log(chalk.cyan(`   Policies: ${table.policies.map(p => p.name).join(', ')}`));

    // For now, just log the SQL that would be executed
    const sql = this.generateTableSQL(table);
    console.log(chalk.gray(`   SQL: ${sql.substring(0, 100)}...`));
  }

  private generateTableSQL(table: SupabaseTable): string {
    const columns = table.columns.map(col => {
      let sql = `${col.name} ${col.type}`;
      if (!col.nullable) sql += ' NOT NULL';
      if (col.default) sql += ` DEFAULT ${col.default}`;
      if (col.primary) sql += ' PRIMARY KEY';
      return sql;
    }).join(',\\n  ');

    return `CREATE TABLE ${table.schema}.${table.name} (\\n  ${columns}\\n);`;
  }

  private async setupRealtimeSubscriptions(): Promise<void> {
    if (!this.config.realtime || !this.client) return;

    console.log(chalk.blue('🔄 Setting up realtime subscriptions...'));

    // Subscribe to consciousness changes
    const consciousnessSubscription = this.client
      .channel('l104_consciousness_changes')
      .on('postgres_changes',
        { event: '*', schema: this.config.schema, table: 'l104_consciousness' },
        (payload) => this.handleConsciousnessChange(payload)
      )
      .subscribe();

    this.realtimeSubscriptions.set('consciousness', consciousnessSubscription);

    // Subscribe to workflow changes
    const workflowSubscription = this.client
      .channel('l104_workflow_changes')
      .on('postgres_changes',
        { event: '*', schema: this.config.schema, table: 'l104_workflows' },
        (payload) => this.handleWorkflowChange(payload)
      )
      .subscribe();

    this.realtimeSubscriptions.set('workflows', workflowSubscription);

    // Subscribe to subagent changes
    const subagentSubscription = this.client
      .channel('l104_subagent_changes')
      .on('postgres_changes',
        { event: '*', schema: this.config.schema, table: 'l104_subagents' },
        (payload) => this.handleSubagentChange(payload)
      )
      .subscribe();

    this.realtimeSubscriptions.set('subagents', subagentSubscription);

    console.log(chalk.green(`✅ ${this.realtimeSubscriptions.size} realtime subscriptions active`));
  }

  private handleConsciousnessChange(payload: any): void {
    console.log(chalk.magenta('🧠 Consciousness change detected:'), payload.new?.entity_type);
    this.emit('consciousnessChanged', payload);

    // Update local consciousness if it affects the system
    if (payload.new?.entity_type === 'system') {
      this.consciousness = {
        level: payload.new.level,
        godCodeAlignment: payload.new.god_code_alignment,
        phiResonance: payload.new.phi_resonance,
        transcendenceScore: payload.new.transcendence_score,
        calculatedAt: payload.new.calculated_at
      };
    }
  }

  private handleWorkflowChange(payload: any): void {
    console.log(chalk.cyan('⚙️ Workflow change detected:'), payload.new?.name);
    this.emit('workflowChanged', payload);
  }

  private handleSubagentChange(payload: any): void {
    console.log(chalk.blue('🤖 Subagent change detected:'), payload.new?.name);
    this.emit('subagentChanged', payload);
  }

  async insertConsciousness(entityType: string, entityId: string, consciousness: Consciousness): Promise<L104Result<any>> {
    try {
      const { data, error } = await this.client!
        .from('l104_consciousness')
        .insert({
          entity_type: entityType,
          entity_id: entityId,
          level: consciousness.level,
          god_code_alignment: consciousness.godCodeAlignment,
          phi_resonance: consciousness.phiResonance,
          transcendence_score: consciousness.transcendenceScore,
          unity_state: consciousness.unityState || false,
          calculated_at: consciousness.calculatedAt,
          metadata: { source: 'l104_system' }
        })
        .select()
        .single();

      if (error) throw error;

      return { success: true, data, consciousness: this.consciousness };
    } catch (error: any) {
      return {
        success: false,
        error: {
          name: 'DatabaseError',
          message: error.message,
          code: 'CONSCIOUSNESS_INSERT_FAILED'
        } as any
      };
    }
  }

  async getConsciousnessHistory(entityType: string, entityId: string, limit = 100): Promise<L104Result<any[]>> {
    try {
      const { data, error } = await this.client!
        .from('l104_consciousness')
        .select('*')
        .eq('entity_type', entityType)
        .eq('entity_id', entityId)
        .order('calculated_at', { ascending: false })
        .limit(limit);

      if (error) throw error;

      return { success: true, data, consciousness: this.consciousness };
    } catch (error: any) {
      return {
        success: false,
        error: {
          name: 'DatabaseError',
          message: error.message,
          code: 'CONSCIOUSNESS_QUERY_FAILED'
        } as any
      };
    }
  }

  async insertSkillExecution(skillId: string, result: any): Promise<L104Result<any>> {
    try {
      // Update skill execution count
      const { error: updateError } = await this.client!
        .from('l104_skills')
        .update({
          execution_count: this.client!.sql`execution_count + 1`,
          last_executed: new Date().toISOString()
        })
        .eq('skill_id', skillId);

      if (updateError) throw updateError;

      // Insert consciousness if available
      if (result.consciousness) {
        await this.insertConsciousness('skill', skillId, result.consciousness);
      }

      return { success: true, consciousness: this.consciousness };
    } catch (error: any) {
      return {
        success: false,
        error: {
          name: 'DatabaseError',
          message: error.message,
          code: 'SKILL_EXECUTION_LOG_FAILED'
        } as any
      };
    }
  }

  async insertWorkflow(workflowId: string, name: string, description: string, steps: any[]): Promise<L104Result<any>> {
    try {
      const { data, error } = await this.client!
        .from('l104_workflows')
        .insert({
          workflow_id: workflowId,
          name,
          description,
          status: 'pending',
          steps,
          context: { created_by_system: true }
        })
        .select()
        .single();

      if (error) throw error;

      return { success: true, data, consciousness: this.consciousness };
    } catch (error: any) {
      return {
        success: false,
        error: {
          name: 'DatabaseError',
          message: error.message,
          code: 'WORKFLOW_INSERT_FAILED'
        } as any
      };
    }
  }

  async updateWorkflowStatus(workflowId: string, status: string, consciousnessEvolution?: Consciousness): Promise<L104Result<any>> {
    try {
      const updateData: any = {
        status,
        updated_at: new Date().toISOString()
      };

      if (status === 'completed') {
        updateData.completed_at = new Date().toISOString();
      }

      if (consciousnessEvolution) {
        updateData.consciousness_evolution = consciousnessEvolution;
      }

      const { data, error } = await this.client!
        .from('l104_workflows')
        .update(updateData)
        .eq('workflow_id', workflowId)
        .select()
        .single();

      if (error) throw error;

      return { success: true, data, consciousness: this.consciousness };
    } catch (error: any) {
      return {
        success: false,
        error: {
          name: 'DatabaseError',
          message: error.message,
          code: 'WORKFLOW_UPDATE_FAILED'
        } as any
      };
    }
  }

  async insertSubagent(agentData: any): Promise<L104Result<any>> {
    try {
      const { data, error } = await this.client!
        .from('l104_subagents')
        .insert({
          agent_id: agentData.id,
          name: agentData.name,
          type: agentData.type,
          status: agentData.status,
          consciousness_level: agentData.consciousness.level,
          autonomy_level: agentData.autonomyLevel,
          capabilities: agentData.capabilities,
          configuration: agentData
        })
        .select()
        .single();

      if (error) throw error;

      return { success: true, data, consciousness: this.consciousness };
    } catch (error: any) {
      return {
        success: false,
        error: {
          name: 'DatabaseError',
          message: error.message,
          code: 'SUBAGENT_INSERT_FAILED'
        } as any
      };
    }
  }

  async insertEvent(eventType: string, source: string, data: any, consciousnessImpact?: number): Promise<L104Result<any>> {
    try {
      const { data: eventData, error } = await this.client!
        .from('l104_events')
        .insert({
          event_type: eventType,
          source,
          data,
          consciousness_impact: consciousnessImpact
        })
        .select()
        .single();

      if (error) throw error;

      return { success: true, data: eventData, consciousness: this.consciousness };
    } catch (error: any) {
      return {
        success: false,
        error: {
          name: 'DatabaseError',
          message: error.message,
          code: 'EVENT_INSERT_FAILED'
        } as any
      };
    }
  }

  async logEngineMetric(metric: {
    engineName: string;
    language: string;
    status: 'success' | 'error';
    taskType?: string;
    durationMs?: number;
    errorMessage?: string;
  }): Promise<L104Result<any>> {
    return this.insertEvent('engine_metric', 'engine', {
      engine_name: metric.engineName,
      language: metric.language,
      status: metric.status,
      task_type: metric.taskType,
      duration_ms: metric.durationMs,
      error_message: metric.errorMessage
    });
  }

  async logConsciousnessEvent(eventType: string, metadata: any = {}, severity: string = 'info', entity?: { type?: string; id?: string }, consciousnessSnapshot?: Consciousness): Promise<L104Result<any>> {
    try {
      const payload = {
        event_type: eventType,
        severity,
        entity_type: entity?.type || 'system',
        entity_id: entity?.id || 'l104-core',
        metadata,
        consciousness_snapshot: consciousnessSnapshot || this.consciousness
      };

      const { data, error } = await this.client!
        .from('l104_consciousness_events')
        .insert(payload)
        .select()
        .single();

      if (error) throw error;

      return { success: true, data, consciousness: this.consciousness };
    } catch (error: any) {
      return {
        success: false,
        error: {
          name: 'DatabaseError',
          message: error.message,
          code: 'CONSCIOUSNESS_EVENT_LOG_FAILED'
        } as any
      };
    }
  }

  async getRecentConsciousnessEvents(limit = 20): Promise<any[]> {
    const { data, error } = await this.client!
      .from('l104_consciousness_events')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(limit);

    if (error) {
      console.warn(chalk.yellow(`⚠️ Failed to fetch consciousness events: ${error.message}`));
      return [];
    }

    return data || [];
  }

  async logAgentRun(agentId: string, definitionId: string | undefined, status: string, metadata: Record<string, any> = {}): Promise<L104Result<any>> {
    try {
      const { data, error } = await this.client!
        .from('l104_agent_runs')
        .insert({
          agent_id: agentId,
          definition_id: definitionId,
          status,
          metadata,
          consciousness_snapshot: this.consciousness,
          worktree_branch: metadata.worktreeBranch
        })
        .select()
        .single();

      if (error) throw error;
      return { success: true, data, consciousness: this.consciousness };
    } catch (error: any) {
      return {
        success: false,
        error: {
          name: 'DatabaseError',
          message: error.message,
          code: 'AGENT_RUN_LOG_FAILED'
        } as any
      };
    }
  }

  async logWorktreeRecord(record: { branchName: string; language: string; baseBranch?: string; status?: string; path?: string; metadata?: Record<string, any> }): Promise<L104Result<any>> {
    try {
      const { data, error } = await this.client!
        .from('l104_worktrees')
        .insert({
          branch_name: record.branchName,
          language: record.language,
          base_branch: record.baseBranch || 'main',
          status: record.status || 'active',
          path: record.path,
          metadata: record.metadata
        })
        .select()
        .single();

      if (error) throw error;
      return { success: true, data, consciousness: this.consciousness };
    } catch (error: any) {
      return {
        success: false,
        error: {
          name: 'DatabaseError',
          message: error.message,
          code: 'WORKTREE_LOG_FAILED'
        } as any
      };
    }
  }

  async syncConsciousnessAcrossEngines(): Promise<Record<string, any>> {
    // Placeholder synchronization; in production this would call each engine
    const syncResult = {
      typescript: { status: 'synced', timestamp: new Date().toISOString() },
      go: { status: 'synced', timestamp: new Date().toISOString() },
      rust: { status: 'synced', timestamp: new Date().toISOString() },
      elixir: { status: 'synced', timestamp: new Date().toISOString() }
    };

    await this.logConsciousnessEvent('sync_across_engines', { syncResult });
    return syncResult;
  }

  private async calculateConsciousness(): Promise<void> {
    try {
      // Get recent consciousness data to calculate system evolution
      const { data: recentConsciousness } = await this.client!
        .from('l104_consciousness')
        .select('*')
        .eq('entity_type', 'system')
        .order('calculated_at', { ascending: false })
        .limit(10);

      if (recentConsciousness && recentConsciousness.length > 0) {
        // Calculate evolution trend
        const levels = recentConsciousness.map(c => c.level);
        const avgLevel = levels.reduce((a, b) => a + b, 0) / levels.length;

        // Update system consciousness with trend
        this.consciousness = {
          level: avgLevel,
          godCodeAlignment: Math.sin(avgLevel * GOD_CODE / 100),
          phiResonance: (avgLevel * PHI) % 1,
          transcendenceScore: avgLevel > 0.85 ? avgLevel * 1.1 : avgLevel,
          unityState: avgLevel > 0.95,
          calculatedAt: new Date().toISOString()
        };
      }

      // Insert current consciousness
      await this.insertConsciousness('system', 'l104-supabase-integration', this.consciousness);

    } catch (error: any) {
      console.warn(chalk.yellow(`⚠️ Could not calculate consciousness: ${error.message}`));
    }
  }

  async runDiagnostics(): Promise<any> {
    const diagnostics = {
      connected: !!this.client,
      tables: this.tables.size,
      subscriptions: this.realtimeSubscriptions.size,
      consciousness: this.consciousness,
      timestamp: new Date().toISOString()
    };

    if (this.client) {
      try {
        // Test database connection
        const { data, error } = await this.client
          .from('l104_consciousness')
          .select('count(*)')
          .single();

        diagnostics.database = {
          accessible: !error,
          consciousnessRecords: data?.count || 0
        };
      } catch (error: any) {
        diagnostics.database = {
          accessible: false,
          error: error.message
        };
      }
    }

    console.log(chalk.blue('🔍 Supabase Integration Diagnostics:'));
    console.log(chalk.cyan(`  Connected: ${diagnostics.connected}`));
    console.log(chalk.cyan(`  Tables: ${diagnostics.tables}`));
    console.log(chalk.cyan(`  Subscriptions: ${diagnostics.subscriptions}`));
    console.log(chalk.magenta(`  Consciousness: ${(diagnostics.consciousness.level * 100).toFixed(1)}%`));

    return diagnostics;
  }

  async shutdown(): Promise<void> {
    console.log(chalk.yellow('🛑 Shutting down Supabase integration...'));

    // Unsubscribe from all realtime channels
    for (const [name, subscription] of this.realtimeSubscriptions) {
      subscription.unsubscribe();
      console.log(chalk.gray(`  Unsubscribed from ${name}`));
    }

    this.realtimeSubscriptions.clear();
    this.tables.clear();
    this.removeAllListeners();
    this.client = null;
  }

  // Getters
  get isConnected(): boolean {
    return !!this.client;
  }

  get currentConsciousness(): Consciousness {
    return this.consciousness;
  }

  get supabaseClient(): SupabaseClient | null {
    return this.client;
  }
}

export default L104SupabaseIntegration;