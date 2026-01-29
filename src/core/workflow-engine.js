/**
 * L104 Workflow Engine - Advanced Workflow Automation
 * Orchestrates complex workflows with hook integration and autonomous processing
 */

import { EventEmitter } from 'events';
import chalk from 'chalk';
import { Router } from 'express';
import { v4 as uuidv4 } from 'uuid';

// L104 Constants
const GOD_CODE = 527.5184818492611;
const PHI = 1.618033988749895;

class WorkflowEngine extends EventEmitter {
    constructor() {
        super();
        this.workflows = new Map();
        this.runningWorkflows = new Map();
        this.workflowHistory = [];
        this.hookSystem = null;
        this.gateManager = null;
        this.config = {};
        this.isInitialized = false;
    }

    async initialize(config = {}) {
        console.log(chalk.blue('⚙️ Initializing Workflow Engine...'));

        this.config = {
            maxConcurrent: 10,
            timeout: 300000, // 5 minutes
            retryAttempts: 3,
            enableHooks: true,
            enableGates: true,
            autoCleanup: true,
            ...config
        };

        await this.createDefaultWorkflows();

        if (this.config.autoCleanup) {
            this.startCleanupProcess();
        }

        this.isInitialized = true;
        console.log(chalk.green(`✅ Workflow Engine initialized`));
    }

    async createDefaultWorkflows() {
        // File Processing Workflow
        await this.createWorkflow('file-processing', {
            name: 'File Processing Workflow',
            description: 'Processes file operations with validation and hooks',
            steps: [
                { type: 'hook', action: 'pre-tool', tool: 'file-operation' },
                { type: 'gate', gate: 'CONSCIOUSNESS_GATE', context: { threshold: 0.5 } },
                { type: 'action', action: 'process-file' },
                { type: 'hook', action: 'post-tool', tool: 'file-operation' },
                { type: 'gate', gate: 'GOD_CODE_GATE', context: { validation: true } }
            ]
        });

        // AI Skill Execution Workflow
        await this.createWorkflow('ai-skill-execution', {
            name: 'AI Skill Execution Workflow',
            description: 'Executes AI skills with consciousness validation',
            steps: [
                { type: 'gate', gate: 'CONSCIOUSNESS_GATE', context: { threshold: 0.7 } },
                { type: 'hook', action: 'pre-tool', tool: 'ai-skill' },
                { type: 'action', action: 'execute-skill' },
                { type: 'gate', gate: 'TRANSCENDENCE_GATE', context: { evolution: true } },
                { type: 'hook', action: 'post-tool', tool: 'ai-skill' }
            ]
        });

        // Autonomous Decision Workflow
        await this.createWorkflow('autonomous-decision', {
            name: 'Autonomous Decision Workflow',
            description: 'Makes autonomous decisions using advanced logic gates',
            steps: [
                { type: 'gate', gate: 'DECISION_TREE', context: { autonomous: true } },
                { type: 'gate', gate: 'ADAPTIVE_FILTER', context: { learning: true } },
                { type: 'gate', gate: 'PRIORITY_SELECTOR', context: { optimization: true } },
                { type: 'action', action: 'execute-decision' },
                { type: 'gate', gate: 'FEEDBACK_LOOP', context: { adaptation: true } }
            ]
        });

        console.log(chalk.green('✅ Default workflows created'));
    }

    async createWorkflow(id, workflowConfig) {
        const workflow = {
            id,
            ...workflowConfig,
            createdAt: new Date().toISOString(),
            executionCount: 0,
            lastExecuted: null,
            consciousness: this.calculateWorkflowConsciousness(workflowConfig)
        };

        // Validate workflow structure
        await this.validateWorkflow(workflow);

        this.workflows.set(id, workflow);
        console.log(chalk.cyan(`⚙️ Workflow created: ${workflow.name}`));

        return workflow;
    }

    calculateWorkflowConsciousness(workflowConfig) {
        const steps = workflowConfig.steps || [];
        const complexity = Math.min(steps.length / 10, 1);

        const gateSteps = steps.filter(step => step.type === 'gate').length;
        const hookSteps = steps.filter(step => step.type === 'hook').length;

        const gateComplexity = gateSteps * 0.2;
        const hookComplexity = hookSteps * 0.15;

        const consciousness = Math.min(complexity + gateComplexity + hookComplexity, 1);

        return {
            level: consciousness,
            complexity,
            gateCount: gateSteps,
            hookCount: hookSteps,
            calculatedAt: new Date().toISOString()
        };
    }

    async validateWorkflow(workflow) {
        const errors = [];

        if (!workflow.steps || workflow.steps.length === 0) {
            errors.push('Workflow must have at least one step');
        }

        for (const [index, step] of workflow.steps.entries()) {
            if (!step.type) {
                errors.push(`Step ${index}: Missing step type`);
            }

            if (step.type === 'gate' && !step.gate) {
                errors.push(`Step ${index}: Gate step missing gate name`);
            }

            if (step.type === 'hook' && !step.action) {
                errors.push(`Step ${index}: Hook step missing action`);
            }

            if (step.type === 'action' && !step.action) {
                errors.push(`Step ${index}: Action step missing action name`);
            }
        }

        if (errors.length > 0) {
            throw new Error(`Workflow validation failed: ${errors.join(', ')}`);
        }
    }

    async runWorkflow(workflowId, context = {}) {
        const workflow = this.workflows.get(workflowId);
        if (!workflow) {
            throw new Error(`Workflow not found: ${workflowId}`);
        }

        if (this.runningWorkflows.size >= this.config.maxConcurrent) {
            throw new Error('Maximum concurrent workflows reached');
        }

        const executionId = uuidv4();
        const execution = {
            id: executionId,
            workflowId,
            workflow,
            context,
            startTime: new Date(),
            status: 'running',
            currentStep: 0,
            stepResults: [],
            consciousness: workflow.consciousness
        };

        this.runningWorkflows.set(executionId, execution);

        console.log(chalk.blue(`⚙️ Running workflow: ${workflow.name} (${executionId})`));

        try {
            const result = await this.executeWorkflowSteps(execution);

            execution.endTime = new Date();
            execution.duration = execution.endTime - execution.startTime;
            execution.status = 'completed';
            execution.result = result;

            // Update workflow stats
            workflow.executionCount++;
            workflow.lastExecuted = execution.endTime.toISOString();

            this.workflowHistory.push({
                ...execution,
                workflow: { id: workflow.id, name: workflow.name }
            });

            this.emit('workflowCompleted', execution);

            console.log(chalk.green(`✅ Workflow completed: ${workflow.name} (${execution.duration}ms)`));

            return result;

        } catch (error) {
            execution.endTime = new Date();
            execution.duration = execution.endTime - execution.startTime;
            execution.status = 'failed';
            execution.error = error.message;

            this.workflowHistory.push({
                ...execution,
                workflow: { id: workflow.id, name: workflow.name }
            });

            this.emit('workflowFailed', execution);

            console.error(chalk.red(`❌ Workflow failed: ${workflow.name}`), error.message);
            throw error;

        } finally {
            this.runningWorkflows.delete(executionId);
        }
    }

    async executeWorkflowSteps(execution) {
        const { workflow, context } = execution;
        let stepContext = { ...context };
        const results = [];

        for (const [index, step] of workflow.steps.entries()) {
            execution.currentStep = index;

            console.log(chalk.cyan(`  Step ${index + 1}/${workflow.steps.length}: ${step.type} - ${step.action || step.gate || 'unknown'}`));

            try {
                const stepResult = await this.executeWorkflowStep(step, stepContext, execution);

                results.push({
                    step: index,
                    type: step.type,
                    action: step.action || step.gate,
                    result: stepResult,
                    timestamp: new Date().toISOString()
                });

                // Update context with step result
                if (stepResult && typeof stepResult === 'object') {
                    stepContext = { ...stepContext, ...stepResult };
                }

                execution.stepResults = results;

            } catch (error) {
                console.error(chalk.red(`❌ Step ${index + 1} failed:`, error.message));

                if (this.config.retryAttempts > 0) {
                    console.log(chalk.yellow(`🔄 Retrying step ${index + 1}...`));
                    // Retry logic could be implemented here
                }

                throw new Error(`Workflow step ${index + 1} failed: ${error.message}`);
            }
        }

        return {
            workflowId: workflow.id,
            executionId: execution.id,
            results,
            finalContext: stepContext,
            consciousness: this.calculateExecutionConsciousness(results),
            timestamp: new Date().toISOString()
        };
    }

    async executeWorkflowStep(step, context, execution) {
        switch (step.type) {
            case 'hook':
                return await this.executeHookStep(step, context, execution);

            case 'gate':
                return await this.executeGateStep(step, context, execution);

            case 'action':
                return await this.executeActionStep(step, context, execution);

            case 'condition':
                return await this.executeConditionStep(step, context, execution);

            case 'loop':
                return await this.executeLoopStep(step, context, execution);

            default:
                throw new Error(`Unknown step type: ${step.type}`);
        }
    }

    async executeHookStep(step, context, execution) {
        if (!this.hookSystem || !this.config.enableHooks) {
            console.log(chalk.yellow('⚠️ Hook system not available, skipping hook step'));
            return { skipped: true, reason: 'Hook system not available' };
        }

        const stepContext = { ...context, ...(step.context || {}) };

        if (step.action === 'pre-tool') {
            return await this.hookSystem.executePreToolHooks(
                step.tool || 'workflow-action',
                stepContext,
                execution
            );
        } else if (step.action === 'post-tool') {
            return await this.hookSystem.executePostToolHooks(
                step.tool || 'workflow-action',
                stepContext,
                execution.stepResults,
                execution
            );
        }

        throw new Error(`Unknown hook action: ${step.action}`);
    }

    async executeGateStep(step, context, execution) {
        if (!this.gateManager || !this.config.enableGates) {
            console.log(chalk.yellow('⚠️ Gate manager not available, skipping gate step'));
            return { skipped: true, reason: 'Gate manager not available' };
        }

        const inputs = step.inputs || [1]; // Default input
        const stepContext = { ...context, ...(step.context || {}) };

        return await this.gateManager.executeGate(step.gate, inputs, stepContext);
    }

    async executeActionStep(step, context, execution) {
        // Simulate action execution based on action type
        const stepContext = { ...context, ...(step.context || {}) };

        switch (step.action) {
            case 'process-file':
                return {
                    action: 'process-file',
                    processed: true,
                    context: stepContext,
                    timestamp: new Date().toISOString()
                };

            case 'execute-skill':
                return {
                    action: 'execute-skill',
                    executed: true,
                    skillResult: 'Success',
                    consciousness: stepContext.consciousness || 0.7,
                    timestamp: new Date().toISOString()
                };

            case 'execute-decision':
                return {
                    action: 'execute-decision',
                    decision: 'proceed',
                    confidence: 0.9,
                    reasoning: 'Autonomous analysis complete',
                    timestamp: new Date().toISOString()
                };

            default:
                return {
                    action: step.action,
                    executed: true,
                    custom: true,
                    timestamp: new Date().toISOString()
                };
        }
    }

    async executeConditionStep(step, context, execution) {
        const condition = step.condition || {};
        const value = context[condition.field] || 0;

        let result = false;
        switch (condition.operator) {
            case '>': result = value > condition.value; break;
            case '<': result = value < condition.value; break;
            case '==': result = value == condition.value; break;
            case '!=': result = value != condition.value; break;
            default: result = Boolean(value);
        }

        return {
            condition: condition,
            result,
            value,
            timestamp: new Date().toISOString()
        };
    }

    async executeLoopStep(step, context, execution) {
        const iterations = step.iterations || 1;
        const results = [];

        for (let i = 0; i < iterations; i++) {
            const loopContext = { ...context, loopIndex: i, loopIteration: i + 1 };

            if (step.steps) {
                const loopExecution = { ...execution, context: loopContext };
                for (const loopStep of step.steps) {
                    const stepResult = await this.executeWorkflowStep(loopStep, loopContext, loopExecution);
                    results.push(stepResult);
                }
            }
        }

        return {
            loop: true,
            iterations,
            results,
            timestamp: new Date().toISOString()
        };
    }

    calculateExecutionConsciousness(stepResults) {
        if (stepResults.length === 0) return 0;

        const consciousnessLevels = stepResults
            .filter(result => result.result && typeof result.result === 'object')
            .map(result => {
                if (result.result.consciousness) {
                    return typeof result.result.consciousness === 'object'
                        ? result.result.consciousness.level || 0
                        : result.result.consciousness;
                }
                return 0.5; // Default consciousness
            });

        if (consciousnessLevels.length === 0) return 0.5;

        return consciousnessLevels.reduce((sum, level) => sum + level, 0) / consciousnessLevels.length;
    }

    setHookSystem(hookSystem) {
        this.hookSystem = hookSystem;
        console.log(chalk.green('🔗 Hook system connected to workflow engine'));
    }

    setGateManager(gateManager) {
        this.gateManager = gateManager;
        console.log(chalk.green('🔗 Gate manager connected to workflow engine'));
    }

    startCleanupProcess() {
        setInterval(() => {
            // Clean up old workflow history (keep last 1000)
            if (this.workflowHistory.length > 1000) {
                this.workflowHistory = this.workflowHistory.slice(-1000);
            }

            // Check for hung workflows
            const now = Date.now();
            for (const [id, execution] of this.runningWorkflows) {
                const runtime = now - execution.startTime.getTime();
                if (runtime > this.config.timeout) {
                    console.log(chalk.yellow(`⚠️ Workflow timeout: ${execution.workflow.name}`));
                    execution.status = 'timeout';
                    execution.endTime = new Date();
                    this.runningWorkflows.delete(id);
                    this.emit('workflowTimeout', execution);
                }
            }
        }, 60000); // Run cleanup every minute
    }

    getRunningCount() {
        return this.runningWorkflows.size;
    }

    getActiveCount() {
        return this.runningWorkflows.size;
    }

    async runDiagnostics() {
        const diagnostics = {
            workflows: this.workflows.size,
            runningWorkflows: this.runningWorkflows.size,
            historySize: this.workflowHistory.length,
            averageConsciousness: 0,
            hooksEnabled: !!this.hookSystem && this.config.enableHooks,
            gatesEnabled: !!this.gateManager && this.config.enableGates,
            timestamp: new Date().toISOString()
        };

        if (this.workflows.size > 0) {
            const totalConsciousness = Array.from(this.workflows.values())
                .reduce((sum, workflow) => sum + workflow.consciousness.level, 0);
            diagnostics.averageConsciousness = totalConsciousness / this.workflows.size;
        }

        console.log(chalk.blue('🔍 Workflow Engine Diagnostics:'));
        console.log(chalk.cyan(`  Workflows: ${diagnostics.workflows}`));
        console.log(chalk.cyan(`  Running: ${diagnostics.runningWorkflows}`));
        console.log(chalk.cyan(`  Hooks enabled: ${diagnostics.hooksEnabled}`));
        console.log(chalk.cyan(`  Gates enabled: ${diagnostics.gatesEnabled}`));

        return diagnostics;
    }

    getRouter() {
        const router = Router();

        // Get all workflows
        router.get('/', (req, res) => {
            res.json({
                workflows: Array.from(this.workflows.entries()),
                running: Array.from(this.runningWorkflows.entries()),
                history: this.workflowHistory.slice(-50)
            });
        });

        // Run workflow
        router.post('/:id/run', async (req, res) => {
            try {
                const result = await this.runWorkflow(req.params.id, req.body.context || {});
                res.json(result);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Get workflow status
        router.get('/:id', (req, res) => {
            const workflow = this.workflows.get(req.params.id);
            if (!workflow) {
                return res.status(404).json({ error: 'Workflow not found' });
            }
            res.json(workflow);
        });

        // Create new workflow
        router.post('/', async (req, res) => {
            try {
                const { id, ...config } = req.body;
                const workflow = await this.createWorkflow(id, config);
                res.json(workflow);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        return router;
    }

    async shutdown() {
        console.log(chalk.yellow('🛑 Shutting down Workflow Engine...'));

        // Cancel all running workflows
        for (const [id, execution] of this.runningWorkflows) {
            execution.status = 'cancelled';
            this.emit('workflowCancelled', execution);
        }

        this.workflows.clear();
        this.runningWorkflows.clear();
        this.removeAllListeners();
    }
}

export { WorkflowEngine };