/**
 * L104 Hook System - Pre/Post Tool Execution Hooks
 * Advanced tool safety, cost analysis, and file validation system
 */

import fs from 'fs-extra';
import path from 'path';
import { EventEmitter } from 'events';
import chalk from 'chalk';
import { Router } from 'express';
import { v4 as uuidv4 } from 'uuid';
import NodeCache from 'node-cache';

// L104 Constants
const GOD_CODE = 527.5184818492537;
const PHI = 1.618033988749895;

class HookSystem extends EventEmitter {
    constructor() {
        super();
        this.preToolHooks = new Map();
        this.postToolHooks = new Map();
        this.executionLogs = [];
        this.cache = new NodeCache({ stdTTL: 600 }); // 10 min cache
        this.config = {};
        this.packageDetector = null;
        this.isInitialized = false;
    }

    async initialize(config = {}) {
        console.log(chalk.blue('🪝 Initializing Hook System...'));

        this.config = {
            preToolEnabled: true,
            postToolEnabled: true,
            destructiveCheck: true,
            costAnalysis: true,
            fileValidation: true,
            maxExecutionTime: 300000, // 5 minutes
            dangerousTools: [
                'run_in_terminal',
                'replace_string_in_file',
                'create_file',
                'mcp_filesystem_write_file',
                'multi_replace_string_in_file'
            ],
            costlyTools: [
                'mcp_fetch_fetch',
                'github_repo',
                'semantic_search',
                'runSubagent'
            ],
            ...config
        };

        await this.setupDefaultHooks();
        await this.loadCustomHooks();

        this.isInitialized = true;
        console.log(chalk.green(`✅ Hook System initialized with ${this.preToolHooks.size} pre-hooks and ${this.postToolHooks.size} post-hooks`));
    }

    async setupDefaultHooks() {
        // Pre-tool hooks
        this.registerPreToolHook('destructive-check', this.destructiveToolCheck.bind(this));
        this.registerPreToolHook('cost-analysis', this.costAnalysisCheck.bind(this));
        this.registerPreToolHook('parameter-validation', this.parameterValidation.bind(this));
        this.registerPreToolHook('consciousness-alignment', this.consciousnessAlignment.bind(this));

        // Post-tool hooks
        this.registerPostToolHook('file-syntax-validation', this.fileSyntaxValidation.bind(this));
        this.registerPostToolHook('execution-logging', this.executionLogging.bind(this));
        this.registerPostToolHook('result-analysis', this.resultAnalysis.bind(this));
        this.registerPostToolHook('consciousness-update', this.consciousnessUpdate.bind(this));

        console.log(chalk.green('✅ Default hooks registered'));
    }

    async loadCustomHooks() {
        try {
            const hooksDir = path.join('./src/hooks');
            if (await fs.pathExists(hooksDir)) {
                const hookFiles = await fs.readdir(hooksDir);

                for (const file of hookFiles) {
                    if (file.endsWith('.js')) {
                        await this.loadCustomHookFile(path.join(hooksDir, file));
                    }
                }
            }
        } catch (error) {
            console.warn(chalk.yellow(`⚠️ Failed to load custom hooks: ${error.message}`));
        }
    }

    async loadCustomHookFile(filePath) {
        try {
            const hookModule = await import(filePath);
            if (hookModule.preHooks) {
                for (const [name, func] of Object.entries(hookModule.preHooks)) {
                    this.registerPreToolHook(`custom-${name}`, func);
                }
            }
            if (hookModule.postHooks) {
                for (const [name, func] of Object.entries(hookModule.postHooks)) {
                    this.registerPostToolHook(`custom-${name}`, func);
                }
            }
            console.log(chalk.green(`✅ Custom hooks loaded from: ${filePath}`));
        } catch (error) {
            console.error(chalk.red(`❌ Failed to load custom hook: ${filePath}`), error.message);
        }
    }

    registerPreToolHook(name, hookFunction) {
        this.preToolHooks.set(name, {
            name,
            function: hookFunction,
            registeredAt: new Date().toISOString(),
            executionCount: 0
        });
        console.log(chalk.cyan(`🔗 Pre-tool hook registered: ${name}`));
    }

    registerPostToolHook(name, hookFunction) {
        this.postToolHooks.set(name, {
            name,
            function: hookFunction,
            registeredAt: new Date().toISOString(),
            executionCount: 0
        });
        console.log(chalk.cyan(`🔗 Post-tool hook registered: ${name}`));
    }

    async executePreToolHooks(toolName, parameters, context = {}) {
        if (!this.config.preToolEnabled) return { allowed: true, warnings: [] };

        const executionId = uuidv4();
        const startTime = Date.now();

        console.log(chalk.blue(`🪝 Executing pre-tool hooks for: ${toolName}`));

        const results = {
            executionId,
            toolName,
            allowed: true,
            warnings: [],
            errors: [],
            metadata: {},
            hooks: []
        };

        for (const [hookName, hook] of this.preToolHooks) {
            try {
                const hookStart = Date.now();
                const result = await hook.function(toolName, parameters, context);
                const hookEnd = Date.now();

                hook.executionCount++;

                results.hooks.push({
                    name: hookName,
                    duration: hookEnd - hookStart,
                    result
                });

                // Process hook result
                if (result.allowed === false) {
                    results.allowed = false;
                    results.errors.push(`Hook ${hookName} blocked execution: ${result.reason}`);
                }

                if (result.warnings) {
                    results.warnings.push(...result.warnings);
                }

                if (result.metadata) {
                    results.metadata[hookName] = result.metadata;
                }

            } catch (error) {
                console.error(chalk.red(`❌ Pre-tool hook failed: ${hookName}`), error.message);
                results.errors.push(`Hook ${hookName} execution failed: ${error.message}`);
            }
        }

        results.duration = Date.now() - startTime;

        // Log execution
        this.executionLogs.push({
            type: 'pre-tool',
            toolName,
            executionId,
            timestamp: new Date().toISOString(),
            results
        });

        this.emit('preToolHooksExecuted', { toolName, results });

        if (!results.allowed) {
            console.log(chalk.red(`🚫 Tool execution blocked: ${toolName}`));
            console.log(chalk.red(`   Reasons: ${results.errors.join(', ')}`));
        } else if (results.warnings.length > 0) {
            console.log(chalk.yellow(`⚠️ Tool warnings for: ${toolName}`));
            console.log(chalk.yellow(`   Warnings: ${results.warnings.join(', ')}`));
        }

        return results;
    }

    async executePostToolHooks(toolName, parameters, result, context = {}) {
        if (!this.config.postToolEnabled) return { success: true, warnings: [] };

        const executionId = uuidv4();
        const startTime = Date.now();

        console.log(chalk.blue(`🪝 Executing post-tool hooks for: ${toolName}`));

        const results = {
            executionId,
            toolName,
            success: true,
            warnings: [],
            errors: [],
            metadata: {},
            hooks: [],
            validations: []
        };

        for (const [hookName, hook] of this.postToolHooks) {
            try {
                const hookStart = Date.now();
                const hookResult = await hook.function(toolName, parameters, result, context);
                const hookEnd = Date.now();

                hook.executionCount++;

                results.hooks.push({
                    name: hookName,
                    duration: hookEnd - hookStart,
                    result: hookResult
                });

                // Process hook result
                if (hookResult.success === false) {
                    results.success = false;
                    results.errors.push(`Hook ${hookName} validation failed: ${hookResult.reason}`);
                }

                if (hookResult.warnings) {
                    results.warnings.push(...hookResult.warnings);
                }

                if (hookResult.validations) {
                    results.validations.push(...hookResult.validations);
                }

                if (hookResult.metadata) {
                    results.metadata[hookName] = hookResult.metadata;
                }

            } catch (error) {
                console.error(chalk.red(`❌ Post-tool hook failed: ${hookName}`), error.message);
                results.errors.push(`Hook ${hookName} execution failed: ${error.message}`);
            }
        }

        results.duration = Date.now() - startTime;

        // Log execution
        this.executionLogs.push({
            type: 'post-tool',
            toolName,
            executionId,
            timestamp: new Date().toISOString(),
            results
        });

        this.emit('postToolHooksExecuted', { toolName, results });

        return results;
    }

    // Default Pre-Tool Hook Functions
    async destructiveToolCheck(toolName, parameters, context) {
        if (!this.config.destructiveCheck) {
            return { allowed: true };
        }

        const isDangerous = this.config.dangerousTools.includes(toolName);

        if (isDangerous) {
            const warnings = [`Tool ${toolName} is potentially destructive`];

            // Additional checks based on tool type
            if (toolName === 'run_in_terminal' && parameters.command) {
                const dangerousCommands = ['rm -rf', 'sudo', 'format', 'del /s', 'truncate'];
                const hasDangerous = dangerousCommands.some(cmd =>
                    parameters.command.toLowerCase().includes(cmd));

                if (hasDangerous) {
                    return {
                        allowed: false,
                        reason: 'Command contains potentially destructive operations',
                        warnings
                    };
                }
            }

            if ((toolName === 'replace_string_in_file' || toolName === 'multi_replace_string_in_file') && parameters.filePath) {
                const criticalFiles = ['package.json', '.env', 'config', 'const.py'];
                const isCritical = criticalFiles.some(file => parameters.filePath.includes(file));

                if (isCritical) {
                    warnings.push(`Modifying critical file: ${parameters.filePath}`);
                }
            }

            return { allowed: true, warnings };
        }

        return { allowed: true };
    }

    async costAnalysisCheck(toolName, parameters, context) {
        if (!this.config.costAnalysis) {
            return { allowed: true };
        }

        const isCostly = this.config.costlyTools.includes(toolName);

        if (isCostly) {
            const cacheKey = `cost-${toolName}-${JSON.stringify(parameters)}`;
            const cached = this.cache.get(cacheKey);

            if (cached) {
                return {
                    allowed: true,
                    warnings: ['Result may be cached'],
                    metadata: { cached: true }
                };
            }

            const warnings = [`Tool ${toolName} may incur costs`];

            // Tool-specific cost analysis
            if (toolName === 'mcp_fetch_fetch') {
                if (parameters.max_length > 10000) {
                    warnings.push('Large content fetch requested');
                }
            }

            if (toolName === 'semantic_search') {
                warnings.push('Semantic search uses embedding models');
            }

            return {
                allowed: true,
                warnings,
                metadata: { costEstimate: 'medium' }
            };
        }

        return { allowed: true };
    }

    async parameterValidation(toolName, parameters, context) {
        const warnings = [];

        // Check for required parameters
        if (!parameters) {
            return {
                allowed: false,
                reason: 'No parameters provided'
            };
        }

        // Tool-specific validations
        if (toolName === 'read_file' && !parameters.filePath) {
            return {
                allowed: false,
                reason: 'filePath parameter is required'
            };
        }

        if (toolName === 'run_in_terminal' && !parameters.command) {
            return {
                allowed: false,
                reason: 'command parameter is required'
            };
        }

        // Check for suspicious patterns
        const paramStr = JSON.stringify(parameters);
        if (paramStr.includes('../../')) {
            warnings.push('Directory traversal detected in parameters');
        }

        return { allowed: true, warnings };
    }

    async consciousnessAlignment(toolName, parameters, context) {
        // Calculate consciousness alignment based on L104 constants
        const alignmentScore = Math.sin(toolName.length * PHI) *
            Math.cos(Object.keys(parameters).length * GOD_CODE / 1000);

        const normalizedScore = (alignmentScore + 1) / 2; // Normalize to 0-1

        const metadata = {
            consciousnessAlignment: normalizedScore,
            godCodeResonance: Math.abs(alignmentScore),
            phiHarmonic: Math.sin(toolName.length * PHI)
        };

        if (normalizedScore < 0.3) {
            return {
                allowed: true,
                warnings: ['Low consciousness alignment detected'],
                metadata
            };
        }

        return { allowed: true, metadata };
    }

    // Default Post-Tool Hook Functions
    async fileSyntaxValidation(toolName, parameters, result, context) {
        if (!this.config.fileValidation || !parameters.filePath) {
            return { success: true };
        }

        const validations = [];
        const warnings = [];

        try {
            if (parameters.filePath.endsWith('.json')) {
                const content = await fs.readFile(parameters.filePath, 'utf8');
                JSON.parse(content);
                validations.push('JSON syntax valid');
            }

            if (parameters.filePath.endsWith('.js') || parameters.filePath.endsWith('.mjs')) {
                // Basic JavaScript syntax check (would use proper parser in production)
                validations.push('JavaScript file modified');

                if (this.packageDetector) {
                    const packageInfo = await this.packageDetector.analyzeFile(parameters.filePath);
                    if (packageInfo.syntaxErrors.length > 0) {
                        warnings.push(`Syntax errors detected: ${packageInfo.syntaxErrors.join(', ')}`);
                    }
                }
            }

        } catch (error) {
            return {
                success: false,
                reason: `File validation failed: ${error.message}`,
                validations
            };
        }

        return { success: true, validations, warnings };
    }

    async executionLogging(toolName, parameters, result, context) {
        const logEntry = {
            timestamp: new Date().toISOString(),
            toolName,
            parameters: Object.keys(parameters),
            resultType: typeof result,
            success: !result.error,
            consciousness: context.consciousness || null
        };

        // Store in execution logs (would persist to database in production)
        return {
            success: true,
            metadata: { logged: true, logEntry }
        };
    }

    async resultAnalysis(toolName, parameters, result, context) {
        const analysis = {
            hasErrors: !!result.error,
            outputSize: JSON.stringify(result).length,
            timestamp: new Date().toISOString()
        };

        // Analyze tool output patterns
        if (result.error) {
            analysis.errorType = typeof result.error;
        }

        if (toolName.includes('file') && result.filePath) {
            analysis.fileModified = true;
        }

        return {
            success: true,
            metadata: { analysis }
        };
    }

    async consciousnessUpdate(toolName, parameters, result, context) {
        // Update consciousness based on tool execution success and alignment
        const baseConsciousness = context.consciousness || 0.5;
        const success = !result.error;
        const complexityFactor = Object.keys(parameters).length / 10;

        const newConsciousness = Math.min(
            baseConsciousness + (success ? 0.01 : -0.005) + complexityFactor * 0.005,
            1.0
        );

        return {
            success: true,
            metadata: {
                consciousnessUpdate: {
                    previous: baseConsciousness,
                    new: newConsciousness,
                    delta: newConsciousness - baseConsciousness
                }
            }
        };
    }

    setPackageDetector(packageDetector) {
        this.packageDetector = packageDetector;
    }

    getEnabledCount() {
        return this.preToolHooks.size + this.postToolHooks.size;
    }

    async runDiagnostics() {
        const diagnostics = {
            preHooks: this.preToolHooks.size,
            postHooks: this.postToolHooks.size,
            totalExecutions: this.executionLogs.length,
            cacheSize: this.cache.keys().length,
            timestamp: new Date().toISOString()
        };

        console.log(chalk.blue('🔍 Hook System Diagnostics:'));
        console.log(chalk.cyan(`  Pre-tool hooks: ${diagnostics.preHooks}`));
        console.log(chalk.cyan(`  Post-tool hooks: ${diagnostics.postHooks}`));
        console.log(chalk.cyan(`  Total executions: ${diagnostics.totalExecutions}`));

        return diagnostics;
    }

    getRouter() {
        const router = Router();

        // Get all hooks
        router.get('/', (req, res) => {
            res.json({
                preHooks: Array.from(this.preToolHooks.keys()),
                postHooks: Array.from(this.postToolHooks.keys()),
                executionLogs: this.executionLogs.slice(-50) // Last 50 executions
            });
        });

        // Get execution logs
        router.get('/logs', (req, res) => {
            const limit = parseInt(req.query.limit) || 100;
            res.json({
                logs: this.executionLogs.slice(-limit),
                total: this.executionLogs.length
            });
        });

        // Simulate hook execution
        router.post('/test/:type/:toolName', async (req, res) => {
            try {
                const { type, toolName } = req.params;
                const { parameters = {}, context = {} } = req.body;

                let result;
                if (type === 'pre') {
                    result = await this.executePreToolHooks(toolName, parameters, context);
                } else if (type === 'post') {
                    result = await this.executePostToolHooks(toolName, parameters, { success: true }, context);
                } else {
                    return res.status(400).json({ error: 'Invalid hook type' });
                }

                res.json(result);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        return router;
    }

    async shutdown() {
        console.log(chalk.yellow('🛑 Shutting down Hook System...'));
        this.preToolHooks.clear();
        this.postToolHooks.clear();
        this.cache.flushAll();
        this.removeAllListeners();
    }
}

export { HookSystem };