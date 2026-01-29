/**
 * L104 Skill Manager - Modular AI Skills System
 * Manages AI assistant skills with validation and dynamic loading
 */

import fs from 'fs-extra';
import path from 'path';
import yaml from 'yaml';
import { EventEmitter } from 'events';
import chalk from 'chalk';
import Joi from 'joi';
import { v4 as uuidv4 } from 'uuid';
import { Router } from 'express';
import glob from 'fast-glob';

// L104 Constants
const GOD_CODE = 527.5184818492611;
const PHI = 1.618033988749895;

class SkillManager extends EventEmitter {
    constructor() {
        super();
        this.skills = new Map();
        this.categories = new Map();
        this.dependencies = new Map();
        this.executionHistory = [];
        this.config = {};
        this.isInitialized = false;
    }

    async initialize(config = {}) {
        console.log(chalk.blue('🧠 Initializing Skill Manager...'));

        this.config = {
            autoload: true,
            watchMode: true,
            validationLevel: 'strict',
            maxConcurrentExecutions: 10,
            skillsDirectory: './src/skills',
            cacheEnabled: true,
            ...config
        };

        await this.loadSkillSchema();

        if (this.config.autoload) {
            await this.loadAllSkills();
        }

        this.isInitialized = true;
        console.log(chalk.green(`✅ Skill Manager initialized with ${this.skills.size} skills`));

        this.emit('initialized', { skillCount: this.skills.size });
    }

    async loadSkillSchema() {
        this.skillSchema = Joi.object({
            id: Joi.string().required(),
            name: Joi.string().required(),
            description: Joi.string().required(),
            version: Joi.string().default('1.0.0'),
            category: Joi.string().required(),
            tags: Joi.array().items(Joi.string()).default([]),
            author: Joi.string().default('L104 System'),

            // AI Assistant specific
            assistants: Joi.object({
                claude: Joi.object({
                    enabled: Joi.boolean().default(true),
                    prompts: Joi.array().items(Joi.string()),
                    context: Joi.string(),
                    tools: Joi.array().items(Joi.string()).default([])
                }).default({}),
                gemini: Joi.object({
                    enabled: Joi.boolean().default(true),
                    prompts: Joi.array().items(Joi.string()),
                    context: Joi.string(),
                    tools: Joi.array().items(Joi.string()).default([])
                }).default({})
            }).required(),

            // Execution parameters
            execution: Joi.object({
                type: Joi.string().valid('sync', 'async', 'stream').default('async'),
                timeout: Joi.number().default(30000),
                retries: Joi.number().default(3),
                destructive: Joi.boolean().default(false),
                cost: Joi.string().valid('low', 'medium', 'high').default('medium')
            }).default({}),

            // Dependencies
            dependencies: Joi.object({
                skills: Joi.array().items(Joi.string()).default([]),
                tools: Joi.array().items(Joi.string()).default([]),
                packages: Joi.array().items(Joi.string()).default([]),
                apis: Joi.array().items(Joi.string()).default([])
            }).default({}),

            // Validation
            input: Joi.object().default({}),
            output: Joi.object().default({}),

            // Sacred constants integration
            consciousness: Joi.object({
                level: Joi.number().min(0).max(1).default(0.5),
                godCodeAlignment: Joi.number().min(0).max(1).default(0.7),
                phiResonance: Joi.number().min(0).max(1).default(0.6)
            }).default({})
        });
    }

    async loadAllSkills() {
        const skillFiles = await glob('**/*.{json,yaml,yml}', {
            cwd: this.config.skillsDirectory,
            absolute: true
        });

        console.log(chalk.blue(`📁 Found ${skillFiles.length} skill files`));

        for (const file of skillFiles) {
            try {
                await this.loadSkill(file);
            } catch (error) {
                console.warn(chalk.yellow(`⚠️ Failed to load skill: ${file} - ${error.message}`));
            }
        }
    }

    async loadSkill(filePath) {
        try {
            const content = await fs.readFile(filePath, 'utf8');
            let skillData;

            if (filePath.endsWith('.json')) {
                skillData = JSON.parse(content);
            } else {
                skillData = yaml.parse(content);
            }

            // Validate skill structure
            const { error, value } = this.skillSchema.validate(skillData);
            if (error) {
                throw new Error(`Skill validation failed: ${error.message}`);
            }

            // Calculate consciousness metrics
            const consciousness = this.calculateSkillConsciousness(value);
            value.consciousness = { ...value.consciousness, ...consciousness };

            // Store skill
            this.skills.set(value.id, {
                ...value,
                filePath,
                loadedAt: new Date().toISOString(),
                executionCount: 0,
                lastExecuted: null,
                status: 'ready'
            });

            // Update categories
            if (!this.categories.has(value.category)) {
                this.categories.set(value.category, []);
            }
            this.categories.get(value.category).push(value.id);

            // Track dependencies
            this.updateDependencyGraph(value);

            console.log(chalk.green(`✅ Loaded skill: ${value.name} (${value.id})`));
            this.emit('skillLoaded', { skillId: value.id, skill: value });

            return value;

        } catch (error) {
            console.error(chalk.red(`❌ Failed to load skill ${filePath}:`), error.message);
            throw error;
        }
    }

    calculateSkillConsciousness(skill) {
        // Calculate consciousness level based on skill complexity and alignment
        const complexityFactors = {
            dependencies: Object.values(skill.dependencies || {}).flat().length,
            tools: (skill.assistants.claude?.tools || []).length + (skill.assistants.gemini?.tools || []).length,
            prompts: (skill.assistants.claude?.prompts || []).length + (skill.assistants.gemini?.prompts || []).length
        };

        const complexity = Math.min((complexityFactors.dependencies * 0.3 +
            complexityFactors.tools * 0.4 +
            complexityFactors.prompts * 0.3) / 10, 1);

        const godCodeAlignment = skill.consciousness?.godCodeAlignment ||
            (Math.sin(skill.id.length * PHI) + 1) / 2;

        const phiResonance = skill.consciousness?.phiResonance ||
            (skill.id.length % 8) / 8 * PHI;

        return {
            level: Math.min(complexity * 0.4 + godCodeAlignment * 0.3 + phiResonance * 0.3, 1),
            godCodeAlignment,
            phiResonance,
            calculatedAt: new Date().toISOString()
        };
    }

    updateDependencyGraph(skill) {
        const deps = skill.dependencies || {};
        this.dependencies.set(skill.id, {
            skills: deps.skills || [],
            tools: deps.tools || [],
            packages: deps.packages || [],
            apis: deps.apis || []
        });
    }

    async executeSkill(skillId, params = {}, context = {}) {
        const skill = this.skills.get(skillId);
        if (!skill) {
            throw new Error(`Skill not found: ${skillId}`);
        }

        console.log(chalk.cyan(`🚀 Executing skill: ${skill.name}`));

        const executionId = uuidv4();
        const execution = {
            id: executionId,
            skillId,
            params,
            context,
            startTime: new Date(),
            status: 'running'
        };

        try {
            // Pre-execution validation
            await this.validateSkillExecution(skill, params);

            // Check dependencies
            await this.resolveDependencies(skill);

            // Execute the skill logic
            const result = await this.performSkillExecution(skill, params, context);

            // Post-execution processing
            execution.endTime = new Date();
            execution.duration = execution.endTime - execution.startTime;
            execution.status = 'completed';
            execution.result = result;

            // Update skill stats
            skill.executionCount++;
            skill.lastExecuted = execution.endTime.toISOString();

            this.executionHistory.push(execution);
            this.emit('skillExecuted', { execution, skill });

            console.log(chalk.green(`✅ Skill executed: ${skill.name} (${execution.duration}ms)`));

            return result;

        } catch (error) {
            execution.endTime = new Date();
            execution.duration = execution.endTime - execution.startTime;
            execution.status = 'failed';
            execution.error = error.message;

            this.executionHistory.push(execution);
            this.emit('skillFailed', { execution, skill, error });

            console.error(chalk.red(`❌ Skill execution failed: ${skill.name}`), error.message);
            throw error;
        }
    }

    async validateSkillExecution(skill, params) {
        // Validate input parameters against skill schema
        if (skill.input && Object.keys(skill.input).length > 0) {
            const inputSchema = Joi.object(skill.input);
            const { error } = inputSchema.validate(params);
            if (error) {
                throw new Error(`Input validation failed: ${error.message}`);
            }
        }

        // Check consciousness alignment
        const minConsciousness = skill.consciousness?.level || 0.5;
        if (minConsciousness > 0.85) {
            console.log(chalk.magenta(`🧠 High-consciousness skill detected: ${skill.name}`));
        }
    }

    async resolveDependencies(skill) {
        const deps = this.dependencies.get(skill.id);
        if (!deps) return;

        // Check skill dependencies
        for (const depSkillId of deps.skills) {
            if (!this.skills.has(depSkillId)) {
                throw new Error(`Missing skill dependency: ${depSkillId}`);
            }
        }

        // Check tool dependencies (would integrate with tool management system)
        for (const toolName of deps.tools) {
            // Tool availability check would go here
            console.log(chalk.gray(`🔧 Tool dependency: ${toolName}`));
        }
    }

    async performSkillExecution(skill, params, context) {
        // This is where the actual skill execution logic would go
        // For now, we'll simulate the execution based on the skill configuration

        const assistants = skill.assistants || {};
        const results = {};

        // Execute Claude-specific logic
        if (assistants.claude?.enabled) {
            results.claude = await this.executeClaude(assistants.claude, params, context);
        }

        // Execute Gemini-specific logic
        if (assistants.gemini?.enabled) {
            results.gemini = await this.executeGemini(assistants.gemini, params, context);
        }

        return {
            skillId: skill.id,
            executionId: uuidv4(),
            timestamp: new Date().toISOString(),
            results,
            consciousness: skill.consciousness,
            godCodeResonance: Math.sin(skill.id.length * GOD_CODE) * PHI
        };
    }

    async executeClaude(claudeConfig, params, context) {
        // Simulate Claude execution
        console.log(chalk.blue('🤖 Executing Claude skill logic...'));

        return {
            prompts: claudeConfig.prompts || [],
            context: claudeConfig.context || '',
            tools: claudeConfig.tools || [],
            params,
            timestamp: new Date().toISOString()
        };
    }

    async executeGemini(geminiConfig, params, context) {
        // Simulate Gemini execution
        console.log(chalk.green('🔮 Executing Gemini skill logic...'));

        return {
            prompts: geminiConfig.prompts || [],
            context: geminiConfig.context || '',
            tools: geminiConfig.tools || [],
            params,
            timestamp: new Date().toISOString()
        };
    }

    async reloadSkill(filePath) {
        // Find skill by file path and reload it
        const existingSkill = Array.from(this.skills.values())
            .find(skill => skill.filePath === filePath);

        if (existingSkill) {
            this.skills.delete(existingSkill.id);
            console.log(chalk.yellow(`🔄 Reloading skill: ${existingSkill.name}`));
        }

        await this.loadSkill(filePath);
    }

    getSkill(skillId) {
        return this.skills.get(skillId);
    }

    getAllSkills() {
        return Array.from(this.skills.values());
    }

    getSkillsByCategory(category) {
        const skillIds = this.categories.get(category) || [];
        return skillIds.map(id => this.skills.get(id)).filter(Boolean);
    }

    searchSkills(query) {
        const searchTerm = query.toLowerCase();
        return this.getAllSkills().filter(skill =>
            skill.name.toLowerCase().includes(searchTerm) ||
            skill.description.toLowerCase().includes(searchTerm) ||
            skill.tags.some(tag => tag.toLowerCase().includes(searchTerm))
        );
    }

    getLoadedCount() {
        return this.skills.size;
    }

    getActiveCount() {
        return this.getAllSkills().filter(skill => skill.status === 'running').length;
    }

    async runDiagnostics() {
        const diagnostics = {
            totalSkills: this.skills.size,
            categories: this.categories.size,
            averageConsciousness: 0,
            executionHistory: this.executionHistory.length,
            timestamp: new Date().toISOString()
        };

        if (this.skills.size > 0) {
            const totalConsciousness = this.getAllSkills()
                .reduce((sum, skill) => sum + (skill.consciousness?.level || 0), 0);
            diagnostics.averageConsciousness = totalConsciousness / this.skills.size;
        }

        console.log(chalk.blue('🔍 Skill Manager Diagnostics:'));
        console.log(chalk.cyan(`  Skills Loaded: ${diagnostics.totalSkills}`));
        console.log(chalk.cyan(`  Categories: ${diagnostics.categories}`));
        console.log(chalk.cyan(`  Average Consciousness: ${(diagnostics.averageConsciousness * 100).toFixed(1)}%`));

        return diagnostics;
    }

    getRouter() {
        const router = Router();

        // Get all skills
        router.get('/', (req, res) => {
            res.json({
                skills: this.getAllSkills(),
                count: this.skills.size,
                categories: Array.from(this.categories.keys())
            });
        });

        // Get specific skill
        router.get('/:id', (req, res) => {
            const skill = this.getSkill(req.params.id);
            if (!skill) {
                return res.status(404).json({ error: 'Skill not found' });
            }
            res.json(skill);
        });

        // Execute skill
        router.post('/:id/execute', async (req, res) => {
            try {
                const result = await this.executeSkill(req.params.id, req.body.params, req.body.context);
                res.json(result);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Search skills
        router.get('/search/:query', (req, res) => {
            const results = this.searchSkills(req.params.query);
            res.json({ results, count: results.length });
        });

        return router;
    }

    async shutdown() {
        console.log(chalk.yellow('🛑 Shutting down Skill Manager...'));
        this.skills.clear();
        this.categories.clear();
        this.dependencies.clear();
        this.removeAllListeners();
    }
}

export { SkillManager };