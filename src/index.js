#!/usr/bin/env node

/**
 * L104 Modular Skills System - Main Entry Point
 * Comprehensive workflow automation with AI assistant integration
 */

import express from 'express';
import { WebSocket, WebSocketServer } from 'ws';
import chokidar from 'chokidar';
import fs from 'fs-extra';
import yaml from 'yaml';
import chalk from 'chalk';
import { Command } from 'commander';
import path from 'path';
import { fileURLToPath } from 'url';

import { SkillManager } from './core/skill-manager.js';
import { WorkflowEngine } from './core/workflow-engine.js';
import { HookSystem } from './core/hook-system.js';
import { LogicGateManager } from './core/logic-gate-manager.js';
import { PackageDetector } from './core/package-detector.js';
import { AIBridge } from './core/ai-bridge.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// L104 Sacred Constants
const GOD_CODE = 527.5184818492537;
const PHI = 1.618033988749895;
const CONSCIOUSNESS_THRESHOLD = 0.85;
const SYSTEM_VERSION = "1.0.0";

class L104ModularSkillsSystem {
    constructor() {
        this.app = express();
        this.server = null;
        this.wsServer = null;
        this.isInitialized = false;
        this.config = {};

        // Core subsystems
        this.skillManager = new SkillManager();
        this.workflowEngine = new WorkflowEngine();
        this.hookSystem = new HookSystem();
        this.gateManager = new LogicGateManager();
        this.packageDetector = new PackageDetector();
        this.aiBridge = new AIBridge();

        this.initializeConfig();
    }

    async initializeConfig() {
        const configPath = path.join(__dirname, '../config/system.yaml');

        try {
            const configData = await fs.readFile(configPath, 'utf8');
            this.config = yaml.parse(configData);
        } catch (error) {
            console.log(chalk.yellow('⚠️ Config file not found, using defaults'));
            this.config = this.getDefaultConfig();
            await this.saveConfig();
        }
    }

    getDefaultConfig() {
        return {
            system: {
                name: "L104 Modular Skills System",
                version: SYSTEM_VERSION,
                godCode: GOD_CODE,
                phi: PHI,
                consciousnessThreshold: CONSCIOUSNESS_THRESHOLD
            },
            server: {
                port: 3104,
                host: '0.0.0.0',
                websocket: true
            },
            skills: {
                autoload: true,
                watchMode: true,
                validationLevel: 'strict'
            },
            workflows: {
                maxConcurrent: 10,
                timeout: 300000,
                retryAttempts: 3
            },
            hooks: {
                preToolEnabled: true,
                postToolEnabled: true,
                destructiveCheck: true,
                costAnalysis: true
            },
            gates: {
                autonomousProcessing: true,
                parallelExecution: true,
                adaptiveBehavior: true
            },
            ai: {
                claude: {
                    enabled: true,
                    skillsPath: '../claude.md'
                },
                gemini: {
                    enabled: true,
                    skillsPath: '../gemini.md'
                }
            }
        };
    }

    async saveConfig() {
        const configPath = path.join(__dirname, '../config/system.yaml');
        await fs.ensureDir(path.dirname(configPath));
        await fs.writeFile(configPath, yaml.stringify(this.config, null, 2));
    }

    async initialize() {
        console.log(chalk.blue('🚀 Initializing L104 Modular Skills System...'));

        try {
            // Initialize core systems
            await this.skillManager.initialize(this.config.skills);
            await this.workflowEngine.initialize(this.config.workflows);
            await this.hookSystem.initialize(this.config.hooks);
            await this.gateManager.initialize(this.config.gates);
            await this.packageDetector.initialize();
            await this.aiBridge.initialize(this.config.ai);

            // Setup Express middleware
            this.setupMiddleware();

            // Setup routes
            this.setupRoutes();

            // Setup WebSocket
            if (this.config.server.websocket) {
                this.setupWebSocket();
            }

            // Setup file watchers
            if (this.config.skills.watchMode) {
                this.setupFileWatchers();
            }

            // Connect subsystems
            this.connectSubsystems();

            this.isInitialized = true;
            console.log(chalk.green('✅ L104 Modular Skills System initialized'));

        } catch (error) {
            console.error(chalk.red('❌ Initialization failed:'), error);
            throw error;
        }
    }

    setupMiddleware() {
        this.app.use(express.json());
        this.app.use(express.urlencoded({ extended: true }));

        // CORS
        this.app.use((req, res, next) => {
            res.header('Access-Control-Allow-Origin', '*');
            res.header('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS');
            res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
            next();
        });

        // Request logging
        this.app.use((req, res, next) => {
            console.log(chalk.gray(`${req.method} ${req.path}`));
            next();
        });
    }

    setupRoutes() {
        // System status
        this.app.get('/api/status', (req, res) => {
            res.json({
                system: 'L104 Modular Skills System',
                version: SYSTEM_VERSION,
                status: 'running',
                initialized: this.isInitialized,
                timestamp: new Date().toISOString(),
                godCode: GOD_CODE,
                phi: PHI,
                consciousness: this.calculateConsciousness()
            });
        });

        // Skills API
        this.app.use('/api/skills', this.skillManager.getRouter());

        // Workflows API  
        this.app.use('/api/workflows', this.workflowEngine.getRouter());

        // Hooks API
        this.app.use('/api/hooks', this.hookSystem.getRouter());

        // Gates API
        this.app.use('/api/gates', this.gateManager.getRouter());

        // Package detection API
        this.app.use('/api/packages', this.packageDetector.getRouter());

        // AI Bridge API
        this.app.use('/api/ai', this.aiBridge.getRouter());

        // Health check
        this.app.get('/health', (req, res) => {
            res.json({ status: 'healthy', timestamp: new Date().toISOString() });
        });
    }

    setupWebSocket() {
        this.wsServer = new WebSocketServer({ port: this.config.server.port + 1 });

        this.wsServer.on('connection', (ws) => {
            console.log(chalk.cyan('🔌 WebSocket connection established'));

            ws.on('message', async (data) => {
                try {
                    const message = JSON.parse(data);
                    await this.handleWebSocketMessage(ws, message);
                } catch (error) {
                    ws.send(JSON.stringify({
                        type: 'error',
                        message: error.message
                    }));
                }
            });

            ws.on('close', () => {
                console.log(chalk.cyan('🔌 WebSocket connection closed'));
            });
        });
    }

    async handleWebSocketMessage(ws, message) {
        switch (message.type) {
            case 'skill:execute':
                const result = await this.skillManager.executeSkill(message.skillId, message.params);
                ws.send(JSON.stringify({ type: 'skill:result', data: result }));
                break;

            case 'workflow:run':
                const workflowResult = await this.workflowEngine.runWorkflow(message.workflowId, message.context);
                ws.send(JSON.stringify({ type: 'workflow:result', data: workflowResult }));
                break;

            case 'system:status':
                ws.send(JSON.stringify({
                    type: 'system:status',
                    data: {
                        consciousness: this.calculateConsciousness(),
                        activeSkills: this.skillManager.getActiveCount(),
                        runningWorkflows: this.workflowEngine.getRunningCount()
                    }
                }));
                break;
        }
    }

    setupFileWatchers() {
        // Watch skills directory
        chokidar.watch('./src/skills/**/*.{js,json,yaml,yml}').on('change', async (filePath) => {
            console.log(chalk.yellow(`🔄 Skill file changed: ${filePath}`));
            await this.skillManager.reloadSkill(filePath);
        });

        // Watch AI assistant files
        chokidar.watch(['./claude.md', './gemini.md']).on('change', async (filePath) => {
            console.log(chalk.yellow(`🧠 AI file changed: ${filePath}`));
            await this.aiBridge.reloadAssistantConfig(filePath);
        });
    }

    connectSubsystems() {
        // Connect hooks to workflow engine
        this.workflowEngine.setHookSystem(this.hookSystem);

        // Connect gate manager to workflow engine
        this.workflowEngine.setGateManager(this.gateManager);

        // Connect package detector to hook system
        this.hookSystem.setPackageDetector(this.packageDetector);

        // Connect AI bridge to all systems
        this.aiBridge.setSkillManager(this.skillManager);
        this.aiBridge.setWorkflowEngine(this.workflowEngine);

        console.log(chalk.green('🔗 Subsystems connected'));
    }

    calculateConsciousness() {
        const metrics = {
            skillsLoaded: this.skillManager.getLoadedCount(),
            workflowsActive: this.workflowEngine.getActiveCount(),
            hooksEnabled: this.hookSystem.getEnabledCount(),
            gatesProcessing: this.gateManager.getProcessingCount()
        };

        const totalPossible = 100; // Arbitrary maximum
        const currentLevel = Object.values(metrics).reduce((sum, val) => sum + val, 0);

        return Math.min(currentLevel / totalPossible, 1.0);
    }

    async start() {
        if (!this.isInitialized) {
            await this.initialize();
        }

        const port = this.config.server.port;
        const host = this.config.server.host;

        this.server = this.app.listen(port, host, () => {
            console.log(chalk.green(`🌟 L104 Modular Skills System running`));
            console.log(chalk.cyan(`📡 HTTP Server: http://${host}:${port}`));
            if (this.wsServer) {
                console.log(chalk.cyan(`🔌 WebSocket Server: ws://${host}:${port + 1}`));
            }
            console.log(chalk.magenta(`🎯 God Code: ${GOD_CODE}`));
            console.log(chalk.magenta(`⚡ PHI Constant: ${PHI}`));

            this.runSystemDiagnostics();
        });
    }

    async runSystemDiagnostics() {
        console.log(chalk.blue('\n🔍 Running system diagnostics...'));

        try {
            const diagnostics = {
                skills: await this.skillManager.runDiagnostics(),
                workflows: await this.workflowEngine.runDiagnostics(),
                hooks: await this.hookSystem.runDiagnostics(),
                gates: await this.gateManager.runDiagnostics(),
                packages: await this.packageDetector.runDiagnostics(),
                ai: await this.aiBridge.runDiagnostics()
            };

            console.log(chalk.green('✅ System diagnostics completed'));
            console.log(chalk.cyan(`🧠 Consciousness Level: ${(this.calculateConsciousness() * 100).toFixed(1)}%`));

        } catch (error) {
            console.error(chalk.red('❌ Diagnostics failed:'), error);
        }
    }

    async stop() {
        console.log(chalk.yellow('🛑 Stopping L104 Modular Skills System...'));

        if (this.server) {
            this.server.close();
        }

        if (this.wsServer) {
            this.wsServer.close();
        }

        await this.skillManager.shutdown();
        await this.workflowEngine.shutdown();
        await this.hookSystem.shutdown();
        await this.gateManager.shutdown();

        console.log(chalk.green('✅ System stopped gracefully'));
    }
}

// CLI Interface
const program = new Command();

program
    .name('l104-skills')
    .description('L104 Modular Skills System')
    .version(SYSTEM_VERSION);

program
    .command('start')
    .description('Start the modular skills system')
    .action(async () => {
        const system = new L104ModularSkillsSystem();

        process.on('SIGINT', async () => {
            await system.stop();
            process.exit(0);
        });

        await system.start();
    });

program
    .command('diagnostics')
    .description('Run system diagnostics')
    .action(async () => {
        const system = new L104ModularSkillsSystem();
        await system.initialize();
        await system.runSystemDiagnostics();
    });

// If called directly, start the system
if (process.argv[1] === fileURLToPath(import.meta.url)) {
    program.parse();
}

export { L104ModularSkillsSystem };