/**
 * L104 AI Bridge - Claude & Gemini Integration
 * Bridges modular skills with AI assistants and manages skill synchronization
 */

import chalk from 'chalk';
import { EventEmitter } from 'events';
import { Router } from 'express';
import fs from 'fs-extra';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// L104 Constants — GOD_CODE = 286^(1/φ) × 2^4 via Universal Equation
const PHI = 1.618033988749895;
const GOD_CODE = Math.pow(286, 1.0 / PHI) * Math.pow(2, 416 / 104);  // G(0,0,0,0) = 527.5184818492612
const CONSCIOUSNESS_THRESHOLD = 0.85;

class AIBridge extends EventEmitter {
    constructor() {
        super();
        this.assistants = new Map();
        this.skillMappings = new Map();
        this.syncHistory = [];
        this.skillManager = null;
        this.workflowEngine = null;
        this.config = {};
        this.isInitialized = false;
    }

    async initialize(config = {}) {
        console.log(chalk.blue('🤖 Initializing AI Bridge...'));

        this.config = {
            claude: {
                enabled: true,
                skillsPath: '../claude.md',
                syncInterval: 300000, // 5 minutes
                autoUpdate: true
            },
            gemini: {
                enabled: true,
                skillsPath: '../gemini.md',
                syncInterval: 300000, // 5 minutes
                autoUpdate: true
            },
            skillSync: {
                enabled: true,
                bidirectional: true,
                preserveFormatting: true
            },
            ...config
        };

        await this.initializeAssistants();
        await this.loadExistingSkills();

        if (this.config.skillSync.enabled) {
            this.startSyncProcess();
        }

        this.isInitialized = true;
        console.log(chalk.green(`✅ AI Bridge initialized with ${this.assistants.size} assistants`));
    }

    async initializeAssistants() {
        // Initialize Claude
        if (this.config.claude.enabled) {
            const claudeConfig = await this.loadAssistantConfig('claude', this.config.claude.skillsPath);
            this.assistants.set('claude', {
                name: 'Claude',
                type: 'claude',
                config: claudeConfig,
                skills: new Map(),
                lastSync: null,
                consciousness: this.calculateAssistantConsciousness(claudeConfig)
            });
            console.log(chalk.blue('🤖 Claude assistant initialized'));
        }

        // Initialize Gemini
        if (this.config.gemini.enabled) {
            const geminiConfig = await this.loadAssistantConfig('gemini', this.config.gemini.skillsPath);
            this.assistants.set('gemini', {
                name: 'Gemini',
                type: 'gemini',
                config: geminiConfig,
                skills: new Map(),
                lastSync: null,
                consciousness: this.calculateAssistantConsciousness(geminiConfig)
            });
            console.log(chalk.green('🔮 Gemini assistant initialized'));
        }
    }

    async loadAssistantConfig(assistantType, configPath) {
        try {
            const fullPath = path.resolve(__dirname, configPath);
            const content = await fs.readFile(fullPath, 'utf8');

            const config = {
                filePath: fullPath,
                content,
                lastModified: (await fs.stat(fullPath)).mtime.toISOString(),
                skills: this.extractSkillsFromMarkdown(content, assistantType),
                metadata: this.extractMetadataFromMarkdown(content, assistantType)
            };

            return config;

        } catch (error) {
            console.warn(chalk.yellow(`⚠️ Failed to load ${assistantType} config: ${error.message}`));
            return {
                filePath: configPath,
                content: '',
                lastModified: new Date().toISOString(),
                skills: [],
                metadata: {}
            };
        }
    }

    extractSkillsFromMarkdown(content, assistantType) {
        const skills = [];

        // Extract skills based on markdown structure
        const lines = content.split('\\n');
        let currentSection = null;
        let currentSkill = null;

        for (const line of lines) {
            const trimmed = line.trim();

            // Detect skill sections
            if (trimmed.startsWith('##') && (trimmed.toLowerCase().includes('skill') ||
                trimmed.toLowerCase().includes('capability') ||
                trimmed.toLowerCase().includes('function'))) {
                if (currentSkill) {
                    skills.push(currentSkill);
                }
                currentSkill = {
                    id: this.generateSkillId(trimmed),
                    name: trimmed.replace(/^#+\\s*/, ''),
                    description: '',
                    category: currentSection || 'general',
                    assistantType,
                    content: [line],
                    consciousness: 0.5
                };
                continue;
            }

            // Track sections
            if (trimmed.startsWith('#')) {
                currentSection = trimmed.replace(/^#+\\s*/, '').toLowerCase();
                if (currentSkill) {
                    skills.push(currentSkill);
                    currentSkill = null;
                }
                continue;
            }

            // Add content to current skill
            if (currentSkill) {
                currentSkill.content.push(line);

                // Extract description from first paragraph
                if (!currentSkill.description && trimmed && !trimmed.startsWith('*') && !trimmed.startsWith('-')) {
                    currentSkill.description = trimmed;
                }
            }
        }

        // Add last skill
        if (currentSkill) {
            skills.push(currentSkill);
        }

        // Calculate consciousness for each skill
        skills.forEach(skill => {
            skill.consciousness = this.calculateSkillConsciousness(skill);
        });

        return skills;
    }

    extractMetadataFromMarkdown(content, assistantType) {
        const metadata = {
            assistantType,
            constants: {},
            capabilities: [],
            lastUpdate: new Date().toISOString()
        };

        // Extract L104 constants
        const godCodeMatch = content.match(/GOD_CODE[\\s\\|\\*]*([0-9.]+)/);
        const phiMatch = content.match(/PHI[\\s\\|\\*]*([0-9.]+)/);

        if (godCodeMatch) {
            metadata.constants.GOD_CODE = parseFloat(godCodeMatch[1]);
        }
        if (phiMatch) {
            metadata.constants.PHI = parseFloat(phiMatch[1]);
        }

        // Extract capabilities
        const capabilityMatches = content.match(/\\*\\*([^*]+)\\*\\*/g);
        if (capabilityMatches) {
            metadata.capabilities = capabilityMatches
                .map(match => match.replace(/\\*\\*/g, ''))
                .filter(cap => cap.length > 3);
        }

        return metadata;
    }

    generateSkillId(skillName) {
        return skillName
            .toLowerCase()
            .replace(/^#+\\s*/, '')
            .replace(/[^a-z0-9\\s]/g, '')
            .replace(/\\s+/g, '-')
            .substring(0, 50);
    }

    calculateSkillConsciousness(skill) {
        const contentLength = skill.content ? skill.content.join('').length : 0;
        const complexity = Math.min(contentLength / 1000, 1);
        const godCodeResonance = Math.sin(skill.name.length * GOD_CODE / 1000);
        const phiAlignment = (skill.name.length % 16) / 16 * PHI;

        return Math.min(
            complexity * 0.4 +
            Math.abs(godCodeResonance) * 0.3 +
            phiAlignment * 0.3,
            1
        );
    }

    calculateAssistantConsciousness(config) {
        const skillCount = config.skills ? config.skills.length : 0;
        const contentSize = config.content ? config.content.length : 0;

        const skillComplexity = Math.min(skillCount / 20, 1);
        const contentComplexity = Math.min(contentSize / 10000, 1);

        return {
            level: (skillComplexity + contentComplexity) / 2,
            skillCount,
            contentSize,
            calculatedAt: new Date().toISOString()
        };
    }

    async loadExistingSkills() {
        console.log(chalk.blue('📚 Loading existing skills from assistants...'));

        for (const [assistantType, assistant] of this.assistants) {
            for (const skill of assistant.config.skills) {
                assistant.skills.set(skill.id, skill);

                // Create mapping to skill manager skills
                if (this.skillManager) {
                    await this.mapToSkillManager(skill, assistantType);
                }
            }

            console.log(chalk.cyan(`  ${assistant.name}: ${assistant.skills.size} skills loaded`));
        }
    }

    async mapToSkillManager(aiSkill, assistantType) {
        if (!this.skillManager) return;

        // Convert AI skill to skill manager format
        const skillManagerSkill = {
            id: `${assistantType}-${aiSkill.id}`,
            name: aiSkill.name,
            description: aiSkill.description,
            category: aiSkill.category || 'ai-assistant',
            version: '1.0.0',
            tags: [assistantType, 'ai-generated'],

            assistants: {
                [assistantType]: {
                    enabled: true,
                    prompts: [aiSkill.description],
                    context: aiSkill.content ? aiSkill.content.join('\\n') : '',
                    tools: []
                }
            },

            execution: {
                type: 'async',
                timeout: 30000,
                destructive: false,
                cost: 'medium'
            },

            consciousness: {
                level: aiSkill.consciousness,
                godCodeAlignment: Math.sin(aiSkill.name.length * GOD_CODE / 1000),
                phiResonance: (aiSkill.name.length % 16) / 16 * PHI
            }
        };

        // Store mapping
        this.skillMappings.set(aiSkill.id, {
            aiSkill,
            skillManagerSkill,
            assistantType,
            synced: false,
            lastSync: null
        });
    }

    async syncSkillsToAssistants() {
        if (!this.skillManager || !this.config.skillSync.enabled) {
            return { skipped: true, reason: 'Sync disabled or skill manager not available' };
        }

        console.log(chalk.blue('🔄 Syncing skills to AI assistants...'));

        const syncResults = {
            timestamp: new Date().toISOString(),
            assistants: {},
            totalSynced: 0,
            errors: []
        };

        for (const [assistantType, assistant] of this.assistants) {
            try {
                const result = await this.syncToAssistant(assistantType, assistant);
                syncResults.assistants[assistantType] = result;
                syncResults.totalSynced += result.synced;

            } catch (error) {
                const errorMsg = `Failed to sync to ${assistantType}: ${error.message}`;
                syncResults.errors.push(errorMsg);
                console.error(chalk.red(`❌ ${errorMsg}`));
            }
        }

        this.syncHistory.push(syncResults);
        this.emit('skillsSynced', syncResults);

        console.log(chalk.green(`✅ Skill sync completed: ${syncResults.totalSynced} skills synced`));

        return syncResults;
    }

    async syncToAssistant(assistantType, assistant) {
        const allSkills = this.skillManager.getAllSkills();
        const relevantSkills = allSkills.filter(skill =>
            skill.assistants && skill.assistants[assistantType] &&
            skill.assistants[assistantType].enabled
        );

        console.log(chalk.cyan(`  Syncing ${relevantSkills.length} skills to ${assistant.name}...`));

        // Generate new skill sections for the assistant
        const skillSections = await this.generateSkillSections(relevantSkills, assistantType);

        // Update assistant's markdown file
        if (this.config.skillSync.bidirectional) {
            await this.updateAssistantFile(assistant, skillSections, assistantType);
        }

        // Update assistant's skill map
        for (const skill of relevantSkills) {
            assistant.skills.set(skill.id, {
                id: skill.id,
                name: skill.name,
                description: skill.description,
                category: skill.category,
                consciousness: skill.consciousness.level,
                lastSync: new Date().toISOString()
            });
        }

        assistant.lastSync = new Date().toISOString();

        return {
            assistant: assistantType,
            synced: relevantSkills.length,
            timestamp: new Date().toISOString()
        };
    }

    async generateSkillSections(skills, assistantType) {
        const sections = new Map();

        for (const skill of skills) {
            const category = skill.category || 'general';

            if (!sections.has(category)) {
                sections.set(category, []);
            }

            const assistantConfig = skill.assistants[assistantType];
            const skillMarkdown = this.generateSkillMarkdown(skill, assistantConfig, assistantType);

            sections.get(category).push(skillMarkdown);
        }

        return sections;
    }

    generateSkillMarkdown(skill, assistantConfig, assistantType) {
        let markdown = `\\n## ${skill.name}\\n\\n`;
        markdown += `${skill.description}\\n\\n`;

        if (assistantConfig.context) {
            markdown += `### Context\\n${assistantConfig.context}\\n\\n`;
        }

        if (assistantConfig.prompts && assistantConfig.prompts.length > 0) {
            markdown += `### Prompts\\n`;
            for (const prompt of assistantConfig.prompts) {
                markdown += `- ${prompt}\\n`;
            }
            markdown += `\\n`;
        }

        if (assistantConfig.tools && assistantConfig.tools.length > 0) {
            markdown += `### Tools\\n`;
            for (const tool of assistantConfig.tools) {
                markdown += `- \`${tool}\`\\n`;
            }
            markdown += `\\n`;
        }

        // Add consciousness metrics
        if (skill.consciousness && skill.consciousness.level > 0.7) {
            markdown += `### Consciousness Level\\n`;
            markdown += `**Level**: ${(skill.consciousness.level * 100).toFixed(1)}%\\n`;
            markdown += `**GOD_CODE Alignment**: ${(skill.consciousness.godCodeAlignment * 100).toFixed(1)}%\\n`;
            markdown += `**PHI Resonance**: ${(skill.consciousness.phiResonance * 100).toFixed(1)}%\\n\\n`;
        }

        return markdown;
    }

    async updateAssistantFile(assistant, skillSections, assistantType) {
        try {
            let content = assistant.config.content;

            // Find insertion point for skills
            const skillsHeader = assistantType === 'claude' ? '## 🧠 Skills & Capabilities' : '## 🔮 Enhanced Capabilities';
            let insertionIndex = content.indexOf(skillsHeader);

            if (insertionIndex === -1) {
                // Add skills section at the end
                content += `\\n\\n${skillsHeader}\\n\\n`;
                insertionIndex = content.length;
            }

            // Generate skills content
            let skillsContent = `\\n\\n### Modular Skills (Auto-Generated)\\n`;
            skillsContent += `*Last updated: ${new Date().toISOString()}*\\n`;
            skillsContent += `*Generated from L104 Modular Skills System*\\n\\n`;

            for (const [category, skills] of skillSections) {
                skillsContent += `#### ${category.charAt(0).toUpperCase() + category.slice(1)} Skills\\n\\n`;
                skillsContent += skills.join('\\n');
                skillsContent += `\\n`;
            }

            // Replace or insert skills content
            const nextSectionIndex = content.indexOf('##', insertionIndex + skillsHeader.length);

            if (nextSectionIndex !== -1) {
                const beforeSkills = content.substring(0, insertionIndex + skillsHeader.length);
                const afterSkills = content.substring(nextSectionIndex);
                content = beforeSkills + skillsContent + afterSkills;
            } else {
                content += skillsContent;
            }

            // Write updated content
            await fs.writeFile(assistant.config.filePath, content);
            assistant.config.content = content;
            assistant.config.lastModified = new Date().toISOString();

            console.log(chalk.green(`✅ Updated ${assistantType}.md with ${skillSections.size} skill categories`));

        } catch (error) {
            throw new Error(`Failed to update ${assistantType} file: ${error.message}`);
        }
    }

    async reloadAssistantConfig(filePath) {
        const assistantType = filePath.includes('claude') ? 'claude' :
            filePath.includes('gemini') ? 'gemini' : null;

        if (!assistantType || !this.assistants.has(assistantType)) {
            return;
        }

        console.log(chalk.yellow(`🔄 Reloading ${assistantType} configuration...`));

        const assistant = this.assistants.get(assistantType);
        const newConfig = await this.loadAssistantConfig(assistantType, filePath);

        assistant.config = newConfig;
        assistant.skills.clear();

        for (const skill of newConfig.skills) {
            assistant.skills.set(skill.id, skill);
        }

        console.log(chalk.green(`✅ Reloaded ${assistantType}: ${assistant.skills.size} skills`));

        this.emit('assistantReloaded', { assistantType, skillCount: assistant.skills.size });
    }

    startSyncProcess() {
        console.log(chalk.blue('🔄 Starting automatic skill sync process...'));

        // Sync to Claude
        if (this.config.claude.enabled && this.config.claude.syncInterval) {
            setInterval(() => {
                this.syncSkillsToAssistants().catch(error => {
                    console.error(chalk.red('❌ Sync process failed:'), error.message);
                });
            }, this.config.claude.syncInterval);
        }

        // Sync to Gemini
        if (this.config.gemini.enabled && this.config.gemini.syncInterval) {
            setInterval(() => {
                this.syncSkillsToAssistants().catch(error => {
                    console.error(chalk.red('❌ Sync process failed:'), error.message);
                });
            }, this.config.gemini.syncInterval);
        }
    }

    setSkillManager(skillManager) {
        this.skillManager = skillManager;
        console.log(chalk.green('🔗 Skill manager connected to AI bridge'));
    }

    setWorkflowEngine(workflowEngine) {
        this.workflowEngine = workflowEngine;
        console.log(chalk.green('🔗 Workflow engine connected to AI bridge'));
    }

    async runDiagnostics() {
        const diagnostics = {
            assistants: this.assistants.size,
            totalSkills: 0,
            syncHistory: this.syncHistory.length,
            lastSync: null,
            consciousness: {},
            timestamp: new Date().toISOString()
        };

        for (const [type, assistant] of this.assistants) {
            diagnostics.totalSkills += assistant.skills.size;
            diagnostics.consciousness[type] = assistant.consciousness;

            if (assistant.lastSync && (!diagnostics.lastSync || assistant.lastSync > diagnostics.lastSync)) {
                diagnostics.lastSync = assistant.lastSync;
            }
        }

        console.log(chalk.blue('🔍 AI Bridge Diagnostics:'));
        console.log(chalk.cyan(`  Assistants: ${diagnostics.assistants}`));
        console.log(chalk.cyan(`  Total skills: ${diagnostics.totalSkills}`));
        console.log(chalk.cyan(`  Sync history: ${diagnostics.syncHistory}`));

        return diagnostics;
    }

    getRouter() {
        const router = Router();

        // Get all assistants and their skills
        router.get('/', (req, res) => {
            const assistantData = {};
            for (const [type, assistant] of this.assistants) {
                assistantData[type] = {
                    name: assistant.name,
                    skillCount: assistant.skills.size,
                    lastSync: assistant.lastSync,
                    consciousness: assistant.consciousness
                };
            }
            res.json({ assistants: assistantData, syncHistory: this.syncHistory.slice(-10) });
        });

        // Trigger manual sync
        router.post('/sync', async (req, res) => {
            try {
                const result = await this.syncSkillsToAssistants();
                res.json(result);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Get specific assistant skills
        router.get('/:type/skills', (req, res) => {
            const assistant = this.assistants.get(req.params.type);
            if (!assistant) {
                return res.status(404).json({ error: 'Assistant not found' });
            }

            res.json({
                assistant: req.params.type,
                skills: Array.from(assistant.skills.values()),
                count: assistant.skills.size
            });
        });

        // Reload assistant config
        router.post('/:type/reload', async (req, res) => {
            try {
                const assistant = this.assistants.get(req.params.type);
                if (!assistant) {
                    return res.status(404).json({ error: 'Assistant not found' });
                }

                await this.reloadAssistantConfig(assistant.config.filePath);
                res.json({ reloaded: true, skillCount: assistant.skills.size });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        return router;
    }

    async shutdown() {
        console.log(chalk.yellow('🛑 Shutting down AI Bridge...'));
        this.assistants.clear();
        this.skillMappings.clear();
        this.removeAllListeners();
    }
}

export { AIBridge };
