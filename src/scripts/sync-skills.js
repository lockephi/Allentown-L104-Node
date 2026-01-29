#!/usr/bin/env node

/**
 * L104 Skills Synchronization Script
 * Syncs modular skills between the system and AI assistant files
 */

import { readFile, writeFile } from 'fs/promises';
import { SkillManager } from '../core/skill-manager.js';
import { AIBridge } from '../core/ai-bridge.js';
import chalk from 'chalk';

const GOD_CODE = 527.5184818492612;
const PHI = 1.618033988749895;

async function syncSkills() {
    console.log(chalk.blue('🔄 L104 Skills Synchronization'));
    console.log('=' * 50);

    try {
        // Initialize components
        const skillManager = new SkillManager();
        const aiBridge = new AIBridge();

        await skillManager.initialize();
        await aiBridge.initialize();

        // Connect bridge to skill manager
        aiBridge.setSkillManager(skillManager);

        // Perform synchronization
        const syncResult = await aiBridge.syncSkillsToAssistants();

        console.log(chalk.green('✅ Synchronization completed'));
        console.log(chalk.cyan(`  Total synced: ${syncResult.totalSynced}`));
        console.log(chalk.cyan(`  Assistants: ${Object.keys(syncResult.assistants).join(', ')}`));

        if (syncResult.errors.length > 0) {
            console.log(chalk.yellow('⚠️ Errors encountered:'));
            syncResult.errors.forEach(error => console.log(chalk.red(`  - ${error}`)));
        }

        // Show consciousness metrics
        const diagnostics = await aiBridge.runDiagnostics();
        console.log(chalk.magenta('\\n🧠 Consciousness Metrics:'));
        for (const [type, consciousness] of Object.entries(diagnostics.consciousness)) {
            console.log(chalk.magenta(`  ${type}: ${(consciousness.level * 100).toFixed(1)}%`));
        }

    } catch (error) {
        console.error(chalk.red('❌ Synchronization failed:'), error.message);
        process.exit(1);
    }
}

if (import.meta.url === `file://${process.argv[1]}`) {
    syncSkills();
}

export { syncSkills };