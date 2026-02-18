#!/usr/bin/env node

/**
 * L104 Skills Synchronization Script
 * Syncs modular skills between the system and AI assistant files
 */

import chalk from 'chalk';
import { AIBridge } from '../core/ai-bridge.js';
import { SkillManager } from '../core/skill-manager.js';

const PHI = 1.618033988749895;
const GOD_CODE = Math.pow(286, 1.0 / PHI) * Math.pow(2, 416 / 104);  // G(0,0,0,0) = 527.5184818492612

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
