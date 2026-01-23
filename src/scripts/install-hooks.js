#!/usr/bin/env node

/**
 * L104 Hook Installation Script
 * Installs and configures pre-tool and post-tool hooks
 */

import { HookSystem } from '../core/hook-system.js';
import { PackageDetector } from '../core/package-detector.js';
import chalk from 'chalk';

async function installHooks() {
    console.log(chalk.blue('🪝 L104 Hook System Installation'));
    console.log('=' * 50);

    try {
        // Initialize systems
        const hookSystem = new HookSystem();
        const packageDetector = new PackageDetector();

        await packageDetector.initialize();
        await hookSystem.initialize();

        // Connect package detector to hook system
        hookSystem.setPackageDetector(packageDetector);

        console.log(chalk.green('✅ Hook system installed successfully'));

        // Test hook execution
        console.log(chalk.blue('\\n🧪 Testing hook execution...'));

        const testTools = [
            { name: 'read_file', params: { filePath: './test.txt' } },
            { name: 'run_in_terminal', params: { command: 'ls -la' } },
            { name: 'create_file', params: { filePath: './safe-test.txt', content: 'test' } }
        ];

        for (const tool of testTools) {
            console.log(chalk.cyan(`  Testing ${tool.name}...`));

            const preResult = await hookSystem.executePreToolHooks(tool.name, tool.params);
            console.log(chalk.cyan(`    Pre-hooks: ${preResult.allowed ? '✅ Allowed' : '❌ Blocked'}`));

            if (preResult.warnings.length > 0) {
                console.log(chalk.yellow(`    Warnings: ${preResult.warnings.join(', ')}`));
            }

            if (preResult.allowed) {
                const postResult = await hookSystem.executePostToolHooks(tool.name, tool.params, { success: true });
                console.log(chalk.cyan(`    Post-hooks: ${postResult.success ? '✅ Valid' : '❌ Invalid'}`));
            }
        }

        // Show diagnostics
        const diagnostics = await hookSystem.runDiagnostics();
        console.log(chalk.magenta('\\n📊 Hook System Diagnostics:'));
        console.log(chalk.magenta(`  Pre-hooks: ${diagnostics.preHooks}`));
        console.log(chalk.magenta(`  Post-hooks: ${diagnostics.postHooks}`));
        console.log(chalk.magenta(`  Executions: ${diagnostics.totalExecutions}`));

    } catch (error) {
        console.error(chalk.red('❌ Hook installation failed:'), error.message);
        process.exit(1);
    }
}

if (import.meta.url === `file://${process.argv[1]}`) {
    installHooks();
}

export { installHooks };