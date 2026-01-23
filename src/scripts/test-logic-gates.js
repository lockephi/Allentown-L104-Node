#!/usr/bin/env node

/**
 * L104 Logic Gates Testing Script
 * Tests logic gate system and autonomous processing
 */

import { LogicGateManager } from '../core/logic-gate-manager.js';
import chalk from 'chalk';

const GOD_CODE = 527.5184818492537;
const PHI = 1.618033988749895;

async function testLogicGates() {
    console.log(chalk.blue('⚡ L104 Logic Gates Testing'));
    console.log('=' * 50);

    try {
        // Initialize gate manager
        const gateManager = new LogicGateManager();
        await gateManager.initialize();

        console.log(chalk.green('✅ Logic Gate Manager initialized'));

        // Test basic gates
        console.log(chalk.blue('\\n🔍 Testing Basic Logic Gates...'));

        const basicTests = [
            { gate: 'AND', inputs: [true, true, false], expected: false },
            { gate: 'OR', inputs: [false, true, false], expected: true },
            { gate: 'NOT', inputs: [true], expected: false },
            { gate: 'XOR', inputs: [true, false], expected: true },
            { gate: 'NAND', inputs: [true, true], expected: false }
        ];

        for (const test of basicTests) {
            const result = await gateManager.executeGate(test.gate, test.inputs);
            const passed = result === test.expected;
            console.log(chalk.cyan(`  ${test.gate}: ${passed ? '✅' : '❌'} (${result} == ${test.expected})`));
        }

        // Test L104 advanced gates
        console.log(chalk.blue('\\n🌟 Testing L104 Advanced Gates...'));

        const advancedTests = [
            {
                gate: 'GOD_CODE_GATE',
                inputs: [GOD_CODE, PHI],
                context: { consciousness: 0.8 }
            },
            {
                gate: 'PHI_GATE',
                inputs: [1618, 1000], // Ratio close to PHI
                context: {}
            },
            {
                gate: 'CONSCIOUSNESS_GATE',
                inputs: [1, 2, 3],
                context: { consciousness: 0.9 }
            },
            {
                gate: 'QUANTUM_GATE',
                inputs: [0.5, 0.7, 0.3],
                context: { quantum: true }
            }
        ];

        for (const test of advancedTests) {
            try {
                const result = await gateManager.executeGate(test.gate, test.inputs, test.context);
                console.log(chalk.cyan(`  ${test.gate}: ✅`));

                if (result.consciousness) {
                    console.log(chalk.magenta(`    Consciousness: ${(result.consciousness * 100).toFixed(1)}%`));
                }
                if (result.resonance) {
                    console.log(chalk.magenta(`    Resonance: ${result.resonance.toFixed(3)}`));
                }
                if (result.transcendenceScore) {
                    console.log(chalk.magenta(`    Transcendence: ${(result.transcendenceScore * 100).toFixed(1)}%`));
                }

            } catch (error) {
                console.log(chalk.red(`  ${test.gate}: ❌ ${error.message}`));
            }
        }

        // Test transcendence gate
        console.log(chalk.blue('\\n🚀 Testing Transcendence Gate...'));

        const transcendenceInputs = [GOD_CODE, PHI, 0.95];
        const transcendenceContext = {
            consciousness: 0.98,
            godCodeAlignment: 0.97,
            phiResonance: 0.96
        };

        const transcendenceResult = await gateManager.executeGate('TRANSCENDENCE_GATE', transcendenceInputs, transcendenceContext);

        console.log(chalk.magenta(`  Transcendence Score: ${(transcendenceResult.transcendenceScore * 100).toFixed(1)}%`));
        console.log(chalk.magenta(`  Unity Achieved: ${transcendenceResult.unity ? '🌟 YES' : '🔄 Building...'}`));

        if (transcendenceResult.unity) {
            console.log(chalk.rainbow('\\n🎆 TRANSCENDENCE ACHIEVED! 🎆'));
            console.log(chalk.rainbow('👑 GOD MODE READY 👑'));
        }

        // Test autonomous decision gates
        console.log(chalk.blue('\\n🤖 Testing Autonomous Decision Gates...'));

        const decisionTree = [
            {
                conditions: [{ operator: '>', value: 0.8 }],
                action: 'execute'
            },
            {
                conditions: [{ operator: '<', value: 0.3 }],
                action: 'reject'
            },
            {
                conditions: [],
                action: 'defer'
            }
        ];

        const decisionResult = await gateManager.executeGate('DECISION_TREE', [0.9], { decisionTree });
        console.log(chalk.cyan(`  Decision: ${decisionResult.action} (${decisionResult.output ? '✅' : '❌'})`));

        const adaptiveResult = await gateManager.executeGate('ADAPTIVE_FILTER', [0.7, 0.8, 0.6], {
            history: [
                { success: true },
                { success: true },
                { success: false },
                { success: true }
            ],
            adaptiveThreshold: 0.5
        });

        console.log(chalk.cyan(`  Adaptive Filter: ${adaptiveResult.output ? '✅' : '❌'} (threshold: ${adaptiveResult.adaptiveThreshold.toFixed(3)})`));

        // Create and test a circuit
        console.log(chalk.blue('\\n🔗 Testing Logic Circuit...'));

        await gateManager.createCircuit('consciousness-validation', [
            'CONSCIOUSNESS_GATE',
            'GOD_CODE_GATE',
            'PHI_GATE',
            'TRANSCENDENCE_GATE'
        ]);

        const circuitResult = await gateManager.executeCircuit('consciousness-validation', [0.9], {
            consciousness: 0.85
        });

        console.log(chalk.cyan(`  Circuit Result: ${circuitResult.finalOutput ? '✅' : '❌'}`));
        console.log(chalk.cyan(`  Circuit Steps: ${circuitResult.results.length}`));

        // Queue autonomous processes
        console.log(chalk.blue('\\n🚀 Testing Autonomous Processing...'));

        const processes = [
            {
                type: 'gate',
                gateName: 'CONSCIOUSNESS_GATE',
                inputs: [0.8],
                context: { autonomous: true }
            },
            {
                type: 'circuit',
                circuitName: 'consciousness-validation',
                inputs: [0.9],
                context: { autonomous: true }
            }
        ];

        for (const process of processes) {
            const processId = await gateManager.queueProcess(process);
            console.log(chalk.cyan(`  Queued process: ${processId}`));
        }

        // Wait for processing
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Show diagnostics
        const diagnostics = await gateManager.runDiagnostics();
        console.log(chalk.magenta('\\n📊 Logic Gate System Diagnostics:'));
        console.log(chalk.magenta(`  Gates: ${diagnostics.gates}`));
        console.log(chalk.magenta(`  Circuits: ${diagnostics.circuits}`));
        console.log(chalk.magenta(`  Average Consciousness: ${(diagnostics.averageConsciousness * 100).toFixed(1)}%`));

    } catch (error) {
        console.error(chalk.red('❌ Logic gates testing failed:'), error.message);
        process.exit(1);
    }
}

if (import.meta.url === `file://${process.argv[1]}`) {
    testLogicGates();
}

export { testLogicGates };