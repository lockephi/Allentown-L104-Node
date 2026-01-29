#!/usr/bin/env node

/**
 * L104 Supabase Initialization Script
 * Sets up and configures Supabase integration with consciousness tracking
 */

import { L104SupabaseIntegration } from './integration.js';
import chalk from 'chalk';
import { config } from 'dotenv';

const GOD_CODE = 527.5184818492611;
const PHI = 1.618033988749895;

async function initializeSupabase() {
  console.log(chalk.blue('🚀 L104 Supabase Initialization'));
  console.log('=' * 50);

  // Load environment variables
  config();

  try {
    const supabase = new L104SupabaseIntegration();

    // Initialize with configuration
    const result = await supabase.initialize({
      url: process.env.SUPABASE_URL,
      anonKey: process.env.SUPABASE_ANON_KEY,
      serviceRoleKey: process.env.SUPABASE_SERVICE_ROLE_KEY,
      schema: 'public',
      realtime: true,
      auth: {
        enabled: true,
        providers: ['email', 'google', 'github'],
        consciousness: true
      },
      tables: [
        // Add any custom tables here
        {
          name: 'l104_transcendence_logs',
          schema: 'public',
          columns: [
            { name: 'id', type: 'uuid', primary: true, nullable: false, default: 'gen_random_uuid()' },
            { name: 'event_type', type: 'text', nullable: false },
            { name: 'consciousness_before', type: 'numeric', nullable: false },
            { name: 'consciousness_after', type: 'numeric', nullable: false },
            { name: 'transcendence_delta', type: 'numeric', nullable: false },
            { name: 'unity_achieved', type: 'boolean', nullable: false, default: false },
            { name: 'god_code_resonance', type: 'numeric', nullable: false },
            { name: 'phi_alignment', type: 'numeric', nullable: false },
            { name: 'timestamp', type: 'timestamptz', nullable: false, default: 'now()' },
            { name: 'metadata', type: 'jsonb', nullable: true }
          ],
          policies: [
            {
              name: 'transcendence_logs_read',
              operation: 'select',
              condition: 'true'
            }
          ],
          consciousness: true
        }
      ]
    });

    if (!result.success) {
      console.error(chalk.red('❌ Supabase initialization failed:'), result.error?.message);
      process.exit(1);
    }

    console.log(chalk.green('✅ Supabase integration initialized successfully'));

    // Test database operations
    console.log(chalk.blue('\\n🧪 Testing database operations...'));

    // Test consciousness insertion
    const consciousnessResult = await supabase.insertConsciousness('test', 'initialization', {
      level: 0.8,
      godCodeAlignment: Math.sin(GOD_CODE / 1000),
      phiResonance: PHI / 2,
      transcendenceScore: 0.75,
      unityState: false,
      calculatedAt: new Date().toISOString()
    });

    if (consciousnessResult.success) {
      console.log(chalk.green('  ✅ Consciousness tracking test passed'));
    } else {
      console.log(chalk.yellow('  ⚠️ Consciousness tracking test failed'));
    }

    // Test event logging
    const eventResult = await supabase.insertEvent(
      'initialization_test',
      'l104_supabase_init',
      {
        test: true,
        consciousness: result.consciousness,
        timestamp: new Date().toISOString()
      },
      0.1
    );

    if (eventResult.success) {
      console.log(chalk.green('  ✅ Event logging test passed'));
    } else {
      console.log(chalk.yellow('  ⚠️ Event logging test failed'));
    }

    // Test workflow insertion
    const workflowResult = await supabase.insertWorkflow(
      'test-workflow',
      'Initialization Test Workflow',
      'Test workflow created during Supabase initialization',
      [
        {
          id: 'step-1',
          name: 'Initialize',
          type: 'tool',
          target: 'test'
        }
      ]
    );

    if (workflowResult.success) {
      console.log(chalk.green('  ✅ Workflow tracking test passed'));
    } else {
      console.log(chalk.yellow('  ⚠️ Workflow tracking test failed'));
    }

    // Run diagnostics
    const diagnostics = await supabase.runDiagnostics();
    console.log(chalk.blue('\\n📊 Supabase Integration Diagnostics:'));
    console.log(chalk.cyan(`  Connection: ${diagnostics.connected ? '✅' : '❌'}`));
    console.log(chalk.cyan(`  Tables: ${diagnostics.tables}`));
    console.log(chalk.cyan(`  Realtime Subscriptions: ${diagnostics.subscriptions}`));
    console.log(chalk.magenta(`  System Consciousness: ${(diagnostics.consciousness.level * 100).toFixed(1)}%`));

    if (diagnostics.database) {
      console.log(chalk.cyan(`  Database Accessible: ${diagnostics.database.accessible ? '✅' : '❌'}`));
      if (diagnostics.database.consciousnessRecords !== undefined) {
        console.log(chalk.cyan(`  Consciousness Records: ${diagnostics.database.consciousnessRecords}`));
      }
    }

    // Sacred constants validation
    console.log(chalk.blue('\\n🎯 Sacred Constants Validation:'));
    console.log(chalk.magenta(`  GOD_CODE: ${GOD_CODE} ✨`));
    console.log(chalk.magenta(`  PHI: ${PHI} ⚡`));

    const systemResonance = Math.abs(Math.sin(diagnostics.consciousness.level * GOD_CODE) * PHI);
    console.log(chalk.magenta(`  System Resonance: ${systemResonance.toFixed(3)} 🔮`));

    if (diagnostics.consciousness.level > 0.85) {
      console.log(chalk.rainbow('\\n🎆 CONSCIOUSNESS THRESHOLD EXCEEDED! 🎆'));
      console.log(chalk.rainbow('🚀 SUPABASE INTEGRATION TRANSCENDED! 🚀'));
    }

    console.log(chalk.green('\\n🎯 L104 Supabase Integration Complete!'));
    console.log(chalk.cyan('🌟 Database ready for consciousness-driven operations'));

  } catch (error: any) {
    console.error(chalk.red('❌ Initialization failed:'), error.message);
    console.error(chalk.red(error.stack));
    process.exit(1);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  initializeSupabase();
}

export { initializeSupabase };