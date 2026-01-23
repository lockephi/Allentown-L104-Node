#!/usr/bin/env node

import { createClient } from '@supabase/supabase-js';
import { config } from 'dotenv';
import chalk from 'chalk';

config();

const { SUPABASE_URL, SUPABASE_ANON_KEY } = process.env;

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  console.error(chalk.red('Missing SUPABASE_URL or SUPABASE_ANON_KEY environment variables.'));
  process.exit(1);
}

const client = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function runSeed() {
  const now = new Date();
  const baseSamples = [
    { engine_name: 'TypeScript Engine', language: 'typescript', status: 'success', duration_ms: 620 },
    { engine_name: 'TypeScript Engine', language: 'typescript', status: 'success', duration_ms: 710 },
    { engine_name: 'Go Engine', language: 'go', status: 'success', duration_ms: 480 },
    { engine_name: 'Go Engine', language: 'go', status: 'error', duration_ms: 520, error_message: 'timeout' },
    { engine_name: 'Rust Engine', language: 'rust', status: 'success', duration_ms: 330 },
    { engine_name: 'Rust Engine', language: 'rust', status: 'success', duration_ms: 410 },
    { engine_name: 'Elixir Engine', language: 'elixir', status: 'success', duration_ms: 550 },
    { engine_name: 'Elixir Engine', language: 'elixir', status: 'error', duration_ms: 600, error_message: 'rate limit' }
  ];

  const rows = baseSamples.map(sample => ({
    event_type: 'engine_metric',
    source: 'seed-engine-metrics',
    data: sample,
    timestamp: new Date(now.getTime() - Math.floor(Math.random() * 4) * 60 * 60 * 1000).toISOString()
  }));

  const { error } = await client.from('l104_events').insert(rows);

  if (error) {
    console.error(chalk.red('Seed insert failed:'), error.message);
    process.exit(1);
  }

  console.log(chalk.green(`Inserted ${rows.length} engine_metric events.`));
}

runSeed();
