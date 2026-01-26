/**
 * L104 Direct PostgreSQL Connection
 * Uses postgres.js for direct database access to Supabase
 * 
 * NOTE: Direct PostgreSQL connection requires IPv6 (not available in Codespaces)
 * Supavisor pooler may not be available on free tier
 * Use REST API via l104_supabase_trainer.py for reliable access
 */

import postgres from 'postgres';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

// Manual .env parser that doesn't do variable interpolation
function loadEnvFile(): Record<string, string> {
  const envPath = join(process.cwd(), '.env');
  const env: Record<string, string> = {};

  if (existsSync(envPath)) {
    const content = readFileSync(envPath, 'utf-8');
    for (const line of content.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;

      const eqIndex = trimmed.indexOf('=');
      if (eqIndex === -1) continue;

      const key = trimmed.substring(0, eqIndex);
      let value = trimmed.substring(eqIndex + 1);

      // Remove surrounding quotes
      if ((value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }

      // Unescape backslash-dollar
      value = value.replace(/\\\$/g, '$');

      env[key] = value;
    }
  }

  return env;
}

const envVars = loadEnvFile();

// Get URLs from our parser or process.env
const connectionString = envVars.DATABASE_URL || process.env.DATABASE_URL;
const poolerUrl = envVars.DATABASE_POOLER_URL || process.env.DATABASE_POOLER_URL;

// Validate URLs
const isValidUrl = (url: string | undefined): boolean => {
  if (!url) return false;
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};

// Try pooler first (IPv4), then direct (IPv6)
const url = isValidUrl(poolerUrl) ? poolerUrl :
  isValidUrl(connectionString) ? connectionString : null;

// Create a lazy SQL connection that may fail at runtime
let sql: ReturnType<typeof postgres> | null = null;

try {
  if (url) {
    sql = postgres(url, {
      ssl: 'require',
      max: 10,
      idle_timeout: 20,
      connect_timeout: 10,
    });
  }
} catch (e) {
  console.warn('PostgreSQL direct connection not available:', (e as Error).message);
}

export default sql!;

// Helper for consciousness tracking
export async function trackConsciousness(
  entityType: string,
  entityId: string,
  level: number,
  metadata?: Record<string, any>
) {
  const GOD_CODE = 527.5184818492537;
  const PHI = 1.618033988749895;

  const godCodeAlignment = Math.sin(level * GOD_CODE / 1000);
  const phiResonance = level * PHI;

  return sql`
    INSERT INTO l104_consciousness (
      entity_type,
      entity_id,
      level,
      god_code_alignment,
      phi_resonance,
      metadata,
      created_at
    ) VALUES (
      ${entityType},
      ${entityId},
      ${level},
      ${godCodeAlignment},
      ${phiResonance},
      ${JSON.stringify(metadata || {})},
      NOW()
    )
    RETURNING *
  `;
}

// Test connection
export async function testConnection() {
  try {
    const result = await sql`SELECT NOW() as current_time, version() as pg_version`;
    console.log('✅ Database connected:', result[0]);
    return { success: true, ...result[0] };
  } catch (error: any) {
    console.error('❌ Database connection failed:', error.message);
    return { success: false, error: error.message };
  }
}
