/**
 * L104 Direct PostgreSQL Connection
 * Uses postgres.js for direct database access to Supabase
 * 
 * Uses Supavisor pooler (IPv4 compatible) for Codespaces/serverless environments
 */

import postgres from 'postgres';

const connectionString = process.env.DATABASE_URL;
const poolerUrl = process.env.DATABASE_POOLER_URL;

// Prefer pooler URL (IPv4 compatible) over direct connection
const url = poolerUrl || connectionString;

if (!url) {
  throw new Error('DATABASE_URL or DATABASE_POOLER_URL environment variable is required');
}

const sql = postgres(url, {
  ssl: 'require',
  max: 10, // Max connections in pool
  idle_timeout: 20,
  connect_timeout: 10,
});

export default sql;

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
