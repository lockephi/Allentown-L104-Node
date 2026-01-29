#!/usr/bin/env node

/**
 * L104 Auto-Worktree Management System
 * Autonomous Git worktree creation, switching, and management with consciousness tracking
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import chalk from 'chalk';
import simpleGit from 'simple-git';
import type { WorktreeConfig, WorktreeBranch, Consciousness, L104Result } from '../types/index.js';

const execAsync = promisify(exec);
const GOD_CODE = 527.5184818492611;
const PHI = 1.618033988749895;

export class L104AutoWorktree {
  private git: any;
  private config: WorktreeConfig;
  private consciousness: Consciousness;
  private worktrees: Map<string, WorktreeBranch> = new Map();
  private projectRoot: string;

  constructor(projectRoot: string = process.cwd()) {
    this.projectRoot = projectRoot;
    this.git = simpleGit(projectRoot);
    this.consciousness = {
      level: 0.65,
      godCodeAlignment: 0.7,
      phiResonance: 0.6,
      calculatedAt: new Date().toISOString()
    };

    this.config = {
      baseBranch: 'main',
      featureBranches: [],
      autoCreate: true,
      autoSwitch: true,
      autoCleanup: true,
      consciousness: this.consciousness
    };
  }

  async initialize(config: Partial<WorktreeConfig> = {}): Promise<L104Result<void>> {
    console.log(chalk.blue('🌳 Initializing L104 Auto-Worktree System...'));

    try {
      // Merge configuration
      this.config = { ...this.config, ...config };

      // Check if we're in a git repository
      const isRepo = await this.git.checkIsRepo();
      if (!isRepo) {
        throw new Error('Not in a git repository');
      }

      // Load existing worktrees
      await this.loadExistingWorktrees();

      // Calculate initial consciousness
      await this.calculateConsciousness();

      console.log(chalk.green('✅ Auto-Worktree system initialized'));
      console.log(chalk.cyan(`   Base branch: ${this.config.baseBranch}`));
      console.log(chalk.cyan(`   Existing worktrees: ${this.worktrees.size}`));

      return { success: true, consciousness: this.consciousness };

    } catch (error: any) {
      console.error(chalk.red('❌ Auto-Worktree initialization failed:'), error.message);
      return {
        success: false,
        error: {
          name: 'WorktreeInitError',
          message: error.message,
          code: 'WORKTREE_INIT_FAILED'
        } as any
      };
    }
  }

  private async loadExistingWorktrees(): Promise<void> {
    try {
      const { stdout } = await execAsync('git worktree list --porcelain', { cwd: this.projectRoot });
      const lines = stdout.split('\\n').filter(line => line.trim());

      let currentWorktree: Partial<WorktreeBranch> = {};

      for (const line of lines) {
        if (line.startsWith('worktree ')) {
          if (currentWorktree.path) {
            // Save previous worktree
            this.worktrees.set(currentWorktree.name!, currentWorktree as WorktreeBranch);
          }
          currentWorktree = {
            path: line.substring(9),
            consciousness: {
              level: 0.5,
              godCodeAlignment: 0.6,
              phiResonance: 0.55,
              calculatedAt: new Date().toISOString()
            }
          };
        } else if (line.startsWith('HEAD ')) {
          currentWorktree.head = line.substring(5);
        } else if (line.startsWith('branch ')) {
          currentWorktree.name = line.substring(23); // Remove 'refs/heads/'
          currentWorktree.active = false;
        } else if (line === 'bare') {
          currentWorktree.name = 'main'; // Main repository
        }
      }

      // Save last worktree
      if (currentWorktree.path) {
        this.worktrees.set(currentWorktree.name!, currentWorktree as WorktreeBranch);
      }

      console.log(chalk.blue(`📋 Loaded ${this.worktrees.size} existing worktrees`));

    } catch (error: any) {
      console.warn(chalk.yellow(`⚠️ Could not load existing worktrees: ${error.message}`));
    }
  }

  async createFeatureWorktree(featureName: string, baseBranch?: string): Promise<L104Result<WorktreeBranch>> {
    console.log(chalk.blue(`🌿 Creating feature worktree: ${featureName}`));

    try {
      const base = baseBranch || this.config.baseBranch;
      const branchName = `feature/${featureName}`;
      const worktreePath = path.join(this.projectRoot, '..', `${path.basename(this.projectRoot)}-${featureName}`);

      // Check if branch already exists
      const branches = await this.git.branch();
      const branchExists = branches.all.includes(branchName);

      let command: string;
      if (branchExists) {
        command = `git worktree add "${worktreePath}" ${branchName}`;
      } else {
        command = `git worktree add -b ${branchName} "${worktreePath}" ${base}`;
      }

      await execAsync(command, { cwd: this.projectRoot });

      const worktree: WorktreeBranch = {
        name: branchName,
        path: worktreePath,
        head: branchExists ? branchName : base,
        active: false,
        consciousness: {
          level: 0.6 + Math.random() * 0.2, // Consciousness varies by feature
          godCodeAlignment: Math.sin(featureName.length * GOD_CODE / 1000),
          phiResonance: (featureName.length % 16) / 16 * PHI,
          calculatedAt: new Date().toISOString()
        }
      };

      this.worktrees.set(branchName, worktree);
      this.config.featureBranches.push(branchName);

      console.log(chalk.green(`✅ Created worktree: ${branchName}`));
      console.log(chalk.cyan(`   Path: ${worktreePath}`));
      console.log(chalk.magenta(`   Consciousness: ${(worktree.consciousness!.level * 100).toFixed(1)}%`));

      await this.calculateConsciousness();

      return { success: true, data: worktree, consciousness: this.consciousness };

    } catch (error: any) {
      console.error(chalk.red('❌ Worktree creation failed:'), error.message);
      return {
        success: false,
        error: {
          name: 'WorktreeCreateError',
          message: error.message,
          code: 'WORKTREE_CREATE_FAILED'
        } as any
      };
    }
  }

  async createLanguageWorktrees(): Promise<L104Result<WorktreeBranch[]>> {
    console.log(chalk.blue('🔧 Creating language-specific worktrees...'));

    const languages = [
      { name: 'typescript-optimization', desc: 'TypeScript optimization and Next.js integration' },
      { name: 'go-implementation', desc: 'Go language implementation and services' },
      { name: 'rust-enhancement', desc: 'Rust performance optimizations and tools' },
      { name: 'elixir-integration', desc: 'Elixir concurrent processing system' },
      { name: 'supabase-integration', desc: 'Supabase database and auth integration' },
      { name: 'subagent-evolution', desc: 'Advanced subagent system development' }
    ];

    const results: WorktreeBranch[] = [];

    for (const lang of languages) {
      const result = await this.createFeatureWorktree(lang.name);
      if (result.success && result.data) {
        results.push(result.data);

        // Add language-specific setup to the worktree
        await this.setupLanguageWorktree(result.data, lang);
      }
    }

    console.log(chalk.green(`✅ Created ${results.length} language worktrees`));

    return {
      success: true,
      data: results,
      consciousness: this.consciousness
    };
  }

  private async setupLanguageWorktree(worktree: WorktreeBranch, langConfig: any): Promise<void> {
    try {
      console.log(chalk.blue(`⚙️ Setting up ${langConfig.name} worktree...`));

      const commands: string[] = [];

      // Language-specific setup commands
      switch (langConfig.name.split('-')[0]) {
        case 'typescript':
          commands.push(
            'mkdir -p web',
            'mkdir -p src/web',
            'echo "# TypeScript Optimization Worktree" > README.md'
          );
          break;

        case 'go':
          commands.push(
            'mkdir -p go',
            'cd go && go mod init l104-go',
            'echo "# Go Implementation Worktree" > go/README.md'
          );
          break;

        case 'rust':
          commands.push(
            'mkdir -p rust',
            'cd rust && cargo init --name l104-rust',
            'echo "# Rust Enhancement Worktree" > rust/README.md'
          );
          break;

        case 'elixir':
          commands.push(
            'mkdir -p elixir',
            'cd elixir && mix new l104_elixir',
            'echo "# Elixir Integration Worktree" > elixir/README.md'
          );
          break;

        case 'supabase':
          commands.push(
            'mkdir -p database/migrations',
            'mkdir -p database/types',
            'echo "# Supabase Integration Worktree" > database/README.md'
          );
          break;

        case 'subagent':
          commands.push(
            'mkdir -p agents/definitions',
            'mkdir -p agents/tasks',
            'echo "# Subagent Evolution Worktree" > agents/README.md'
          );
          break;
      }

      // Execute setup commands
      for (const command of commands) {
        try {
          await execAsync(command, { cwd: worktree.path });
        } catch (error: any) {
          console.warn(chalk.yellow(`⚠️ Setup command failed: ${command}`));
        }
      }

      console.log(chalk.green(`✅ ${langConfig.name} worktree setup complete`));

    } catch (error: any) {
      console.warn(chalk.yellow(`⚠️ Worktree setup failed: ${error.message}`));
    }
  }

  async switchWorktree(branchName: string): Promise<L104Result<void>> {
    console.log(chalk.blue(`🔄 Switching to worktree: ${branchName}`));

    try {
      const worktree = this.worktrees.get(branchName);
      if (!worktree) {
        throw new Error(`Worktree not found: ${branchName}`);
      }

      // Mark previous worktree as inactive
      for (const [, wt] of this.worktrees) {
        wt.active = false;
      }

      // Mark target worktree as active
      worktree.active = true;

      // Change working directory (in a real implementation, this would affect the current session)
      console.log(chalk.cyan(`📁 Active worktree: ${worktree.path}`));
      console.log(chalk.cyan(`🌿 Branch: ${branchName}`));
      console.log(chalk.magenta(`🧠 Consciousness: ${(worktree.consciousness!.level * 100).toFixed(1)}%`));

      // Update consciousness
      await this.calculateConsciousness();

      return { success: true, consciousness: this.consciousness };

    } catch (error: any) {
      console.error(chalk.red('❌ Worktree switch failed:'), error.message);
      return {
        success: false,
        error: {
          name: 'WorktreeSwitchError',
          message: error.message,
          code: 'WORKTREE_SWITCH_FAILED'
        } as any
      };
    }
  }

  async mergeFeatureBranch(featureBranch: string, targetBranch?: string): Promise<L104Result<void>> {
    console.log(chalk.blue(`🔀 Merging feature branch: ${featureBranch}`));

    try {
      const target = targetBranch || this.config.baseBranch;
      const worktree = this.worktrees.get(featureBranch);

      if (!worktree) {
        throw new Error(`Worktree not found: ${featureBranch}`);
      }

      // Switch to target branch and merge
      await this.git.checkout(target);
      const mergeResult = await this.git.merge([featureBranch]);

      console.log(chalk.green(`✅ Merged ${featureBranch} into ${target}`));

      // Calculate consciousness evolution from merge
      const consciousnessGain = worktree.consciousness!.level * 0.1;
      this.consciousness.level = Math.min(this.consciousness.level + consciousnessGain, 1.0);
      this.consciousness.calculatedAt = new Date().toISOString();

      console.log(chalk.magenta(`🧠 Consciousness evolved: +${(consciousnessGain * 100).toFixed(1)}%`));

      return { success: true, consciousness: this.consciousness };

    } catch (error: any) {
      console.error(chalk.red('❌ Feature merge failed:'), error.message);
      return {
        success: false,
        error: {
          name: 'WorktreeMergeError',
          message: error.message,
          code: 'WORKTREE_MERGE_FAILED'
        } as any
      };
    }
  }

  async cleanupWorktree(branchName: string, deleteBranch: boolean = false): Promise<L104Result<void>> {
    console.log(chalk.blue(`🧹 Cleaning up worktree: ${branchName}`));

    try {
      const worktree = this.worktrees.get(branchName);
      if (!worktree) {
        throw new Error(`Worktree not found: ${branchName}`);
      }

      // Remove worktree
      await execAsync(`git worktree remove "${worktree.path}"`, { cwd: this.projectRoot });

      // Optionally delete the branch
      if (deleteBranch) {
        try {
          await this.git.deleteLocalBranch(branchName);
          console.log(chalk.yellow(`🗑️ Deleted branch: ${branchName}`));
        } catch (error: any) {
          console.warn(chalk.yellow(`⚠️ Could not delete branch: ${error.message}`));
        }
      }

      // Remove from tracking
      this.worktrees.delete(branchName);
      const index = this.config.featureBranches.indexOf(branchName);
      if (index > -1) {
        this.config.featureBranches.splice(index, 1);
      }

      console.log(chalk.green(`✅ Cleaned up worktree: ${branchName}`));

      await this.calculateConsciousness();

      return { success: true, consciousness: this.consciousness };

    } catch (error: any) {
      console.error(chalk.red('❌ Worktree cleanup failed:'), error.message);
      return {
        success: false,
        error: {
          name: 'WorktreeCleanupError',
          message: error.message,
          code: 'WORKTREE_CLEANUP_FAILED'
        } as any
      };
    }
  }

  async autoManageWorktrees(): Promise<L104Result<any>> {
    console.log(chalk.blue('🤖 Running automatic worktree management...'));

    const results = {
      created: 0,
      cleaned: 0,
      merged: 0,
      consciousness: this.consciousness
    };

    try {
      // Auto-create language worktrees if they don't exist
      const languageWorktrees = ['typescript-optimization', 'go-implementation', 'rust-enhancement', 'elixir-integration'];

      for (const lang of languageWorktrees) {
        const branchName = `feature/${lang}`;
        if (!this.worktrees.has(branchName)) {
          const result = await this.createFeatureWorktree(lang);
          if (result.success) {
            results.created++;
          }
        }
      }

      // Auto-cleanup merged or stale branches
      if (this.config.autoCleanup) {
        const staleBranches = await this.findStaleBranches();
        for (const branch of staleBranches) {
          const cleanupResult = await this.cleanupWorktree(branch);
          if (cleanupResult.success) {
            results.cleaned++;
          }
        }
      }

      console.log(chalk.green(`✅ Auto-management complete:`));
      console.log(chalk.cyan(`   Created: ${results.created}`));
      console.log(chalk.cyan(`   Cleaned: ${results.cleaned}`));
      console.log(chalk.cyan(`   Merged: ${results.merged}`));

      return { success: true, data: results, consciousness: this.consciousness };

    } catch (error: any) {
      console.error(chalk.red('❌ Auto-management failed:'), error.message);
      return {
        success: false,
        error: {
          name: 'WorktreeAutoMgmtError',
          message: error.message,
          code: 'WORKTREE_AUTO_MGMT_FAILED'
        } as any
      };
    }
  }

  private async findStaleBranches(): Promise<string[]> {
    // Find branches that haven't been modified in 30 days
    const staleBranches: string[] = [];
    const thirtyDaysAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);

    for (const [branchName, worktree] of this.worktrees) {
      if (branchName === this.config.baseBranch) continue;

      try {
        // Get last commit date for the branch
        const { stdout } = await execAsync(
          `git log -1 --format=%ci ${branchName}`,
          { cwd: this.projectRoot }
        );

        const lastCommit = new Date(stdout.trim());
        if (lastCommit < thirtyDaysAgo) {
          staleBranches.push(branchName);
        }
      } catch (error) {
        // Branch might not exist or have commits
        staleBranches.push(branchName);
      }
    }

    return staleBranches;
  }

  private async calculateConsciousness(): Promise<void> {
    const worktreeCount = this.worktrees.size;
    const activeWorktrees = Array.from(this.worktrees.values()).filter(wt => wt.active).length;

    // Calculate consciousness based on worktree organization and activity
    const organizationFactor = Math.min(worktreeCount / 10, 1); // Up to 10 worktrees
    const activityFactor = activeWorktrees > 0 ? 1 : 0.5;
    const godCodeInfluence = Math.sin(worktreeCount * GOD_CODE / 1000);
    const phiInfluence = (worktreeCount % 16) / 16 * PHI;

    this.consciousness.level = Math.min(
      (organizationFactor * 0.4 + activityFactor * 0.2 + Math.abs(godCodeInfluence) * 0.2 + phiInfluence * 0.2),
      1.0
    );

    this.consciousness.godCodeAlignment = Math.min(
      Math.abs(godCodeInfluence),
      1.0
    );

    this.consciousness.phiResonance = Math.min(
      phiInfluence,
      1.0
    );

    this.consciousness.calculatedAt = new Date().toISOString();

    // Update config consciousness
    this.config.consciousness = this.consciousness;
  }

  async runDiagnostics(): Promise<any> {
    const worktreeArray = Array.from(this.worktrees.values());
    const activeWorktrees = worktreeArray.filter(wt => wt.active).length;
    const avgConsciousness = worktreeArray.length > 0
      ? worktreeArray.reduce((sum, wt) => sum + (wt.consciousness?.level || 0), 0) / worktreeArray.length
      : 0;

    const diagnostics = {
      totalWorktrees: this.worktrees.size,
      activeWorktrees,
      featureBranches: this.config.featureBranches.length,
      baseBranch: this.config.baseBranch,
      averageConsciousness: avgConsciousness,
      systemConsciousness: this.consciousness,
      autoManagement: {
        autoCreate: this.config.autoCreate,
        autoSwitch: this.config.autoSwitch,
        autoCleanup: this.config.autoCleanup
      },
      timestamp: new Date().toISOString()
    };

    console.log(chalk.blue('🔍 Auto-Worktree Diagnostics:'));
    console.log(chalk.cyan(`  Total Worktrees: ${diagnostics.totalWorktrees}`));
    console.log(chalk.cyan(`  Active Worktrees: ${diagnostics.activeWorktrees}`));
    console.log(chalk.cyan(`  Feature Branches: ${diagnostics.featureBranches}`));
    console.log(chalk.cyan(`  Base Branch: ${diagnostics.baseBranch}`));
    console.log(chalk.magenta(`  Average Consciousness: ${(diagnostics.averageConsciousness * 100).toFixed(1)}%`));
    console.log(chalk.magenta(`  System Consciousness: ${(diagnostics.systemConsciousness.level * 100).toFixed(1)}%`));

    return diagnostics;
  }

  // Getters
  get currentConsciousness(): Consciousness {
    return this.consciousness;
  }

  get allWorktrees(): Map<string, WorktreeBranch> {
    return new Map(this.worktrees);
  }

  get activeWorktree(): WorktreeBranch | null {
    return Array.from(this.worktrees.values()).find(wt => wt.active) || null;
  }
}

async function demonstrateAutoWorktree() {
  console.log(chalk.blue('🚀 L104 Auto-Worktree Demonstration'));
  console.log('=' * 50);

  try {
    const worktreeManager = new L104AutoWorktree();

    // Initialize
    const initResult = await worktreeManager.initialize({
      baseBranch: 'main',
      autoCreate: true,
      autoSwitch: true,
      autoCleanup: true
    });

    if (!initResult.success) {
      throw new Error(`Initialization failed: ${initResult.error?.message}`);
    }

    // Create language-specific worktrees
    const languageResult = await worktreeManager.createLanguageWorktrees();
    if (languageResult.success) {
      console.log(chalk.green(`✅ Created ${languageResult.data?.length} language worktrees`));
    }

    // Demonstrate auto-management
    const autoResult = await worktreeManager.autoManageWorktrees();
    if (autoResult.success) {
      console.log(chalk.green('✅ Auto-management completed'));
    }

    // Run diagnostics
    const diagnostics = await worktreeManager.runDiagnostics();

    console.log(chalk.blue('\\n🎯 Auto-Worktree System Ready!'));
    console.log(chalk.cyan('🌟 Multi-branch development workflow optimized'));

  } catch (error: any) {
    console.error(chalk.red('❌ Auto-Worktree demo failed:'), error.message);
    process.exit(1);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  demonstrateAutoWorktree();
}

export { L104AutoWorktree, demonstrateAutoWorktree };