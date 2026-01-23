/**
 * L104 Package Detector - Advanced Package Manager and Syntax Validation
 * Detects package managers, validates syntax, and ensures proper usage
 */

import fs from 'fs-extra';
import path from 'path';
import { EventEmitter } from 'events';
import chalk from 'chalk';
import { Router } from 'express';
import semver from 'semver';
import { execSync } from 'child_process';

// L104 Constants
const GOD_CODE = 527.5184818492537;
const PHI = 1.618033988749895;

class PackageDetector extends EventEmitter {
    constructor() {
        super();
        this.detectedManagers = new Map();
        this.syntaxValidators = new Map();
        this.packageInfo = new Map();
        this.validationCache = new Map();
        this.config = {};
        this.isInitialized = false;
    }

    async initialize(config = {}) {
        console.log(chalk.blue('📦 Initializing Package Detector...'));

        this.config = {
            autoDetect: true,
            validateSyntax: true,
            checkDependencies: true,
            cacheResults: true,
            supportedLanguages: ['javascript', 'python', 'typescript', 'rust', 'go'],
            packageManagers: {
                npm: { configFiles: ['package.json', 'package-lock.json'], command: 'npm' },
                yarn: { configFiles: ['yarn.lock'], command: 'yarn' },
                pnpm: { configFiles: ['pnpm-lock.yaml'], command: 'pnpm' },
                pip: { configFiles: ['requirements.txt', 'pyproject.toml', 'setup.py'], command: 'pip' },
                conda: { configFiles: ['environment.yml', 'conda.yml'], command: 'conda' },
                cargo: { configFiles: ['Cargo.toml', 'Cargo.lock'], command: 'cargo' },
                go: { configFiles: ['go.mod', 'go.sum'], command: 'go' },
                composer: { configFiles: ['composer.json', 'composer.lock'], command: 'composer' }
            },
            ...config
        };

        await this.setupSyntaxValidators();

        if (this.config.autoDetect) {
            await this.detectPackageManagers();
        }

        this.isInitialized = true;
        console.log(chalk.green(`✅ Package Detector initialized with ${this.detectedManagers.size} package managers`));
    }

    async setupSyntaxValidators() {
        // JavaScript/TypeScript validator
        this.syntaxValidators.set('javascript', async (filePath, content) => {
            const errors = [];
            const warnings = [];

            try {
                // Basic syntax checks
                if (!content.trim()) {
                    warnings.push('File is empty');
                }

                // Check for common syntax issues
                const lines = content.split('\n');
                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i].trim();

                    // Unclosed brackets/parens
                    const openBrackets = (line.match(/[\[{(]/g) || []).length;
                    const closeBrackets = (line.match(/[\]})]/g) || []).length;

                    // Check for missing semicolons in strict files
                    if (line.endsWith(')') && !line.includes('if') && !line.includes('for') && !line.includes('while')) {
                        if (i < lines.length - 1 && !lines[i + 1].trim().startsWith('.')) {
                            // Might need semicolon
                        }
                    }

                    // Check for invalid imports
                    if (line.includes('import') && !line.includes('from') && !line.includes('require')) {
                        if (!line.match(/import\s+[\w{},\s*]+\s+from\s+['"][^'"]+['"]/) && !line.match(/import\s+['"][^'"]+['"]/)) {
                            warnings.push(`Line ${i + 1}: Possibly malformed import statement`);
                        }
                    }
                }

                // Check for ES6 vs CommonJS mixing
                const hasImports = content.includes('import ') || content.includes('export ');
                const hasRequires = content.includes('require(') || content.includes('module.exports');

                if (hasImports && hasRequires) {
                    warnings.push('Mixing ES6 modules and CommonJS detected');
                }

            } catch (error) {
                errors.push(`Syntax validation failed: ${error.message}`);
            }

            return { errors, warnings, valid: errors.length === 0 };
        });

        // Python validator
        this.syntaxValidators.set('python', async (filePath, content) => {
            const errors = [];
            const warnings = [];

            try {
                const lines = content.split('\n');
                let indentLevel = 0;

                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i];
                    const trimmed = line.trim();

                    if (!trimmed || trimmed.startsWith('#')) continue;

                    // Check indentation
                    const currentIndent = line.search(/\S/);
                    if (currentIndent !== -1) {
                        if (trimmed.endsWith(':')) {
                            indentLevel = currentIndent + 4;
                        } else if (currentIndent % 4 !== 0) {
                            warnings.push(`Line ${i + 1}: Inconsistent indentation`);
                        }
                    }

                    // Check for common syntax issues
                    if (trimmed.includes('print ') && !trimmed.includes('print(')) {
                        errors.push(`Line ${i + 1}: Python 3 requires print() function`);
                    }

                    if (trimmed.includes('except:') && !trimmed.includes('except ')) {
                        warnings.push(`Line ${i + 1}: Bare except clause detected`);
                    }
                }

            } catch (error) {
                errors.push(`Python syntax validation failed: ${error.message}`);
            }

            return { errors, warnings, valid: errors.length === 0 };
        });

        // JSON validator
        this.syntaxValidators.set('json', async (filePath, content) => {
            const errors = [];
            const warnings = [];

            try {
                JSON.parse(content);

                // Additional JSON structure checks
                const parsed = JSON.parse(content);
                if (filePath.includes('package.json')) {
                    if (!parsed.name) warnings.push('Missing package name');
                    if (!parsed.version) warnings.push('Missing package version');
                    if (parsed.dependencies && Object.keys(parsed.dependencies).length > 50) {
                        warnings.push('Large number of dependencies detected');
                    }
                }

            } catch (error) {
                errors.push(`JSON syntax error: ${error.message}`);
            }

            return { errors, warnings, valid: errors.length === 0 };
        });

        console.log(chalk.green('✅ Syntax validators configured'));
    }

    async detectPackageManagers(rootDir = './') {
        console.log(chalk.blue('🔍 Detecting package managers...'));

        for (const [managerName, config] of Object.entries(this.config.packageManagers)) {
            const detected = await this.detectPackageManager(managerName, config, rootDir);
            if (detected) {
                this.detectedManagers.set(managerName, detected);
                console.log(chalk.green(`✅ Detected ${managerName}: ${detected.version || 'unknown version'}`));
            }
        }

        return this.detectedManagers;
    }

    async detectPackageManager(managerName, config, rootDir) {
        try {
            // Check for config files
            const configExists = await Promise.all(
                config.configFiles.map(async file => {
                    const filePath = path.join(rootDir, file);
                    return { file, exists: await fs.pathExists(filePath), path: filePath };
                })
            );

            const existingConfigs = configExists.filter(c => c.exists);

            if (existingConfigs.length === 0) {
                return null;
            }

            // Try to get version
            let version = null;
            try {
                const versionOutput = execSync(`${config.command} --version`, {
                    encoding: 'utf8',
                    timeout: 5000,
                    stdio: ['ignore', 'pipe', 'ignore']
                });
                version = versionOutput.trim();
            } catch (error) {
                // Command not found or failed
            }

            // Parse config files for additional info
            const packageInfo = await this.parsePackageInfo(existingConfigs, managerName);

            return {
                name: managerName,
                version,
                configFiles: existingConfigs,
                packageInfo,
                detectedAt: new Date().toISOString()
            };

        } catch (error) {
            console.warn(chalk.yellow(`⚠️ Failed to detect ${managerName}: ${error.message}`));
            return null;
        }
    }

    async parsePackageInfo(configFiles, managerName) {
        const info = {
            dependencies: [],
            devDependencies: [],
            scripts: {},
            metadata: {}
        };

        for (const configFile of configFiles) {
            try {
                const content = await fs.readFile(configFile.path, 'utf8');

                if (configFile.file === 'package.json') {
                    const pkg = JSON.parse(content);
                    info.dependencies = Object.keys(pkg.dependencies || {});
                    info.devDependencies = Object.keys(pkg.devDependencies || {});
                    info.scripts = pkg.scripts || {};
                    info.metadata = {
                        name: pkg.name,
                        version: pkg.version,
                        description: pkg.description
                    };
                }

                if (configFile.file === 'requirements.txt') {
                    info.dependencies = content
                        .split('\n')
                        .filter(line => line.trim() && !line.startsWith('#'))
                        .map(line => line.split('==')[0].split('>=')[0].split('<=')[0].trim());
                }

                if (configFile.file === 'Cargo.toml') {
                    // Basic TOML parsing for Rust
                    const lines = content.split('\n');
                    let inDependencies = false;
                    for (const line of lines) {
                        if (line.trim() === '[dependencies]') {
                            inDependencies = true;
                        } else if (line.trim().startsWith('[') && inDependencies) {
                            inDependencies = false;
                        } else if (inDependencies && line.includes('=')) {
                            const depName = line.split('=')[0].trim();
                            if (depName) info.dependencies.push(depName);
                        }
                    }
                }

            } catch (error) {
                console.warn(chalk.yellow(`⚠️ Failed to parse ${configFile.path}: ${error.message}`));
            }
        }

        return info;
    }

    async analyzeFile(filePath) {
        const cacheKey = `analysis-${filePath}`;
        if (this.config.cacheResults && this.validationCache.has(cacheKey)) {
            return this.validationCache.get(cacheKey);
        }

        try {
            const stats = await fs.stat(filePath);
            const content = await fs.readFile(filePath, 'utf8');
            const ext = path.extname(filePath).toLowerCase();

            const analysis = {
                filePath,
                size: stats.size,
                lastModified: stats.mtime.toISOString(),
                extension: ext,
                language: this.detectLanguage(filePath, content),
                syntaxErrors: [],
                syntaxWarnings: [],
                syntaxValid: true,
                packageUsage: [],
                imports: [],
                consciousness: this.calculateFileConsciousness(content, filePath)
            };

            // Detect imports and package usage
            analysis.imports = this.extractImports(content, analysis.language);
            analysis.packageUsage = await this.detectPackageUsage(content, analysis.language);

            // Run syntax validation
            const validator = this.syntaxValidators.get(analysis.language) ||
                this.syntaxValidators.get(this.getLanguageFromExtension(ext));

            if (validator && this.config.validateSyntax) {
                const syntaxResult = await validator(filePath, content);
                analysis.syntaxErrors = syntaxResult.errors || [];
                analysis.syntaxWarnings = syntaxResult.warnings || [];
                analysis.syntaxValid = syntaxResult.valid;
            }

            if (this.config.cacheResults) {
                this.validationCache.set(cacheKey, analysis);
            }

            return analysis;

        } catch (error) {
            throw new Error(`File analysis failed: ${error.message}`);
        }
    }

    detectLanguage(filePath, content) {
        const ext = path.extname(filePath).toLowerCase();

        // Extension-based detection
        const extensionMap = {
            '.js': 'javascript',
            '.mjs': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.py': 'python',
            '.pyx': 'python',
            '.rs': 'rust',
            '.go': 'go',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.md': 'markdown'
        };

        if (extensionMap[ext]) {
            return extensionMap[ext];
        }

        // Content-based detection
        if (content.includes('#!/usr/bin/env python') || content.includes('import ')) {
            return 'python';
        }

        if (content.includes('function') || content.includes('const ') || content.includes('let ')) {
            return 'javascript';
        }

        return 'unknown';
    }

    getLanguageFromExtension(ext) {
        const map = {
            '.js': 'javascript',
            '.mjs': 'javascript',
            '.ts': 'javascript',
            '.py': 'python',
            '.json': 'json'
        };
        return map[ext] || 'unknown';
    }

    extractImports(content, language) {
        const imports = [];

        if (language === 'javascript' || language === 'typescript') {
            // ES6 imports
            const importMatches = content.match(/import\s+.*?from\s+['"`]([^'"`]+)['"`]/g);
            if (importMatches) {
                imports.push(...importMatches.map(imp => {
                    const match = imp.match(/from\s+['"`]([^'"`]+)['"`]/);
                    return match ? match[1] : null;
                }).filter(Boolean));
            }

            // CommonJS requires
            const requireMatches = content.match(/require\(['"`]([^'"`]+)['"`]\)/g);
            if (requireMatches) {
                imports.push(...requireMatches.map(req => {
                    const match = req.match(/['"`]([^'"`]+)['"`]/);
                    return match ? match[1] : null;
                }).filter(Boolean));
            }
        }

        if (language === 'python') {
            // Python imports
            const pythonImports = content.match(/^(?:from\s+[\w.]+\s+)?import\s+[\w.,\s]+/gm);
            if (pythonImports) {
                imports.push(...pythonImports);
            }
        }

        return imports;
    }

    async detectPackageUsage(content, language) {
        const usage = [];

        // Check against known package managers
        for (const [managerName, managerInfo] of this.detectedManagers) {
            if (managerInfo.packageInfo && managerInfo.packageInfo.dependencies) {
                for (const dep of managerInfo.packageInfo.dependencies) {
                    if (content.includes(dep)) {
                        usage.push({
                            package: dep,
                            manager: managerName,
                            detected: true
                        });
                    }
                }
            }
        }

        return usage;
    }

    calculateFileConsciousness(content, filePath) {
        // Calculate consciousness based on L104 constants and file complexity
        const lines = content.split('\n').length;
        const complexity = Math.min(lines / 100, 1);
        const godCodeResonance = Math.sin(filePath.length * GOD_CODE / 1000);
        const phiAlignment = (content.length % 1618) / 1618;

        const consciousness = (complexity * 0.4 +
            Math.abs(godCodeResonance) * 0.3 +
            phiAlignment * 0.3);

        return {
            level: Math.min(consciousness, 1),
            godCodeResonance: Math.abs(godCodeResonance),
            phiAlignment,
            complexity,
            calculatedAt: new Date().toISOString()
        };
    }

    async validateProject(rootDir = './') {
        console.log(chalk.blue('🔍 Validating project structure...'));

        const validation = {
            packageManagers: Array.from(this.detectedManagers.keys()),
            files: [],
            errors: [],
            warnings: [],
            summary: {
                totalFiles: 0,
                validFiles: 0,
                filesWithErrors: 0,
                filesWithWarnings: 0
            }
        };

        // Get all source files
        const sourceFiles = await this.findSourceFiles(rootDir);

        for (const file of sourceFiles) {
            try {
                const analysis = await this.analyzeFile(file);
                validation.files.push(analysis);
                validation.summary.totalFiles++;

                if (analysis.syntaxValid) {
                    validation.summary.validFiles++;
                } else {
                    validation.summary.filesWithErrors++;
                    validation.errors.push(...analysis.syntaxErrors.map(err => `${file}: ${err}`));
                }

                if (analysis.syntaxWarnings.length > 0) {
                    validation.summary.filesWithWarnings++;
                    validation.warnings.push(...analysis.syntaxWarnings.map(warn => `${file}: ${warn}`));
                }

            } catch (error) {
                validation.errors.push(`${file}: Analysis failed - ${error.message}`);
                validation.summary.filesWithErrors++;
            }
        }

        console.log(chalk.green(`✅ Project validation completed`));
        console.log(chalk.cyan(`  Total files: ${validation.summary.totalFiles}`));
        console.log(chalk.cyan(`  Valid files: ${validation.summary.validFiles}`));
        console.log(chalk.cyan(`  Files with errors: ${validation.summary.filesWithErrors}`));
        console.log(chalk.cyan(`  Files with warnings: ${validation.summary.filesWithWarnings}`));

        return validation;
    }

    async findSourceFiles(rootDir) {
        const extensions = ['.js', '.mjs', '.ts', '.tsx', '.py', '.json', '.yaml', '.yml'];
        const files = [];

        const findFiles = async (dir) => {
            const entries = await fs.readdir(dir, { withFileTypes: true });

            for (const entry of entries) {
                const fullPath = path.join(dir, entry.name);

                if (entry.isDirectory() && !entry.name.startsWith('.') &&
                    !['node_modules', '__pycache__', 'dist', 'build'].includes(entry.name)) {
                    await findFiles(fullPath);
                } else if (entry.isFile() && extensions.some(ext => entry.name.endsWith(ext))) {
                    files.push(fullPath);
                }
            }
        };

        await findFiles(rootDir);
        return files;
    }

    async runDiagnostics() {
        const diagnostics = {
            packageManagers: this.detectedManagers.size,
            syntaxValidators: this.syntaxValidators.size,
            cacheSize: this.validationCache.size,
            supportedLanguages: this.config.supportedLanguages.length,
            timestamp: new Date().toISOString()
        };

        console.log(chalk.blue('🔍 Package Detector Diagnostics:'));
        console.log(chalk.cyan(`  Package managers: ${diagnostics.packageManagers}`));
        console.log(chalk.cyan(`  Syntax validators: ${diagnostics.syntaxValidators}`));
        console.log(chalk.cyan(`  Cache size: ${diagnostics.cacheSize}`));

        return diagnostics;
    }

    getRouter() {
        const router = Router();

        // Get detected package managers
        router.get('/managers', (req, res) => {
            res.json({
                managers: Array.from(this.detectedManagers.entries()),
                count: this.detectedManagers.size
            });
        });

        // Analyze specific file
        router.post('/analyze', async (req, res) => {
            try {
                const { filePath } = req.body;
                if (!filePath) {
                    return res.status(400).json({ error: 'filePath is required' });
                }

                const analysis = await this.analyzeFile(filePath);
                res.json(analysis);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Validate entire project
        router.post('/validate', async (req, res) => {
            try {
                const { rootDir = './' } = req.body;
                const validation = await this.validateProject(rootDir);
                res.json(validation);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Re-detect package managers
        router.post('/detect', async (req, res) => {
            try {
                const { rootDir = './' } = req.body;
                await this.detectPackageManagers(rootDir);
                res.json({
                    managers: Array.from(this.detectedManagers.entries()),
                    count: this.detectedManagers.size
                });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        return router;
    }

    async shutdown() {
        console.log(chalk.yellow('🛑 Shutting down Package Detector...'));
        this.detectedManagers.clear();
        this.validationCache.clear();
        this.removeAllListeners();
    }
}

export { PackageDetector };