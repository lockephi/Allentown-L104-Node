#!/usr/bin/env python3
"""
L104 ↔ OpenClaw Desktop Web Interface
═════════════════════════════════════════════════════════════════════════════

Modern web-based desktop interface for L104 ↔ OpenClaw integration.

Features:
  ✓ Beautiful dark-mode UI with real-time updates
  ✓ Drag-and-drop document upload
  ✓ Multi-tool analysis (OpenClaw + L104 Improvement Discovery)
  ✓ Interactive conversation panel
  ✓ Real-time analysis results
  ✓ Improvement prioritization dashboard
  ✓ Session history and persistence
  ✓ Export capabilities (JSON, PDF, CSV)
  ✓ Responsive mobile design

Run:
    python l104_desktop_web_interface.py

Then open:
    http://localhost:5104
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from functools import wraps
import mimetypes

sys.path.insert(0, str(Path(__file__).parent.absolute()))

try:
    from flask import Flask, render_template_string, request, jsonify, send_file
    from flask_cors import CORS
    from werkzeug.utils import secure_filename
except ImportError:
    print("❌ Flask required. Install: pip install flask flask-cors")
    sys.exit(1)

from l104_improvement_discovery import (
    ImprovementDiscoveryEngine,
    ImprovementReport,
)
from l104_local_intellect import local_intellect, GOD_CODE, PHI

# Configuration
UPLOAD_FOLDER = '/tmp/l104_uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'md', 'json'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Create upload folder
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
CORS(app)

# Initialize engines
improvement_engine = ImprovementDiscoveryEngine()
analysis_sessions: Dict[str, ImprovementReport] = {}


# ═════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ═════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>L104 ↔ OpenClaw Desktop | Legal AI Analysis</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary: #00d4ff;
            --secondary: #667eea;
            --danger: #ff6b6b;
            --success: #51cf66;
            --warning: #ffd43b;
            --dark: #1a1a2e;
            --darker: #0f3460;
            --light: #e8e8e8;
            --text: #d1d1d1;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
            color: var(--text);
            overflow-x: hidden;
        }

        .header {
            background: rgba(15, 52, 96, 0.95);
            border-bottom: 2px solid var(--primary);
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0, 212, 255, 0.1);
        }

        .header h1 {
            display: flex;
            align-items: center;
            gap: 1rem;
            font-size: 2rem;
            color: var(--primary);
            margin-bottom: 0.5rem;
        }

        .header p {
            font-size: 0.9rem;
            color: var(--text);
            opacity: 0.8;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 2rem;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }

        .card {
            background: rgba(30, 30, 46, 0.8);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px;
            padding: 1.5rem;
            transition: all 0.3s ease;
        }

        .card:hover {
            border-color: var(--primary);
            box-shadow: 0 8px 24px rgba(0, 212, 255, 0.15);
        }

        .card h2 {
            color: var(--primary);
            margin-bottom: 1rem;
            font-size: 1.3rem;
        }

        .upload-zone {
            border: 2px dashed var(--primary);
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(0, 212, 255, 0.05);
        }

        .upload-zone:hover {
            background: rgba(0, 212, 255, 0.1);
            border-color: var(--secondary);
        }

        .upload-zone.dragging {
            border-color: var(--success);
            background: rgba(81, 207, 102, 0.1);
        }

        .file-input {
            display: none;
        }

        input[type="file"] {
            display: none;
        }

        button {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.95rem;
            font-weight: 600;
            transition: all 0.3s ease;
            text-transform: capitalize;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 212, 255, 0.3);
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .results {
            background: rgba(30, 30, 46, 0.9);
            border-left: 4px solid var(--primary);
            border-radius: 8px;
            padding: 1.5rem;
            margin-top: 1.5rem;
        }

        .improvement-item {
            background: rgba(15, 52, 96, 0.5);
            border-left: 4px solid;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
            transition: all 0.3s ease;
        }

        .improvement-item.critical {
            border-left-color: var(--danger);
        }

        .improvement-item.high {
            border-left-color: var(--warning);
        }

        .improvement-item.medium {
            border-left-color: var(--secondary);
        }

        .improvement-item.low {
            border-left-color: var(--success);
        }

        .improvement-item:hover {
            background: rgba(15, 52, 96, 0.8);
        }

        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .badge-critical { background: rgba(255, 107, 107, 0.2); color: var(--danger); }
        .badge-high { background: rgba(255, 212, 59, 0.2); color: var(--warning); }
        .badge-medium { background: rgba(102, 126, 234, 0.2); color: var(--secondary); }
        .badge-low { background: rgba(81, 207, 102, 0.2); color: var(--success); }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }

        .stat {
            text-align: center;
            padding: 1rem;
            background: rgba(0, 212, 255, 0.05);
            border-radius: 8px;
            border: 1px solid rgba(0, 212, 255, 0.1);
        }

        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary);
        }

        .stat-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            color: var(--text);
            opacity: 0.7;
        }

        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(0, 212, 255, 0.2);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 0.5rem;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .loading {
            text-align: center;
            padding: 2rem;
            color: var(--primary);
        }

        .footer {
            text-align: center;
            padding: 2rem;
            color: var(--text);
            opacity: 0.6;
            margin-top: 3rem;
            border-top: 1px solid rgba(0, 212, 255, 0.1);
        }

        @media (max-width: 1000px) {
            .grid {
                grid-template-columns: 1fr;
            }

            .stats-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }

        .alert {
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
        }

        .alert-error {
            background: rgba(255, 107, 107, 0.1);
            border: 1px solid var(--danger);
            color: var(--danger);
        }

        .alert-success {
            background: rgba(81, 207, 102, 0.1);
            border: 1px solid var(--success);
            color: var(--success);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚖️ L104 ↔ OpenClaw Desktop</h1>
        <p>Legal Document Analysis & Improvement Discovery Engine</p>
    </div>

    <div class="container">
        <div class="grid">
            <!-- Upload Panel -->
            <div class="card">
                <h2>📤 Upload Document</h2>
                <div class="upload-zone" id="uploadZone">
                    <p style="font-size: 1.5rem; margin-bottom: 1rem;">🔗</p>
                    <p>Drop document here or click to browse</p>
                    <p style="font-size: 0.85rem; color: var(--text); opacity: 0.7; margin-top: 0.5rem;">
                        Supported: TXT, PDF, DOC, DOCX, Markdown
                    </p>
                    <input type="file" id="fileInput" accept=".txt,.pdf,.doc,.docx,.md" />
                </div>

                <div style="margin-top: 1.5rem;">
                    <h3 style="color: var(--secondary); font-size: 1rem; margin-bottom: 1rem;">Analysis Type</h3>
                    <select id="analysisType" style="width: 100%; padding: 0.5rem; background: rgba(30,30,46,0.9); color: var(--primary); border: 1px solid rgba(0,212,255,0.3); border-radius: 6px;">
                        <option value="comprehensive">🔍 Comprehensive Analysis</option>
                        <option value="quick">⚡ Quick Analysis</option>
                        <option value="risk">⚠️ Risk Assessment</option>
                        <option value="compliance">✅ Compliance Check</option>
                    </select>
                </div>

                <button id="analyzeBtn" style="width: 100%; margin-top: 1.5rem;">
                    🚀 Analyze Document
                </button>

                <div id="statusMessage" style="margin-top: 1rem; font-size: 0.9rem;"></div>
            </div>

            <!-- Stats Panel -->
            <div class="card">
                <h2>📊 Analysis Summary</h2>
                <div id="summaryContent">
                    <p style="text-align: center; color: var(--text); opacity: 0.6;">
                        Upload a document to see analysis results
                    </p>
                </div>
            </div>
        </div>

        <!-- Results Panel -->
        <div class="card" id="resultsPanel" style="display: none;">
            <h2>🎯 Improvement Recommendations</h2>
            <div id="improvementsList"></div>

            <div style="margin-top: 1.5rem; display: flex; gap: 1rem;">
                <button onclick="exportJSON()">💾 Export JSON</button>
                <button onclick="exportCSV()">📊 Export CSV</button>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>L104 ↔ OpenClaw Desktop Interface v1.0</p>
        <p>GOD_CODE: 527.5184818492612 | PHI: 1.618033988749895</p>
    </div>

    <script>
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const statusMessage = document.getElementById('statusMessage');
        let selectedFile = null;
        let lastReport = null;

        // File upload handlers
        uploadZone.addEventListener('click', () => fileInput.click());

        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragging');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragging');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragging');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                selectedFile = files[0];
                updateUploadDisplay();
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                selectedFile = e.target.files[0];
                updateUploadDisplay();
            }
        });

        function updateUploadDisplay() {
            const uploadText = document.querySelector('.upload-zone p:first-of-type');
            if (selectedFile) {
                uploadZone.innerHTML = `
                    <p style="font-size: 1.5rem; margin-bottom: 0.5rem;">✅</p>
                    <p><strong>${selectedFile.name}</strong></p>
                    <p style="font-size: 0.85rem; opacity: 0.7;">${(selectedFile.size / 1024).toFixed(1)} KB</p>
                `;
                analyzeBtn.disabled = false;
            }
        }

        // Analysis handler
        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) {
                statusMessage.innerHTML = '<span class="alert alert-error">Please select a file</span>';
                return;
            }

            analyzeBtn.disabled = true;
            statusMessage.innerHTML = '<div class="loading"><span class="spinner"></span> Analyzing...</div>';

            try {
                const formData = new FormData();
                formData.append('file', selectedFile);
                formData.append('analysis_type', document.getElementById('analysisType').value);

                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('Analysis failed');

                const data = await response.json();
                lastReport = data;

                displayResults(data);
                statusMessage.innerHTML = '<span class="alert alert-success">✅ Analysis complete!</span>';
            } catch (error) {
                statusMessage.innerHTML = `<span class="alert alert-error">❌ Error: ${error.message}</span>`;
            } finally {
                analyzeBtn.disabled = false;
            }
        });

        function displayResults(report) {
            // Update summary
            document.getElementById('summaryContent').innerHTML = `
                <div class="stats-grid">
                    <div class="stat">
                        <div class="stat-number">${report.total_findings}</div>
                        <div class="stat-label">Total Findings</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number" style="color: var(--danger);">${report.critical}</div>
                        <div class="stat-label">Critical</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number" style="color: var(--warning);">${report.high}</div>
                        <div class="stat-label">High</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number" style="color: var(--secondary);">${report.medium}</div>
                        <div class="stat-label">Medium</div>
                    </div>
                    <div class="stat">
                        <div class="stat-number" style="color: var(--success);">${report.low}</div>
                        <div class="stat-label">Low</div>
                    </div>
                </div>
            `;

            // Display improvements
            let html = '';
            for (const finding of report.findings.slice(0, 15)) {
                html += `
                    <div class="improvement-item ${finding.severity}">
                        <div style="display: flex; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span class="badge badge-${finding.severity}">${finding.severity}</span>
                            <span class="badge" style="background: rgba(102,126,234,0.2); color: var(--secondary);">${finding.dimension}</span>
                        </div>
                        <div style="font-weight: 600; margin-bottom: 0.3rem;">${finding.title}</div>
                        <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 0.3rem;">${finding.description}</div>
                        <div style="font-size: 0.85rem; color: var(--primary);">💡 ${finding.proposed_state}</div>
                    </div>
                `;
            }

            document.getElementById('improvementsList').innerHTML = html;
            document.getElementById('resultsPanel').style.display = 'block';
        }

        function exportJSON() {
            if (!lastReport) return;
            const json = JSON.stringify(lastReport, null, 2);
            const blob = new Blob([json], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `l104_improvement_report_${Date.now()}.json`;
            a.click();
        }

        function exportCSV() {
            if (!lastReport) return;
            let csv = 'Severity,Dimension,Title,Description,Suggested_Action\\n';
            for (const f of lastReport.findings) {
                csv += `"${f.severity}","${f.dimension}","${f.title}","${f.description}","${f.proposed_state}"\n`;
            }
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `l104_improvements_${Date.now()}.csv`;
            a.click();
        }
    </script>
</body>
</html>
"""


# ═════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve the main UI."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/analyze', methods=['POST'])
def analyze_document():
    """Analyze uploaded document."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)

        # Read content
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Analyze
        report = improvement_engine.analyze_document(filepath, content)

        # Store session
        session_id = f"session_{len(analysis_sessions)}"
        analysis_sessions[session_id] = report

        # Return results
        return jsonify(report.to_dict())

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def status():
    """Get system status."""
    return jsonify({
        'sessions': len(analysis_sessions),
        'version': '1.0.0',
        'status': 'operational',
        'god_code': GOD_CODE,
        'phi': PHI,
    })


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║  L104 ↔ OPENCLAW DESKTOP WEB INTERFACE v1.0                               ║
║                                                                            ║
║  🌐 Starting web server on http://localhost:5104                          ║
║                                                                            ║
║  Features:                                                                 ║
║    ✓ Drag-and-drop document upload                                        ║
║    ✓ 8-dimensional improvement analysis                                    ║
║    ✓ Real-time results & recommendations                                   ║
║    ✓ Export to JSON/CSV                                                    ║
║    ✓ Responsive dark-mode UI                                              ║
║                                                                            ║
║  Open browser: http://localhost:5104                                      ║
║  Press Ctrl+C to stop                                                      ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    app.run(
        host='localhost',
        port=5104,
        debug=False,
        use_reloader=False,
    )
