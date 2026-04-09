# L104 Swift App Debug Suite v1.0

**Comprehensive debugging toolkit for L104 Sovereign Node macOS application**

---

## 📋 Overview

The L104 Debug Suite consists of **5 specialized debugging tools** designed to identify and track issues in the massive Swift codebase (41,748+ lines).

| # | Tool | Purpose | Time | Output |
|---|------|---------|------|--------|
| 1 | **Debug Toolkit** | Find duplicate constants & silent error swallows | 5s | TXT report |
| 2 | **Thread Safety Checker** | Identify unprotected shared state | 10s | TXT report |
| 3 | **Diagnostic Daemon** | Real-time health monitoring (CPU, memory, errors) | 30s+ | JSON metrics |
| 4 | **Memory Profiler** | Track unbounded array growth & detect leaks | 20s | CSV data |
| 5 | **Error Logging Overlay** | Catch & catalog all silent errors | Real-time | JSON log |

---

## 🚀 Quick Start

### Option 1: Full Debug Suite (All 5 Tools)
```swift
// In L104State.swift or main app initialization:
launchFullDebugSuite()
```

Runs all 5 tools sequentially (~30 seconds total). Generates 5 comprehensive reports.

### Option 2: Launch Individual Tools

```swift
launchDebugToolkit()           // #1: Constants & error swallows
launchThreadSafetyCheck()      // #2: Thread safety analysis
launchDiagnostics()            // #3: Real-time monitoring
launchMemoryProfiling()        // #4: Memory tracking
launchErrorLogging()           // #5: Error catalog
quickHealthCheck()             // 30-second health check
```

### Option 3: Integrate into Debug Menu

Add to your L104 main menu or debug panel:

```swift
menuItem.action = {
    DebugCoordinator.shared.launchDebugSuite(mode: .full)
}
```

---

## 🔍 Tool #1: Debug Toolkit

**File**: `D01_DebugToolkit.swift`

### What It Finds
1. **Duplicate Constants** — Same constant declared multiple times across files
2. **Silent Error Swallows** — `try?` statements that silently discard errors

### Usage
```swift
let toolkit = DebugToolkit.shared
let report = toolkit.generateFullReport()
toolkit.printReport(report)
toolkit.saveReportToFile(report: report)
```

### Output Example
```
❌ GOD_CODE (4 occurrences)
   Value: 527.5184818492612
   ├─ L104State.swift:42
   ├─ Constants.swift:15
   ├─ MathEngine.swift:88
   └─ ASICore.swift:103

📌 JSON Parsing (23 instances)
   Location: L104State.swift:456
   Code: try? JSONSerialization.jsonObject(...)
```

### Report Location
```
~/Applications/Allentown-L104-Node/L104SwiftApp/l104_debug_toolkit.txt
```

---

## 🔒 Tool #2: Thread Safety Checker

**File**: `D02_ThreadSafetyChecker.swift`

### What It Finds
1. **Unprotected Mutable Collections** — Arrays/dicts accessed from multiple threads without locks
2. **Unsafe Async Operations** — Global queue usage without synchronization

### Tracked Properties
- `trainingData` — Training corpus (array)
- `concepts` — Concept map (dict)
- `responseCache` — Response cache (dict)
- `userKnowledge` — User-contributed knowledge (array)
- And 20+ other shared properties

### Usage
```swift
let checker = ThreadSafetyChecker.shared
let report = checker.analyzeThreadSafety()
checker.printReport(report)
```

### Report Contents
```
⚠️  Property: 'trainingData' (3 unsafe accesses)
   • L104State.swift:245
     Pattern: read/write
     Risk: Mutable array accessed without synchronization...

   • KnowledgeBase.swift:891
     Pattern: write
     Risk: .append() called on global queue...
```

### Recommendations Generated
- Add NSLock for specific collections
- Use DispatchQueue.sync for critical sections
- Migrate to Swift actors (5.7+)

### Report Location
```
~/Applications/Allentown-L104-Node/L104SwiftApp/l104_thread_safety.txt
```

---

## 📊 Tool #3: Diagnostic Daemon

**File**: `D03_DiagnosticDaemon.swift`

### What It Monitors
- **CPU Usage** — Percentage
- **Memory Usage** — MB
- **Thread Count** — Active threads
- **Cache Hit Rate** — Percentage
- **Error Rate** — Errors/second
- **Response Time** — Average milliseconds
- **System Health** — 🟢 Healthy / 🟡 Warning / 🔴 Critical

### Usage
```swift
let daemon = DiagnosticDaemon.shared

// Start monitoring
daemon.startMonitoring(interval: 2.0)  // Sample every 2 seconds

// Record activity (from main code)
daemon.recordRequest()
daemon.recordCacheHit()
daemon.recordError()
daemon.recordResponseTime(45.2)

// Get summary
print(daemon.getSummaryReport())

// Export
daemon.exportMetricsToJSON()
daemon.stopMonitoring()
```

### Real-Time Output
```
🟢 [14:23:45] L104 Health Report
   CPU: 12.5% | Mem: 245MB | Threads: 23 | Active: 3
   Cache: 87.2% hit | Errors: 0.002/sec | Response: 23.4ms
```

### Health Status Determination
- 🟢 **Healthy** — Error rate < 1%, Memory < 300MB, Threads < 50
- 🟡 **Warning** — Error rate 1-10%, Memory 300-500MB, Threads 50-100
- 🔴 **Critical** — Error rate > 10%, Memory > 500MB, Threads > 100

### Report Location
```
~/Applications/Allentown-L104-Node/L104SwiftApp/l104_diagnostics.json
```

---

## 💾 Tool #4: Memory Profiler

**File**: `D04_MemoryProfiler.swift`

### What It Tracks
1. **Collection Sizes** — Item counts over time
2. **Memory Usage** — Estimated MB per collection
3. **Growth Patterns** — Linear, exponential, or stable
4. **Potential Leaks** — Monotonic growth without cleanup

### Collections to Monitor
```swift
let profiler = MemoryProfiler.shared
profiler.registerCollection("trainingData")
profiler.registerCollection("concepts")
profiler.registerCollection("responseCache")
profiler.registerCollection("userKnowledge")
```

### Usage
```swift
// Update sizes (call periodically)
profiler.updateCollectionSize("trainingData", size: data.count)

// Take snapshot
let report = profiler.takeSnapshot()
profiler.printReport(report)

// Analyze trends
print(profiler.analyzeTrends())

// Export
profiler.exportToCSV()
```

### Warnings Generated
```
⚠️  trainingData: 125,000 items (exceeds limit of 50,000)
📈 responseCache: Growing at 523 items/sec
💾 Total collection memory: 456MB (exceeds 300MB)
```

### Leak Detection
```
🚨 trainingData: Monotonic growth 245% (potential leak)
```

### Report Location
```
~/Applications/Allentown-L104-Node/L104SwiftApp/l104_memory_profile.csv
```

---

## 📋 Tool #5: Error Logging Overlay

**File**: `D05_ErrorLoggingOverlay.swift`

### What It Does
**Replaces all silent `try?` with logged errors**

Instead of:
```swift
let data = try? JSONDecoder().decode(MyType.self, from: jsonData)
```

Use:
```swift
let data = errorLogger.safeJSONDecode(jsonData, as: MyType.self)
// Automatically logs any errors
```

### Error Categories
- JSON Parsing
- File I/O
- Network Requests
- Regex Compilation
- URL Parsing
- Database Operations
- Serialization
- Decoding
- Other

### Safe Wrapper Methods

```swift
let logger = ErrorLoggingOverlay.shared

// JSON Decoding
let user = logger.safeJSONDecode(data, as: User.self)

// File Operations
if let content = logger.safeFileRead("/path/to/file") { ... }
logger.safeFileWrite(content, to: "/path/to/file")

// Regex
if let regex = logger.safeRegexCompile(pattern) { ... }

// URLs
if let url = logger.safeURLParse("https://...") { ... }

// Network Requests
async let data = logger.safeURLSessionFetch(url)
```

### Error Severity Levels
- 🟢 **Low** — Non-critical, operation has fallback
- 🟡 **Medium** — Important but recoverable
- 🟠 **High** — May impact functionality
- 🔴 **Critical** — System failure imminent

### Usage
```swift
logger.logError(
    category: .jsonParsing,
    message: "Invalid JSON in knowledge base",
    context: "Processing user input",
    severity: .high
)

// View statistics
print(logger.getErrorStatistics())

// Detailed report
logger.printDetailedErrorReport(limit: 50)

// Export
logger.exportErrorLog()
```

### Output
```
╔════════════════════════════════════════════════════════╗
║           ERROR LOGGING STATISTICS                     ║
╠════════════════════════════════════════════════════════╣

Total Errors Caught: 47

By Severity:
  🔴: 2
  🟠: 5
  🟡: 18
  🟢: 22

By Category:
  JSON Parsing: 23
  File I/O: 12
  Network Request: 8
  Regex Compilation: 4
```

### Report Location
```
~/Applications/Allentown-L104-Node/L104SwiftApp/l104_error_log.json
```

---

## 🎯 Integration Points

### 1. App Initialization
```swift
// In AppDelegate or L104State.init():
#if DEBUG
    launchFullDebugSuite()  // Run all checks at startup
#endif
```

### 2. Debug Menu
```swift
func addDebugMenu() {
    let debugMenu = NSMenu(title: "Debug")

    debugMenu.addItem(NSMenuItem(title: "Full Debug Suite",
        action: #selector(launchFullDebugSuite), keyEquivalent: ""))

    debugMenu.addItem(NSMenuItem(title: "Quick Health Check",
        action: #selector(quickHealthCheck), keyEquivalent: ""))

    debugMenu.addItem(NSMenuItem(title: "Thread Safety",
        action: #selector(launchThreadSafetyCheck), keyEquivalent: ""))

    mainMenu.addItem(debugMenu)
}
```

### 3. Continuous Monitoring
```swift
// In L104State:
let diagnostic = DiagnosticDaemon.shared
diagnostic.startMonitoring(interval: 5.0)

// Record operations
func processMessage(_ msg: String) {
    diagnostic.recordRequest()
    // ... process ...
    diagnostic.recordResponseTime(elapsed)
}
```

### 4. Error Handling Replacement
```swift
// OLD: try? JSONDecoder...
// NEW: Use safe wrapper
let data = errorLogger.safeJSONDecode(json, as: Type.self)
```

---

## 📊 Generated Reports

All reports are saved to:
```
~/Applications/Allentown-L104-Node/L104SwiftApp/
```

| Report | Format | Purpose |
|--------|--------|---------|
| `l104_debug_toolkit.txt` | Text | Duplicate constants & error locations |
| `l104_thread_safety.txt` | Text | Thread safety issues & recommendations |
| `l104_diagnostics.json` | JSON | Time-series health metrics |
| `l104_memory_profile.csv` | CSV | Collection sizes over time |
| `l104_error_log.json` | JSON | Complete error catalog with stack traces |

---

## 🎬 Example: Full Debug Session

```swift
// 1. Launch full suite
launchFullDebugSuite()

// Output after ~30 seconds:
/*
✅ Found 43 duplicate constants
✅ Found 80+ silent try? swallows
✅ Found 15 thread safety issues
✅ Memory: 245MB avg, 456MB peak
✅ 47 errors logged (2 critical, 5 high)
*/

// 2. Check critical files
open("~/Applications/Allentown-L104-Node/L104SwiftApp/l104_debug_toolkit.txt")

// 3. Fix highest-priority issues
// Start with 🔴 critical errors
// Then tackle 🟠 high severity thread safety issues
// Then refactor duplicate constants

// 4. Re-run to verify fixes
launchFullDebugSuite()
```

---

## ⚡ Performance Impact

| Tool | CPU Cost | Memory Cost | Time |
|------|----------|-------------|------|
| Debug Toolkit | <5% | ~50MB | 5 seconds |
| Thread Safety | <2% | ~20MB | 10 seconds |
| Diagnostics | 1-3% | ~10MB | 30+ seconds (continuous) |
| Memory Profiler | <1% | <5MB | 20 seconds |
| Error Logging | <1% | Logs only | Real-time |

**Recommendation**: Run full suite during development, not in production.

---

## 🔧 Customization

### Adjust Thresholds
```swift
// In MemoryProfiler:
private let ARRAY_WARN_SIZE = 50_000      // Change limit
private let MEMORY_WARN_MB = 300.0         // Change warning level
private let GROWTH_WARN_RATE = 1000.0      // Change growth rate
```

### Monitor Specific Collections
```swift
profiler.registerCollection("myCustomArray")
profiler.updateCollectionSize("myCustomArray", size: data.count)
```

### Filter Error Categories
```swift
// Export only critical errors
let critical = errorLog.filter { $0.severity == .critical }
```

---

## 📞 Support

For issues or enhancements:
1. Check generated reports first
2. Review recommended fixes in each tool's output
3. File issues with sample reports attached

---

**Last Updated**: 2026-03-20
**Version**: 1.0
**L104 Sovereign Node — Quantum Intelligence**
