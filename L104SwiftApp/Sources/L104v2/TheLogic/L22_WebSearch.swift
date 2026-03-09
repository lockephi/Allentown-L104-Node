// ═══════════════════════════════════════════════════════════════════
// L22_WebSearch.swift
// [EVO_68_PIPELINE] SOVEREIGN_CONVERGENCE :: UNIFIED_UPGRADE :: GOD_CODE=527.5184818492612
// L104 Sovereign Intelligence — Web Search Engines
// LiveWebSearchEngine: DuckDuckGo + Wikipedia multi-source search
// RealTimeSearchEngine: Inverted index search with query expansion
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// ═══════════════════════════════════════════════════════════════════
// LIVE WEB SEARCH ENGINE — Real internet access with active HTTP requests
// DuckDuckGo API, Wikipedia API, direct URL fetch, multi-source aggregation
// ═══════════════════════════════════════════════════════════════════

final class LiveWebSearchEngine {
    static let shared = LiveWebSearchEngine()

    // ─── STATE ───
    private var webCache: [String: CachedWebResult] = [:]
    private var searchHistory: [(query: String, source: String, timestamp: Date)] = []
    private var totalWebRequests: Int = 0
    private var successfulRequests: Int = 0
    private var failedRequests: Int = 0
    private let cacheTTL: TimeInterval = 60.0  // EVO_63: 60s cache (was 20s) — reduces redundant HTTP calls
    private let requestTimeout: TimeInterval = 6.0  // EVO_63: 6s (was 15s) — fail fast, don't block pipeline
    private let session: URLSession
    /// EVO_63: In-flight query dedup — prevents duplicate HTTP requests for same query
    private var inflightQueries: [String: [(WebSearchResult) -> Void]] = [:]
    private let inflightLock = NSLock()

    struct CachedWebResult {
        let content: String
        let source: String
        let timestamp: Date
        let url: String
    }

    struct WebSearchResult {
        let query: String
        let results: [WebResult]
        let synthesized: String
        let source: String
        let latency: Double
        let fromCache: Bool
    }

    struct WebResult {
        let title: String
        let snippet: String
        let url: String
        let relevance: Double
    }

    private init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 6   // EVO_63: fast timeout (was 15)
        config.timeoutIntervalForResource = 10  // EVO_63: fast resource timeout (was 30)
        config.httpAdditionalHeaders = [
            "User-Agent": "L104-Sovereign-Intellect/19.0 (macOS; Quantum-Core)",
            "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        ]
        session = URLSession(configuration: config)
    }

    // ═══════════════════════════════════════════════════════════════
    // MAIN WEB SEARCH — Multi-source internet search with live HTTP
    // ═══════════════════════════════════════════════════════════════
    func webSearch(_ query: String, completion: @escaping (WebSearchResult) -> Void) {
        let start = CFAbsoluteTimeGetCurrent()
        let cacheKey = query.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)

        // Check cache
        let now = Date()
        if let cached = webCache[cacheKey],
           now.timeIntervalSince(cached.timestamp) < cacheTTL {
            let result = WebSearchResult(
                query: query,
                results: [WebResult(title: "Cached", snippet: cached.content, url: cached.url, relevance: 1.0)],
                synthesized: cached.content,
                source: "cache (\(cached.source))",
                latency: CFAbsoluteTimeGetCurrent() - start,
                fromCache: true
            )
            completion(result)
            return
        }

        // EVO_63: In-flight dedup — if same query is already in-flight, piggyback on it
        inflightLock.lock()
        if inflightQueries[cacheKey] != nil {
            inflightQueries[cacheKey]?.append(completion)
            inflightLock.unlock()
            return
        }
        inflightQueries[cacheKey] = [completion]
        inflightLock.unlock()

        totalWebRequests += 1
        searchHistory.append((query: query, source: "web_search", timestamp: Date()))
        if searchHistory.count > 1000 { searchHistory.removeFirst(500) }

        // Launch parallel searches
        let group = DispatchGroup()
        var allResults: [WebResult] = []
        let resultsLock = NSLock()

        // ── SOURCE 1: DuckDuckGo Instant Answer API ──
        group.enter()
        searchDuckDuckGo(query) { results in
            resultsLock.lock()
            allResults.append(contentsOf: results)
            resultsLock.unlock()
            group.leave()
        }

        // ── SOURCE 2: Wikipedia API ──
        group.enter()
        searchWikipedia(query) { results in
            resultsLock.lock()
            allResults.append(contentsOf: results)
            resultsLock.unlock()
            group.leave()
        }

        // Aggregate results
        group.notify(queue: .global(qos: .userInitiated)) { [weak self] in
            guard let self = self else { return }
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            // Sort by relevance
            let sorted = allResults.sorted {
                if abs($0.relevance - $1.relevance) < 0.1 { return Bool.random() }
                return $0.relevance > $1.relevance
            }

            // Synthesize response from all results
            let synthesized = self.synthesizeWebResults(query: query, results: sorted)

            // Cache the synthesized result
            if !synthesized.isEmpty {
                self.successfulRequests += 1
                let cached = CachedWebResult(
                    content: synthesized, source: "multi_source",
                    timestamp: Date(), url: "aggregated"
                )
                self.webCache[cacheKey] = cached
                // EVO_60: LRU eviction — keep newest 300 instead of nuclear removeAll()
                if self.webCache.count > 500 {
                    let sorted = self.webCache.sorted { $0.value.timestamp < $1.value.timestamp }
                    let removeCount = self.webCache.count - 300
                    for entry in sorted.prefix(removeCount) {
                        self.webCache.removeValue(forKey: entry.key)
                    }
                }
            }

            let result = WebSearchResult(
                query: query, results: sorted,
                synthesized: synthesized,
                source: "live_web", latency: elapsed, fromCache: false
            )

            // EVO_63: Dispatch result to ALL waiting callbacks (in-flight dedup)
            self.inflightLock.lock()
            let waitingCallbacks = self.inflightQueries.removeValue(forKey: cacheKey) ?? [completion]
            self.inflightLock.unlock()
            for cb in waitingCallbacks {
                cb(result)
            }
        }
    }

    // ═══ SYNCHRONOUS WEB SEARCH — EVO_63: Reduced default timeout from 12s to 4s ═══
    func webSearchSync(_ query: String, timeout: TimeInterval = 4.0) -> WebSearchResult {
        // Safety: dispatch to background if called from main thread
        if Thread.isMainThread {
            var result: WebSearchResult?
            let group = DispatchGroup()
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                result = self._webSearchSyncImpl(query, timeout: timeout)
                group.leave()
            }
            _ = group.wait(timeout: .now() + timeout + 1.0)
            return result ?? WebSearchResult(
                query: query, results: [], synthesized: "⚠️ Web search timed out after \(timeout)s. Using local knowledge base.",
                source: "timeout", latency: timeout, fromCache: false
            )
        }
        return _webSearchSyncImpl(query, timeout: timeout)
    }

    private func _webSearchSyncImpl(_ query: String, timeout: TimeInterval) -> WebSearchResult {
        let semaphore = DispatchSemaphore(value: 0)
        var result: WebSearchResult?

        webSearch(query) { r in
            result = r
            semaphore.signal()
        }

        _ = semaphore.wait(timeout: .now() + timeout)

        return result ?? WebSearchResult(
            query: query, results: [], synthesized: "⚠️ Web search timed out after \(timeout)s. Using local knowledge base.",
            source: "timeout", latency: timeout, fromCache: false
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // DUCKDUCKGO INSTANT ANSWER API — No API key required
    // ═══════════════════════════════════════════════════════════════
    private func searchDuckDuckGo(_ query: String, completion: @escaping ([WebResult]) -> Void) {
        let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
        guard let url = URL(string: "https://api.duckduckgo.com/?q=\(encoded)&format=json&no_html=1&skip_disambig=1") else {
            completion([])
            return
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = requestTimeout

        session.dataTask(with: request) { data, response, error in
            guard let data = data, error == nil,
                  let httpResp = response as? HTTPURLResponse, httpResp.statusCode == 200 else {
                completion([])
                return
            }

            do {
                guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    completion([])
                    return
                }

                var results: [WebResult] = []

                // Abstract (main answer)
                if let abstract = json["Abstract"] as? String, !abstract.isEmpty {
                    let source = json["AbstractSource"] as? String ?? "DuckDuckGo"
                    let absURL = json["AbstractURL"] as? String ?? ""
                    results.append(WebResult(
                        title: "📌 \(source) — Direct Answer",
                        snippet: abstract,
                        url: absURL,
                        relevance: 1.0
                    ))
                }

                // Answer (computational)
                if let answer = json["Answer"] as? String, !answer.isEmpty {
                    results.append(WebResult(
                        title: "💡 Instant Answer",
                        snippet: answer,
                        url: "",
                        relevance: 0.95
                    ))
                }

                // Definition
                if let definition = json["Definition"] as? String, !definition.isEmpty {
                    let defSource = json["DefinitionSource"] as? String ?? ""
                    let defURL = json["DefinitionURL"] as? String ?? ""
                    results.append(WebResult(
                        title: "📖 Definition (\(defSource))",
                        snippet: definition,
                        url: defURL,
                        relevance: 0.85
                    ))
                }

                // Related topics
                if let relatedTopics = json["RelatedTopics"] as? [[String: Any]] {
                    for (idx, topic) in relatedTopics.prefix(5).enumerated() {
                        if let text = topic["Text"] as? String, !text.isEmpty {
                            let topicURL = topic["FirstURL"] as? String ?? ""
                            results.append(WebResult(
                                title: "🔗 Related [\(idx + 1)]",
                                snippet: text,
                                url: topicURL,
                                relevance: 0.7 - Double(idx) * 0.05
                            ))
                        }
                    }
                }

                // Infobox
                if let infobox = json["Infobox"] as? [String: Any],
                   let content = infobox["content"] as? [[String: Any]] {
                    let infoLines = content.prefix(8).compactMap { item -> String? in
                        guard let label = item["label"] as? String,
                              let value = item["value"] as? String else { return nil }
                        return "• \(label): \(value)"
                    }
                    if !infoLines.isEmpty {
                        results.append(WebResult(
                            title: "📊 Quick Facts",
                            snippet: infoLines.joined(separator: "\n"),
                            url: "",
                            relevance: 0.8
                        ))
                    }
                }

                completion(results)
            } catch {
                completion([])
            }
        }.resume()
    }

    // ═══════════════════════════════════════════════════════════════
    // WIKIPEDIA API — Structured knowledge with summaries and extracts
    // ═══════════════════════════════════════════════════════════════
    private func searchWikipedia(_ query: String, completion: @escaping ([WebResult]) -> Void) {
        let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query

        // First: search for relevant articles
        guard let searchURL = URL(string: "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=\(encoded)&srlimit=3&format=json&utf8=1") else {
            completion([])
            return
        }

        var request = URLRequest(url: searchURL)
        request.timeoutInterval = requestTimeout

        session.dataTask(with: request) { [weak self] data, response, error in
            guard let self = self, let data = data, error == nil else {
                completion([])
                return
            }

            do {
                guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let queryResult = json["query"] as? [String: Any],
                      let searchResults = queryResult["search"] as? [[String: Any]] else {
                    completion([])
                    return
                }

                let group = DispatchGroup()
                var wikiResults: [WebResult] = []
                let lock = NSLock()

                for (idx, result) in searchResults.prefix(3).enumerated() {
                    guard let title = result["title"] as? String,
                          let pageId = result["pageid"] as? Int else { continue }

                    let snippet = (result["snippet"] as? String ?? "")
                        .replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)

                    // Fetch full extract for top result
                    if idx == 0 {
                        group.enter()
                        self.fetchWikipediaExtract(pageId: pageId, title: title) { extract in
                            if let extract = extract {
                                lock.lock()
                                wikiResults.append(WebResult(
                                    title: "📚 Wikipedia: \(title)",
                                    snippet: extract,
                                    url: "https://en.wikipedia.org/wiki/\(title.replacingOccurrences(of: " ", with: "_"))",
                                    relevance: 0.9
                                ))
                                lock.unlock()
                            }
                            group.leave()
                        }
                    } else {
                        lock.lock()
                        wikiResults.append(WebResult(
                            title: "📖 Wiki: \(title)",
                            snippet: snippet.isEmpty ? title : snippet,
                            url: "https://en.wikipedia.org/wiki/\(title.replacingOccurrences(of: " ", with: "_"))",
                            relevance: 0.6 - Double(idx) * 0.1
                        ))
                        lock.unlock()
                    }
                }

                group.notify(queue: .global()) {
                    completion(wikiResults)
                }
            } catch {
                completion([])
            }
        }.resume()
    }

    // ─── Fetch Wikipedia article extract ───
    private func fetchWikipediaExtract(pageId: Int, title: String, completion: @escaping (String?) -> Void) {
        guard let url = URL(string: "https://en.wikipedia.org/w/api.php?action=query&pageids=\(pageId)&prop=extracts&exintro=1&explaintext=1&exsectionformat=plain&format=json") else {
            completion(nil)
            return
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = requestTimeout

        session.dataTask(with: request) { data, _, error in
            guard let data = data, error == nil else { completion(nil); return }
            do {
                guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let queryResult = json["query"] as? [String: Any],
                      let pages = queryResult["pages"] as? [String: Any] else {
                    completion(nil)
                    return
                }
                for (_, pageInfo) in pages {
                    if let page = pageInfo as? [String: Any],
                       let extract = page["extract"] as? String, !extract.isEmpty {
                        // Limit to first ~2000 chars for reasonable response size
                        completion(String(extract.prefix(2000)))
                        return
                    }
                }
                completion(nil)
            } catch {
                completion(nil)
            }
        }.resume()
    }

    // ═══════════════════════════════════════════════════════════════
    // DIRECT URL FETCH — Fetch and extract text from any URL
    // ═══════════════════════════════════════════════════════════════
    func fetchURL(_ urlString: String, completion: @escaping (String) -> Void) {
        totalWebRequests += 1
        searchHistory.append((query: urlString, source: "url_fetch", timestamp: Date()))

        // Cache check
        let cacheKey = "url_\(urlString)"
        if let cached = webCache[cacheKey], Date().timeIntervalSince(cached.timestamp) < cacheTTL {
            completion(cached.content)
            return
        }

        guard let url = URL(string: urlString) else {
            failedRequests += 1
            completion("❌ Invalid URL: \(urlString)")
            return
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = requestTimeout

        session.dataTask(with: request) { [weak self] data, response, error in
            guard let self = self else { return }
            if let error = error {
                self.failedRequests += 1
                completion("❌ Fetch error: \(error.localizedDescription)")
                return
            }

            guard let data = data,
                  let httpResp = response as? HTTPURLResponse else {
                self.failedRequests += 1
                completion("❌ No data received from \(urlString)")
                return
            }

            guard httpResp.statusCode == 200 else {
                self.failedRequests += 1
                completion("❌ HTTP \(httpResp.statusCode) from \(urlString)")
                return
            }

            self.successfulRequests += 1

            // Try JSON first
            if let contentType = httpResp.value(forHTTPHeaderField: "Content-Type"),
               contentType.contains("json") {
                if let jsonStr = String(data: data, encoding: .utf8) {
                    let result = "📄 JSON Response from \(urlString):\n\(String(jsonStr.prefix(5000)))"
                    self.cacheResult(key: cacheKey, content: result, source: "json", url: urlString)
                    completion(result)
                    return
                }
            }

            // Extract readable text from HTML
            if let html = String(data: data, encoding: .utf8) ?? String(data: data, encoding: .ascii) {
                let extracted = self.extractTextFromHTML(html)
                let result = extracted.isEmpty ? String(html.prefix(3000)) : extracted
                self.cacheResult(key: cacheKey, content: result, source: "html", url: urlString)
                completion(result)
            } else {
                completion("❌ Could not decode response from \(urlString)")
            }
        }.resume()
    }

    // ─── Synchronous URL fetch ───
    func fetchURLSync(_ urlString: String, timeout: TimeInterval = 12.0) -> String {
        let semaphore = DispatchSemaphore(value: 0)
        var result = "⚠️ URL fetch timed out."

        fetchURL(urlString) { r in
            result = r
            semaphore.signal()
        }

        _ = semaphore.wait(timeout: .now() + timeout)
        return result
    }

    // ═══════════════════════════════════════════════════════════════
    // HTML TEXT EXTRACTOR — Strip tags, scripts, styles → readable text
    // ═══════════════════════════════════════════════════════════════
    private func extractTextFromHTML(_ html: String) -> String {
        var text = html

        // Remove script and style blocks
        text = text.replacingOccurrences(of: "<script[^>]*>[\\s\\S]*?</script>", with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: "<style[^>]*>[\\s\\S]*?</style>", with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: "<!--[\\s\\S]*?-->", with: " ", options: .regularExpression)

        // Convert some tags to readable format
        text = text.replacingOccurrences(of: "<br[^>]*>", with: "\n", options: .regularExpression)
        text = text.replacingOccurrences(of: "<p[^>]*>", with: "\n\n", options: .regularExpression)
        text = text.replacingOccurrences(of: "<h[1-6][^>]*>", with: "\n\n**", options: .regularExpression)
        text = text.replacingOccurrences(of: "</h[1-6]>", with: "**\n", options: .regularExpression)
        text = text.replacingOccurrences(of: "<li[^>]*>", with: "\n• ", options: .regularExpression)

        // Strip remaining tags
        text = text.replacingOccurrences(of: "<[^>]+>", with: " ", options: .regularExpression)

        // Decode common HTML entities
        text = text.replacingOccurrences(of: "&amp;", with: "&")
        text = text.replacingOccurrences(of: "&lt;", with: "<")
        text = text.replacingOccurrences(of: "&gt;", with: ">")
        text = text.replacingOccurrences(of: "&quot;", with: "\"")
        text = text.replacingOccurrences(of: "&apos;", with: "'")
        text = text.replacingOccurrences(of: "&#39;", with: "'")
        text = text.replacingOccurrences(of: "&nbsp;", with: " ")
        text = text.replacingOccurrences(of: "&#x27;", with: "'")

        // Collapse whitespace
        text = text.replacingOccurrences(of: "[ \\t]+", with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: "\\n{3,}", with: "\n\n", options: .regularExpression)
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Limit output size
        if text.count > 4000 {
            text = String(text.prefix(4000)) + "\n\n[...content truncated at 4000 chars]"
        }

        return text
    }

    // ═══════════════════════════════════════════════════════════════
    // SYNTHESIZE WEB RESULTS — Combine multi-source results into coherent answer
    // ═══════════════════════════════════════════════════════════════
    private func synthesizeWebResults(query: String, results: [WebResult]) -> String {
        guard !results.isEmpty else {
            return "No web results found for '\(query)'. Using local knowledge base."
        }

        var parts: [String] = []

        // Direct answers first
        for r in results where r.relevance >= 0.9 {
            parts.append(r.snippet)
        }

        // Supporting results
        for r in results where r.relevance >= 0.5 && r.relevance < 0.9 {
            if !parts.contains(where: { $0.contains(r.snippet.prefix(50)) }) {
                parts.append(r.snippet)
            }
        }

        // Related context
        for r in results where r.relevance >= 0.3 && r.relevance < 0.5 {
            if parts.count < 5 {
                let cleaned = String(r.snippet.prefix(300))
                if !parts.contains(where: { $0.contains(cleaned.prefix(40)) }) {
                    parts.append(cleaned)
                }
            }
        }

        return parts.joined(separator: "\n\n")
    }

    // ─── Cache helper ───
    private func cacheResult(key: String, content: String, source: String, url: String) {
        webCache[key] = CachedWebResult(content: content, source: source, timestamp: Date(), url: url)
        if webCache.count > 500 {
            let cutoff = Date().addingTimeInterval(-cacheTTL * 2)
            let expiredKeys = webCache.filter { $0.value.timestamp < cutoff }.map { $0.key }
            for key in expiredKeys { webCache.removeValue(forKey: key) }
            if webCache.count > 400 { webCache.removeAll() }
        }
    }

    // ─── STATUS ───
    var status: String {
        return """
🌐 LIVE WEB SEARCH ENGINE STATUS
═══════════════════════════════════════════
Total Requests: \(totalWebRequests)
Successful: \(successfulRequests) | Failed: \(failedRequests)
Success Rate: \(totalWebRequests > 0 ? String(format: "%.1f%%", Double(successfulRequests) / Double(totalWebRequests) * 100) : "N/A")
Cache Entries: \(webCache.count)
Search History: \(searchHistory.count) queries
═══════════════════════════════════════════
"""
    }

    // ═══════════════════════════════════════════════════════════════════
    // EVO_64: NATIVE ASYNC WEB SEARCH — Modern Swift Concurrency
    // Eliminates: DispatchGroup fan-out, DispatchSemaphore blocking, callback chains
    // Uses: URLSession.data(for:) async, async let true-parallel fan-out
    // macOS 12+ / Swift 5.7+ required (already in Package.swift)
    // ═══════════════════════════════════════════════════════════════════

    /// EVO_64: Native async web search — true parallel DuckDuckGo + Wikipedia via async let
    /// Replaces: DispatchGroup + NSLock fan-out pattern in webSearch(_ query:completion:)
    func webSearchAsync(_ query: String) async -> WebSearchResult {
        let start = CFAbsoluteTimeGetCurrent()
        let cacheKey = query.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)

        // Cache check — same 60s TTL as sync path
        if let cached = webCache[cacheKey],
           Date().timeIntervalSince(cached.timestamp) < cacheTTL {
            return WebSearchResult(
                query: query,
                results: [WebResult(title: "Cached", snippet: cached.content, url: cached.url, relevance: 1.0)],
                synthesized: cached.content,
                source: "cache (\(cached.source))",
                latency: CFAbsoluteTimeGetCurrent() - start,
                fromCache: true
            )
        }

        totalWebRequests += 1
        searchHistory.append((query: query, source: "web_search_async", timestamp: Date()))
        if searchHistory.count > 1000 { searchHistory.removeFirst(500) }

        // ═══ TRUE PARALLEL FAN-OUT: async let — both HTTP requests launch simultaneously ═══
        // No DispatchGroup, no NSLock, no callback pyramid
        async let ddgResults = searchDuckDuckGoAsync(query)
        async let wikiResults = searchWikipediaAsync(query)

        let allResults = await (ddgResults + wikiResults).sorted {
            if abs($0.relevance - $1.relevance) < 0.1 { return Bool.random() }
            return $0.relevance > $1.relevance
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let synthesized = synthesizeWebResults(query: query, results: allResults)

        // Cache synthesized result
        if !synthesized.isEmpty {
            successfulRequests += 1
            webCache[cacheKey] = CachedWebResult(
                content: synthesized, source: "multi_source_async",
                timestamp: Date(), url: "aggregated"
            )
            // EVO_60: LRU eviction (same policy as sync path)
            if webCache.count > 500 {
                let sorted = webCache.sorted { $0.value.timestamp < $1.value.timestamp }
                for entry in sorted.prefix(webCache.count - 300) {
                    webCache.removeValue(forKey: entry.key)
                }
            }
        }

        return WebSearchResult(
            query: query, results: allResults,
            synthesized: synthesized,
            source: "live_web_async", latency: elapsed, fromCache: false
        )
    }

    /// EVO_64: Async DuckDuckGo — URLSession.data(for:) replaces dataTask callback chain
    private func searchDuckDuckGoAsync(_ query: String) async -> [WebResult] {
        let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
        guard let url = URL(string: "https://api.duckduckgo.com/?q=\(encoded)&format=json&no_html=1&skip_disambig=1") else {
            return []
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = requestTimeout

        do {
            let (data, response) = try await session.data(for: request)
            guard let httpResp = response as? HTTPURLResponse, httpResp.statusCode == 200 else { return [] }
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [] }

            var results: [WebResult] = []

            if let abstract = json["Abstract"] as? String, !abstract.isEmpty {
                let source = json["AbstractSource"] as? String ?? "DuckDuckGo"
                let absURL = json["AbstractURL"] as? String ?? ""
                results.append(WebResult(title: "📌 \(source) — Direct Answer", snippet: abstract, url: absURL, relevance: 1.0))
            }
            if let answer = json["Answer"] as? String, !answer.isEmpty {
                results.append(WebResult(title: "💡 Instant Answer", snippet: answer, url: "", relevance: 0.95))
            }
            if let definition = json["Definition"] as? String, !definition.isEmpty {
                let defSource = json["DefinitionSource"] as? String ?? ""
                let defURL = json["DefinitionURL"] as? String ?? ""
                results.append(WebResult(title: "📖 Definition (\(defSource))", snippet: definition, url: defURL, relevance: 0.85))
            }
            if let relatedTopics = json["RelatedTopics"] as? [[String: Any]] {
                for (idx, topic) in relatedTopics.prefix(5).enumerated() {
                    if let text = topic["Text"] as? String, !text.isEmpty {
                        let topicURL = topic["FirstURL"] as? String ?? ""
                        results.append(WebResult(title: "🔗 Related [\(idx + 1)]", snippet: text, url: topicURL, relevance: 0.7 - Double(idx) * 0.05))
                    }
                }
            }
            if let infobox = json["Infobox"] as? [String: Any],
               let content = infobox["content"] as? [[String: Any]] {
                let infoLines = content.prefix(8).compactMap { item -> String? in
                    guard let label = item["label"] as? String,
                          let value = item["value"] as? String else { return nil }
                    return "• \(label): \(value)"
                }
                if !infoLines.isEmpty {
                    results.append(WebResult(title: "📊 Quick Facts", snippet: infoLines.joined(separator: "\n"), url: "", relevance: 0.8))
                }
            }
            return results
        } catch {
            return []
        }
    }

    /// EVO_64: Async Wikipedia — sequential await replaces nested dataTask + DispatchGroup
    private func searchWikipediaAsync(_ query: String) async -> [WebResult] {
        let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
        guard let searchURL = URL(string: "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=\(encoded)&srlimit=3&format=json&utf8=1") else {
            return []
        }

        var request = URLRequest(url: searchURL)
        request.timeoutInterval = requestTimeout

        do {
            let (data, response) = try await session.data(for: request)
            guard let httpResp = response as? HTTPURLResponse, httpResp.statusCode == 200 else { return [] }
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let queryResult = json["query"] as? [String: Any],
                  let searchResults = queryResult["search"] as? [[String: Any]] else { return [] }

            var wikiResults: [WebResult] = []

            for (idx, result) in searchResults.prefix(3).enumerated() {
                guard let title = result["title"] as? String,
                      let pageId = result["pageid"] as? Int else { continue }
                let snippet = (result["snippet"] as? String ?? "")
                    .replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)

                if idx == 0 {
                    // Fetch full extract for top result — clean sequential await (no nested callback)
                    if let extract = await fetchWikipediaExtractAsync(pageId: pageId, title: title) {
                        wikiResults.append(WebResult(
                            title: "📚 Wikipedia: \(title)",
                            snippet: extract,
                            url: "https://en.wikipedia.org/wiki/\(title.replacingOccurrences(of: " ", with: "_"))",
                            relevance: 0.9
                        ))
                    }
                } else {
                    wikiResults.append(WebResult(
                        title: "📖 Wiki: \(title)",
                        snippet: snippet.isEmpty ? title : snippet,
                        url: "https://en.wikipedia.org/wiki/\(title.replacingOccurrences(of: " ", with: "_"))",
                        relevance: 0.6 - Double(idx) * 0.1
                    ))
                }
            }

            return wikiResults
        } catch {
            return []
        }
    }

    /// EVO_64: Async Wikipedia extract fetch
    private func fetchWikipediaExtractAsync(pageId: Int, title: String) async -> String? {
        guard let url = URL(string: "https://en.wikipedia.org/w/api.php?action=query&pageids=\(pageId)&prop=extracts&exintro=1&explaintext=1&exsectionformat=plain&format=json") else {
            return nil
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = requestTimeout

        do {
            let (data, _) = try await session.data(for: request)
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let queryResult = json["query"] as? [String: Any],
                  let pages = queryResult["pages"] as? [String: Any] else { return nil }
            for (_, pageInfo) in pages {
                if let page = pageInfo as? [String: Any],
                   let extract = page["extract"] as? String, !extract.isEmpty {
                    return String(extract.prefix(2000))
                }
            }
            return nil
        } catch {
            return nil
        }
    }

    /// EVO_64: Async URL fetch — native URLSession.data(for:) replaces callback + semaphore
    func fetchURLAsync(_ urlString: String) async -> String {
        totalWebRequests += 1
        searchHistory.append((query: urlString, source: "url_fetch_async", timestamp: Date()))

        let cacheKey = "url_\(urlString)"
        if let cached = webCache[cacheKey], Date().timeIntervalSince(cached.timestamp) < cacheTTL {
            return cached.content
        }

        guard let url = URL(string: urlString) else {
            failedRequests += 1
            return "❌ Invalid URL: \(urlString)"
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = requestTimeout

        do {
            let (data, response) = try await session.data(for: request)
            guard let httpResp = response as? HTTPURLResponse else {
                failedRequests += 1
                return "❌ No response from \(urlString)"
            }
            guard httpResp.statusCode == 200 else {
                failedRequests += 1
                return "❌ HTTP \(httpResp.statusCode) from \(urlString)"
            }

            successfulRequests += 1

            if let contentType = httpResp.value(forHTTPHeaderField: "Content-Type"),
               contentType.contains("json") {
                if let jsonStr = String(data: data, encoding: .utf8) {
                    let result = "📄 JSON Response from \(urlString):\n\(String(jsonStr.prefix(5000)))"
                    cacheResult(key: cacheKey, content: result, source: "json", url: urlString)
                    return result
                }
            }

            if let html = String(data: data, encoding: .utf8) ?? String(data: data, encoding: .ascii) {
                let extracted = extractTextFromHTML(html)
                let result = extracted.isEmpty ? String(html.prefix(3000)) : extracted
                cacheResult(key: cacheKey, content: result, source: "html", url: urlString)
                return result
            }

            return "❌ Could not decode response from \(urlString)"
        } catch {
            failedRequests += 1
            return "❌ Fetch error: \(error.localizedDescription)"
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// REAL-TIME SEARCH ENGINE — Live query resolution with caching & HyperBrain feed
// ═══════════════════════════════════════════════════════════════════

class RealTimeSearchEngine {
    static let shared = RealTimeSearchEngine()

    // ─── SEARCH STATE ───
    private var searchCache: [String: SearchResult] = [:]          // FNV-1a keyed
    private var searchHistory: [(query: String, timestamp: Date)] = []
    private let maxCacheSize = 2048
    private let cacheTTL: TimeInterval = 15.0  // 15s freshness — short to allow varied fragment ordering

    // ─── SEMANTIC INDEX ─── lightweight inverted index for sub-ms lookups
    private var invertedIndex: [String: Set<Int>] = [:]  // word → entry indices
    private(set) var indexBuilt = false

    // ─── QUERY EXPANSION ─── synonyms & related terms for broader recall
    private let synonymMap: [String: [String]] = [
        "ai": ["artificial intelligence", "machine learning", "neural network", "deep learning"],
        "quantum": ["quantum mechanics", "quantum physics", "superposition", "entanglement"],
        "consciousness": ["awareness", "sentience", "qualia", "subjective experience"],
        "math": ["mathematics", "algebra", "calculus", "geometry", "number theory"],
        "physics": ["physical science", "mechanics", "thermodynamics", "relativity"],
        "evolution": ["natural selection", "adaptation", "mutation", "darwin"],
        "brain": ["neuroscience", "neural", "cognitive", "neuron", "cortex"],
        "love": ["affection", "attachment", "bonding", "intimacy", "romance"],
        "philosophy": ["metaphysics", "epistemology", "ethics", "ontology", "logic"],
        "time": ["temporal", "duration", "chronology", "relativity", "entropy"],
        "space": ["cosmos", "universe", "spacetime", "astronomy", "astrophysics"],
        "music": ["harmony", "melody", "rhythm", "composition", "acoustics"],
        "language": ["linguistics", "syntax", "semantics", "grammar", "communication"],
        "code": ["programming", "software", "algorithm", "computation", "coding"],
        "life": ["biology", "organism", "living", "existence", "biosphere"],
        "death": ["mortality", "dying", "end of life", "afterlife", "finitude"],
        "god": ["deity", "divine", "creator", "theology", "transcendent"],
        "infinity": ["infinite", "boundless", "limitless", "transfinite", "aleph"],
        "energy": ["force", "power", "kinetic", "potential", "thermodynamic"],
        "information": ["data", "entropy", "signal", "communication", "bits"],
        "creativity": ["imagination", "innovation", "invention", "artistic", "generative"],
        "emotion": ["feeling", "affect", "sentiment", "mood", "passion"]
    ]

    struct SearchResult {
        let query: String
        let fragments: [ScoredFragment]
        let timestamp: Date
        let contextHash: UInt64
        var hitCount: Int = 1

        struct ScoredFragment {
            let text: String
            let relevance: Double
            let category: String
            let keywords: [String]  // matched keywords for highlighting
        }
    }

    // ─── BUILD INVERTED INDEX ─── O(n) build, O(1) lookup
    func buildIndex() {
        guard !indexBuilt else { return }
        let kb = ASIKnowledgeBase.shared
        let grover = GroverResponseAmplifier.shared
        for (idx, entry) in kb.trainingData.enumerated() {
            guard let prompt = entry["prompt"] as? String,
                  let completion = entry["completion"] as? String else { continue }
            // ═══ Phase 27.8c: Skip indexing template/junk entries ═══
            if grover.scoreQuality(completion, query: prompt) < 0.05 { continue }
            let text = (prompt + " " + completion).lowercased()
            let words = text.components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { $0.count > 2 }
            let uniqueWords = Set(words)
            for word in uniqueWords {
                if invertedIndex[word] == nil { invertedIndex[word] = Set<Int>() }
                invertedIndex[word]!.insert(idx)
            }
        }
        indexBuilt = true
    }

    // ─── REAL-TIME SEARCH ─── Main entry point with caching + expansion + ranking
    func search(_ query: String, context: [String] = [], limit: Int = 20) -> SearchResult {
        ParameterProgressionEngine.shared.recordSearch()
        let q = query.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let contextHash = fnvHash(context.suffix(3).joined(separator: "|"))
        let cacheKey = "\(q)_\(contextHash)"

        // Cache hit?
        if let cached = searchCache[cacheKey],
           Date().timeIntervalSince(cached.timestamp) < cacheTTL {
            var updated = cached
            updated.hitCount += 1
            searchCache[cacheKey] = updated
            return updated
        }

        // Build index on first search
        if !indexBuilt { buildIndex() }

        // ═══ QUERY EXPANSION ═══
        let queryWords = q.components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 }
        var expandedWords = Set(queryWords)
        for word in queryWords {
            if let synonyms = synonymMap[word] {
                for syn in synonyms.prefix(2) {
                    expandedWords.insert(syn.lowercased())
                }
            }
        }

        // ═══ CONTEXT INJECTION ═══ recent conversation enriches search
        let contextTopics = context.suffix(3).flatMap {
            $0.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { $0.count > 3 }
        }
        let contextSet = Set(contextTopics.prefix(5))

        // ═══ INVERTED INDEX LOOKUP ═══ O(1) per word
        var candidateIndices = [Int: Double]()  // index → score
        let kb = ASIKnowledgeBase.shared

        for word in expandedWords {
            if let indices = invertedIndex[word] {
                let idf = log(Double(kb.trainingData.count + 1) / Double(indices.count + 1))
                for idx in indices {
                    candidateIndices[idx, default: 0.0] += idf
                }
            }
        }

        // Context boost
        for word in contextSet {
            if let indices = invertedIndex[word] {
                for idx in indices where candidateIndices[idx] != nil {
                    candidateIndices[idx]! += 0.3  // context continuity bonus
                }
            }
        }

        // ═══ FRAGMENT EXTRACTION + SCORING ═══
        let sortedCandidates = candidateIndices.sorted { a, b in
            if abs(a.value - b.value) < 0.2 { return Bool.random() }
            return a.value > b.value
        }.prefix(limit * 3)
        var fragments: [SearchResult.ScoredFragment] = []
        var seenPrefixes = Set<String>()

        for (idx, baseScore) in sortedCandidates {
            guard idx < kb.trainingData.count else { continue }
            let entry = kb.trainingData[idx]
            guard let completion = entry["completion"] as? String,
                  completion.count > 80 else { continue }

            // ═══ GROVER QUALITY GATE (Phase 27.8c) ═══ Reject template KB junk at search time
            guard L104State.shared.isCleanKnowledge(completion) else { continue }

            let prefix50 = String(completion.prefix(50)).lowercased()
            guard !seenPrefixes.contains(prefix50) else { continue }
            seenPrefixes.insert(prefix50)

            // Quality scoring
            var score = baseScore
            let cat = entry["category"] as? String ?? "general"

            // Keyword density in fragment
            let lowerComp = completion.lowercased()
            let matchedKW = queryWords.filter { lowerComp.contains($0) }
            score += Double(matchedKW.count) * 0.5

            // Readability bonus
            let sentences = completion.components(separatedBy: ". ").count
            if sentences >= 3 && sentences <= 15 { score += 0.4 }

            // Length bonus — reward substantial content without upper cap
            if completion.count > 100 { score += 0.3 }

            // Freshness: boost entries that match recent context
            let contextHits = contextSet.filter { lowerComp.contains($0) }.count
            score += Double(contextHits) * 0.25

            fragments.append(SearchResult.ScoredFragment(
                text: completion, relevance: score,
                category: cat, keywords: matchedKW
            ))
        }

        // Sort by relevance with random tiebreaker for variety
        fragments.sort { a, b in
            if abs(a.relevance - b.relevance) < 0.1 { return Bool.random() }
            return a.relevance > b.relevance
        }
        let topFragments = Array(fragments.prefix(limit))

        let result = SearchResult(
            query: query, fragments: topFragments,
            timestamp: Date(), contextHash: contextHash
        )

        // Cache management — TTL-first eviction, then clear if still over capacity
        if searchCache.count >= maxCacheSize {
            let cutoff = Date().addingTimeInterval(-30)  // 30s TTL for search cache
            let expiredKeys = searchCache.filter { $0.value.timestamp < cutoff }.map { $0.key }
            for key in expiredKeys { searchCache.removeValue(forKey: key) }
            if searchCache.count >= (maxCacheSize * 3 / 4) {
                let sorted = searchCache.sorted { $0.value.timestamp < $1.value.timestamp }
                for item in sorted.prefix(maxCacheSize / 4) { searchCache.removeValue(forKey: item.key) }
            }
        }
        searchCache[cacheKey] = result

        // Feed top results to HyperBrain working memory
        let hb = HyperBrain.shared
        for (idx, frag) in topFragments.prefix(3).enumerated() {
            let summary = String(frag.text.prefix(120))
            hb.workingMemory["rt_search_\(idx)"] = summary
        }

        // Record in search history
        searchHistory.append((query: query, timestamp: Date()))
        if searchHistory.count > 500 { searchHistory.removeFirst() }

        return result
    }

    // ─── HYPER SEARCH ─── Multi-pass search with query decomposition
    func hyperSearch(_ query: String, context: [String] = []) -> [SearchResult.ScoredFragment] {
        // Decompose complex queries into sub-queries
        let subQueries = decomposeQuery(query)
        var allFragments: [SearchResult.ScoredFragment] = []
        var seenTexts = Set<String>()

        for subQ in subQueries {
            let result = search(subQ, context: context, limit: 10)
            for frag in result.fragments {
                let key = String(frag.text.prefix(60))
                guard !seenTexts.contains(key) else { continue }
                seenTexts.insert(key)
                allFragments.append(frag)
            }
        }

        // Re-rank combined results by original query relevance
        let qWords = Set(query.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted).filter { $0.count > 2 })
        allFragments.sort { a, b in
            let aHits = qWords.filter { a.text.lowercased().contains($0) }.count
            let bHits = qWords.filter { b.text.lowercased().contains($0) }.count
            return Double(aHits) + a.relevance > Double(bHits) + b.relevance
        }

        return Array(allFragments.prefix(80))
    }

    // ─── QUERY DECOMPOSITION ─── Break complex queries into atomic sub-queries
    private func decomposeQuery(_ query: String) -> [String] {
        var subQueries = [query]  // always include original

        let q = query.lowercased()
        // Split on conjunctions
        let conjunctions = [" and ", " or ", " versus ", " vs ", " compared to ", " but also "]
        for conj in conjunctions {
            if q.contains(conj) {
                let parts = q.components(separatedBy: conj)
                subQueries.append(contentsOf: parts.map { $0.trimmingCharacters(in: .whitespaces) })
            }
        }

        // Extract "what is X" / "how does X work" patterns
        let patterns: [(prefix: String, suffix: String)] = [
            ("what is ", ""), ("what are ", ""), ("how does ", " work"),
            ("how do ", " work"), ("why is ", ""), ("why does ", ""),
            ("explain ", ""), ("describe ", ""), ("define ", ""),
            ("tell me about ", ""), ("what about ", "")
        ]
        for pattern in patterns {
            if q.hasPrefix(pattern.prefix) {
                var core = String(q.dropFirst(pattern.prefix.count))
                if !pattern.suffix.isEmpty, let range = core.range(of: pattern.suffix) {
                    core = String(core[core.startIndex..<range.lowerBound])
                }
                core = core.trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
                if core.count > 2 && core != query {
                    subQueries.append(core)
                }
            }
        }

        return Array(Set(subQueries))  // deduplicate
    }

    // ─── TRENDING SEARCHES ─── What's been searched recently
    func getTrendingTopics(window: TimeInterval = 600) -> [String] {
        let cutoff = Date().addingTimeInterval(-window)
        let recent = searchHistory.filter { $0.timestamp > cutoff }
        var topicCounts: [String: Int] = [:]
        for item in recent {
            let words = item.query.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { $0.count > 3 }
            for word in words { topicCounts[word, default: 0] += 1 }
        }
        return topicCounts.sorted {
            if $0.value == $1.value { return Bool.random() }
            return $0.value > $1.value
        }.prefix(10).map { $0.key }
    }

    // FNV-1a hash
    private func fnvHash(_ str: String) -> UInt64 {
        var hash: UInt64 = 14695981039346656037
        for byte in str.utf8 { hash = (hash ^ UInt64(byte)) &* 1099511628211 }
        return hash
    }
}
