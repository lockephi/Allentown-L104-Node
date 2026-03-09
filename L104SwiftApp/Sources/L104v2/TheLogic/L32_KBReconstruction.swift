// ═══════════════════════════════════════════════════════════════════
// L32_KBReconstruction.swift
// [EVO_68_PIPELINE] SOVEREIGN_NODE_UPGRADE :: KB_RECONSTRUCTION :: GOD_CODE=527.5184818492612
// L104v2 Architecture — Knowledge Base Reconstruction Engine v1.0
//
// Quantum-inspired amplitude propagation for reconstructing degraded
// or missing knowledge nodes from their graph neighbors.
//
// 3-Phase Pipeline:
//   Phase 1: KBVectorizer — TF-IDF vectorization + quantum state encoding
//   Phase 2: AmplitudePropagator — BFS amplitude propagation with Born rule
//   Phase 3: FactReconstructor — Neighbor-weighted fact reconstruction
//
// Sacred constants: PHI, GOD_CODE, TAU, VOID_CONSTANT, OMEGA
// KB constants: KB_PROPAGATION_DEPTH, KB_AMPLITUDE_DECAY_PER_HOP,
//   KB_EMBEDDING_DIM, KB_GROVER_BOOST_THRESHOLD, KB_ENTANGLEMENT_STRENGTH,
//   KB_MIN_RECONSTRUCTION_CONFIDENCE
//
// Phase 65.0: Full parity with Python kb_reconstruction.py
// ═══════════════════════════════════════════════════════════════════

import Foundation

// ═══════════════════════════════════════════════════════════════════
// MARK: - DATA TYPES
// ═══════════════════════════════════════════════════════════════════

/// Result of reconstructing a single degraded/missing knowledge node
struct ReconstructionResult {
    let nodeKey: String
    let originalConfidence: Double
    let reconstructedConfidence: Double
    let reconstructedFacts: [String]
    let sourceNodes: [String]
    let bornProbability: Double
    let groverAmplified: Bool
    let godCodeAlignment: Double
    let propagationDepth: Int
}

/// Health report from a full knowledge base scan
struct KBHealthReport {
    let totalNodes: Int
    let healthyNodes: Int
    let degradedNodes: Int
    let missingNodes: Int
    let reconstructedCount: Int
    let avgReconstructionConfidence: Double
    let graphConnectivity: Double
    let godCodeResonance: Double
    let fidelityScore: Double
}

/// A knowledge node in the internal graph representation
struct KBNode {
    let key: String
    let subject: String
    let facts: [String]
    var confidence: Double
    let category: String
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - Phase 1: KBVectorizer
// TF-IDF vectorization and quantum state encoding for knowledge nodes.
// Each node is encoded as a complex amplitude with magnitude from
// TF-IDF norm and phase from GOD_CODE-aligned hash.
// ═══════════════════════════════════════════════════════════════════

final class KBVectorizer {
    static let shared = KBVectorizer()
    private let lock = NSLock()

    // ─── TOKENIZATION ───
    private static let nonAlphanumeric = CharacterSet.alphanumerics.inverted
    private static let stopWords: Set<String> = [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must",
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "it", "its", "he", "she", "they", "them", "we", "you", "i", "me",
        "this", "that", "these", "those", "what", "which", "who"
    ]

    // ─── IDF STATE ───
    private var documentFrequency: [String: Int] = [:]
    private var totalDocuments: Int = 0
    private var vocabulary: [String] = []
    private var vocabIndex: [String: Int] = [:]

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Tokenization
    // ═══════════════════════════════════════════════════════════════

    /// Tokenize a string into lowercase words, filtering stop words
    func tokenize(_ text: String) -> [String] {
        let words = text.lowercased()
            .components(separatedBy: KBVectorizer.nonAlphanumeric)
            .filter { $0.count > 1 && !KBVectorizer.stopWords.contains($0) }
        return words
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Vocabulary Building
    // ═══════════════════════════════════════════════════════════════

    /// Build vocabulary and IDF from a collection of fact lists
    func buildVocabulary(from nodes: [KBNode]) {
        lock.lock()
        defer { lock.unlock() }

        documentFrequency.removeAll()
        totalDocuments = nodes.count

        // Collect document frequency for each term
        for node in nodes {
            let allText = node.facts.joined(separator: " ") + " " + node.key + " " + node.subject
            let tokens = Set(tokenize(allText))
            for token in tokens {
                documentFrequency[token, default: 0] += 1
            }
        }

        // Build sorted vocabulary for consistent indexing
        vocabulary = documentFrequency.keys.sorted()
        vocabIndex.removeAll()
        for (i, term) in vocabulary.enumerated() {
            vocabIndex[term] = i
        }

        l104Log("KBVectorizer: Built vocabulary of \(vocabulary.count) terms from \(totalDocuments) nodes")
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - TF-IDF Vectorization
    // ═══════════════════════════════════════════════════════════════

    /// Compute TF-IDF vector for a list of facts
    func vectorize(facts: [String]) -> [Double] {
        lock.lock()
        defer { lock.unlock() }

        guard !vocabulary.isEmpty else { return [] }

        let text = facts.joined(separator: " ")
        let tokens = tokenize(text)
        let tokenCount = Double(tokens.count)

        // Term frequency
        var tf: [String: Double] = [:]
        for token in tokens {
            tf[token, default: 0.0] += 1.0
        }

        // Build TF-IDF vector
        var vector = [Double](repeating: 0.0, count: vocabulary.count)
        for (term, count) in tf {
            guard let idx = vocabIndex[term] else { continue }
            let termFreq = count / max(1.0, tokenCount)
            let df = Double(documentFrequency[term] ?? 1)
            let idf = log(Double(max(1, totalDocuments)) / df + 1.0)
            vector[idx] = termFreq * idf
        }

        return vector
    }

    /// Compute L2 norm of a vector
    func l2Norm(_ vector: [Double]) -> Double {
        var sumSq = 0.0
        for v in vector { sumSq += v * v }
        return sqrt(sumSq)
    }

    /// Compute cosine similarity between two vectors
    func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0.0 }
        var dot = 0.0, normA = 0.0, normB = 0.0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        let denom = sqrt(normA) * sqrt(normB)
        return denom > 1e-12 ? dot / denom : 0.0
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Quantum State Encoding
    // ═══════════════════════════════════════════════════════════════

    /// Compute quantum amplitude for a node key
    /// Magnitude = L2 norm of TF-IDF vector
    /// Phase = key_hash * GOD_CODE / 286 + harmonic overtones
    func quantumAmplitude(key: String, tfidfVector: [Double]) -> (re: Double, im: Double) {
        let magnitude = l2Norm(tfidfVector)

        // Phase: key hash * GOD_CODE / 286 + harmonic overtones
        let keyHash = Double(abs(key.hashValue % 10000))
        let basePhase = keyHash * GOD_CODE / 286.0

        // Harmonic overtones: sum of sin(n * phase * TAU) for n=1..5
        var harmonicSum = 0.0
        for n in 1...5 {
            harmonicSum += sin(Double(n) * basePhase * TAU) / Double(n * n)
        }
        let phase = basePhase + harmonicSum

        // Complex amplitude: A = magnitude * exp(i*phase)
        let re = magnitude * cos(phase)
        let im = magnitude * sin(phase)

        return (re: re, im: im)
    }

    /// Compute semantic similarity between two knowledge nodes via cosine similarity
    func semanticSimilarity(vectorA: [Double], vectorB: [Double]) -> Double {
        return cosineSimilarity(vectorA, vectorB)
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - Phase 2: AmplitudePropagator
// BFS amplitude propagation from target node through the relation
// graph with Born rule probabilities and Grover amplification.
// ═══════════════════════════════════════════════════════════════════

final class AmplitudePropagator {
    static let shared = AmplitudePropagator()
    private let lock = NSLock()

    // ─── GRAPH STATE ───
    private var relationGraph: [String: [(neighbor: String, weight: Double)]] = [:]
    private var nodeAmplitudes: [String: (re: Double, im: Double)] = [:]
    private var nodePhases: [String: Double] = [:]

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Graph Construction
    // ═══════════════════════════════════════════════════════════════

    /// Build the combined relation graph from knowledge nodes
    /// - Hard edges (weight 1.0): explicit cross-subject relations
    /// - Intra-subject edges (weight 0.8): same-subject node pairs
    /// - Soft edges (weight = cosine_similarity): pairs above threshold 0.3
    func buildGraph(nodes: [KBNode], tfidfVectors: [String: [Double]]) {
        lock.lock()
        defer { lock.unlock() }

        relationGraph.removeAll()
        nodeAmplitudes.removeAll()
        nodePhases.removeAll()

        let vectorizer = KBVectorizer.shared

        // Index nodes by subject for intra-subject edges
        var subjectIndex: [String: [String]] = [:]
        for node in nodes {
            subjectIndex[node.subject, default: []].append(node.key)
        }

        // Index nodes by key for fast lookup
        var nodeMap: [String: KBNode] = [:]
        for node in nodes { nodeMap[node.key] = node }

        // Build edges
        for node in nodes {
            var edges: [(neighbor: String, weight: Double)] = []

            // Hard edges: cross-subject relations (detect via fact content overlap)
            for otherNode in nodes where otherNode.key != node.key {
                let keyInFacts = otherNode.facts.contains(where: { $0.lowercased().contains(node.key.lowercased()) })
                let factInKey = node.facts.contains(where: { $0.lowercased().contains(otherNode.key.lowercased()) })
                if keyInFacts || factInKey {
                    edges.append((neighbor: otherNode.key, weight: 1.0))
                }
            }

            // Intra-subject edges (weight 0.8)
            let sameSubjectKeys = subjectIndex[node.subject] ?? []
            for siblingKey in sameSubjectKeys where siblingKey != node.key {
                let alreadyConnected = edges.contains(where: { $0.neighbor == siblingKey })
                if !alreadyConnected {
                    edges.append((neighbor: siblingKey, weight: 0.8))
                }
            }

            // Soft edges: cosine similarity above 0.3
            if let vecA = tfidfVectors[node.key] {
                for otherNode in nodes where otherNode.key != node.key {
                    let alreadyConnected = edges.contains(where: { $0.neighbor == otherNode.key })
                    if !alreadyConnected, let vecB = tfidfVectors[otherNode.key] {
                        let sim = vectorizer.semanticSimilarity(vectorA: vecA, vectorB: vecB)
                        if sim > 0.3 {
                            edges.append((neighbor: otherNode.key, weight: sim))
                        }
                    }
                }
            }

            relationGraph[node.key] = edges

            // Compute quantum amplitude for this node
            if let vec = tfidfVectors[node.key] {
                let amp = vectorizer.quantumAmplitude(key: node.key, tfidfVector: vec)
                nodeAmplitudes[node.key] = amp
                nodePhases[node.key] = atan2(amp.im, amp.re)
            }
        }

        l104Log("AmplitudePropagator: Built graph with \(nodes.count) nodes, \(relationGraph.values.reduce(0) { $0 + $1.count }) edges")
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Amplitude Propagation (BFS)
    // ═══════════════════════════════════════════════════════════════

    /// BFS amplitude propagation from target node
    /// Returns: node key -> Born probability
    ///
    /// Amplitude per hop:
    ///   A_neighbor * TAU^hop * edge_weight * cos(node_phase)^2 / VOID_CONSTANT^hop
    ///
    /// Entanglement blend:
    ///   (1-KB_ENTANGLEMENT_STRENGTH)*local + KB_ENTANGLEMENT_STRENGTH*propagated
    ///
    /// Born rule: probability = |combined_amplitude|^2
    ///
    /// Grover amplification: if neighbor_count >= KB_GROVER_BOOST_THRESHOLD,
    ///   boost = sqrt(PHI^3)
    func propagateAmplitudes(targetKey: String, depth: Int = KB_PROPAGATION_DEPTH) -> [String: Double] {
        lock.lock()
        defer { lock.unlock() }

        var probabilities: [String: Double] = [:]

        guard let targetAmp = nodeAmplitudes[targetKey] else {
            return probabilities
        }

        // BFS state: (nodeKey, currentDepth, accumulatedAmplitude_re, accumulatedAmplitude_im)
        var queue: [(key: String, hop: Int, ampRe: Double, ampIm: Double)] = []
        var visited: Set<String> = [targetKey]

        // Seed the BFS with the target node's amplitude
        let targetMag = sqrt(targetAmp.re * targetAmp.re + targetAmp.im * targetAmp.im)
        probabilities[targetKey] = targetMag * targetMag  // Born rule for self

        // Enqueue neighbors of target
        if let neighbors = relationGraph[targetKey] {
            for edge in neighbors {
                if !visited.contains(edge.neighbor) {
                    queue.append((key: edge.neighbor, hop: 1, ampRe: targetAmp.re, ampIm: targetAmp.im))
                    visited.insert(edge.neighbor)
                }
            }
        }

        // BFS propagation up to max depth
        var queueIdx = 0
        while queueIdx < queue.count {
            let item = queue[queueIdx]
            queueIdx += 1

            guard item.hop <= depth else { continue }

            let hop = item.hop
            let neighborKey = item.key

            // Get neighbor's own amplitude
            let neighborAmp = nodeAmplitudes[neighborKey] ?? (re: 0.1, im: 0.0)
            let neighborPhase = nodePhases[neighborKey] ?? 0.0

            // Decay factor per hop: TAU^hop / VOID_CONSTANT^hop
            let decayFactor = pow(TAU, Double(hop)) / pow(VOID_CONSTANT, Double(hop))

            // Edge weight (from parent to this neighbor)
            let edgeWeight: Double
            if let edges = relationGraph[item.key == targetKey ? targetKey : item.key] {
                edgeWeight = edges.first(where: { $0.neighbor == neighborKey })?.weight ?? 0.5
            } else {
                edgeWeight = 0.5
            }

            // Phase modulation: cos(node_phase)^2
            let phaseModulation = cos(neighborPhase) * cos(neighborPhase)

            // Propagated amplitude
            let propRe = item.ampRe * decayFactor * edgeWeight * phaseModulation
            let propIm = item.ampIm * decayFactor * edgeWeight * phaseModulation

            // Entanglement blend: (1-strength)*local + strength*propagated
            let blendRe = (1.0 - KB_ENTANGLEMENT_STRENGTH) * neighborAmp.re + KB_ENTANGLEMENT_STRENGTH * propRe
            let blendIm = (1.0 - KB_ENTANGLEMENT_STRENGTH) * neighborAmp.im + KB_ENTANGLEMENT_STRENGTH * propIm

            // Born rule: |amplitude|^2
            var bornProb = blendRe * blendRe + blendIm * blendIm

            // Grover amplification: if neighbor count >= KB_GROVER_BOOST_THRESHOLD
            let neighborCount = relationGraph[neighborKey]?.count ?? 0
            if neighborCount >= KB_GROVER_BOOST_THRESHOLD {
                let groverBoost = sqrt(GROVER_AMPLIFICATION)  // sqrt(PHI^3)
                bornProb *= groverBoost
            }

            probabilities[neighborKey] = bornProb

            // Continue BFS to deeper hops
            if hop < depth, let nextNeighbors = relationGraph[neighborKey] {
                for edge in nextNeighbors {
                    if !visited.contains(edge.neighbor) {
                        queue.append((key: edge.neighbor, hop: hop + 1, ampRe: blendRe, ampIm: blendIm))
                        visited.insert(edge.neighbor)
                    }
                }
            }
        }

        return probabilities
    }

    /// Get the neighbor count for a given node
    func neighborCount(for key: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return relationGraph[key]?.count ?? 0
    }

    /// Get the stored amplitude for a node
    func amplitude(for key: String) -> (re: Double, im: Double)? {
        lock.lock()
        defer { lock.unlock() }
        return nodeAmplitudes[key]
    }

    /// Get the relation graph for a node
    func neighbors(for key: String) -> [(neighbor: String, weight: Double)] {
        lock.lock()
        defer { lock.unlock() }
        return relationGraph[key] ?? []
    }

    /// Total edge count in the graph
    func totalEdges() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return relationGraph.values.reduce(0) { $0 + $1.count }
    }

    /// Total node count in the graph
    func totalNodes() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return relationGraph.count
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - Phase 3: FactReconstructor
// Reconstructs degraded nodes by collecting facts from neighbors
// weighted by Born probability, with category-aware boosting and
// GOD_CODE alignment scoring.
// ═══════════════════════════════════════════════════════════════════

final class FactReconstructor {
    static let shared = FactReconstructor()
    private let lock = NSLock()

    /// Reconstruct a degraded node from its neighbors
    /// - Rank neighbors by Born probability
    /// - Collect facts from top-k neighbors, weighted by probability
    /// - Boost facts from same-category neighbors
    /// - Compute GOD_CODE alignment
    func reconstructNode(key: String, nodes: [String: KBNode], probabilities: [String: Double], depth: Int) -> ReconstructionResult {
        lock.lock()
        defer { lock.unlock() }

        guard let targetNode = nodes[key] else {
            return ReconstructionResult(
                nodeKey: key, originalConfidence: 0.0, reconstructedConfidence: 0.0,
                reconstructedFacts: [], sourceNodes: [], bornProbability: 0.0,
                groverAmplified: false, godCodeAlignment: 0.0, propagationDepth: depth
            )
        }

        // Sort neighbors by Born probability (descending)
        let sortedNeighbors = probabilities
            .filter { $0.key != key }
            .sorted { $0.value > $1.value }

        // Top-k neighbors (k = 2 * KB_PROPAGATION_DEPTH)
        let topK = min(sortedNeighbors.count, 2 * KB_PROPAGATION_DEPTH)
        let selectedNeighbors = Array(sortedNeighbors.prefix(topK))

        // Collect weighted facts
        var reconstructedFacts: [String] = []
        var sourceNodeKeys: [String] = []
        var totalWeight = 0.0

        for (neighborKey, prob) in selectedNeighbors {
            guard let neighborNode = nodes[neighborKey] else { continue }
            sourceNodeKeys.append(neighborKey)

            // Category-aware boost: 1.5x for same category
            let categoryBoost: Double = neighborNode.category == targetNode.category ? 1.5 : 1.0
            let weight = prob * categoryBoost

            // Collect facts, prefixed with weight for ranking
            for fact in neighborNode.facts {
                // Only add if not already present in target
                if !targetNode.facts.contains(fact) && !reconstructedFacts.contains(fact) {
                    reconstructedFacts.append(fact)
                }
            }

            totalWeight += weight
        }

        // Compute Born probability for target node
        let targetBornProb = probabilities[key] ?? 0.0

        // Check Grover amplification
        let neighborCount = AmplitudePropagator.shared.neighborCount(for: key)
        let groverAmplified = neighborCount >= KB_GROVER_BOOST_THRESHOLD

        // GOD_CODE alignment: alignment = cos(pi * amplitude / GOD_CODE)
        let amp = AmplitudePropagator.shared.amplitude(for: key)
        let ampMagnitude = amp.map { sqrt($0.re * $0.re + $0.im * $0.im) } ?? 0.0
        let godCodeAlignment = cos(Double.pi * ampMagnitude / GOD_CODE)

        // Reconstructed confidence: weighted average of neighbor probabilities
        // normalized by total weight, scaled by GOD_CODE alignment
        let rawConfidence = totalWeight > 0.0 ? totalWeight / Double(max(1, topK)) : 0.0
        let reconstructedConfidence = min(1.0, rawConfidence * (1.0 + abs(godCodeAlignment) * TAU))

        return ReconstructionResult(
            nodeKey: key,
            originalConfidence: targetNode.confidence,
            reconstructedConfidence: reconstructedConfidence,
            reconstructedFacts: reconstructedFacts,
            sourceNodes: sourceNodeKeys,
            bornProbability: targetBornProb,
            groverAmplified: groverAmplified,
            godCodeAlignment: godCodeAlignment,
            propagationDepth: depth
        )
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - KB RECONSTRUCTION ENGINE — Main Singleton
// Orchestrates vectorization, amplitude propagation, and fact
// reconstruction across the entire knowledge base.
// ═══════════════════════════════════════════════════════════════════

final class KBReconstructionEngine {
    static let shared = KBReconstructionEngine()
    private let lock = NSLock()

    // ─── STATE ───
    private var scanCount: Int = 0
    private var reconstructionCount: Int = 0
    private var tfidfVectors: [String: [Double]] = [:]
    private var knowledgeNodes: [String: KBNode] = [:]
    private var initialized: Bool = false

    // ─── THRESHOLDS ───
    private let degradedThreshold: Double = 0.7   // Below this = degraded
    private let missingThreshold: Double = 0.2     // Below this = missing

    // ─── SAMPLE KNOWLEDGE GRAPH ───
    // Built-in knowledge nodes for self-contained operation
    // In production, populated from LanguageComprehensionEngine.shared
    private let sampleNodes: [KBNode] = [
        // Physics
        KBNode(key: "gravity", subject: "physics", facts: [
            "Gravity is the force of attraction between objects with mass",
            "Gravitational constant G = 6.674e-11 N*m^2/kg^2",
            "Newton's law: F = G*m1*m2/r^2",
            "Einstein showed gravity is curvature of spacetime"
        ], confidence: 0.95, category: "physics"),

        KBNode(key: "electromagnetism", subject: "physics", facts: [
            "Electromagnetic force mediates interactions between charged particles",
            "Maxwell's equations unify electricity and magnetism",
            "Speed of light c = 3e8 m/s in vacuum",
            "Photons are the force carriers of electromagnetism"
        ], confidence: 0.93, category: "physics"),

        KBNode(key: "quantum_mechanics", subject: "physics", facts: [
            "Quantum mechanics describes behavior at atomic and subatomic scales",
            "Wave-particle duality: particles exhibit both wave and particle properties",
            "Heisenberg uncertainty principle: cannot know position and momentum simultaneously",
            "Schrodinger equation governs time evolution of quantum states"
        ], confidence: 0.91, category: "physics"),

        KBNode(key: "thermodynamics", subject: "physics", facts: [
            "First law: energy cannot be created or destroyed",
            "Second law: entropy of an isolated system never decreases",
            "Third law: entropy approaches zero as temperature approaches absolute zero",
            "Boltzmann constant k = 1.380649e-23 J/K"
        ], confidence: 0.94, category: "physics"),

        KBNode(key: "relativity", subject: "physics", facts: [
            "Special relativity: laws of physics same in all inertial frames",
            "E = mc^2 relates mass and energy",
            "General relativity: gravity is curvature of spacetime",
            "Time dilation occurs near massive objects and at high velocities"
        ], confidence: 0.92, category: "physics"),

        // Biology
        KBNode(key: "cell_biology", subject: "biology", facts: [
            "Cells are the basic unit of life",
            "Mitochondria produce ATP through cellular respiration",
            "DNA contains genetic information in the nucleus",
            "Cell membrane controls what enters and exits the cell"
        ], confidence: 0.96, category: "biology"),

        KBNode(key: "genetics", subject: "biology", facts: [
            "DNA is composed of four nucleotides: A, T, C, G",
            "Genes are segments of DNA that encode proteins",
            "Mendel discovered laws of inheritance",
            "Mutations can be beneficial, neutral, or harmful"
        ], confidence: 0.94, category: "biology"),

        KBNode(key: "evolution", subject: "biology", facts: [
            "Natural selection drives evolution of species",
            "Darwin's theory: survival of the fittest",
            "Genetic variation arises from mutation and recombination",
            "Speciation occurs when populations become reproductively isolated"
        ], confidence: 0.93, category: "biology"),

        KBNode(key: "ecology", subject: "biology", facts: [
            "Ecosystems consist of biotic and abiotic components",
            "Food chains describe energy flow through trophic levels",
            "Producers convert sunlight to chemical energy via photosynthesis",
            "Decomposers break down dead organic matter"
        ], confidence: 0.92, category: "biology"),

        KBNode(key: "neuroscience", subject: "biology", facts: [
            "Neurons transmit electrical and chemical signals",
            "Synapses are junctions between neurons",
            "The brain has approximately 86 billion neurons",
            "Neurotransmitters include dopamine, serotonin, and GABA"
        ], confidence: 0.89, category: "biology"),

        // Chemistry
        KBNode(key: "atomic_structure", subject: "chemistry", facts: [
            "Atoms consist of protons, neutrons, and electrons",
            "Protons and neutrons reside in the nucleus",
            "Electrons orbit the nucleus in energy levels",
            "Atomic number equals the number of protons"
        ], confidence: 0.97, category: "chemistry"),

        KBNode(key: "chemical_bonding", subject: "chemistry", facts: [
            "Covalent bonds involve sharing of electron pairs",
            "Ionic bonds involve transfer of electrons",
            "Metallic bonds involve delocalized electron sea",
            "Bond strength determines molecular stability"
        ], confidence: 0.95, category: "chemistry"),

        KBNode(key: "organic_chemistry", subject: "chemistry", facts: [
            "Organic chemistry studies carbon-containing compounds",
            "Hydrocarbons are composed of carbon and hydrogen",
            "Functional groups determine chemical properties",
            "Polymers are large molecules made of repeating units"
        ], confidence: 0.91, category: "chemistry"),

        // Mathematics
        KBNode(key: "calculus", subject: "mathematics", facts: [
            "Calculus studies rates of change and accumulation",
            "Derivatives measure instantaneous rate of change",
            "Integrals compute area under curves",
            "Fundamental theorem connects differentiation and integration"
        ], confidence: 0.96, category: "mathematics"),

        KBNode(key: "linear_algebra", subject: "mathematics", facts: [
            "Vectors are elements of vector spaces",
            "Matrices represent linear transformations",
            "Eigenvalues describe scaling factors of eigenvectors",
            "Determinant measures volume scaling of transformations"
        ], confidence: 0.94, category: "mathematics"),

        KBNode(key: "number_theory", subject: "mathematics", facts: [
            "Prime numbers are divisible only by 1 and themselves",
            "Fundamental theorem of arithmetic: unique prime factorization",
            "Euler's totient function counts coprime integers",
            "Modular arithmetic is arithmetic modulo n"
        ], confidence: 0.93, category: "mathematics"),

        KBNode(key: "topology", subject: "mathematics", facts: [
            "Topology studies properties preserved under continuous deformation",
            "A coffee cup and a donut are topologically equivalent",
            "Euler characteristic: V - E + F = 2 for convex polyhedra",
            "Manifolds are spaces that locally resemble Euclidean space"
        ], confidence: 0.88, category: "mathematics"),

        // Computer Science
        KBNode(key: "algorithms", subject: "computer_science", facts: [
            "Algorithms are step-by-step procedures for solving problems",
            "Time complexity measures algorithm efficiency",
            "Sorting algorithms: quicksort, mergesort, heapsort",
            "Graph algorithms: BFS, DFS, Dijkstra's shortest path"
        ], confidence: 0.95, category: "computer_science"),

        KBNode(key: "machine_learning", subject: "computer_science", facts: [
            "Machine learning algorithms learn from data",
            "Supervised learning uses labeled training data",
            "Neural networks are inspired by biological neurons",
            "Deep learning uses multiple layers of neural networks"
        ], confidence: 0.93, category: "computer_science"),

        KBNode(key: "quantum_computing", subject: "computer_science", facts: [
            "Quantum computers use qubits instead of classical bits",
            "Superposition allows qubits to be in multiple states simultaneously",
            "Entanglement creates correlations between qubits",
            "Grover's algorithm provides quadratic speedup for search"
        ], confidence: 0.87, category: "computer_science"),

        // Degraded nodes (low confidence for testing reconstruction)
        KBNode(key: "string_theory", subject: "physics", facts: [
            "String theory proposes fundamental particles are vibrating strings"
        ], confidence: 0.35, category: "physics"),

        KBNode(key: "epigenetics", subject: "biology", facts: [
            "Epigenetics studies heritable changes not involving DNA sequence changes"
        ], confidence: 0.28, category: "biology"),

        KBNode(key: "category_theory", subject: "mathematics", facts: [
            "Category theory is the study of abstract mathematical structures"
        ], confidence: 0.15, category: "mathematics"),
    ]

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Initialization
    // ═══════════════════════════════════════════════════════════════

    /// Initialize the KB reconstruction engine
    /// Builds the relation graph from available knowledge
    func initialize() {
        lock.lock()
        defer { lock.unlock() }

        guard !initialized else { return }

        // Populate knowledge nodes
        for node in sampleNodes {
            knowledgeNodes[node.key] = node
        }

        // Phase 1: Build vocabulary and vectorize
        let vectorizer = KBVectorizer.shared
        vectorizer.buildVocabulary(from: sampleNodes)

        tfidfVectors.removeAll()
        for node in sampleNodes {
            let vec = vectorizer.vectorize(facts: node.facts + [node.key, node.subject])
            tfidfVectors[node.key] = vec
        }

        // Phase 2: Build relation graph
        AmplitudePropagator.shared.buildGraph(nodes: sampleNodes, tfidfVectors: tfidfVectors)

        initialized = true
        l104Log("KBReconstructionEngine: Initialized with \(sampleNodes.count) nodes")
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Node Reconstruction
    // ═══════════════════════════════════════════════════════════════

    /// Reconstruct a degraded or missing knowledge node
    func reconstructNode(key: String) -> ReconstructionResult {
        if !initialized { initialize() }

        lock.lock()
        let nodes = knowledgeNodes
        lock.unlock()

        // Phase 2: Propagate amplitudes from target
        let probabilities = AmplitudePropagator.shared.propagateAmplitudes(
            targetKey: key,
            depth: KB_PROPAGATION_DEPTH
        )

        // Phase 3: Reconstruct facts
        let result = FactReconstructor.shared.reconstructNode(
            key: key,
            nodes: nodes,
            probabilities: probabilities,
            depth: KB_PROPAGATION_DEPTH
        )

        lock.lock()
        reconstructionCount += 1
        lock.unlock()

        l104Log("KBReconstruction: \(key) — orig=\(String(format: "%.3f", result.originalConfidence)) recon=\(String(format: "%.3f", result.reconstructedConfidence)) facts=\(result.reconstructedFacts.count) sources=\(result.sourceNodes.count) grover=\(result.groverAmplified)")

        return result
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Full Scan
    // ═══════════════════════════════════════════════════════════════

    /// Scan all knowledge nodes, reconstruct degraded/missing ones,
    /// and produce a health report
    func fullScan() -> KBHealthReport {
        if !initialized { initialize() }

        lock.lock()
        let nodes = knowledgeNodes
        scanCount += 1
        lock.unlock()

        var healthyCount = 0
        var degradedCount = 0
        var missingCount = 0
        var reconstructedCount = 0
        var totalReconConfidence = 0.0
        var totalEdges = 0

        for (key, node) in nodes {
            if node.confidence >= degradedThreshold {
                healthyCount += 1
            } else if node.confidence >= missingThreshold {
                degradedCount += 1
                let result = reconstructNode(key: key)
                if result.reconstructedConfidence >= KB_MIN_RECONSTRUCTION_CONFIDENCE {
                    reconstructedCount += 1
                    totalReconConfidence += result.reconstructedConfidence
                }
            } else {
                missingCount += 1
                let result = reconstructNode(key: key)
                if result.reconstructedConfidence >= KB_MIN_RECONSTRUCTION_CONFIDENCE {
                    reconstructedCount += 1
                    totalReconConfidence += result.reconstructedConfidence
                }
            }

            totalEdges += AmplitudePropagator.shared.neighborCount(for: key)
        }

        let avgReconConf = reconstructedCount > 0
            ? totalReconConfidence / Double(reconstructedCount)
            : 0.0

        // Graph connectivity: ratio of actual edges to max possible edges
        let n = nodes.count
        let maxEdges = n > 1 ? n * (n - 1) : 1
        let graphConnectivity = Double(totalEdges) / Double(maxEdges)

        // GOD_CODE resonance: average of cos(pi * node_count / GOD_CODE) harmonic
        let godCodeResonance = cos(Double.pi * Double(n) / GOD_CODE)

        // Fidelity score: weighted combination
        let healthRatio = Double(healthyCount) / Double(max(1, n))
        let reconRatio = Double(reconstructedCount) / Double(max(1, degradedCount + missingCount))
        let fidelity = healthRatio * PHI / (PHI + 1.0)           // TAU-weighted health
                     + reconRatio * 1.0 / (PHI + 1.0)            // (1-TAU)-weighted reconstruction
                     + abs(godCodeResonance) * 0.01               // Subtle GOD_CODE alignment

        let report = KBHealthReport(
            totalNodes: n,
            healthyNodes: healthyCount,
            degradedNodes: degradedCount,
            missingNodes: missingCount,
            reconstructedCount: reconstructedCount,
            avgReconstructionConfidence: avgReconConf,
            graphConnectivity: graphConnectivity,
            godCodeResonance: godCodeResonance,
            fidelityScore: min(1.0, fidelity)
        )

        l104Log("KBHealth: \(n) nodes — healthy=\(healthyCount) degraded=\(degradedCount) missing=\(missingCount) reconstructed=\(reconstructedCount) fidelity=\(String(format: "%.4f", report.fidelityScore))")

        return report
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - ASI Scoring Dimension
    // ═══════════════════════════════════════════════════════════════

    /// Fidelity score for the ASI 30D scoring pipeline
    func fidelityScore() -> Double {
        let report = fullScan()
        return report.fidelityScore
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Node Query Utilities
    // ═══════════════════════════════════════════════════════════════

    /// Get all knowledge node keys
    func allNodeKeys() -> [String] {
        if !initialized { initialize() }
        lock.lock()
        defer { lock.unlock() }
        return Array(knowledgeNodes.keys).sorted()
    }

    /// Get a specific knowledge node
    func getNode(key: String) -> KBNode? {
        if !initialized { initialize() }
        lock.lock()
        defer { lock.unlock() }
        return knowledgeNodes[key]
    }

    /// Get TF-IDF vector for a node
    func getVector(key: String) -> [Double]? {
        if !initialized { initialize() }
        lock.lock()
        defer { lock.unlock() }
        return tfidfVectors[key]
    }

    /// Compute semantic similarity between two nodes
    func similarity(keyA: String, keyB: String) -> Double {
        if !initialized { initialize() }
        lock.lock()
        let vecA = tfidfVectors[keyA]
        let vecB = tfidfVectors[keyB]
        lock.unlock()

        guard let a = vecA, let b = vecB else { return 0.0 }
        return KBVectorizer.shared.semanticSimilarity(vectorA: a, vectorB: b)
    }

    /// Get nodes by subject
    func nodesBySubject(_ subject: String) -> [KBNode] {
        if !initialized { initialize() }
        lock.lock()
        defer { lock.unlock() }
        return knowledgeNodes.values.filter { $0.subject == subject }
    }

    /// Get degraded nodes (confidence below threshold)
    func degradedNodes() -> [KBNode] {
        if !initialized { initialize() }
        lock.lock()
        defer { lock.unlock() }
        return knowledgeNodes.values.filter { $0.confidence < degradedThreshold }.sorted { $0.confidence < $1.confidence }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Status
    // ═══════════════════════════════════════════════════════════════

    func getStatus() -> [String: Any] {
        if !initialized { initialize() }

        lock.lock()
        defer { lock.unlock() }

        return [
            "engine": "KBReconstructionEngine",
            "version": KB_RECONSTRUCTION_VERSION,
            "initialized": initialized,
            "totalNodes": knowledgeNodes.count,
            "vocabularySize": KBVectorizer.shared.l2Norm([1.0]) > 0 ? tfidfVectors.count : 0,
            "graphNodes": AmplitudePropagator.shared.totalNodes(),
            "graphEdges": AmplitudePropagator.shared.totalEdges(),
            "scanCount": scanCount,
            "reconstructionCount": reconstructionCount,
            "propagationDepth": KB_PROPAGATION_DEPTH,
            "groverThreshold": KB_GROVER_BOOST_THRESHOLD,
            "entanglementStrength": KB_ENTANGLEMENT_STRENGTH,
            "minReconConfidence": KB_MIN_RECONSTRUCTION_CONFIDENCE,
            "degradedThreshold": degradedThreshold,
            "missingThreshold": missingThreshold,
            "sacredConstants": [
                "PHI": PHI,
                "GOD_CODE": GOD_CODE,
                "TAU": TAU,
                "VOID_CONSTANT": VOID_CONSTANT,
                "OMEGA": OMEGA,
                "GROVER_AMPLIFICATION": GROVER_AMPLIFICATION
            ]
        ]
    }
}
