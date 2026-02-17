// ═══════════════════════════════════════════════════════════════
// B23_MathEngines3.swift
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 · TheBrain · v2 Architecture
//
// Extracted from L104Native.swift lines 15084-16375
// Classes: GraphTheoryEngine, SpecialFunctionsEngine, ControlTheoryEngine
// ═══════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class GraphTheoryEngine {
    static let shared = GraphTheoryEngine()
    private var computations: Int = 0

    // ═══════════════════════════════════════════════════════════════
    // MARK: GRAPH REPRESENTATION
    // ═══════════════════════════════════════════════════════════════

    /// Create adjacency matrix from edge list
    func adjacencyMatrix(vertices: Int, edges: [(Int, Int)], directed: Bool = false) -> [[Int]] {
        computations += 1
        var matrix = [[Int]](repeating: [Int](repeating: 0, count: vertices), count: vertices)
        for (u, v) in edges {
            matrix[u][v] = 1
            if !directed { matrix[v][u] = 1 }
        }
        return matrix
    }

    /// Adjacency matrix from weighted edge list
    func weightedAdjacencyMatrix(vertices: Int, edges: [(Int, Int, Double)], directed: Bool = false) -> [[Double]] {
        computations += 1
        var matrix = [[Double]](repeating: [Double](repeating: Double.infinity, count: vertices), count: vertices)
        for i in 0..<vertices { matrix[i][i] = 0 }
        for (u, v, w) in edges {
            matrix[u][v] = w
            if !directed { matrix[v][u] = w }
        }
        return matrix
    }

    /// Degree of each vertex in an adjacency matrix
    func degrees(_ adj: [[Int]]) -> [Int] {
        computations += 1
        return adj.map { $0.reduce(0, +) }
    }

    /// Laplacian matrix: L = D - A
    func laplacianMatrix(_ adj: [[Int]]) -> [[Int]] {
        computations += 1
        let n = adj.count
        let deg = degrees(adj)
        var L = [[Int]](repeating: [Int](repeating: 0, count: n), count: n)
        for i in 0..<n {
            for j in 0..<n {
                if i == j { L[i][j] = deg[i] }
                else { L[i][j] = -adj[i][j] }
            }
        }
        return L
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: SHORTEST PATHS
    // ═══════════════════════════════════════════════════════════════

    /// Dijkstra's algorithm: shortest paths from source
    func dijkstra(adj: [[Double]], source: Int) -> (distances: [Double], predecessors: [Int]) {
        computations += 1
        let n = adj.count
        var dist = [Double](repeating: Double.infinity, count: n)
        var pred = [Int](repeating: -1, count: n)
        var visited = [Bool](repeating: false, count: n)
        dist[source] = 0

        for _ in 0..<n {
            // Find unvisited vertex with minimum distance
            var u = -1
            var minDist: Double = Double.infinity
            for v in 0..<n {
                if !visited[v] && dist[v] < minDist {
                    minDist = dist[v]
                    u = v
                }
            }
            guard u >= 0 else { break }
            visited[u] = true

            for v in 0..<n {
                if !visited[v] && adj[u][v] < Double.infinity {
                    let alt: Double = dist[u] + adj[u][v]
                    if alt < dist[v] {
                        dist[v] = alt
                        pred[v] = u
                    }
                }
            }
        }
        return (dist, pred)
    }

    /// Floyd-Warshall: all-pairs shortest paths
    func floydWarshall(adj: [[Double]]) -> [[Double]] {
        computations += 1
        var dist = adj
        let n = adj.count
        for k in 0..<n {
            for i in 0..<n {
                for j in 0..<n {
                    if dist[i][k] + dist[k][j] < dist[i][j] {
                        dist[i][j] = dist[i][k] + dist[k][j]
                    }
                }
            }
        }
        return dist
    }

    /// Bellman-Ford: shortest paths with negative weights (detects negative cycles)
    func bellmanFord(vertices: Int, edges: [(Int, Int, Double)], source: Int) -> (distances: [Double], hasNegativeCycle: Bool) {
        computations += 1
        var dist = [Double](repeating: Double.infinity, count: vertices)
        dist[source] = 0

        for _ in 0..<vertices - 1 {
            for (u, v, w) in edges {
                if dist[u] < Double.infinity && dist[u] + w < dist[v] {
                    dist[v] = dist[u] + w
                }
            }
        }
        // Check for negative cycles
        for (u, v, w) in edges {
            if dist[u] < Double.infinity && dist[u] + w < dist[v] {
                return (dist, true)
            }
        }
        return (dist, false)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: GRAPH CONNECTIVITY
    // ═══════════════════════════════════════════════════════════════

    /// BFS: breadth-first search — returns visited order
    func bfs(_ adj: [[Int]], start: Int) -> [Int] {
        computations += 1
        let n = adj.count
        var visited = [Bool](repeating: false, count: n)
        var queue = [start]
        var head = 0
        var order: [Int] = []
        visited[start] = true

        while head < queue.count {
            let u = queue[head]
            head += 1
            order.append(u)
            for v in 0..<n where adj[u][v] > 0 && !visited[v] {
                visited[v] = true
                queue.append(v)
            }
        }
        return order
    }

    /// DFS: depth-first search — returns visited order
    func dfs(_ adj: [[Int]], start: Int) -> [Int] {
        computations += 1
        let n = adj.count
        var visited = [Bool](repeating: false, count: n)
        var order: [Int] = []
        func visit(_ u: Int) {
            visited[u] = true
            order.append(u)
            for v in 0..<n where adj[u][v] > 0 && !visited[v] {
                visit(v)
            }
        }
        visit(start)
        return order
    }

    /// Is the graph connected? (undirected)
    func isConnected(_ adj: [[Int]]) -> Bool {
        computations += 1
        guard !adj.isEmpty else { return true }
        return bfs(adj, start: 0).count == adj.count
    }

    /// Number of connected components
    func connectedComponents(_ adj: [[Int]]) -> Int {
        computations += 1
        let n = adj.count
        var visited = [Bool](repeating: false, count: n)
        var components = 0
        for i in 0..<n {
            if !visited[i] {
                components += 1
                // BFS from i
                var queue = [i]
                var head = 0
                visited[i] = true
                while head < queue.count {
                    let u = queue[head]
                    head += 1
                    for v in 0..<n where adj[u][v] > 0 && !visited[v] {
                        visited[v] = true
                        queue.append(v)
                    }
                }
            }
        }
        return components
    }

    /// Topological sort (DAG only) using Kahn's algorithm
    func topologicalSort(_ adj: [[Int]]) -> [Int]? {
        computations += 1
        let n = adj.count
        var inDegree = [Int](repeating: 0, count: n)
        for i in 0..<n {
            for j in 0..<n {
                inDegree[j] += adj[i][j]
            }
        }
        var queue: [Int] = []
        for i in 0..<n where inDegree[i] == 0 { queue.append(i) }
        var order: [Int] = []
        var head = 0
        while head < queue.count {
            let u = queue[head]
            head += 1
            order.append(u)
            for v in 0..<n where adj[u][v] > 0 {
                inDegree[v] -= 1
                if inDegree[v] == 0 { queue.append(v) }
            }
        }
        return order.count == n ? order : nil  // nil means cycle exists
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: SPANNING TREES
    // ═══════════════════════════════════════════════════════════════

    /// Kruskal's minimum spanning tree
    func kruskalMST(vertices: Int, edges: [(Int, Int, Double)]) -> (edges: [(Int, Int, Double)], totalWeight: Double) {
        computations += 1
        let sorted = edges.sorted { $0.2 < $1.2 }
        var parent = Array(0..<vertices)
        var rank = [Int](repeating: 0, count: vertices)

        func find(_ x: Int) -> Int {
            if parent[x] != x { parent[x] = find(parent[x]) }
            return parent[x]
        }
        func union(_ x: Int, _ y: Int) -> Bool {
            let px = find(x), py = find(y)
            if px == py { return false }
            if rank[px] < rank[py] { parent[px] = py }
            else if rank[px] > rank[py] { parent[py] = px }
            else { parent[py] = px; rank[px] += 1 }
            return true
        }

        var mst: [(Int, Int, Double)] = []
        var totalWeight: Double = 0
        for (u, v, w) in sorted {
            if union(u, v) {
                mst.append((u, v, w))
                totalWeight += w
                if mst.count == vertices - 1 { break }
            }
        }
        return (mst, totalWeight)
    }

    /// Prim's minimum spanning tree
    func primMST(adj: [[Double]]) -> (edges: [(Int, Int, Double)], totalWeight: Double) {
        computations += 1
        let n = adj.count
        var inMST = [Bool](repeating: false, count: n)
        var key = [Double](repeating: Double.infinity, count: n)
        var parent = [Int](repeating: -1, count: n)
        key[0] = 0

        for _ in 0..<n {
            var u = -1
            var minKey: Double = Double.infinity
            for v in 0..<n where !inMST[v] && key[v] < minKey {
                minKey = key[v]; u = v
            }
            guard u >= 0 else { break }
            inMST[u] = true
            for v in 0..<n {
                if !inMST[v] && adj[u][v] < Double.infinity && adj[u][v] < key[v] {
                    key[v] = adj[u][v]
                    parent[v] = u
                }
            }
        }
        var mstEdges: [(Int, Int, Double)] = []
        var total: Double = 0
        for v in 1..<n where parent[v] >= 0 {
            mstEdges.append((parent[v], v, adj[parent[v]][v]))
            total += adj[parent[v]][v]
        }
        return (mstEdges, total)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: GRAPH PROPERTIES
    // ═══════════════════════════════════════════════════════════════

    /// Is the graph bipartite? Returns (true, partition) or (false, [])
    func isBipartite(_ adj: [[Int]]) -> (bipartite: Bool, coloring: [Int]) {
        computations += 1
        let n = adj.count
        var color = [Int](repeating: -1, count: n)

        for start in 0..<n {
            if color[start] != -1 { continue }
            color[start] = 0
            var queue = [start]
            var head = 0
            while head < queue.count {
                let u = queue[head]
                head += 1
                for v in 0..<n where adj[u][v] > 0 {
                    if color[v] == -1 {
                        color[v] = 1 - color[u]
                        queue.append(v)
                    } else if color[v] == color[u] {
                        return (false, [])
                    }
                }
            }
        }
        return (true, color)
    }

    /// Has Euler circuit? (undirected: connected + all degrees even)
    func hasEulerCircuit(_ adj: [[Int]]) -> Bool {
        computations += 1
        guard isConnected(adj) else { return false }
        let degs = degrees(adj)
        return degs.allSatisfy { $0 % 2 == 0 }
    }

    /// Has Euler path? (undirected: connected + exactly 0 or 2 odd-degree vertices)
    func hasEulerPath(_ adj: [[Int]]) -> Bool {
        computations += 1
        guard isConnected(adj) else { return false }
        let oddCount = degrees(adj).filter { $0 % 2 != 0 }.count
        return oddCount == 0 || oddCount == 2
    }

    /// Graph diameter: max of all shortest path distances
    func diameter(adj: [[Double]]) -> Double {
        computations += 1
        let allPairs = floydWarshall(adj: adj)
        var maxDist: Double = 0
        for row in allPairs {
            for d in row where d < Double.infinity {
                maxDist = max(maxDist, d)
            }
        }
        return maxDist
    }

    /// Vertex eccentricity: max distance from vertex to any other vertex
    func eccentricity(adj: [[Double]], vertex: Int) -> Double {
        computations += 1
        let (dists, _) = dijkstra(adj: adj, source: vertex)
        return dists.filter { $0 < Double.infinity }.max() ?? 0
    }

    /// Clustering coefficient for a vertex: |edges among neighbors| / C(deg, 2)
    func clusteringCoefficient(_ adj: [[Int]], vertex: Int) -> Double {
        computations += 1
        let n = adj.count
        var neighbors: [Int] = []
        for v in 0..<n where adj[vertex][v] > 0 { neighbors.append(v) }
        let k = neighbors.count
        guard k >= 2 else { return 0 }
        var triangles = 0
        for i in 0..<k {
            for j in (i+1)..<k {
                if adj[neighbors[i]][neighbors[j]] > 0 { triangles += 1 }
            }
        }
        return 2.0 * Double(triangles) / Double(k * (k - 1))
    }

    /// Average clustering coefficient of entire graph
    func averageClusteringCoefficient(_ adj: [[Int]]) -> Double {
        computations += 1
        let n = adj.count
        guard n > 0 else { return 0 }
        var sum: Double = 0
        for v in 0..<n { sum += clusteringCoefficient(adj, vertex: v) }
        return sum / Double(n)
    }

    /// Page Rank algorithm
    func pageRank(_ adj: [[Int]], dampingFactor d: Double = 0.85, maxIter: Int = 100, tol: Double = 1e-8) -> [Double] {
        computations += 1
        let n = adj.count
        guard n > 0 else { return [] }
        let outDegree = adj.map { $0.reduce(0, +) }
        var rank = [Double](repeating: 1.0 / Double(n), count: n)

        for _ in 0..<maxIter {
            var newRank = [Double](repeating: (1.0 - d) / Double(n), count: n)
            for i in 0..<n {
                if outDegree[i] > 0 {
                    for j in 0..<n where adj[i][j] > 0 {
                        newRank[j] += d * rank[i] / Double(outDegree[i])
                    }
                } else {
                    // Dangling node: distribute equally
                    for j in 0..<n { newRank[j] += d * rank[i] / Double(n) }
                }
            }
            var diff: Double = 0
            for i in 0..<n { diff += abs(newRank[i] - rank[i]) }
            rank = newRank
            if diff < tol { break }
        }
        return rank
    }

    var status: String {
        """
        ╔═══════════════════════════════════════════════════════════╗
        ║  🕸️ GRAPH THEORY & COMBINATORICS v42.1                    ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Computations:     \(computations)
        ║  Representation:
        ║    • Adjacency matrix, weighted, degree, Laplacian
        ║  Shortest Paths:
        ║    • Dijkstra, Floyd-Warshall, Bellman-Ford
        ║  Traversal:
        ║    • BFS, DFS, topological sort
        ║  Spanning Trees:
        ║    • Kruskal, Prim (MST)
        ║  Properties:
        ║    • Connected, bipartite, Euler circuit/path
        ║    • Diameter, eccentricity, clustering coefficient
        ║    • PageRank
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - 🔮 SPECIAL FUNCTIONS & QUANTUM COMPUTING MATH ENGINE
// Phase 42.2: Bessel, Legendre, Hermite, Laguerre, spherical harmonics,
// quantum gates, qubit operations, entanglement, error function extensions
// ═══════════════════════════════════════════════════════════════════════════════

class SpecialFunctionsEngine {
    static let shared = SpecialFunctionsEngine()
    private var computations: Int = 0

    // ═══════════════════════════════════════════════════════════════
    // MARK: ORTHOGONAL POLYNOMIALS
    // ═══════════════════════════════════════════════════════════════

    /// Legendre polynomial P_n(x) via recurrence: (n+1)P_{n+1} = (2n+1)xP_n - nP_{n-1}
    func legendre(n: Int, x: Double) -> Double {
        computations += 1
        guard n >= 0 else { return 0 }
        if n == 0 { return 1 }
        if n == 1 { return x }
        var p0: Double = 1, p1: Double = x
        for k in 1..<n {
            let p2: Double = ((2.0 * Double(k) + 1.0) * x * p1 - Double(k) * p0) / Double(k + 1)
            p0 = p1; p1 = p2
        }
        return p1
    }

    /// Associated Legendre polynomial P_l^m(x)
    func associatedLegendre(l: Int, m: Int, x: Double) -> Double {
        computations += 1
        let absM = abs(m)
        guard absM <= l else { return 0 }

        // Start with P_m^m
        var pmm: Double = 1
        if absM > 0 {
            let somx2: Double = Foundation.sqrt(1.0 - x * x)
            var fact: Double = 1
            for i in 1...absM {
                pmm *= -fact * somx2
                fact += 2.0
                _ = i // suppress warning
            }
        }
        if l == absM { return pmm }

        // P_{m+1}^m
        var pmmp1: Double = x * Double(2 * absM + 1) * pmm
        if l == absM + 1 { return pmmp1 }

        // Recurrence upward
        for ll in (absM + 2)...l {
            let pll: Double = (x * Double(2 * ll - 1) * pmmp1 - Double(ll + absM - 1) * pmm) / Double(ll - absM)
            pmm = pmmp1
            pmmp1 = pll
        }
        return pmmp1
    }

    /// Hermite polynomial H_n(x) (physicist's convention): H_{n+1} = 2xH_n - 2nH_{n-1}
    func hermite(n: Int, x: Double) -> Double {
        computations += 1
        if n == 0 { return 1 }
        if n == 1 { return 2 * x }
        var h0: Double = 1, h1: Double = 2 * x
        for k in 1..<n {
            let h2: Double = 2.0 * x * h1 - 2.0 * Double(k) * h0
            h0 = h1; h1 = h2
        }
        return h1
    }

    /// Laguerre polynomial L_n(x): L_{n+1} = ((2n+1-x)L_n - nL_{n-1})/(n+1)
    func laguerre(n: Int, x: Double) -> Double {
        computations += 1
        if n == 0 { return 1 }
        if n == 1 { return 1 - x }
        var l0: Double = 1, l1: Double = 1 - x
        for k in 1..<n {
            let l2: Double = ((2.0 * Double(k) + 1.0 - x) * l1 - Double(k) * l0) / Double(k + 1)
            l0 = l1; l1 = l2
        }
        return l1
    }

    /// Associated Laguerre polynomial L_n^α(x)
    func associatedLaguerre(n: Int, alpha: Double, x: Double) -> Double {
        computations += 1
        if n == 0 { return 1 }
        if n == 1 { return 1 + alpha - x }
        var l0: Double = 1, l1: Double = 1 + alpha - x
        for k in 1..<n {
            let kd: Double = Double(k)
            let l2: Double = ((2.0 * kd + 1.0 + alpha - x) * l1 - (kd + alpha) * l0) / (kd + 1.0)
            l0 = l1; l1 = l2
        }
        return l1
    }

    /// Chebyshev polynomial T_n(x): T_{n+1} = 2xT_n - T_{n-1}
    func chebyshevT(n: Int, x: Double) -> Double {
        computations += 1
        if n == 0 { return 1 }
        if n == 1 { return x }
        var t0: Double = 1, t1: Double = x
        for _ in 1..<n {
            let t2: Double = 2.0 * x * t1 - t0
            t0 = t1; t1 = t2
        }
        return t1
    }

    /// Chebyshev polynomial U_n(x): U_{n+1} = 2xU_n - U_{n-1}
    func chebyshevU(n: Int, x: Double) -> Double {
        computations += 1
        if n == 0 { return 1 }
        if n == 1 { return 2 * x }
        var u0: Double = 1, u1: Double = 2 * x
        for _ in 1..<n {
            let u2: Double = 2.0 * x * u1 - u0
            u0 = u1; u1 = u2
        }
        return u1
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: BESSEL FUNCTIONS
    // ═══════════════════════════════════════════════════════════════

    /// Bessel function J_n(x) of the first kind via series expansion
    func besselJ(n: Int, x: Double) -> Double {
        computations += 1
        let nd: Double = Double(n)
        var sum: Double = 0
        for k in 0..<50 {
            let kd: Double = Double(k)
            let sign: Double = (k % 2 == 0) ? 1.0 : -1.0
            // Direct computation to avoid overflow
            var term: Double = sign * pow(x / 2.0, nd + 2.0 * kd)
            for i in 1...max(1, k) { term /= Double(i) }
            for i in 1...max(1, n + k) { term /= Double(i) }
            sum += term
            if abs(term) < 1e-15 * abs(sum) { break }
        }
        return sum
    }

    /// Bessel function Y_n(x) of the second kind (Neumann function)
    /// Y_0(x) approximation via series
    func besselY0(x: Double) -> Double {
        computations += 1
        guard x > 0 else { return -Double.infinity }
        let j0: Double = besselJ(n: 0, x: x)
        let gamma: Double = 0.5772156649  // Euler-Mascheroni
        var sum: Double = 0
        var harmonic: Double = 0
        for k in 0..<30 {
            let kd: Double = Double(k)
            if k > 0 { harmonic += 1.0 / kd }
            let sign: Double = (k % 2 == 0) ? 1.0 : -1.0
            var term: Double = sign * harmonic * pow(x / 2.0, 2.0 * kd)
            for i in 1...max(1, k) { term /= Double(i) }
            for i in 1...max(1, k) { term /= Double(i) }
            sum += term
        }
        return (2.0 / .pi) * ((log(x / 2.0) + gamma) * j0 + sum)
    }

    /// Modified Bessel function I_n(x) of the first kind
    func besselI(n: Int, x: Double) -> Double {
        computations += 1
        let nd: Double = Double(n)
        var sum: Double = 0
        for k in 0..<50 {
            let kd: Double = Double(k)
            var term: Double = pow(x / 2.0, nd + 2.0 * kd)
            for i in 1...max(1, k) { term /= Double(i) }
            for i in 1...max(1, n + k) { term /= Double(i) }
            sum += term
            if abs(term) < 1e-15 * abs(sum) { break }
        }
        return sum
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: SPHERICAL HARMONICS
    // ═══════════════════════════════════════════════════════════════

    /// Real spherical harmonic Y_l^m(θ, φ) — returns real part
    func sphericalHarmonic(l: Int, m: Int, theta: Double, phi: Double) -> Double {
        computations += 1
        let absM = abs(m)
        // Normalization: √((2l+1)/(4π) · (l-|m|)!/(l+|m|)!)
        var factRatio: Double = 1
        for i in (l - absM + 1)...(l + absM) {
            factRatio *= Double(i)
        }
        let norm: Double = Foundation.sqrt((2.0 * Double(l) + 1.0) / (4.0 * .pi) / factRatio)
        let plm: Double = associatedLegendre(l: l, m: absM, x: cos(theta))

        if m > 0 {
            return norm * plm * Foundation.sqrt(2.0) * cos(Double(m) * phi)
        } else if m < 0 {
            return norm * plm * Foundation.sqrt(2.0) * sin(Double(absM) * phi)
        } else {
            return norm * plm
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: SPECIAL FUNCTIONS
    // ═══════════════════════════════════════════════════════════════

    /// Airy function Ai(x) — via series for small x
    func airyAi(x: Double) -> Double {
        computations += 1
        let c1: Double = 0.3550281  // 1/(3^(2/3)·Γ(2/3))
        let c2: Double = -0.2588194  // -1/(3^(1/3)·Γ(1/3))
        var f: Double = c1, g: Double = c2 * x
        var fk: Double = c1, gk: Double = c2 * x
        for k in 1..<30 {
            let kd: Double = Double(k)
            fk *= x * x * x / ((3.0 * kd - 1.0) * (3.0 * kd))
            gk *= x * x * x / ((3.0 * kd) * (3.0 * kd + 1.0))
            f += fk; g += gk
            if abs(fk) + abs(gk) < 1e-15 { break }
        }
        return f + g
    }

    /// Digamma function ψ(x) = d/dx ln(Γ(x)) — via recurrence + asymptotic
    func digamma(_ x: Double) -> Double {
        computations += 1
        var result: Double = 0
        var z: Double = x
        // Recurrence: ψ(x+1) = ψ(x) + 1/x
        while z < 7 {
            result -= 1.0 / z
            z += 1.0
        }
        // Asymptotic: ψ(z) ≈ ln(z) - 1/(2z) - 1/(12z²) + 1/(120z⁴) - ...
        result += log(z) - 0.5 / z
        let z2: Double = z * z
        result -= 1.0 / (12.0 * z2)
        result += 1.0 / (120.0 * z2 * z2)
        result -= 1.0 / (252.0 * z2 * z2 * z2)
        return result
    }

    /// Polygamma function ψ^(n)(x) — nth derivative of digamma
    func polygamma(order n: Int, x: Double) -> Double {
        computations += 1
        if n == 0 { return digamma(x) }
        // ψ^(n)(x) = (-1)^(n+1) · n! · Σ_{k=0}^∞ 1/(x+k)^(n+1)
        let sign: Double = (n % 2 == 0) ? -1.0 : 1.0
        var factN: Double = 1
        for i in 1...n { factN *= Double(i) }
        var sum: Double = 0
        for k in 0..<200 {
            let term: Double = 1.0 / pow(x + Double(k), Double(n + 1))
            sum += term
            if abs(term) < 1e-15 { break }
        }
        return sign * factN * sum
    }

    /// Elliptic integral K(m) — complete elliptic integral of the first kind
    /// K(m) = ∫₀^(π/2) dθ/√(1 - m·sin²θ) via AGM
    func ellipticK(m: Double) -> Double {
        computations += 1
        guard m < 1.0 else { return Double.infinity }
        var a: Double = 1.0, b: Double = Foundation.sqrt(1.0 - m)
        for _ in 0..<50 {
            let an: Double = (a + b) / 2.0
            let bn: Double = Foundation.sqrt(a * b)
            if abs(an - bn) < 1e-15 { a = an; break }
            a = an; b = bn
        }
        return .pi / (2.0 * a)
    }

    /// Elliptic integral E(m) — complete elliptic integral of the second kind
    func ellipticE(m: Double) -> Double {
        computations += 1
        guard m <= 1.0 else { return 0 }
        if m == 1.0 { return 1.0 }
        var a: Double = 1.0, b: Double = Foundation.sqrt(1.0 - m)
        var c: Double = Foundation.sqrt(m)
        var sum: Double = m / 2.0
        var pow2: Double = 1.0
        for _ in 0..<50 {
            let an: Double = (a + b) / 2.0
            let bn: Double = Foundation.sqrt(a * b)
            c = (a - b) / 2.0
            pow2 *= 2.0
            sum += pow2 * c * c
            if abs(c) < 1e-15 { a = an; break }
            a = an; b = bn
        }
        return .pi / (2.0 * a) * (1.0 - sum)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: QUANTUM COMPUTING MATH
    // ═══════════════════════════════════════════════════════════════

    /// Quantum state: 2x1 complex vector |ψ⟩ = α|0⟩ + β|1⟩
    typealias Qubit = (alpha: Complex, beta: Complex)

    /// Standard basis states
    static let ket0: Qubit = (Complex(1, 0), Complex(0, 0))
    static let ket1: Qubit = (Complex(0, 0), Complex(1, 0))
    static let ketPlus: Qubit = (Complex(1.0 / Foundation.sqrt(2.0), 0), Complex(1.0 / Foundation.sqrt(2.0), 0))
    static let ketMinus: Qubit = (Complex(1.0 / Foundation.sqrt(2.0), 0), Complex(-1.0 / Foundation.sqrt(2.0), 0))

    /// 2x2 quantum gate as [[Complex]]
    typealias Gate2x2 = [[Complex]]

    /// Pauli-X (NOT) gate: |0⟩↔|1⟩
    func pauliX() -> Gate2x2 {
        computations += 1
        return [[Complex.zero, Complex.one], [Complex.one, Complex.zero]]
    }

    /// Pauli-Y gate: σ_y = [[0, -i], [i, 0]]
    func pauliY() -> Gate2x2 {
        computations += 1
        return [[Complex.zero, Complex(0, -1)], [Complex(0, 1), Complex.zero]]
    }

    /// Pauli-Z gate: |0⟩→|0⟩, |1⟩→-|1⟩
    func pauliZ() -> Gate2x2 {
        computations += 1
        return [[Complex.one, Complex.zero], [Complex.zero, Complex(-1, 0)]]
    }

    /// Hadamard gate: H = (1/√2)[[1,1],[1,-1]]
    func hadamard() -> Gate2x2 {
        computations += 1
        let h: Double = 1.0 / Foundation.sqrt(2.0)
        return [[Complex(h, 0), Complex(h, 0)], [Complex(h, 0), Complex(-h, 0)]]
    }

    /// Phase gate S: [[1,0],[0,i]]
    func phaseS() -> Gate2x2 {
        computations += 1
        return [[Complex.one, Complex.zero], [Complex.zero, Complex.i]]
    }

    /// T gate (π/8 gate): [[1,0],[0,e^(iπ/4)]]
    func tGate() -> Gate2x2 {
        computations += 1
        return [[Complex.one, Complex.zero], [Complex.zero, Complex.euler(.pi / 4.0)]]
    }

    /// Rotation gate Rz(θ): [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
    func rotationZ(theta: Double) -> Gate2x2 {
        computations += 1
        return [[Complex.euler(-theta / 2.0), Complex.zero], [Complex.zero, Complex.euler(theta / 2.0)]]
    }

    /// Rotation gate Rx(θ): [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
    func rotationX(theta: Double) -> Gate2x2 {
        computations += 1
        let c: Double = cos(theta / 2.0)
        let s: Double = sin(theta / 2.0)
        return [[Complex(c, 0), Complex(0, -s)], [Complex(0, -s), Complex(c, 0)]]
    }

    /// Rotation gate Ry(θ): [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
    func rotationY(theta: Double) -> Gate2x2 {
        computations += 1
        let c: Double = cos(theta / 2.0)
        let s: Double = sin(theta / 2.0)
        return [[Complex(c, 0), Complex(-s, 0)], [Complex(s, 0), Complex(c, 0)]]
    }

    /// Apply 2x2 gate to qubit: |ψ'⟩ = U|ψ⟩
    func applyGate(_ gate: Gate2x2, to qubit: Qubit) -> Qubit {
        computations += 1
        let newAlpha: Complex = gate[0][0] * qubit.alpha + gate[0][1] * qubit.beta
        let newBeta: Complex = gate[1][0] * qubit.alpha + gate[1][1] * qubit.beta
        return (newAlpha, newBeta)
    }

    /// Measurement probabilities: P(|0⟩) = |α|², P(|1⟩) = |β|²
    func measurementProbabilities(_ qubit: Qubit) -> (p0: Double, p1: Double) {
        computations += 1
        let p0: Double = qubit.alpha.magnitude * qubit.alpha.magnitude
        let p1: Double = qubit.beta.magnitude * qubit.beta.magnitude
        return (p0, p1)
    }

    /// Bloch sphere coordinates: (θ, φ) from qubit state
    func blochCoordinates(_ qubit: Qubit) -> (theta: Double, phi: Double) {
        computations += 1
        let p0: Double = qubit.alpha.magnitude * qubit.alpha.magnitude
        let theta: Double = 2.0 * acos(min(1.0, Foundation.sqrt(p0)))
        let phi: Double = qubit.beta.phase - qubit.alpha.phase
        return (theta, phi)
    }

    /// Von Neumann entropy: S = -Tr(ρ log₂ ρ) for a pure state = 0
    /// For mixed state with eigenvalues λᵢ: S = -Σ λᵢ log₂(λᵢ)
    func vonNeumannEntropy(eigenvalues: [Double]) -> Double {
        computations += 1
        return -eigenvalues.reduce(0.0) { sum, lambda in
            lambda > 0 ? sum + lambda * log2(lambda) : sum
        }
    }

    /// Fidelity between two pure states: F = |⟨ψ|φ⟩|²
    func fidelity(_ psi: Qubit, _ phi: Qubit) -> Double {
        computations += 1
        let inner: Complex = psi.alpha * phi.alpha + psi.beta * phi.beta
        return inner.magnitude * inner.magnitude
    }

    /// Concurrence for a 2-qubit state (entanglement measure)
    /// For Bell states: C = 1 (maximally entangled)
    /// For product states: C = 0
    func concurrence(coefficients: [Complex]) -> Double {
        computations += 1
        guard coefficients.count == 4 else { return 0 }
        // C = 2|α₀₀·α₁₁ - α₀₁·α₁₀|
        let term: Complex = coefficients[0] * coefficients[3] - coefficients[1] * coefficients[2]
        return 2.0 * term.magnitude
    }

    var status: String {
        """
        ╔═══════════════════════════════════════════════════════════╗
        ║  🔮 SPECIAL FUNCTIONS & QUANTUM COMPUTING v42.2           ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Computations:     \(computations)
        ║  Orthogonal Polynomials:
        ║    • Legendre (P_n, P_l^m), Hermite, Laguerre
        ║    • Chebyshev (T_n, U_n), associated Laguerre
        ║  Bessel Functions:
        ║    • J_n(x), Y_0(x), I_n(x)
        ║  Special Functions:
        ║    • Spherical harmonics Y_l^m, Airy Ai
        ║    • Digamma, polygamma, elliptic K & E
        ║  Quantum Computing:
        ║    • Pauli X/Y/Z, Hadamard, S, T gates
        ║    • Rx/Ry/Rz rotations, gate application
        ║    • Measurement, Bloch sphere, fidelity
        ║    • Von Neumann entropy, concurrence
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MARK: - 🎛️ CONTROL THEORY ENGINE
// Phase 43.0: Transfer functions, PID control, stability analysis, Laplace
// transforms, Bode analysis, state-space, Nyquist, root locus, Ziegler-Nichols
// ═══════════════════════════════════════════════════════════════════════════════

class ControlTheoryEngine {
    static let shared = ControlTheoryEngine()
    private var computations: Int = 0

    // ═══ Transfer Functions ═══

    /// Evaluate a polynomial at s: coeffs[0]*s^n + coeffs[1]*s^(n-1) + ... + coeffs[n]
    func evaluatePolynomial(_ coeffs: [Double], at s: Complex) -> Complex {
        computations += 1
        var result = Complex(0, 0)
        for c in coeffs {
            result = result * s + Complex(c, 0)
        }
        return result
    }

    /// Transfer function H(s) = num(s)/den(s) evaluated at complex s
    func transferFunction(numerator: [Double], denominator: [Double], at s: Complex) -> Complex {
        computations += 1
        let num = evaluatePolynomial(numerator, at: s)
        let den = evaluatePolynomial(denominator, at: s)
        guard den.magnitude > 1e-15 else { return Complex(.infinity, 0) }
        return num / den
    }

    /// DC gain: H(0) = num[last]/den[last]
    func dcGain(numerator: [Double], denominator: [Double]) -> Double {
        computations += 1
        guard let numLast = numerator.last, let denLast = denominator.last, abs(denLast) > 1e-15 else { return .infinity }
        return numLast / denLast
    }

    /// Poles of the transfer function (roots of denominator) — quadratic
    func polesQuadratic(a: Double, b: Double, c: Double) -> [Complex] {
        computations += 1
        let disc = b * b - 4 * a * c
        if disc >= 0 {
            let sqrtDisc = Foundation.sqrt(disc)
            return [Complex((-b + sqrtDisc) / (2 * a), 0),
                    Complex((-b - sqrtDisc) / (2 * a), 0)]
        } else {
            let sqrtDisc = Foundation.sqrt(-disc)
            return [Complex(-b / (2 * a), sqrtDisc / (2 * a)),
                    Complex(-b / (2 * a), -sqrtDisc / (2 * a))]
        }
    }

    /// Check if system is stable (all poles have negative real parts)
    func isStable(poles: [Complex]) -> Bool {
        computations += 1
        return poles.allSatisfy { $0.real < 0 }
    }

    /// Routh-Hurwitz stability criterion for polynomial a0*s^n + a1*s^(n-1) + ... + an
    /// Returns true if all first-column elements are positive (stable system)
    func routhHurwitz(coefficients: [Double]) -> (stable: Bool, firstColumn: [Double]) {
        computations += 1
        let n = coefficients.count
        guard n >= 2 else { return (coefficients.allSatisfy { $0 > 0 }, coefficients) }

        let rows = n
        let cols = (n + 1) / 2
        var table = [[Double]](repeating: [Double](repeating: 0, count: cols), count: rows)

        // Fill first two rows
        for j in 0..<cols {
            if 2 * j < n { table[0][j] = coefficients[2 * j] }
            if 2 * j + 1 < n { table[1][j] = coefficients[2 * j + 1] }
        }

        // Compute remaining rows
        for i in 2..<rows {
            let a0 = table[i - 1][0]
            guard abs(a0) > 1e-15 else { break }
            for j in 0..<(cols - 1) {
                let top1 = table[i - 2][0]
                let top2 = (j + 1 < cols) ? table[i - 2][j + 1] : 0
                let bot1 = table[i - 1][0]
                let bot2 = (j + 1 < cols) ? table[i - 1][j + 1] : 0
                table[i][j] = (bot1 * top2 - top1 * bot2) / bot1
            }
        }

        var firstCol: [Double] = []
        for i in 0..<rows {
            firstCol.append(table[i][0])
        }

        let stable = firstCol.allSatisfy { $0 > 0 } || firstCol.allSatisfy { $0 < 0 }
        return (stable, firstCol)
    }

    // ═══ PID Control ═══

    /// PID controller output: u(t) = Kp*e + Ki*∫e dt + Kd*de/dt
    /// Given error history, returns control signal
    func pidOutput(kp: Double, ki: Double, kd: Double, error: Double, integralError: Double, derivativeError: Double) -> Double {
        computations += 1
        return kp * error + ki * integralError + kd * derivativeError
    }

    /// PID transfer function: C(s) = Kp + Ki/s + Kd*s = (Kd*s² + Kp*s + Ki)/s
    func pidTransferFunction(kp: Double, ki: Double, kd: Double, at s: Complex) -> Complex {
        computations += 1
        guard s.magnitude > 1e-15 else { return Complex(.infinity, 0) }
        let proportional = Complex(kp, 0)
        let integral = Complex(ki, 0) / s
        let derivative = Complex(kd, 0) * s
        return proportional + integral + derivative
    }

    /// Ziegler-Nichols tuning (based on critical gain Ku and oscillation period Tu)
    func zieglerNicholsPID(ku: Double, tu: Double) -> (kp: Double, ki: Double, kd: Double) {
        computations += 1
        let kp = 0.6 * ku
        let ti = tu / 2.0
        let td = tu / 8.0
        return (kp, kp / ti, kp * td)
    }

    /// Ziegler-Nichols P-only tuning
    func zieglerNicholsP(ku: Double) -> Double {
        computations += 1
        return 0.5 * ku
    }

    /// Ziegler-Nichols PI tuning
    func zieglerNicholsPI(ku: Double, tu: Double) -> (kp: Double, ki: Double) {
        computations += 1
        let kp = 0.45 * ku
        return (kp, kp / (tu / 1.2))
    }

    /// Cohen-Coon PID tuning (based on process gain K, time constant tau, dead time theta)
    func cohenCoonPID(K: Double, tau: Double, theta: Double) -> (kp: Double, ki: Double, kd: Double) {
        computations += 1
        let r = theta / tau
        let kp = (1.0 / K) * (tau / theta) * (1.35 + r / 4.0)
        let ti = theta * (2.5 - 2.0 * r) / (1.0 - 0.39 * r)
        let td = 0.37 * theta / (1.0 - 0.81 * r)
        return (kp, kp / ti, kp * td)
    }

    // ═══ Frequency Domain Analysis ═══

    /// Bode magnitude in dB: 20*log10(|H(jω)|)
    func bodeMagnitude(numerator: [Double], denominator: [Double], omega: Double) -> Double {
        computations += 1
        let s = Complex(0, omega)
        let h = transferFunction(numerator: numerator, denominator: denominator, at: s)
        return 20.0 * Foundation.log10(max(h.magnitude, 1e-15))
    }

    /// Bode phase in degrees: arg(H(jω))
    func bodePhase(numerator: [Double], denominator: [Double], omega: Double) -> Double {
        computations += 1
        let s = Complex(0, omega)
        let h = transferFunction(numerator: numerator, denominator: denominator, at: s)
        return atan2(h.imag, h.real) * 180.0 / .pi
    }

    /// Gain margin: dB at phase crossover (where phase = -180°)
    func gainMargin(numerator: [Double], denominator: [Double], omegaRange: (Double, Double) = (0.001, 1000), steps: Int = 10000) -> (marginDB: Double, omegaCrossover: Double) {
        computations += 1
        let logStart = Foundation.log10(omegaRange.0)
        let logEnd = Foundation.log10(omegaRange.1)
        var bestOmega = 0.0
        var bestPhaseDiff = Double.infinity

        for i in 0..<steps {
            let logOmega = logStart + (logEnd - logStart) * Double(i) / Double(steps - 1)
            let omega = Foundation.pow(10.0, logOmega)
            let phase = bodePhase(numerator: numerator, denominator: denominator, omega: omega)
            let diff = abs(phase + 180.0)
            if diff < bestPhaseDiff {
                bestPhaseDiff = diff
                bestOmega = omega
            }
        }

        let magDB = bodeMagnitude(numerator: numerator, denominator: denominator, omega: bestOmega)
        return (-magDB, bestOmega)
    }

    /// Phase margin: degrees above -180° at gain crossover (where |H| = 0 dB)
    func phaseMargin(numerator: [Double], denominator: [Double], omegaRange: (Double, Double) = (0.001, 1000), steps: Int = 10000) -> (marginDeg: Double, omegaCrossover: Double) {
        computations += 1
        let logStart = Foundation.log10(omegaRange.0)
        let logEnd = Foundation.log10(omegaRange.1)
        var bestOmega = 0.0
        var bestMagDiff = Double.infinity

        for i in 0..<steps {
            let logOmega = logStart + (logEnd - logStart) * Double(i) / Double(steps - 1)
            let omega = Foundation.pow(10.0, logOmega)
            let magDB = bodeMagnitude(numerator: numerator, denominator: denominator, omega: omega)
            let diff = abs(magDB)
            if diff < bestMagDiff {
                bestMagDiff = diff
                bestOmega = omega
            }
        }

        let phase = bodePhase(numerator: numerator, denominator: denominator, omega: bestOmega)
        return (180.0 + phase, bestOmega)
    }

    // ═══ State-Space ═══

    /// State-space representation: dx/dt = Ax + Bu, y = Cx + Du
    /// Evaluate next state using Euler method: x(t+dt) = x(t) + dt*(A*x + B*u)
    func stateSpaceStep(A: [[Double]], B: [[Double]], x: [Double], u: [Double], dt: Double) -> [Double] {
        computations += 1
        let n = x.count
        var xNext = [Double](repeating: 0, count: n)
        for i in 0..<n {
            var dxdt = 0.0
            for j in 0..<n {
                dxdt += A[i][j] * x[j]
            }
            for j in 0..<u.count {
                if j < B[i].count {
                    dxdt += B[i][j] * u[j]
                }
            }
            xNext[i] = x[i] + dt * dxdt
        }
        return xNext
    }

    /// Output equation: y = Cx + Du
    func stateSpaceOutput(C: [[Double]], D: [[Double]], x: [Double], u: [Double]) -> [Double] {
        computations += 1
        let p = C.count
        var y = [Double](repeating: 0, count: p)
        for i in 0..<p {
            for j in 0..<x.count {
                if j < C[i].count { y[i] += C[i][j] * x[j] }
            }
            for j in 0..<u.count {
                if i < D.count && j < D[i].count { y[i] += D[i][j] * u[j] }
            }
        }
        return y
    }

    /// Controllability matrix: [B, AB, A²B, ..., A^(n-1)B] — system is controllable if rank = n
    func controllabilityMatrix(A: [[Double]], B: [[Double]]) -> [[Double]] {
        computations += 1
        let n = A.count
        let m = B[0].count
        var ctrb = [[Double]](repeating: [Double](repeating: 0, count: n * m), count: n)

        // Start with B
        var power = B
        for col in 0..<m {
            for row in 0..<n {
                ctrb[row][col] = power[row][col]
            }
        }

        // A^k * B for k = 1..n-1
        for k in 1..<n {
            var next = [[Double]](repeating: [Double](repeating: 0, count: m), count: n)
            for i in 0..<n {
                for j in 0..<m {
                    for p in 0..<n {
                        next[i][j] += A[i][p] * power[p][j]
                    }
                }
            }
            power = next
            for col in 0..<m {
                for row in 0..<n {
                    ctrb[row][k * m + col] = power[row][col]
                }
            }
        }
        return ctrb
    }

    // ═══ Laplace & Time Domain ═══

    /// First-order step response: y(t) = K * (1 - e^(-t/τ))
    func firstOrderStepResponse(K: Double, tau: Double, t: Double) -> Double {
        computations += 1
        return K * (1.0 - Foundation.exp(-t / tau))
    }

    /// Second-order step response (underdamped ζ < 1)
    func secondOrderStepResponse(K: Double, wn: Double, zeta: Double, t: Double) -> Double {
        computations += 1
        guard zeta < 1.0 else {
            // Critically/overdamped: approximate
            let s1 = -wn * (zeta + Foundation.sqrt(zeta * zeta - 1))
            let s2 = -wn * (zeta - Foundation.sqrt(zeta * zeta - 1))
            return K * (1.0 + (s1 * Foundation.exp(s2 * t) - s2 * Foundation.exp(s1 * t)) / (s2 - s1))
        }
        let wd = wn * Foundation.sqrt(1 - zeta * zeta)
        let env = Foundation.exp(-zeta * wn * t)
        return K * (1.0 - env * (Foundation.cos(wd * t) + (zeta / Foundation.sqrt(1 - zeta * zeta)) * Foundation.sin(wd * t)))
    }

    /// Rise time estimate for second-order underdamped: tr ≈ (π - arccos(ζ)) / ωd
    func riseTime(wn: Double, zeta: Double) -> Double {
        computations += 1
        let wd = wn * Foundation.sqrt(1 - zeta * zeta)
        return (.pi - Foundation.acos(zeta)) / wd
    }

    /// Settling time (2% criterion): ts ≈ 4 / (ζωn)
    func settlingTime(wn: Double, zeta: Double) -> Double {
        computations += 1
        return 4.0 / (zeta * wn)
    }

    /// Peak time: tp = π / ωd
    func peakTime(wn: Double, zeta: Double) -> Double {
        computations += 1
        let wd = wn * Foundation.sqrt(1 - zeta * zeta)
        return .pi / wd
    }

    /// Maximum overshoot percentage: Mp = exp(-ζπ / √(1-ζ²)) × 100
    func overshoot(zeta: Double) -> Double {
        computations += 1
        return Foundation.exp(-zeta * .pi / Foundation.sqrt(1 - zeta * zeta)) * 100.0
    }

    /// Bandwidth frequency (3dB): ωbw ≈ ωn * √(1-2ζ² + √(4ζ⁴-4ζ²+2))
    func bandwidth(wn: Double, zeta: Double) -> Double {
        computations += 1
        let z2 = zeta * zeta
        return wn * Foundation.sqrt(1 - 2*z2 + Foundation.sqrt(4*z2*z2 - 4*z2 + 2))
    }

    // ═══ Lead-Lag Compensator ═══

    /// Lead compensator: C(s) = Kc * (s + z) / (s + p) where p > z (adds phase lead)
    func leadCompensator(kc: Double, zero: Double, pole: Double, at s: Complex) -> Complex {
        computations += 1
        let num = s + Complex(zero, 0)
        let den = s + Complex(pole, 0)
        guard den.magnitude > 1e-15 else { return Complex(.infinity, 0) }
        return Complex(kc, 0) * num / den
    }

    /// Lag compensator: C(s) = Kc * (s + z) / (s + p) where p < z (increases DC gain)
    func lagCompensator(kc: Double, zero: Double, pole: Double, at s: Complex) -> Complex {
        computations += 1
        return leadCompensator(kc: kc, zero: zero, pole: pole, at: s)
    }

    /// Maximum phase lead: φmax = arcsin((p-z)/(p+z)) (for lead compensator, p > z)
    func maxPhaseLead(zero: Double, pole: Double) -> Double {
        computations += 1
        return Foundation.asin((pole - zero) / (pole + zero)) * 180.0 / .pi
    }

    var status: String {
        """
        ╔═══════════════════════════════════════════════════════════╗
        ║  🎛️ CONTROL THEORY ENGINE v43.0                          ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Computations:     \(computations)
        ║  Transfer Functions:
        ║    • H(s) evaluation, DC gain, quadratic poles
        ║    • Stability check, Routh-Hurwitz criterion
        ║  PID Control:
        ║    • PID output, PID transfer function
        ║    • Ziegler-Nichols (P, PI, PID), Cohen-Coon
        ║  Frequency Domain:
        ║    • Bode magnitude & phase, gain margin, phase margin
        ║  State-Space:
        ║    • State evolution (Euler), output equation
        ║    • Controllability matrix
        ║  Time Domain:
        ║    • 1st/2nd order step response
        ║    • Rise time, settling time, peak time, overshoot
        ║    • Bandwidth
        ║  Compensators:
        ║    • Lead/Lag compensators, max phase lead
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}
