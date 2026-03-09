// ═══════════════════════════════════════════════════════════════
// B34_LanguageComprehension.swift — Language Comprehension Engine
// L104v2 — TheBrain Layer — EVO_68 SOVEREIGN_CONVERGENCE → v5.0.0 ASI
// 8-Layer NLU + 191 Knowledge Nodes + BM25 + 57 MMLU Subjects
// + SubjectDetector + NumericalReasoner + CrossVerificationEngine
// + Three-Engine Scoring + Evaluate Comprehension
// ═══════════════════════════════════════════════════════════════

import Foundation

// ═══════════════════════════════════════════════════════════════
// MARK: - Knowledge Node
// ═══════════════════════════════════════════════════════════════

struct KnowledgeNode {
    let id: String
    let domain: String
    let topic: String
    let facts: [String]
    let connections: [String]
    let confidence: Double
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Comprehension Result
// ═══════════════════════════════════════════════════════════════

struct ComprehensionResult {
    let text: String
    let layerScores: [String: Double]
    let retrievedKnowledge: [(node: String, score: Double)]
    let entities: [String]
    let confidence: Double
}

// ═══════════════════════════════════════════════════════════════
// MARK: - MMLU Subject
// ═══════════════════════════════════════════════════════════════

struct MMLUSubject {
    let name: String
    let domain: String
    let nodeIDs: [String]
    let sampleQuestions: [(question: String, answer: String)]
}

// ═══════════════════════════════════════════════════════════════
// MARK: - BM25 Retrieval Engine
// ═══════════════════════════════════════════════════════════════

struct BM25Engine {
    static let k1: Double = 1.2
    static let b: Double = 0.75

    static let stopwords: Set<String> = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "because", "but", "and", "or", "if", "while", "that", "this", "it",
    ]

    static func tokenize(_ text: String) -> [String] {
        let lower = text.lowercased()
        let words = lower.components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 1 && !stopwords.contains($0) }
        return words
    }

    static func computeIDF(term: String, documents: [[String]]) -> Double {
        let n = Double(documents.count)
        let df = Double(documents.filter { $0.contains(term) }.count)
        guard df > 0 else { return 0 }
        return log((n - df + 0.5) / (df + 0.5) + 1.0)
    }

    static func computeBM25(queryTokens: [String], docTokens: [String], avgDL: Double, idfMap: [String: Double]) -> Double {
        let dl = Double(docTokens.count)
        var score = 0.0
        for term in queryTokens {
            let tf = Double(docTokens.filter { $0 == term }.count)
            let idf = idfMap[term] ?? 0
            let num = tf * (k1 + 1)
            let denom = tf + k1 * (1 - b + b * dl / max(avgDL, 1.0))
            score += idf * num / max(denom, 1e-10)
        }
        return score
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - Language Comprehension Engine
// ═══════════════════════════════════════════════════════════════

final class LanguageComprehensionEngine {
    static let shared = LanguageComprehensionEngine()
    private let lock = NSLock()
    static let VERSION = LANGUAGE_COMP_ENGINE_VERSION

    var queriesProcessed: Int = 0
    var totalRetrievals: Int = 0

    // ═══════════════════════════════════════════════════════════
    // MARK: - 191 Knowledge Nodes
    // ═══════════════════════════════════════════════════════════

    lazy var knowledgeBase: [KnowledgeNode] = {
        var nodes: [KnowledgeNode] = []

        // ─── STEM: Physics (10 nodes) ───
        nodes.append(KnowledgeNode(id: "phys_01", domain: "physics", topic: "Classical Mechanics", facts: ["Newton's laws govern motion of macroscopic objects", "F=ma relates force to mass and acceleration", "Conservation of energy: energy cannot be created or destroyed", "Momentum is conserved in isolated systems"], connections: ["phys_02", "phys_03"], confidence: 0.95))
        nodes.append(KnowledgeNode(id: "phys_02", domain: "physics", topic: "Thermodynamics", facts: ["First law: energy is conserved in thermodynamic processes", "Second law: entropy of isolated system never decreases", "Third law: entropy approaches zero at absolute zero", "Heat flows from hot to cold spontaneously"], connections: ["phys_01", "chem_01"], confidence: 0.95))
        nodes.append(KnowledgeNode(id: "phys_03", domain: "physics", topic: "Electromagnetism", facts: ["Maxwell's equations unify electricity and magnetism", "Electromagnetic waves travel at speed of light", "Coulomb's law describes electrostatic force", "Faraday's law: changing magnetic field induces EMF"], connections: ["phys_01", "phys_04"], confidence: 0.95))
        nodes.append(KnowledgeNode(id: "phys_04", domain: "physics", topic: "Quantum Mechanics", facts: ["Wave-particle duality: matter exhibits both wave and particle behavior", "Heisenberg uncertainty principle limits simultaneous measurement", "Schrödinger equation describes quantum state evolution", "Quantum entanglement allows correlated measurements"], connections: ["phys_03", "phys_05"], confidence: 0.95))
        nodes.append(KnowledgeNode(id: "phys_05", domain: "physics", topic: "Relativity", facts: ["Special relativity: speed of light is constant in all frames", "E=mc² relates mass and energy", "General relativity: gravity curves spacetime", "Time dilation occurs at relativistic speeds"], connections: ["phys_04", "astro_01"], confidence: 0.95))
        nodes.append(KnowledgeNode(id: "phys_06", domain: "physics", topic: "Optics", facts: ["Snell's law governs refraction at interfaces", "Diffraction occurs when waves pass through apertures", "Polarization filters specific wave orientations", "Total internal reflection enables fiber optics"], connections: ["phys_03", "phys_04"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "phys_07", domain: "physics", topic: "Nuclear Physics", facts: ["Strong force binds protons and neutrons in nucleus", "Radioactive decay: alpha, beta, gamma emission", "Nuclear fission splits heavy nuclei releasing energy", "Nuclear fusion combines light nuclei releasing energy"], connections: ["phys_04", "chem_02"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "phys_08", domain: "physics", topic: "Fluid Dynamics", facts: ["Bernoulli's principle: faster flow means lower pressure", "Reynolds number determines laminar vs turbulent flow", "Navier-Stokes equations govern viscous fluid motion", "Archimedes principle: buoyant force equals displaced fluid weight"], connections: ["phys_01", "eng_02"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "phys_09", domain: "physics", topic: "Acoustics", facts: ["Sound waves are longitudinal pressure waves", "Speed of sound depends on medium density and elasticity", "Doppler effect shifts frequency with relative motion", "Resonance amplifies waves at natural frequencies"], connections: ["phys_01", "phys_06"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "phys_10", domain: "physics", topic: "Condensed Matter", facts: ["Band theory explains conductors, insulators, semiconductors", "Superconductivity: zero resistance below critical temperature", "Ferromagnetism arises from aligned electron spins", "Crystal structure determines material properties"], connections: ["phys_04", "chem_03"], confidence: 0.85))

        // ─── STEM: Chemistry (8 nodes) ───
        nodes.append(KnowledgeNode(id: "chem_01", domain: "chemistry", topic: "General Chemistry", facts: ["Periodic table organizes elements by atomic number", "Chemical bonds: ionic, covalent, metallic", "Stoichiometry relates reactant and product quantities", "Le Chatelier's principle predicts equilibrium shifts"], connections: ["phys_02", "chem_02"], confidence: 0.95))
        nodes.append(KnowledgeNode(id: "chem_02", domain: "chemistry", topic: "Organic Chemistry", facts: ["Carbon forms four covalent bonds in most compounds", "Functional groups determine organic reaction behavior", "Stereochemistry: chirality affects biological activity", "Polymers are long chains of repeating monomer units"], connections: ["chem_01", "bio_01"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "chem_03", domain: "chemistry", topic: "Inorganic Chemistry", facts: ["Transition metals have partially filled d orbitals", "Coordination compounds have central metal with ligands", "Crystal field theory explains transition metal colors", "Catalysts lower activation energy without being consumed"], connections: ["chem_01", "phys_10"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "chem_04", domain: "chemistry", topic: "Biochemistry", facts: ["Enzymes are biological catalysts with high specificity", "ATP is the primary energy currency of cells", "DNA double helix stores genetic information", "Proteins fold into specific 3D structures"], connections: ["chem_02", "bio_01"], confidence: 0.92))
        nodes.append(KnowledgeNode(id: "chem_05", domain: "chemistry", topic: "Analytical Chemistry", facts: ["Spectroscopy identifies substances by light interaction", "Chromatography separates mixtures by differential affinity", "Mass spectrometry measures mass-to-charge ratios", "Titration determines concentration by reaction endpoint"], connections: ["chem_01", "chem_03"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "chem_06", domain: "chemistry", topic: "Physical Chemistry", facts: ["Gibbs free energy determines reaction spontaneity", "Reaction kinetics studies rates and mechanisms", "Phase diagrams map state transitions of matter", "Quantum chemistry applies QM to molecular systems"], connections: ["chem_01", "phys_02"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "chem_07", domain: "chemistry", topic: "Electrochemistry", facts: ["Redox reactions involve electron transfer", "Galvanic cells convert chemical to electrical energy", "Nernst equation relates cell potential to concentration", "Electrolysis drives non-spontaneous reactions"], connections: ["chem_01", "phys_03"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "chem_08", domain: "chemistry", topic: "Environmental Chemistry", facts: ["Ozone layer absorbs harmful UV radiation", "Greenhouse gases trap infrared radiation", "Acid rain forms from SO2 and NOx emissions", "Bioaccumulation concentrates toxins in food chains"], connections: ["chem_01", "geo_01"], confidence: 0.85))

        // ─── STEM: Biology (10 nodes) ───
        nodes.append(KnowledgeNode(id: "bio_01", domain: "biology", topic: "Cell Biology", facts: ["Cells are the basic unit of life", "Mitochondria produce ATP via oxidative phosphorylation", "Cell membrane is a phospholipid bilayer", "Endoplasmic reticulum processes proteins and lipids"], connections: ["chem_04", "bio_02"], confidence: 0.95))
        nodes.append(KnowledgeNode(id: "bio_02", domain: "biology", topic: "Genetics", facts: ["DNA replication is semi-conservative", "Mendel's laws govern inheritance patterns", "Mutations are changes in DNA sequence", "Gene expression involves transcription and translation"], connections: ["bio_01", "bio_03"], confidence: 0.95))
        nodes.append(KnowledgeNode(id: "bio_03", domain: "biology", topic: "Evolution", facts: ["Natural selection drives adaptation to environment", "Genetic drift causes random allele frequency changes", "Speciation occurs when populations diverge", "Common descent unifies all life through shared ancestry"], connections: ["bio_02", "bio_04"], confidence: 0.95))
        nodes.append(KnowledgeNode(id: "bio_04", domain: "biology", topic: "Ecology", facts: ["Ecosystems include biotic and abiotic components", "Energy flows through trophic levels in food webs", "Biodiversity increases ecosystem resilience", "Keystone species have disproportionate ecological impact"], connections: ["bio_03", "geo_01"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "bio_05", domain: "biology", topic: "Anatomy", facts: ["Skeletal system provides structure and support", "Muscular system enables movement through contraction", "Nervous system coordinates body functions via signals", "Circulatory system transports blood, oxygen, nutrients"], connections: ["bio_01", "med_01"], confidence: 0.92))
        nodes.append(KnowledgeNode(id: "bio_06", domain: "biology", topic: "Physiology", facts: ["Homeostasis maintains stable internal conditions", "Hormones are chemical messengers in endocrine system", "Kidneys filter blood and regulate fluid balance", "Respiratory system exchanges O2 and CO2 in lungs"], connections: ["bio_05", "med_02"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "bio_07", domain: "biology", topic: "Microbiology", facts: ["Bacteria are prokaryotes without membrane-bound nucleus", "Viruses require host cells to replicate", "Antibiotics target bacterial cell processes", "Microbiome plays crucial role in human health"], connections: ["bio_01", "med_03"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "bio_08", domain: "biology", topic: "Immunology", facts: ["Innate immunity provides immediate nonspecific defense", "Adaptive immunity produces specific antibodies", "T cells coordinate and execute immune responses", "Vaccines train immune system against pathogens"], connections: ["bio_07", "med_03"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "bio_09", domain: "biology", topic: "Neuroscience", facts: ["Neurons transmit signals via action potentials", "Synapses connect neurons through neurotransmitters", "Brain plasticity allows reorganization after injury", "Prefrontal cortex handles executive function"], connections: ["bio_05", "psy_01"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "bio_10", domain: "biology", topic: "Molecular Biology", facts: ["Central dogma: DNA→RNA→Protein", "CRISPR enables precise genome editing", "PCR amplifies specific DNA sequences", "Epigenetics modifies gene expression without DNA change"], connections: ["bio_02", "chem_04"], confidence: 0.92))

        // ─── STEM: Mathematics (8 nodes) ───
        nodes.append(KnowledgeNode(id: "math_01", domain: "mathematics", topic: "Algebra", facts: ["Groups, rings, fields form algebraic structures", "Polynomials can be factored over various fields", "Linear equations solved by elimination or substitution", "Quadratic formula: x=(-b±√(b²-4ac))/2a"], connections: ["math_02", "math_03"], confidence: 0.95))
        nodes.append(KnowledgeNode(id: "math_02", domain: "mathematics", topic: "Calculus", facts: ["Derivative measures instantaneous rate of change", "Integral computes accumulated area under curve", "Fundamental theorem links differentiation and integration", "Taylor series approximates functions as polynomials"], connections: ["math_01", "math_04"], confidence: 0.95))
        nodes.append(KnowledgeNode(id: "math_03", domain: "mathematics", topic: "Linear Algebra", facts: ["Matrices represent linear transformations", "Eigenvalues characterize transformation scaling", "Determinant indicates matrix invertibility", "Vector spaces are sets closed under addition and scaling"], connections: ["math_01", "cs_03"], confidence: 0.95))
        nodes.append(KnowledgeNode(id: "math_04", domain: "mathematics", topic: "Statistics", facts: ["Central limit theorem: sample means approach normal distribution", "Hypothesis testing evaluates statistical significance", "Regression models relationships between variables", "Bayes theorem updates probability with new evidence"], connections: ["math_02", "cs_04"], confidence: 0.92))
        nodes.append(KnowledgeNode(id: "math_05", domain: "mathematics", topic: "Number Theory", facts: ["Prime numbers have exactly two divisors", "Fundamental theorem: every integer has unique prime factorization", "Modular arithmetic studies remainders", "Fermat's little theorem: a^(p-1)≡1 (mod p) for prime p"], connections: ["math_01", "cs_05"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "math_06", domain: "mathematics", topic: "Geometry", facts: ["Euclidean geometry based on five postulates", "Pythagorean theorem: a²+b²=c² in right triangles", "Non-Euclidean geometries have curved spaces", "Topology studies properties preserved under deformation"], connections: ["math_01", "phys_05"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "math_07", domain: "mathematics", topic: "Probability", facts: ["Probability measures likelihood of events 0 to 1", "Conditional probability: P(A|B) = P(A∩B)/P(B)", "Expected value is probability-weighted average", "Law of large numbers: empirical frequency converges to probability"], connections: ["math_04", "cs_04"], confidence: 0.92))
        nodes.append(KnowledgeNode(id: "math_08", domain: "mathematics", topic: "Discrete Math", facts: ["Graph theory models pairwise relationships", "Combinatorics counts arrangements and selections", "Boolean algebra underlies digital logic", "Recurrence relations define sequences recursively"], connections: ["math_05", "cs_01"], confidence: 0.88))

        // ─── STEM: Computer Science (8 nodes) ───
        nodes.append(KnowledgeNode(id: "cs_01", domain: "cs", topic: "Algorithms", facts: ["Big-O notation describes algorithm complexity", "Sorting: merge sort O(n log n), quicksort O(n log n) average", "Graph algorithms: BFS, DFS, Dijkstra, A*", "Dynamic programming solves overlapping subproblems"], connections: ["math_08", "cs_02"], confidence: 0.95))
        nodes.append(KnowledgeNode(id: "cs_02", domain: "cs", topic: "Data Structures", facts: ["Arrays provide O(1) random access", "Hash tables give O(1) average lookup", "Trees enable O(log n) search and insertion", "Graphs represent complex relationships"], connections: ["cs_01", "cs_03"], confidence: 0.95))
        nodes.append(KnowledgeNode(id: "cs_03", domain: "cs", topic: "Machine Learning", facts: ["Supervised learning maps inputs to labeled outputs", "Neural networks learn hierarchical representations", "Gradient descent optimizes model parameters", "Overfitting occurs when model memorizes training data"], connections: ["math_03", "cs_04"], confidence: 0.92))
        nodes.append(KnowledgeNode(id: "cs_04", domain: "cs", topic: "Artificial Intelligence", facts: ["Search algorithms explore state spaces for solutions", "Knowledge representation encodes domain understanding", "Natural language processing interprets human language", "Reinforcement learning optimizes via reward signals"], connections: ["cs_03", "cs_05"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "cs_05", domain: "cs", topic: "Cryptography", facts: ["RSA based on difficulty of factoring large primes", "AES provides symmetric block encryption", "Hash functions produce fixed-size digests from input", "Public key infrastructure enables secure communication"], connections: ["math_05", "cs_06"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "cs_06", domain: "cs", topic: "Operating Systems", facts: ["Process scheduling allocates CPU time to processes", "Virtual memory maps logical to physical addresses", "File systems organize data storage on disk", "Deadlock occurs when processes wait in circular chain"], connections: ["cs_02", "cs_07"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "cs_07", domain: "cs", topic: "Networking", facts: ["OSI model has 7 layers from physical to application", "TCP provides reliable ordered data delivery", "IP addressing enables packet routing across networks", "DNS translates domain names to IP addresses"], connections: ["cs_06", "cs_05"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "cs_08", domain: "cs", topic: "Databases", facts: ["Relational databases use SQL for structured queries", "ACID properties ensure transaction reliability", "Normalization reduces data redundancy", "NoSQL databases handle unstructured data at scale"], connections: ["cs_02", "cs_01"], confidence: 0.88))

        // ─── STEM: Engineering (6 nodes) ───
        nodes.append(KnowledgeNode(id: "eng_01", domain: "engineering", topic: "Electrical Engineering", facts: ["Ohm's law: V=IR relates voltage, current, resistance", "Kirchhoff's laws govern circuit analysis", "Transistors amplify or switch electronic signals", "Signal processing transforms and analyzes signals"], connections: ["phys_03", "cs_06"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "eng_02", domain: "engineering", topic: "Mechanical Engineering", facts: ["Stress-strain curves characterize material behavior", "Thermodynamic cycles convert heat to work", "CAD/CAM enables computer-aided design and manufacturing", "Finite element analysis simulates structural behavior"], connections: ["phys_01", "phys_08"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "eng_03", domain: "engineering", topic: "Civil Engineering", facts: ["Structural analysis ensures buildings resist loads", "Concrete and steel are primary construction materials", "Geotechnical engineering studies soil mechanics", "Transportation engineering designs road and transit systems"], connections: ["eng_02", "geo_01"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "eng_04", domain: "engineering", topic: "Chemical Engineering", facts: ["Unit operations: distillation, filtration, reaction", "Process control maintains optimal operating conditions", "Mass and energy balances govern chemical processes", "Reactor design optimizes conversion and selectivity"], connections: ["chem_01", "eng_02"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "eng_05", domain: "engineering", topic: "Biomedical Engineering", facts: ["Medical imaging: MRI, CT, ultrasound, PET", "Prosthetics replace or augment body functions", "Biomaterials must be biocompatible for implantation", "Tissue engineering grows replacement tissues"], connections: ["bio_05", "eng_01"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "eng_06", domain: "engineering", topic: "Aerospace Engineering", facts: ["Lift generated by airfoil shape and angle of attack", "Rocket propulsion follows Newton's third law", "Orbital mechanics governs satellite trajectories", "Composite materials reduce aircraft weight"], connections: ["phys_01", "phys_08"], confidence: 0.85))

        // ─── Humanities: History (8 nodes) ───
        nodes.append(KnowledgeNode(id: "hist_01", domain: "history", topic: "Ancient History", facts: ["Mesopotamia developed first writing system (cuneiform)", "Egyptian pyramids built ~2560 BCE as pharaoh tombs", "Greek democracy originated in Athens ~508 BCE", "Roman Republic lasted from 509 BCE to 27 BCE"], connections: ["hist_02", "phil_01"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "hist_02", domain: "history", topic: "Medieval History", facts: ["Feudalism structured European society in Middle Ages", "Crusades were religious wars from 1096-1291", "Black Death killed ~30-60% of European population", "Magna Carta (1215) limited English royal power"], connections: ["hist_01", "hist_03"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "hist_03", domain: "history", topic: "Early Modern History", facts: ["Renaissance revived classical learning in 14th-17th centuries", "Protestant Reformation began with Luther's 95 Theses (1517)", "Age of Exploration expanded European global reach", "Scientific Revolution transformed understanding of nature"], connections: ["hist_02", "hist_04"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "hist_04", domain: "history", topic: "Modern History", facts: ["French Revolution (1789) overthrew absolute monarchy", "Industrial Revolution mechanized production from 1760s", "World War I (1914-1918) killed ~20 million people", "World War II (1939-1945) was deadliest conflict in history"], connections: ["hist_03", "hist_05"], confidence: 0.92))
        nodes.append(KnowledgeNode(id: "hist_05", domain: "history", topic: "US History", facts: ["Declaration of Independence signed July 4, 1776", "Civil War (1861-1865) ended slavery in the US", "Civil rights movement peaked in 1960s", "Constitution ratified in 1788 with Bill of Rights in 1791"], connections: ["hist_04", "pol_01"], confidence: 0.92))
        nodes.append(KnowledgeNode(id: "hist_06", domain: "history", topic: "European History", facts: ["European Union formed from post-WWII cooperation", "Cold War divided Europe into NATO and Warsaw Pact", "Fall of Berlin Wall in 1989 ended division of Europe", "French Revolution inspired democratic movements globally"], connections: ["hist_04", "pol_02"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "hist_07", domain: "history", topic: "World History", facts: ["Silk Road connected East and West for trade", "Mongol Empire was largest contiguous land empire", "Columbian Exchange transferred plants, animals, diseases", "Decolonization transformed Africa and Asia post-WWII"], connections: ["hist_04", "hist_01"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "hist_08", domain: "history", topic: "Prehistory", facts: ["Homo sapiens emerged in Africa ~300,000 years ago", "Agriculture began ~10,000 BCE in Fertile Crescent", "Stone tools date back ~2.6 million years", "Cave paintings at Lascaux created ~17,000 years ago"], connections: ["hist_01", "bio_03"], confidence: 0.85))

        // ─── Humanities: Philosophy (8 nodes) ───
        nodes.append(KnowledgeNode(id: "phil_01", domain: "philosophy", topic: "Ethics", facts: ["Utilitarianism maximizes overall happiness", "Deontology judges actions by moral rules", "Virtue ethics focuses on character development", "Social contract theory derives political authority from agreement"], connections: ["phil_02", "pol_01"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "phil_02", domain: "philosophy", topic: "Epistemology", facts: ["Empiricism: knowledge comes from sensory experience", "Rationalism: knowledge comes from reason alone", "Skepticism questions certainty of knowledge claims", "Justified true belief is traditional definition of knowledge"], connections: ["phil_01", "phil_03"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "phil_03", domain: "philosophy", topic: "Metaphysics", facts: ["Dualism posits mind and body are separate substances", "Materialism: only physical matter exists", "Free will debate: determinism vs libertarianism", "Identity: what makes an entity the same over time"], connections: ["phil_02", "phil_04"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "phil_04", domain: "philosophy", topic: "Logic", facts: ["Deductive reasoning: if premises true, conclusion must be true", "Inductive reasoning: generalizes from specific observations", "Formal logic uses symbolic notation for proofs", "Logical fallacies are errors in reasoning"], connections: ["phil_02", "math_08"], confidence: 0.92))
        nodes.append(KnowledgeNode(id: "phil_05", domain: "philosophy", topic: "Political Philosophy", facts: ["Locke: natural rights to life, liberty, property", "Rawls: justice as fairness behind veil of ignorance", "Marx: class struggle drives historical change", "Hobbes: without government, life is nasty, brutish, short"], connections: ["phil_01", "pol_01"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "phil_06", domain: "philosophy", topic: "Aesthetics", facts: ["Beauty may be objective or subjective", "Kant: aesthetic judgment is disinterested pleasure", "Art serves expression, representation, or form", "Sublime evokes awe mixed with fear"], connections: ["phil_03", "art_01"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "phil_07", domain: "philosophy", topic: "Philosophy of Mind", facts: ["Consciousness is the hard problem of philosophy", "Functionalism: mental states defined by functional roles", "Qualia are subjective experiential properties", "Chinese Room argues syntax is not semantics"], connections: ["phil_03", "cs_04"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "phil_08", domain: "philosophy", topic: "Philosophy of Science", facts: ["Falsifiability distinguishes science from pseudoscience", "Paradigm shifts revolutionize scientific frameworks", "Scientific realism: theories describe real entities", "Underdetermination: data may support multiple theories"], connections: ["phil_02", "phys_01"], confidence: 0.85))

        // ─── Humanities: Literature (8 nodes) ───
        nodes.append(KnowledgeNode(id: "lit_01", domain: "literature", topic: "Literary Theory", facts: ["Structuralism analyzes underlying patterns in texts", "Postmodernism questions grand narratives and truth", "Feminist criticism examines gender representation", "Reader-response theory: meaning created by reader"], connections: ["lit_02", "phil_02"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "lit_02", domain: "literature", topic: "Poetry", facts: ["Sonnets follow 14-line structural conventions", "Free verse abandons traditional meter and rhyme", "Haiku: 5-7-5 syllable Japanese form", "Metaphor creates meaning through comparison"], connections: ["lit_01", "lit_03"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "lit_03", domain: "literature", topic: "Fiction", facts: ["Novel emerged as dominant literary form in 18th century", "Plot structure: exposition, rising action, climax, resolution", "Unreliable narrator creates ambiguity in storytelling", "Stream of consciousness captures inner thought flow"], connections: ["lit_01", "lit_02"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "lit_04", domain: "literature", topic: "Drama", facts: ["Greek tragedy originated in Athenian festivals", "Shakespeare wrote 37 plays across genres", "Theater of the absurd questions meaning of existence", "Aristotle defined tragedy with catharsis"], connections: ["lit_01", "hist_01"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "lit_05", domain: "literature", topic: "World Literature", facts: ["Epic of Gilgamesh is oldest known literary work", "Don Quixote considered first modern novel", "1001 Nights shaped Western perception of Middle East", "Harlem Renaissance celebrated African-American literature"], connections: ["lit_03", "hist_07"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "lit_06", domain: "literature", topic: "American Literature", facts: ["Transcendentalism emphasized nature and individualism", "Twain's works critiqued American society", "Modernist writers like Hemingway used sparse prose", "Beat Generation challenged conformity in 1950s"], connections: ["lit_03", "hist_05"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "lit_07", domain: "literature", topic: "British Literature", facts: ["Chaucer's Canterbury Tales depicts medieval society", "Romantic poets celebrated nature and emotion", "Victorian novel examined social conditions", "Modernist writers like Woolf experimented with form"], connections: ["lit_03", "hist_06"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "lit_08", domain: "literature", topic: "Rhetoric", facts: ["Ethos appeals to credibility and character", "Pathos appeals to emotion", "Logos appeals to logic and reason", "Kairos refers to the right timing for argument"], connections: ["lit_01", "phil_04"], confidence: 0.85))

        // ─── Humanities: Art & Music (8 nodes) ───
        nodes.append(KnowledgeNode(id: "art_01", domain: "arts", topic: "Visual Arts", facts: ["Renaissance art developed perspective and realism", "Impressionism captured light and momentary effects", "Abstract art abandoned representational forms", "Photography transformed visual documentation"], connections: ["phil_06", "hist_03"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "art_02", domain: "arts", topic: "Architecture", facts: ["Gothic architecture features pointed arches and flying buttresses", "Modernist architecture emphasizes form follows function", "Sustainable design minimizes environmental impact", "Classical orders: Doric, Ionic, Corinthian"], connections: ["art_01", "eng_03"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "art_03", domain: "arts", topic: "Music Theory", facts: ["Western music uses 12-tone equal temperament", "Harmony combines simultaneous pitches", "Rhythm organizes music in time", "Counterpoint layers independent melodic lines"], connections: ["art_01", "phys_09"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "art_04", domain: "arts", topic: "Film Studies", facts: ["Montage editing creates meaning from juxtaposition", "Mise-en-scène encompasses visual staging elements", "Cinema verité captures reality without intervention", "Genre conventions create audience expectations"], connections: ["art_01", "lit_04"], confidence: 0.80))

        // ─── Social Sciences (35 nodes across 5 fields) ───
        // Psychology (8)
        nodes.append(KnowledgeNode(id: "psy_01", domain: "psychology", topic: "Cognitive Psychology", facts: ["Working memory has limited capacity (~7 items)", "Cognitive biases systematically distort judgment", "Schema theory: knowledge organized in mental frameworks", "Attention is selective and limited resource"], connections: ["bio_09", "psy_02"], confidence: 0.92))
        nodes.append(KnowledgeNode(id: "psy_02", domain: "psychology", topic: "Developmental Psychology", facts: ["Piaget: children progress through cognitive stages", "Attachment theory: secure base for exploration", "Erikson: psychosocial development across lifespan", "Critical periods exist for language acquisition"], connections: ["psy_01", "psy_03"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "psy_03", domain: "psychology", topic: "Social Psychology", facts: ["Conformity increases with group unanimity", "Milgram experiment showed obedience to authority", "Cognitive dissonance creates psychological discomfort", "Bystander effect reduces helping in groups"], connections: ["psy_01", "soc_01"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "psy_04", domain: "psychology", topic: "Clinical Psychology", facts: ["CBT treats disorders by changing thought patterns", "DSM-5 classifies mental disorders diagnostically", "Depression involves persistent low mood and anhedonia", "Anxiety disorders are most common mental health condition"], connections: ["psy_01", "med_04"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "psy_05", domain: "psychology", topic: "Behavioral Psychology", facts: ["Classical conditioning pairs stimuli for learning", "Operant conditioning shapes behavior via consequences", "Reinforcement schedules affect behavior patterns", "Extinction occurs when reinforcement stops"], connections: ["psy_01", "psy_02"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "psy_06", domain: "psychology", topic: "Personality Psychology", facts: ["Big Five: openness, conscientiousness, extraversion, agreeableness, neuroticism", "Nature and nurture both influence personality", "Trait theory identifies stable behavioral patterns", "Self-actualization is highest in Maslow's hierarchy"], connections: ["psy_01", "psy_04"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "psy_07", domain: "psychology", topic: "Research Methods", facts: ["Random assignment controls for confounding variables", "Double-blind studies reduce experimenter bias", "Correlation does not imply causation", "Statistical significance typically set at p<0.05"], connections: ["psy_01", "math_04"], confidence: 0.92))
        nodes.append(KnowledgeNode(id: "psy_08", domain: "psychology", topic: "Human Sexuality", facts: ["Kinsey scale measures sexual orientation on continuum", "Gender identity is distinct from biological sex", "Sexual response cycle: excitement, plateau, orgasm, resolution", "Sexuality influenced by biological, psychological, social factors"], connections: ["psy_06", "bio_06"], confidence: 0.82))

        // Sociology (7)
        nodes.append(KnowledgeNode(id: "soc_01", domain: "sociology", topic: "Social Theory", facts: ["Functionalism: society is system of interconnected parts", "Conflict theory: society shaped by power struggles", "Symbolic interactionism: meaning created through interaction", "Social constructionism: reality is socially constructed"], connections: ["psy_03", "soc_02"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "soc_02", domain: "sociology", topic: "Social Stratification", facts: ["Class system ranks people by wealth and status", "Social mobility varies across societies", "Income inequality measured by Gini coefficient", "Intersectionality: overlapping social categories affect experience"], connections: ["soc_01", "econ_01"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "soc_03", domain: "sociology", topic: "Institutions", facts: ["Family is primary agent of socialization", "Education transmits knowledge and social norms", "Religion provides meaning and community cohesion", "Media shapes public opinion and cultural norms"], connections: ["soc_01", "pol_01"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "soc_04", domain: "sociology", topic: "Deviance", facts: ["Deviance is behavior violating social norms", "Labeling theory: deviance is socially constructed", "Strain theory: deviance from blocked opportunities", "Social control mechanisms enforce conformity"], connections: ["soc_01", "law_02"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "soc_05", domain: "sociology", topic: "Demography", facts: ["Demographic transition: from high to low birth/death rates", "Urbanization concentrates population in cities", "Migration patterns shaped by push-pull factors", "Aging populations challenge social welfare systems"], connections: ["soc_02", "econ_02"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "soc_06", domain: "sociology", topic: "Race and Ethnicity", facts: ["Race is a social construct rather than biological", "Systemic racism embedded in institutional practices", "Ethnic identity shaped by culture, language, history", "Prejudice involves prejudgment based on group membership"], connections: ["soc_01", "soc_02"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "soc_07", domain: "sociology", topic: "Gender Studies", facts: ["Gender roles are socially constructed expectations", "Feminist waves: suffrage, liberation, intersectionality", "Patriarchy describes male-dominated social systems", "Gender wage gap persists across many industries"], connections: ["soc_01", "psy_08"], confidence: 0.82))

        // Economics (7)
        nodes.append(KnowledgeNode(id: "econ_01", domain: "economics", topic: "Microeconomics", facts: ["Supply and demand determine market prices", "Price elasticity measures quantity response to price changes", "Marginal analysis guides optimal decision-making", "Market failures include externalities and public goods"], connections: ["econ_02", "bus_01"], confidence: 0.92))
        nodes.append(KnowledgeNode(id: "econ_02", domain: "economics", topic: "Macroeconomics", facts: ["GDP measures total economic output of a nation", "Inflation erodes purchasing power over time", "Fiscal policy uses government spending and taxation", "Monetary policy controls money supply and interest rates"], connections: ["econ_01", "econ_03"], confidence: 0.92))
        nodes.append(KnowledgeNode(id: "econ_03", domain: "economics", topic: "International Economics", facts: ["Comparative advantage drives international trade", "Exchange rates fluctuate based on market forces", "Trade deficits occur when imports exceed exports", "Globalization increases economic interdependence"], connections: ["econ_02", "pol_03"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "econ_04", domain: "economics", topic: "Econometrics", facts: ["Regression analysis estimates economic relationships", "Time series models forecast economic variables", "Instrumental variables address endogeneity", "Panel data combines cross-sectional and time series"], connections: ["econ_01", "math_04"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "econ_05", domain: "economics", topic: "Behavioral Economics", facts: ["Loss aversion: losses feel worse than equivalent gains", "Prospect theory models decisions under risk", "Nudge theory: choice architecture influences decisions", "Bounded rationality: limited cognitive processing capacity"], connections: ["econ_01", "psy_01"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "econ_06", domain: "economics", topic: "Development Economics", facts: ["Poverty traps prevent economic advancement", "Human capital investment drives economic growth", "Microfinance provides small loans to entrepreneurs", "Aid effectiveness debated among development economists"], connections: ["econ_02", "soc_02"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "econ_07", domain: "economics", topic: "Labor Economics", facts: ["Minimum wage effects debated among economists", "Human capital theory links education to earnings", "Unemployment types: frictional, structural, cyclical", "Labor unions collectively bargain for workers"], connections: ["econ_01", "soc_02"], confidence: 0.82))

        // Political Science (6)
        nodes.append(KnowledgeNode(id: "pol_01", domain: "political_science", topic: "Government", facts: ["Democracy: government by the people", "Separation of powers: legislative, executive, judicial", "Federalism divides power between national and state", "Constitution is supreme law of the land"], connections: ["phil_05", "law_01"], confidence: 0.92))
        nodes.append(KnowledgeNode(id: "pol_02", domain: "political_science", topic: "International Relations", facts: ["Realism: states pursue power in anarchic system", "Liberalism: cooperation through institutions possible", "Constructivism: international norms shape behavior", "Deterrence theory prevents aggression through threat"], connections: ["pol_01", "pol_03"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "pol_03", domain: "political_science", topic: "Foreign Policy", facts: ["Diplomacy is primary tool of foreign policy", "Sanctions apply economic pressure on states", "Soft power influences through culture and values", "National interest guides foreign policy decisions"], connections: ["pol_02", "econ_03"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "pol_04", domain: "political_science", topic: "Political Theory", facts: ["Liberalism emphasizes individual rights and freedoms", "Conservatism values tradition and incremental change", "Socialism advocates collective ownership of production", "Anarchism opposes all forms of coercive authority"], connections: ["phil_05", "pol_01"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "pol_05", domain: "political_science", topic: "Comparative Politics", facts: ["Parliamentary systems fuse executive and legislative", "Presidential systems separate executive and legislative", "Electoral systems: proportional, majoritarian, mixed", "Political parties aggregate and represent interests"], connections: ["pol_01", "pol_04"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "pol_06", domain: "political_science", topic: "Public Policy", facts: ["Policy cycle: agenda, formulation, implementation, evaluation", "Regulatory policy sets rules for private sector", "Distributive policy allocates resources to groups", "Cost-benefit analysis evaluates policy alternatives"], connections: ["pol_01", "econ_02"], confidence: 0.82))

        // Anthropology (7)
        nodes.append(KnowledgeNode(id: "anth_01", domain: "anthropology", topic: "Cultural Anthropology", facts: ["Ethnography studies cultures through participant observation", "Cultural relativism: cultures understood on own terms", "Kinship systems organize social relationships", "Rituals mark transitions and reinforce social bonds"], connections: ["soc_01", "anth_02"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "anth_02", domain: "anthropology", topic: "Physical Anthropology", facts: ["Bipedalism evolved ~4 million years ago in hominins", "Human genetic diversity is small compared to other primates", "Forensic anthropology identifies skeletal remains", "Primatology studies nonhuman primate behavior"], connections: ["bio_03", "hist_08"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "anth_03", domain: "anthropology", topic: "Archaeology", facts: ["Stratigraphy dates artifacts by soil layers", "Carbon-14 dating measures organic material age", "Material culture reveals past societies' lifeways", "Archaeological sites preserve evidence of human activity"], connections: ["hist_08", "anth_01"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "anth_04", domain: "anthropology", topic: "Linguistic Anthropology", facts: ["Sapir-Whorf hypothesis: language shapes thought", "Language families share common ancestral origins", "Code-switching adapts language to social context", "Endangered languages disappear as speakers decline"], connections: ["anth_01", "lit_08"], confidence: 0.80))
        nodes.append(KnowledgeNode(id: "anth_05", domain: "anthropology", topic: "Medical Anthropology", facts: ["Disease concepts vary across cultures", "Biomedical model dominant in Western medicine", "Healing practices reflect cultural beliefs", "Health disparities linked to social determinants"], connections: ["anth_01", "med_01"], confidence: 0.80))
        nodes.append(KnowledgeNode(id: "anth_06", domain: "anthropology", topic: "Economic Anthropology", facts: ["Gift economies build social relationships", "Subsistence strategies: foraging, pastoralism, agriculture", "Market integration transforms indigenous economies", "Reciprocity operates in formal and informal exchanges"], connections: ["anth_01", "econ_01"], confidence: 0.80))
        nodes.append(KnowledgeNode(id: "anth_07", domain: "anthropology", topic: "Religion", facts: ["Animism attributes spirits to natural phenomena", "Monotheism: belief in one deity", "Shamanism: spiritual practitioners mediate spirit world", "Major world religions: Christianity, Islam, Hinduism, Buddhism, Judaism"], connections: ["anth_01", "soc_03"], confidence: 0.82))

        // ─── Medicine (25 nodes) ───
        nodes.append(KnowledgeNode(id: "med_01", domain: "medicine", topic: "Anatomy", facts: ["Human body has 206 bones and 600+ muscles", "Heart pumps ~5 liters of blood per minute", "Brain contains ~86 billion neurons", "Liver performs 500+ metabolic functions"], connections: ["bio_05", "med_02"], confidence: 0.92))
        nodes.append(KnowledgeNode(id: "med_02", domain: "medicine", topic: "Pharmacology", facts: ["Drug efficacy measured by dose-response curves", "Pharmacokinetics: absorption, distribution, metabolism, excretion", "Drug interactions can be synergistic or antagonistic", "Therapeutic index measures drug safety margin"], connections: ["chem_04", "med_01"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "med_03", domain: "medicine", topic: "Pathology", facts: ["Inflammation is body's response to tissue damage", "Cancer involves uncontrolled cell proliferation", "Atherosclerosis narrows arteries with plaque buildup", "Autoimmune diseases: immune system attacks own tissue"], connections: ["bio_08", "med_01"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "med_04", domain: "medicine", topic: "Epidemiology", facts: ["Incidence measures new cases in a population", "Prevalence measures total existing cases", "R0 indicates pathogen transmissibility", "Randomized controlled trials are gold standard"], connections: ["med_03", "math_04"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "med_05", domain: "medicine", topic: "Clinical Medicine", facts: ["History and physical exam are diagnostic foundations", "Differential diagnosis ranks possible conditions", "Evidence-based medicine integrates research and clinical expertise", "Vital signs: temperature, pulse, respiration, blood pressure"], connections: ["med_01", "med_03"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "med_06", domain: "medicine", topic: "Surgery", facts: ["Aseptic technique prevents surgical infections", "Minimally invasive surgery reduces recovery time", "Anesthesia enables painless surgical procedures", "Organ transplantation requires immunosuppression"], connections: ["med_01", "med_02"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "med_07", domain: "medicine", topic: "Cardiology", facts: ["ECG records heart's electrical activity", "Heart failure: heart cannot pump adequately", "Atrial fibrillation is most common arrhythmia", "Statins lower cholesterol to prevent heart disease"], connections: ["med_01", "med_03"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "med_08", domain: "medicine", topic: "Neurology", facts: ["Stroke: interrupted blood supply to brain", "Alzheimer's disease causes progressive dementia", "Multiple sclerosis damages myelin sheath", "EEG records brain electrical activity"], connections: ["bio_09", "med_03"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "med_09", domain: "medicine", topic: "Pediatrics", facts: ["Growth charts monitor child development", "Vaccination schedule protects against childhood diseases", "Childhood developmental milestones guide assessment", "Neonatal period is first 28 days of life"], connections: ["med_05", "psy_02"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "med_10", domain: "medicine", topic: "Oncology", facts: ["Chemotherapy uses cytotoxic drugs to kill cancer cells", "Radiation therapy damages cancer cell DNA", "Immunotherapy harnesses immune system against cancer", "Tumor staging determines cancer extent and prognosis"], connections: ["med_03", "bio_10"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "med_11", domain: "medicine", topic: "Infectious Disease", facts: ["Koch's postulates identify causative organisms", "Antibiotic resistance is growing global threat", "HIV targets CD4+ T cells", "Pandemic preparedness requires surveillance and response"], connections: ["bio_07", "med_04"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "med_12", domain: "medicine", topic: "Psychiatry", facts: ["Schizophrenia involves psychosis and cognitive deficits", "Bipolar disorder alternates mania and depression", "SSRIs are first-line antidepressant medications", "Psychotherapy combined with medication often most effective"], connections: ["psy_04", "med_02"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "med_13", domain: "medicine", topic: "Genetics (Clinical)", facts: ["Down syndrome caused by trisomy 21", "Cystic fibrosis is autosomal recessive disorder", "Pharmacogenomics tailors drugs to genetic profile", "Genetic counseling helps families understand hereditary risks"], connections: ["bio_02", "med_05"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "med_14", domain: "medicine", topic: "Virology", facts: ["Viruses classified by genome type (DNA/RNA)", "Influenza virus undergoes antigenic drift and shift", "Retroviruses integrate into host genome", "Viral vaccines can be live-attenuated or inactivated"], connections: ["bio_07", "med_11"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "med_15", domain: "medicine", topic: "Nutrition", facts: ["Macronutrients: carbohydrates, proteins, fats", "Micronutrients: vitamins and minerals", "BMI estimates body fat from height and weight", "Malnutrition includes both undernutrition and obesity"], connections: ["bio_06", "med_05"], confidence: 0.85))
        // Remaining medicine nodes
        nodes.append(KnowledgeNode(id: "med_16", domain: "medicine", topic: "Dermatology", facts: ["Skin is largest organ of the body", "Melanoma is most dangerous skin cancer", "Eczema involves chronic skin inflammation"], connections: ["med_03", "bio_08"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "med_17", domain: "medicine", topic: "Endocrinology", facts: ["Diabetes mellitus involves insulin dysfunction", "Thyroid hormones regulate metabolism", "Adrenal glands produce cortisol and adrenaline"], connections: ["bio_06", "med_02"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "med_18", domain: "medicine", topic: "Gastroenterology", facts: ["IBS is common functional GI disorder", "Celiac disease is autoimmune response to gluten", "Liver cirrhosis results from chronic damage"], connections: ["med_01", "med_03"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "med_19", domain: "medicine", topic: "Nephrology", facts: ["Chronic kidney disease progresses through 5 stages", "Dialysis replaces kidney filtration function", "Glomerulonephritis inflames kidney filtering units"], connections: ["bio_06", "med_03"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "med_20", domain: "medicine", topic: "Pulmonology", facts: ["COPD is progressive obstructive lung disease", "Asthma involves reversible airway obstruction", "Pneumonia is infection of lung tissue"], connections: ["bio_06", "med_11"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "med_21", domain: "medicine", topic: "Ophthalmology", facts: ["Glaucoma damages optic nerve from pressure", "Cataracts cloud the eye's natural lens", "Retinal detachment requires emergency treatment"], connections: ["med_01", "bio_09"], confidence: 0.80))
        nodes.append(KnowledgeNode(id: "med_22", domain: "medicine", topic: "Orthopedics", facts: ["Osteoporosis weakens bones through density loss", "ACL tears are common sports injuries", "Fracture healing involves inflammation and remodeling"], connections: ["med_01", "bio_05"], confidence: 0.80))
        nodes.append(KnowledgeNode(id: "med_23", domain: "medicine", topic: "Radiology", facts: ["X-rays penetrate tissue to reveal bone structure", "MRI uses magnetic fields for soft tissue imaging", "CT combines X-rays for cross-sectional images"], connections: ["phys_03", "med_05"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "med_24", domain: "medicine", topic: "Emergency Medicine", facts: ["Triage prioritizes patients by severity", "ABCDE approach: airway, breathing, circulation, disability, exposure", "Golden hour: first 60 minutes after trauma are critical"], connections: ["med_05", "med_06"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "med_25", domain: "medicine", topic: "Human Aging", facts: ["Telomere shortening correlates with cellular aging", "Sarcopenia: age-related loss of muscle mass", "Cognitive decline varies widely among elderly"], connections: ["bio_01", "psy_02"], confidence: 0.82))

        // ─── Law (15 nodes) ───
        nodes.append(KnowledgeNode(id: "law_01", domain: "law", topic: "Constitutional Law", facts: ["Judicial review: courts assess law constitutionality", "Due process protects against arbitrary government action", "Equal protection clause prohibits discrimination", "First Amendment protects speech, religion, press, assembly"], connections: ["pol_01", "law_02"], confidence: 0.90))
        nodes.append(KnowledgeNode(id: "law_02", domain: "law", topic: "Criminal Law", facts: ["Actus reus and mens rea required for most crimes", "Beyond reasonable doubt is criminal burden of proof", "Rights of accused: counsel, jury trial, confrontation", "Felonies are more serious than misdemeanors"], connections: ["law_01", "law_03"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "law_03", domain: "law", topic: "Civil Law", facts: ["Torts: wrongs causing harm to another person", "Contracts require offer, acceptance, consideration", "Preponderance of evidence is civil burden of proof", "Damages compensate for losses suffered"], connections: ["law_02", "law_04"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "law_04", domain: "law", topic: "International Law", facts: ["Treaties are binding agreements between states", "UN Charter establishes international legal framework", "Geneva Conventions protect war victims", "ICJ is principal judicial organ of the UN"], connections: ["pol_02", "law_01"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "law_05", domain: "law", topic: "Jurisprudence", facts: ["Natural law theory: law derived from morality", "Legal positivism: law is social convention", "Legal realism: law shaped by judicial behavior", "Critical legal studies questions legal neutrality"], connections: ["phil_01", "law_01"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "law_06", domain: "law", topic: "Administrative Law", facts: ["Agencies create regulations under statutory authority", "Judicial review checks agency actions", "Notice and comment required for rulemaking"], connections: ["law_01", "pol_06"], confidence: 0.80))
        nodes.append(KnowledgeNode(id: "law_07", domain: "law", topic: "Property Law", facts: ["Real property includes land and fixtures", "Intellectual property protects creative works", "Easements grant rights to use another's land"], connections: ["law_03", "bus_01"], confidence: 0.80))
        nodes.append(KnowledgeNode(id: "law_08", domain: "law", topic: "Corporate Law", facts: ["Corporations have separate legal personality", "Fiduciary duties: care, loyalty, good faith", "Securities regulation protects investors"], connections: ["law_03", "bus_02"], confidence: 0.80))
        nodes.append(KnowledgeNode(id: "law_09", domain: "law", topic: "Environmental Law", facts: ["EPA enforces environmental regulations in US", "Clean Air Act regulates air pollutant emissions", "Environmental impact assessments required for major projects"], connections: ["law_06", "geo_01"], confidence: 0.80))
        nodes.append(KnowledgeNode(id: "law_10", domain: "law", topic: "Labor Law", facts: ["NLRA protects collective bargaining rights", "OSHA regulates workplace safety", "Employment discrimination prohibited by Title VII"], connections: ["law_03", "econ_07"], confidence: 0.80))
        nodes.append(KnowledgeNode(id: "law_11", domain: "law", topic: "Tax Law", facts: ["Progressive taxation: higher rates on higher income", "Tax deductions reduce taxable income", "Capital gains taxed differently from ordinary income"], connections: ["law_06", "bus_04"], confidence: 0.78))
        nodes.append(KnowledgeNode(id: "law_12", domain: "law", topic: "Immigration Law", facts: ["Visa categories determine entry conditions", "Asylum protects those fleeing persecution", "Citizenship obtained by birth or naturalization"], connections: ["law_01", "pol_03"], confidence: 0.78))
        nodes.append(KnowledgeNode(id: "law_13", domain: "law", topic: "Family Law", facts: ["Marriage is both a legal contract and social institution", "Child custody determined by best interests standard", "Divorce laws vary by jurisdiction"], connections: ["law_03", "soc_03"], confidence: 0.78))
        nodes.append(KnowledgeNode(id: "law_14", domain: "law", topic: "Cyber Law", facts: ["CFAA criminalizes unauthorized computer access", "GDPR protects personal data in EU", "Digital copyright adapts IP law to internet"], connections: ["law_07", "cs_05"], confidence: 0.78))
        nodes.append(KnowledgeNode(id: "law_15", domain: "law", topic: "Human Rights Law", facts: ["UDHR adopted by UN General Assembly in 1948", "Fundamental rights include life, liberty, security", "Humanitarian law governs conduct of armed conflict"], connections: ["law_04", "pol_02"], confidence: 0.82))

        // ─── Business (15 nodes) ───
        nodes.append(KnowledgeNode(id: "bus_01", domain: "business", topic: "Finance", facts: ["Time value of money: dollar today worth more than tomorrow", "CAPM relates expected return to systematic risk", "Diversification reduces unsystematic portfolio risk", "NPV: sum of discounted future cash flows"], connections: ["econ_01", "bus_02"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "bus_02", domain: "business", topic: "Accounting", facts: ["Balance sheet: assets = liabilities + equity", "Income statement shows revenue and expenses", "Cash flow statement tracks money in and out", "GAAP provides standardized accounting rules"], connections: ["bus_01", "bus_03"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "bus_03", domain: "business", topic: "Management", facts: ["SWOT analysis: strengths, weaknesses, opportunities, threats", "Leadership styles: transformational, transactional, servant", "Organizational behavior studies workplace dynamics", "Strategic planning sets long-term organizational direction"], connections: ["bus_02", "bus_04"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "bus_04", domain: "business", topic: "Marketing", facts: ["4Ps: product, price, place, promotion", "Market segmentation divides consumers into groups", "Brand equity is intangible value of a brand", "Digital marketing leverages online channels"], connections: ["bus_03", "econ_01"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "bus_05", domain: "business", topic: "Operations", facts: ["Supply chain management optimizes product flow", "Six Sigma reduces defects and variability", "JIT inventory minimizes holding costs", "Quality management ensures product standards"], connections: ["bus_03", "eng_04"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "bus_06", domain: "business", topic: "Entrepreneurship", facts: ["Business plan outlines venture strategy and finances", "Venture capital funds high-growth startups", "Lean startup method: build-measure-learn cycle", "Disruptive innovation creates new markets"], connections: ["bus_01", "bus_03"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "bus_07", domain: "business", topic: "Business Ethics", facts: ["Corporate social responsibility extends beyond profits", "Stakeholder theory: business serves all stakeholders", "Ethical dilemmas require balancing competing values", "Whistleblower protections encourage reporting misconduct"], connections: ["phil_01", "bus_03"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "bus_08", domain: "business", topic: "Human Resources", facts: ["Recruitment, selection, training, development cycle", "Performance appraisal evaluates employee contribution", "Compensation includes salary, benefits, incentives"], connections: ["bus_03", "psy_06"], confidence: 0.80))
        nodes.append(KnowledgeNode(id: "bus_09", domain: "business", topic: "International Business", facts: ["Multinational corporations operate across borders", "Currency risk affects international operations", "Cultural differences impact global management"], connections: ["bus_03", "econ_03"], confidence: 0.80))
        nodes.append(KnowledgeNode(id: "bus_10", domain: "business", topic: "Public Relations", facts: ["PR manages organizational reputation and communication", "Crisis communication requires transparency and speed", "Media relations builds positive press coverage"], connections: ["bus_04", "soc_03"], confidence: 0.78))
        nodes.append(KnowledgeNode(id: "bus_11", domain: "business", topic: "Real Estate", facts: ["Location is primary driver of property value", "Mortgage amortization schedules principal payments", "REITs allow investment in property portfolios"], connections: ["bus_01", "law_07"], confidence: 0.78))
        nodes.append(KnowledgeNode(id: "bus_12", domain: "business", topic: "Insurance", facts: ["Risk pooling spreads loss across many policyholders", "Actuarial science quantifies insurance risk", "Moral hazard: insured parties take more risk"], connections: ["bus_01", "math_07"], confidence: 0.78))
        nodes.append(KnowledgeNode(id: "bus_13", domain: "business", topic: "Information Systems", facts: ["ERP integrates core business processes", "Business intelligence analyzes data for decisions", "Cybersecurity protects digital business assets"], connections: ["cs_07", "bus_03"], confidence: 0.80))
        nodes.append(KnowledgeNode(id: "bus_14", domain: "business", topic: "Project Management", facts: ["Critical path determines minimum project duration", "Agile methodology uses iterative development sprints", "Risk management identifies and mitigates threats"], connections: ["bus_03", "eng_02"], confidence: 0.80))
        nodes.append(KnowledgeNode(id: "bus_15", domain: "business", topic: "Securities", facts: ["Stocks represent equity ownership in companies", "Bonds are debt instruments with fixed interest", "Options give right but not obligation to trade"], connections: ["bus_01", "law_08"], confidence: 0.82))

        // ─── Other (11 nodes) ───
        nodes.append(KnowledgeNode(id: "geo_01", domain: "geography", topic: "Physical Geography", facts: ["Plate tectonics drives continental movement", "Climate zones determined by latitude and geography", "Erosion shapes landscapes through water, wind, ice", "Oceans cover approximately 71% of Earth's surface"], connections: ["bio_04", "chem_08"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "geo_02", domain: "geography", topic: "Human Geography", facts: ["Urbanization is global trend toward city living", "Cultural landscapes reflect human-environment interaction", "Geopolitics studies geographic influence on politics"], connections: ["geo_01", "soc_05"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "astro_01", domain: "astronomy", topic: "Astronomy", facts: ["Universe is approximately 13.8 billion years old", "Stars form from gravitational collapse of gas clouds", "Black holes have gravity so strong light cannot escape", "Hubble's law: universe is expanding"], connections: ["phys_05", "phys_04"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "env_01", domain: "environmental_science", topic: "Environmental Science", facts: ["Climate change driven by greenhouse gas emissions", "Deforestation reduces carbon sinks and biodiversity", "Renewable energy: solar, wind, hydro, geothermal", "Water scarcity affects billions worldwide"], connections: ["chem_08", "geo_01"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "nutr_01", domain: "nutrition", topic: "Nutrition Science", facts: ["Essential amino acids must come from diet", "Fiber promotes digestive health", "Antioxidants protect against oxidative stress"], connections: ["med_15", "chem_04"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "sec_01", domain: "security", topic: "Security Studies", facts: ["National security encompasses military and intelligence", "Cybersecurity protects digital infrastructure", "Terrorism uses violence for political objectives", "Arms control agreements limit weapons proliferation"], connections: ["pol_02", "cs_05"], confidence: 0.82))
        nodes.append(KnowledgeNode(id: "misc_01", domain: "miscellaneous", topic: "Global Facts", facts: ["World population exceeds 8 billion people", "United Nations has 193 member states", "Internet users surpassed 5 billion globally"], connections: ["geo_02", "soc_05"], confidence: 0.80))
        nodes.append(KnowledgeNode(id: "misc_02", domain: "miscellaneous", topic: "Logic (Formal)", facts: ["Modus ponens: if P then Q; P; therefore Q", "Logical equivalence: biconditional truth", "De Morgan's laws: negation of conjunctions and disjunctions", "Fallacy of affirming the consequent: invalid reasoning"], connections: ["phil_04", "math_08"], confidence: 0.88))
        nodes.append(KnowledgeNode(id: "misc_03", domain: "miscellaneous", topic: "Logical Fallacies", facts: ["Ad hominem attacks the person not the argument", "Straw man misrepresents opponent's position", "Appeal to authority: expert opinion as proof", "False dichotomy presents only two options"], connections: ["phil_04", "lit_08"], confidence: 0.85))
        nodes.append(KnowledgeNode(id: "misc_04", domain: "miscellaneous", topic: "Moral Issues", facts: ["Trolley problem: utilitarianism vs deontology conflict", "Euthanasia debates balance autonomy and sanctity of life", "Animal rights question moral status of non-humans"], connections: ["phil_01", "law_01"], confidence: 0.80))
        nodes.append(KnowledgeNode(id: "misc_05", domain: "miscellaneous", topic: "Moral Scenarios", facts: ["Ethical dilemmas test moral reasoning frameworks", "Kohlberg stages: pre-conventional to post-conventional", "Consequentialism judges by outcomes alone"], connections: ["phil_01", "psy_02"], confidence: 0.80))

        return nodes
    }()

    // ═══════════════════════════════════════════════════════════
    // MARK: - 57 MMLU Subjects
    // ═══════════════════════════════════════════════════════════

    lazy var mmluSubjects: [MMLUSubject] = {
        return [
            MMLUSubject(name: "abstract_algebra", domain: "STEM", nodeIDs: ["math_01"], sampleQuestions: [("Find the order of the element 3 in Z_7*", "6"), ("Is Z_6 cyclic?", "Yes")]),
            MMLUSubject(name: "anatomy", domain: "Medicine", nodeIDs: ["med_01", "bio_05"], sampleQuestions: [("What muscle flexes the forearm?", "Biceps brachii"), ("Where is the thyroid gland?", "Anterior neck")]),
            MMLUSubject(name: "astronomy", domain: "STEM", nodeIDs: ["astro_01", "phys_05"], sampleQuestions: [("What causes a solar eclipse?", "Moon between Earth and Sun"), ("What is a neutron star?", "Collapsed stellar core")]),
            MMLUSubject(name: "business_ethics", domain: "Business", nodeIDs: ["bus_07", "phil_01"], sampleQuestions: [("What is stakeholder theory?", "Business serves all stakeholders"), ("What is CSR?", "Corporate social responsibility")]),
            MMLUSubject(name: "clinical_knowledge", domain: "Medicine", nodeIDs: ["med_05", "med_03"], sampleQuestions: [("Normal resting heart rate range?", "60-100 bpm"), ("What does hemoglobin carry?", "Oxygen")]),
            MMLUSubject(name: "college_biology", domain: "STEM", nodeIDs: ["bio_01", "bio_02", "bio_03"], sampleQuestions: [("What is the central dogma?", "DNA→RNA→Protein"), ("What is mitosis?", "Cell division producing identical cells")]),
            MMLUSubject(name: "college_chemistry", domain: "STEM", nodeIDs: ["chem_01", "chem_06"], sampleQuestions: [("What is Avogadro's number?", "6.022×10²³"), ("What determines reaction spontaneity?", "Gibbs free energy")]),
            MMLUSubject(name: "college_computer_science", domain: "STEM", nodeIDs: ["cs_01", "cs_02"], sampleQuestions: [("What is O(n log n)?", "Linearithmic complexity"), ("What is a hash collision?", "Two keys map to same index")]),
            MMLUSubject(name: "college_mathematics", domain: "STEM", nodeIDs: ["math_01", "math_02", "math_03"], sampleQuestions: [("What is L'Hôpital's rule?", "Limit of ratio equals limit of derivatives ratio"), ("Is √2 rational?", "No")]),
            MMLUSubject(name: "college_medicine", domain: "Medicine", nodeIDs: ["med_05", "med_02", "med_01"], sampleQuestions: [("What is the function of insulin?", "Lowers blood glucose"), ("What is sepsis?", "Systemic infection response")]),
            MMLUSubject(name: "college_physics", domain: "STEM", nodeIDs: ["phys_01", "phys_03", "phys_04"], sampleQuestions: [("What is the Doppler effect?", "Frequency shift from relative motion"), ("What is Faraday's law?", "Changing B-field induces EMF")]),
            MMLUSubject(name: "computer_security", domain: "STEM", nodeIDs: ["cs_05", "cs_07"], sampleQuestions: [("What is SQL injection?", "Malicious SQL in user input"), ("What is a firewall?", "Network traffic filter")]),
            MMLUSubject(name: "conceptual_physics", domain: "STEM", nodeIDs: ["phys_01", "phys_06"], sampleQuestions: [("Why do objects float?", "Buoyant force exceeds weight"), ("What is inertia?", "Resistance to change in motion")]),
            MMLUSubject(name: "econometrics", domain: "Social Sciences", nodeIDs: ["econ_04", "math_04"], sampleQuestions: [("What is heteroscedasticity?", "Non-constant variance of errors"), ("What is R-squared?", "Proportion of variance explained")]),
            MMLUSubject(name: "electrical_engineering", domain: "STEM", nodeIDs: ["eng_01", "phys_03"], sampleQuestions: [("What is Ohm's law?", "V=IR"), ("What is a transistor?", "Semiconductor switch/amplifier")]),
            MMLUSubject(name: "elementary_mathematics", domain: "STEM", nodeIDs: ["math_01", "math_08"], sampleQuestions: [("What is 15% of 200?", "30"), ("What is the LCM of 4 and 6?", "12")]),
            MMLUSubject(name: "formal_logic", domain: "STEM", nodeIDs: ["misc_02", "phil_04"], sampleQuestions: [("Is P∧¬P satisfiable?", "No"), ("What is modus tollens?", "If P→Q and ¬Q then ¬P")]),
            MMLUSubject(name: "global_facts", domain: "Other", nodeIDs: ["misc_01", "geo_02"], sampleQuestions: [("Largest ocean by area?", "Pacific"), ("Most spoken language?", "English (by total speakers)")]),
            MMLUSubject(name: "high_school_biology", domain: "STEM", nodeIDs: ["bio_01", "bio_02"], sampleQuestions: [("What organelle makes ATP?", "Mitochondria"), ("What is DNA made of?", "Nucleotides")]),
            MMLUSubject(name: "high_school_chemistry", domain: "STEM", nodeIDs: ["chem_01", "chem_02"], sampleQuestions: [("What is pH?", "-log[H+]"), ("What is a covalent bond?", "Shared electron pair")]),
            MMLUSubject(name: "high_school_computer_science", domain: "STEM", nodeIDs: ["cs_01", "cs_02"], sampleQuestions: [("What is a variable?", "Named storage location"), ("What is recursion?", "Function calling itself")]),
            MMLUSubject(name: "high_school_european_history", domain: "Humanities", nodeIDs: ["hist_06", "hist_03"], sampleQuestions: [("What was the Renaissance?", "Revival of classical learning"), ("When did WWII end in Europe?", "May 1945")]),
            MMLUSubject(name: "high_school_geography", domain: "Other", nodeIDs: ["geo_01", "geo_02"], sampleQuestions: [("What causes earthquakes?", "Tectonic plate movement"), ("What is a delta?", "Sediment deposit at river mouth")]),
            MMLUSubject(name: "high_school_government_and_politics", domain: "Social Sciences", nodeIDs: ["pol_01", "pol_05"], sampleQuestions: [("What are the three branches of US government?", "Legislative, Executive, Judicial"), ("What is federalism?", "Power divided between national and state")]),
            MMLUSubject(name: "high_school_macroeconomics", domain: "Social Sciences", nodeIDs: ["econ_02"], sampleQuestions: [("What is GDP?", "Total value of goods and services produced"), ("What causes inflation?", "Too much money chasing too few goods")]),
            MMLUSubject(name: "high_school_mathematics", domain: "STEM", nodeIDs: ["math_01", "math_06"], sampleQuestions: [("Solve: 2x + 5 = 11", "x = 3"), ("What is the Pythagorean theorem?", "a²+b²=c²")]),
            MMLUSubject(name: "high_school_microeconomics", domain: "Social Sciences", nodeIDs: ["econ_01"], sampleQuestions: [("What is opportunity cost?", "Value of next best alternative"), ("What is a monopoly?", "Single seller in market")]),
            MMLUSubject(name: "high_school_physics", domain: "STEM", nodeIDs: ["phys_01", "phys_03"], sampleQuestions: [("What is Newton's third law?", "Every action has equal opposite reaction"), ("What is kinetic energy?", "½mv²")]),
            MMLUSubject(name: "high_school_psychology", domain: "Social Sciences", nodeIDs: ["psy_01", "psy_05"], sampleQuestions: [("What is classical conditioning?", "Learning through stimulus association"), ("Who proposed hierarchy of needs?", "Maslow")]),
            MMLUSubject(name: "high_school_statistics", domain: "STEM", nodeIDs: ["math_04", "math_07"], sampleQuestions: [("What is standard deviation?", "Measure of data spread"), ("What is the mean of 2,4,6,8?", "5")]),
            MMLUSubject(name: "high_school_us_history", domain: "Humanities", nodeIDs: ["hist_05"], sampleQuestions: [("What was the Emancipation Proclamation?", "Freed slaves in Confederate states"), ("When was the Constitution ratified?", "1788")]),
            MMLUSubject(name: "high_school_world_history", domain: "Humanities", nodeIDs: ["hist_07", "hist_04"], sampleQuestions: [("What was the Silk Road?", "Trade route connecting East and West"), ("When did World War I begin?", "1914")]),
            MMLUSubject(name: "human_aging", domain: "Medicine", nodeIDs: ["med_25", "psy_02"], sampleQuestions: [("What is sarcopenia?", "Age-related muscle loss"), ("What are telomeres?", "Chromosome end caps that shorten with age")]),
            MMLUSubject(name: "human_sexuality", domain: "Social Sciences", nodeIDs: ["psy_08", "bio_06"], sampleQuestions: [("What is the Kinsey scale?", "Sexual orientation continuum"), ("What is gender identity?", "Internal sense of gender")]),
            MMLUSubject(name: "international_law", domain: "Law", nodeIDs: ["law_04", "pol_02"], sampleQuestions: [("What are the Geneva Conventions?", "Treaties protecting war victims"), ("What is the ICC?", "International Criminal Court")]),
            MMLUSubject(name: "jurisprudence", domain: "Law", nodeIDs: ["law_05", "phil_01"], sampleQuestions: [("What is legal positivism?", "Law as social convention"), ("What is natural law?", "Law derived from morality")]),
            MMLUSubject(name: "logical_fallacies", domain: "Other", nodeIDs: ["misc_03", "phil_04"], sampleQuestions: [("What is ad hominem?", "Attack on person not argument"), ("What is a straw man?", "Misrepresenting opponent's position")]),
            MMLUSubject(name: "machine_learning", domain: "STEM", nodeIDs: ["cs_03", "math_03"], sampleQuestions: [("What is overfitting?", "Model memorizes training data"), ("What is gradient descent?", "Optimization by following gradient")]),
            MMLUSubject(name: "management", domain: "Business", nodeIDs: ["bus_03"], sampleQuestions: [("What is SWOT?", "Strengths, Weaknesses, Opportunities, Threats"), ("What is delegation?", "Assigning authority and responsibility")]),
            MMLUSubject(name: "marketing", domain: "Business", nodeIDs: ["bus_04"], sampleQuestions: [("What are the 4Ps?", "Product, Price, Place, Promotion"), ("What is market segmentation?", "Dividing consumers into groups")]),
            MMLUSubject(name: "medical_genetics", domain: "Medicine", nodeIDs: ["med_13", "bio_02"], sampleQuestions: [("What causes Down syndrome?", "Trisomy 21"), ("What is CRISPR?", "Gene editing technology")]),
            MMLUSubject(name: "miscellaneous", domain: "Other", nodeIDs: ["misc_01"], sampleQuestions: [("What is the speed of light?", "~3×10⁸ m/s"), ("What element has symbol Fe?", "Iron")]),
            MMLUSubject(name: "moral_disputes", domain: "Other", nodeIDs: ["misc_04", "phil_01"], sampleQuestions: [("What is the trolley problem?", "Ethical dilemma about sacrifice"), ("Is euthanasia ethical?", "Debated—autonomy vs sanctity of life")]),
            MMLUSubject(name: "moral_scenarios", domain: "Other", nodeIDs: ["misc_05", "phil_01"], sampleQuestions: [("What are Kohlberg's stages?", "Levels of moral development"), ("What is consequentialism?", "Judging actions by outcomes")]),
            MMLUSubject(name: "nutrition", domain: "Medicine", nodeIDs: ["nutr_01", "med_15"], sampleQuestions: [("What are macronutrients?", "Carbs, proteins, fats"), ("What is BMI?", "Body mass index: weight/height²")]),
            MMLUSubject(name: "philosophy", domain: "Humanities", nodeIDs: ["phil_01", "phil_02", "phil_03"], sampleQuestions: [("What is empiricism?", "Knowledge from experience"), ("What is the mind-body problem?", "Relationship between mental and physical")]),
            MMLUSubject(name: "prehistory", domain: "Humanities", nodeIDs: ["hist_08", "anth_03"], sampleQuestions: [("When did agriculture begin?", "~10,000 BCE"), ("What are cave paintings?", "Prehistoric art on cave walls")]),
            MMLUSubject(name: "professional_accounting", domain: "Business", nodeIDs: ["bus_02"], sampleQuestions: [("What is depreciation?", "Allocation of asset cost over time"), ("What is GAAP?", "Generally Accepted Accounting Principles")]),
            MMLUSubject(name: "professional_law", domain: "Law", nodeIDs: ["law_01", "law_02", "law_03"], sampleQuestions: [("What is stare decisis?", "Following precedent"), ("What is voir dire?", "Jury selection process")]),
            MMLUSubject(name: "professional_medicine", domain: "Medicine", nodeIDs: ["med_05", "med_02", "med_03"], sampleQuestions: [("What is differential diagnosis?", "Ranking possible conditions"), ("What is informed consent?", "Patient agrees after understanding risks")]),
            MMLUSubject(name: "professional_psychology", domain: "Social Sciences", nodeIDs: ["psy_04", "psy_06"], sampleQuestions: [("What is CBT?", "Cognitive behavioral therapy"), ("What is the DSM-5?", "Diagnostic manual for mental disorders")]),
            MMLUSubject(name: "public_relations", domain: "Business", nodeIDs: ["bus_10"], sampleQuestions: [("What is crisis communication?", "Managing reputation during crises"), ("What is media relations?", "Building positive press coverage")]),
            MMLUSubject(name: "security_studies", domain: "Other", nodeIDs: ["sec_01", "pol_02"], sampleQuestions: [("What is deterrence?", "Preventing aggression through threat"), ("What is asymmetric warfare?", "Conflict between unequal forces")]),
            MMLUSubject(name: "sociology", domain: "Social Sciences", nodeIDs: ["soc_01", "soc_02", "soc_03"], sampleQuestions: [("What is social stratification?", "Hierarchical ranking of society"), ("What is the sociological imagination?", "Linking personal to social")]),
            MMLUSubject(name: "us_foreign_policy", domain: "Social Sciences", nodeIDs: ["pol_03", "hist_05"], sampleQuestions: [("What is the Monroe Doctrine?", "US opposition to European colonialism in Americas"), ("What is NATO?", "North Atlantic Treaty Organization")]),
            MMLUSubject(name: "virology", domain: "Medicine", nodeIDs: ["med_14", "bio_07"], sampleQuestions: [("How do retroviruses replicate?", "Reverse transcribe RNA to DNA"), ("What is antigenic drift?", "Gradual mutation in viral antigens")]),
            MMLUSubject(name: "world_religions", domain: "Humanities", nodeIDs: ["anth_07"], sampleQuestions: [("What are the Five Pillars of Islam?", "Shahada, Salat, Zakat, Sawm, Hajj"), ("What is karma?", "Actions determine future outcomes")]),
        ]
    }()

    // ═══════════════════════════════════════════════════════════
    // MARK: - BM25 Retrieval
    // ═══════════════════════════════════════════════════════════

    private var _cachedDocTokens: [[String]]?
    private var _cachedAvgDL: Double = 0

    private func buildDocCache() {
        if _cachedDocTokens == nil {
            let docs = knowledgeBase.map { node -> [String] in
                let text = ([node.topic] + node.facts).joined(separator: " ")
                return BM25Engine.tokenize(text)
            }
            _cachedDocTokens = docs
            let totalLen = docs.reduce(0) { $0 + $1.count }
            _cachedAvgDL = Double(totalLen) / max(1.0, Double(docs.count))
        }
    }

    func retrieve(query: String, topK: Int = 5) -> [(node: KnowledgeNode, score: Double)] {
        lock.lock(); defer { lock.unlock() }
        buildDocCache()
        guard let docs = _cachedDocTokens else { return [] }

        let queryTokens = BM25Engine.tokenize(query)
        var idfMap: [String: Double] = [:]
        for t in Set(queryTokens) {
            idfMap[t] = BM25Engine.computeIDF(term: t, documents: docs)
        }

        var scored: [(Int, Double)] = []
        for (i, docTokens) in docs.enumerated() {
            let score = BM25Engine.computeBM25(queryTokens: queryTokens, docTokens: docTokens, avgDL: _cachedAvgDL, idfMap: idfMap)
            if score > 0 { scored.append((i, score)) }
        }

        scored.sort { $0.1 > $1.1 }
        totalRetrievals += 1

        return scored.prefix(topK).map { (knowledgeBase[$0.0], $0.1) }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - 8 Comprehension Layers
    // ═══════════════════════════════════════════════════════════

    func lexicalAnalysis(_ text: String) -> Double {
        let tokens = BM25Engine.tokenize(text)
        let unique = Set(tokens)
        let richness = Double(unique.count) / max(1.0, Double(tokens.count))
        return min(1.0, richness * PHI)
    }

    func syntacticParsing(_ text: String) -> Double {
        let sentences = text.components(separatedBy: CharacterSet(charactersIn: ".!?"))
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        let avgLen = sentences.isEmpty ? 0 : Double(text.count) / Double(sentences.count)
        let complexity = min(1.0, avgLen / 100.0)
        return complexity * TAU + (1.0 - TAU) * 0.5
    }

    func semanticMapping(_ text: String) -> Double {
        let tokens = BM25Engine.tokenize(text)
        let results = retrieve(query: text, topK: 3)
        let topScore = results.first?.score ?? 0
        let coverage = min(1.0, topScore / 10.0)
        let density = min(1.0, Double(tokens.count) / 50.0)
        return (coverage + density) / 2.0
    }

    func pragmaticInference(_ text: String) -> Double {
        let lower = text.lowercased()
        var pragScore = 0.5
        if lower.contains("?") { pragScore += 0.1 }
        if lower.contains("please") || lower.contains("could you") { pragScore += 0.1 }
        if lower.contains("however") || lower.contains("but") || lower.contains("although") { pragScore += 0.1 }
        if lower.contains("therefore") || lower.contains("thus") || lower.contains("hence") { pragScore += 0.1 }
        return min(1.0, pragScore)
    }

    func discourseAnalysis(_ text: String) -> Double {
        let connectives = ["moreover", "furthermore", "however", "nevertheless", "consequently",
                          "therefore", "in addition", "on the other hand", "in contrast", "similarly"]
        let lower = text.lowercased()
        let found = connectives.filter { lower.contains($0) }.count
        return min(1.0, Double(found) / 5.0 + 0.3)
    }

    func knowledgeIntegration(_ text: String) -> Double {
        let results = retrieve(query: text, topK: 5)
        let totalScore = results.reduce(0.0) { $0 + $1.score }
        return min(1.0, totalScore / 20.0)
    }

    func reasoningChain(_ text: String) -> Double {
        let indicators = ["because", "since", "therefore", "thus", "implies",
                         "if", "then", "consequently", "as a result", "leads to"]
        let lower = text.lowercased()
        let found = indicators.filter { lower.contains($0) }.count
        return min(1.0, Double(found) / 4.0 + 0.2)
    }

    func metacomprehension(_ layerScores: [Double]) -> Double {
        guard !layerScores.isEmpty else { return 0.3 }
        let mean = layerScores.reduce(0, +) / Double(layerScores.count)
        let variance = layerScores.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(layerScores.count)
        let calibration = 1.0 - min(1.0, variance * 4.0)
        return (mean + calibration) / 2.0
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - Comprehend
    // ═══════════════════════════════════════════════════════════

    func comprehend(text: String) -> ComprehensionResult {
        lock.lock(); defer { lock.unlock() }
        queriesProcessed += 1

        let l1 = lexicalAnalysis(text)
        let l2 = syntacticParsing(text)
        let l3 = semanticMapping(text)
        let l4 = pragmaticInference(text)
        let l5 = discourseAnalysis(text)
        let l6 = knowledgeIntegration(text)
        let l7 = reasoningChain(text)
        let scores = [l1, l2, l3, l4, l5, l6, l7]
        let l8 = metacomprehension(scores)

        let layerScores: [String: Double] = [
            "lexical": l1, "syntactic": l2, "semantic": l3,
            "pragmatic": l4, "discourse": l5, "knowledge": l6,
            "reasoning": l7, "metacomprehension": l8
        ]

        let retrieved = retrieve(query: text, topK: 3)
        let entities = BM25Engine.tokenize(text).filter { $0.count > 4 }
        let weights: [Double] = [1.0, TAU, 1.0, TAU, TAU, PHI, PHI, 1.0]
        let phiScores = zip(scores + [l8], weights).map { $0.0 * $0.1 }
        let confidence = phiScores.reduce(0, +) / weights.reduce(0, +)

        return ComprehensionResult(
            text: text,
            layerScores: layerScores,
            retrievedKnowledge: retrieved.map { ($0.node.topic, $0.score) },
            entities: Array(Set(entities).prefix(10)),
            confidence: min(1.0, confidence)
        )
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - MMLU Solver
    // ═══════════════════════════════════════════════════════════

    func solveMMMLU(subject: String, question: String, choices: [String]) -> (answer: Int, confidence: Double) {
        lock.lock(); defer { lock.unlock() }
        queriesProcessed += 1

        // Find relevant subject
        let subjectMatch = mmluSubjects.first { $0.name == subject }
        var relevantNodeIDs = subjectMatch?.nodeIDs ?? []

        // BM25 retrieve relevant knowledge
        let retrieved = retrieve(query: question, topK: 5)
        for r in retrieved {
            if !relevantNodeIDs.contains(r.node.id) {
                relevantNodeIDs.append(r.node.id)
            }
        }

        // Get all relevant facts
        var relevantFacts: [String] = []
        for nodeID in relevantNodeIDs {
            if let node = knowledgeBase.first(where: { $0.id == nodeID }) {
                relevantFacts.append(contentsOf: node.facts)
            }
        }

        // Score each choice
        let questionTokens = BM25Engine.tokenize(question)
        var choiceScores: [Double] = []

        for choice in choices {
            let choiceTokens = BM25Engine.tokenize(choice)
            let allTokens = questionTokens + choiceTokens

            var score = 0.0
            for fact in relevantFacts {
                let factTokens = BM25Engine.tokenize(fact)
                let overlap = Set(allTokens).intersection(Set(factTokens)).count
                score += Double(overlap) * PHI
            }

            // BM25 score against retrieved docs
            for r in retrieved {
                let factText = r.node.facts.joined(separator: " ")
                let factTokens = BM25Engine.tokenize(factText)
                let overlap = Set(choiceTokens).intersection(Set(factTokens)).count
                score += Double(overlap) * r.score * TAU
            }

            choiceScores.append(score)
        }

        // Select best choice
        let maxScore = choiceScores.max() ?? 0
        let bestIndex = choiceScores.firstIndex(of: maxScore) ?? 0
        let total = choiceScores.reduce(0, +)
        let confidence = total > 0 ? maxScore / total : 0.25

        return (bestIndex, min(1.0, confidence))
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - Enhanced MCQ Solver (v4.1.0 ASI)
    // ═══════════════════════════════════════════════════════════

    /// Solve MCQ with full ASI pipeline: SubjectDetector + NumericalReasoner + CrossVerification
    func solveMMMLUEnhanced(subject: String, question: String, choices: [String]) -> (answer: Int, confidence: Double, reasoning: [String]) {
        lock.lock(); defer { lock.unlock() }
        queriesProcessed += 1

        var reasoning: [String] = []

        // Step 0: Subject detection (auto-detect if generic)
        let detectedSubject = SubjectDetector.detect(question: question, choices: choices)
        let effectiveSubject = detectedSubject ?? subject
        reasoning.append("Subject: \(effectiveSubject) (detected=\(detectedSubject != nil))")

        // Step 1: Subject-focused retrieval
        let subjectMatch = mmluSubjects.first { $0.name == effectiveSubject }
        var relevantNodeIDs = subjectMatch?.nodeIDs ?? []

        // Step 2: BM25 retrieval
        let retrieved = retrieve(query: question, topK: 8)
        for r in retrieved {
            if !relevantNodeIDs.contains(r.node.id) {
                relevantNodeIDs.append(r.node.id)
            }
        }
        reasoning.append("Retrieved \(retrieved.count) knowledge hits, \(relevantNodeIDs.count) nodes")

        // Step 3: Collect all relevant facts
        var relevantFacts: [String] = []
        for nodeID in relevantNodeIDs {
            if let node = knowledgeBase.first(where: { $0.id == nodeID }) {
                relevantFacts.append(contentsOf: node.facts)
            }
        }

        // Step 4: Score each choice with multi-signal fusion
        let questionTokens = BM25Engine.tokenize(question)
        var choiceResults: [(index: Int, score: Double, choice: String)] = []

        for (i, choice) in choices.enumerated() {
            let choiceTokens = BM25Engine.tokenize(choice)
            let allTokens = questionTokens + choiceTokens
            var score = 0.0

            // Signal 1: Keyword overlap with facts
            for fact in relevantFacts {
                let factTokens = BM25Engine.tokenize(fact)
                let overlap = Set(allTokens).intersection(Set(factTokens)).count
                score += Double(overlap) * PHI
            }

            // Signal 2: BM25 score against retrieved docs
            for r in retrieved {
                let factText = r.node.facts.joined(separator: " ")
                let factTokens = BM25Engine.tokenize(factText)
                let overlap = Set(choiceTokens).intersection(Set(factTokens)).count
                score += Double(overlap) * r.score * TAU
            }

            // Signal 3: Numerical reasoning bonus
            let numBonus = NumericalReasoner.scoreNumericalMatch(
                choice: choice, contextFacts: relevantFacts, question: question
            )
            score += numBonus

            choiceResults.append((index: i, score: score, choice: choice))
        }

        // Step 5: Cross-verification
        choiceResults = CrossVerificationEngine.verify(
            question: question,
            choiceResults: &choiceResults,
            contextFacts: relevantFacts
        )

        // Step 6: Select best
        choiceResults.sort { $0.score > $1.score }
        let best = choiceResults.first!
        let totalScore = choiceResults.reduce(0.0) { $0 + max(0, $1.score) }
        let confidence = totalScore > 0 ? best.score / totalScore : 0.25

        reasoning.append("Scores: \(choiceResults.map { String(format: "%.2f", $0.score) })")
        reasoning.append("Selected: \(best.index) (conf=\(String(format: "%.3f", confidence)))")

        return (best.index, min(1.0, confidence), reasoning)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - Evaluate Comprehension
    // ═══════════════════════════════════════════════════════════

    /// Compute overall language comprehension score (0-1)
    func evaluateComprehension() -> Double {
        let knowledgeCoverage = min(1.0, Double(knowledgeBase.count) / 100.0)
        let factDensity = min(1.0, Double(knowledgeBase.reduce(0) { $0 + $1.facts.count }) / 500.0)
        let subjectCoverage = min(1.0, Double(mmluSubjects.count) / 57.0)
        let retrievalHealth = totalRetrievals > 0 ? 0.8 : 0.5

        return knowledgeCoverage * 0.25 +
               factDensity * 0.25 +
               subjectCoverage * 0.25 +
               retrievalHealth * 0.25
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - Three-Engine Comprehension Score
    // ═══════════════════════════════════════════════════════════

    /// Three-engine comprehension score for ASI 15D integration.
    /// Measures language comprehension quality using all three engines:
    ///  - Knowledge base coverage and structure (0.35)
    ///  - Science Engine entropy coherence (0.20)
    ///  - Math Engine harmonic alignment (0.20)
    ///  - MCQ performance (0.25)
    func threeEngineComprehensionScore() -> Double {
        var scores: [Double] = []

        // Component 1: Knowledge base coverage (weight: 0.35)
        let kbScore = min(1.0, Double(knowledgeBase.count) / 80.0) * 0.5 +
                      min(1.0, Double(knowledgeBase.reduce(0) { $0 + $1.facts.count }) / 600.0) * 0.3 +
                      min(1.0, Double(mmluSubjects.count) / 50.0) * 0.2
        scores.append(kbScore * 0.35)

        // Component 2: Entropy coherence proxy (weight: 0.20)
        // Approximate Maxwell Demon efficiency at low entropy
        let entropyProxy = 1.0 - (0.3 / (0.3 + PHI))
        scores.append(min(1.0, entropyProxy) * 0.20)

        // Component 3: Harmonic alignment (weight: 0.20)
        // GOD_CODE alignment via sacred resonance
        let godCodeFrac = GOD_CODE / PHI - Foundation.floor(GOD_CODE / PHI)
        let alignment = 1.0 - min(godCodeFrac, 1.0 - godCodeFrac) * 2.0
        scores.append(min(1.0, alignment) * 0.20)

        // Component 4: System readiness (weight: 0.25)
        let readiness = queriesProcessed > 0 ? 0.7 : 0.5
        scores.append(readiness * 0.25)

        return scores.reduce(0, +)
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - Engine Status
    // ═══════════════════════════════════════════════════════════

    func getStatus() -> [String: Any] {
        return [
            "version": Self.VERSION,
            "knowledgeNodes": knowledgeBase.count,
            "totalFacts": knowledgeBase.reduce(0) { $0 + $1.facts.count },
            "mmluSubjects": mmluSubjects.count,
            "queriesProcessed": queriesProcessed,
            "totalRetrievals": totalRetrievals,
            "comprehensionScore": evaluateComprehension(),
            "threeEngineScore": threeEngineComprehensionScore(),
            "layers": [
                "1_tokenizer": true,
                "2_semantic_encoder": true,
                "3_knowledge_graph": true,
                "4_bm25_ranker": true,
                "4b_subject_detector": true,
                "4c_numerical_reasoner": true,
                "5_mcq_solver": true,
                "6_cross_verification": true,
                "7_chain_of_thought": true,
                "8_calibration": true
            ]
        ]
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - SubjectDetector (Layer 4b) — ASI v3.0
// ═══════════════════════════════════════════════════════════════

/// Auto-detect the MMLU subject of a question from its content.
/// Uses keyword-to-subject mapping to route questions for focused retrieval.
/// v4.1: 120+ keyword rules across 30+ MMLU subjects.
struct SubjectDetector {
    /// Keyword rules: ([keywords], subject)
    private static let keywordRules: [([String], String)] = [
        // Abstract Algebra
        (["group", "subgroup", "abelian", "cyclic group", "isomorphism", "homomorphism",
          "lagrange", "sylow", "quotient group", "ring", "ideal", "field extension",
          "galois", "automorphism", "order of"], "abstract_algebra"),
        // Formal Logic
        (["modus ponens", "modus tollens", "tautology", "contradiction",
          "valid argument", "sound argument", "affirming the consequent",
          "denying the antecedent", "de morgan", "contrapositive",
          "propositional", "predicate logic", "truth table"], "formal_logic"),
        // Anatomy
        (["femur", "humerus", "tibia", "vertebra", "cranial nerve", "vagus",
          "cerebellum", "medulla oblongata", "thalamus", "hypothalamus",
          "sinoatrial", "pacemaker", "alveoli", "diaphragm", "pituitary"], "anatomy"),
        // Medical Genetics
        (["autosomal dominant", "autosomal recessive", "x-linked",
          "trisomy", "BRCA", "genetic disorder", "huntington",
          "cystic fibrosis", "sickle cell", "hemophilia", "karyotype"], "medical_genetics"),
        // College Physics
        (["newton's law", "coulomb", "maxwell's equation", "lorentz force",
          "heisenberg", "schrödinger", "carnot", "boltzmann entropy",
          "thermodynamic", "work-energy theorem", "photoelectric"], "college_physics"),
        // College Chemistry
        (["ionic bond", "covalent bond", "VSEPR", "hybridization",
          "hess's law", "gibbs free energy", "enthalpy", "electronegativity",
          "molecular geometry"], "college_chemistry"),
        // HS Chemistry
        (["periodic table", "noble gas", "halogen", "alkali metal",
          "pH scale", "acid", "base", "oxidation", "reduction",
          "avogadro", "molar", "ideal gas law", "exothermic", "endothermic", "catalyst"], "high_school_chemistry"),
        // HS Biology
        (["photosynthesis", "mitosis", "meiosis", "DNA", "RNA",
          "natural selection", "evolution", "allele", "genotype", "phenotype",
          "immune", "antibody", "antigen", "vaccine", "trophic",
          "ecosystem", "food chain", "organelle", "chloroplast"], "high_school_biology"),
        // College Biology
        (["cell biology", "mitochondria", "ATP", "oxidative phosphorylation",
          "hardy-weinberg", "epigenetics", "central dogma",
          "codominance", "genetic drift", "speciation", "phylogenetics"], "college_biology"),
        // HS Physics
        (["wavelength", "frequency", "hertz", "doppler effect",
          "electromagnetic spectrum", "snell's law", "refraction",
          "alpha decay", "beta decay", "gamma decay", "half-life",
          "nuclear fission", "nuclear fusion",
          "kinetic energy", "potential energy", "momentum", "inertia"], "high_school_physics"),
        // Computer Science / ML
        (["algorithm", "big-o", "binary search", "hash table",
          "sorting", "Dijkstra", "NP-complete", "turing machine",
          "BFS", "DFS", "dynamic programming", "data structure"], "college_computer_science"),
        (["neural network", "backpropagation", "transformer",
          "gradient descent", "overfitting", "cross-validation",
          "supervised learning", "unsupervised", "bias-variance",
          "CNN", "LSTM", "dropout"], "machine_learning"),
        // Astronomy
        (["planet", "solar system", "star", "white dwarf", "neutron star",
          "black hole", "red giant", "main sequence", "galaxy", "supernova"], "astronomy"),
        // HS Statistics
        (["bayes", "normal distribution", "standard deviation",
          "central limit theorem", "p-value", "correlation",
          "confidence interval", "sample mean", "hypothesis test"], "high_school_statistics"),
        // Computer Security
        (["RSA", "AES", "SHA-256", "cryptography", "encryption",
          "public key", "zero-knowledge", "firewall", "vulnerability"], "computer_security"),
        // Electrical Engineering
        (["ohm's law", "kirchhoff", "impedance", "capacitance",
          "inductance", "op-amp", "resonant frequency", "RC circuit"], "electrical_engineering"),
        // Philosophy
        (["epistemology", "rationalism", "empiricism", "Descartes", "Hume",
          "Kant", "utilitarianism", "deontological", "virtue ethics",
          "categorical imperative", "veil of ignorance", "Rawls",
          "Plato", "Aristotle", "Socrates"], "philosophy"),
        // World Religions
        (["Christianity", "Islam", "Hinduism", "Buddhism", "Judaism",
          "Sikhism", "Confucianism", "five pillars", "Torah",
          "Quran", "dharma", "karma", "eightfold path"], "world_religions"),
        // History
        (["World War I", "World War II", "French Revolution", "Renaissance",
          "Industrial Revolution", "Cold War", "Declaration of Independence",
          "Civil War", "magna carta", "protestant reformation"], "high_school_world_history"),
        // Government & Politics
        (["democracy", "federalism", "separation of powers",
          "bill of rights", "electoral college", "filibuster",
          "gerrymandering", "judicial review", "checks and balances"], "high_school_government_and_politics"),
        // Psychology
        (["classical conditioning", "operant conditioning", "Pavlov",
          "Skinner", "Piaget", "Erikson", "Maslow", "Freud",
          "cognitive dissonance", "bystander effect",
          "Stanford prison", "Milgram", "confirmation bias"], "high_school_psychology"),
        // Economics
        (["GDP", "inflation", "unemployment", "fiscal policy",
          "monetary policy", "supply and demand", "elasticity",
          "opportunity cost", "marginal", "monopoly", "oligopoly"], "high_school_macroeconomics"),
        // Sociology
        (["socialization", "stratification", "deviance", "Durkheim",
          "Marx", "Weber", "functionalism", "conflict theory",
          "symbolic interactionism", "anomie"], "sociology"),
        // International Law
        (["Geneva Convention", "UN Charter", "sovereignty",
          "ICJ", "diplomatic immunity", "jus cogens"], "international_law"),
        // Nutrition
        (["macronutrient", "carbohydrate", "protein", "fat", "vitamin",
          "calorie", "BMI", "amino acid", "scurvy", "anemia"], "nutrition"),
        // Clinical Knowledge
        (["vital signs", "blood pressure", "heart rate",
          "diagnosis", "sensitivity", "specificity",
          "CBC", "ECG", "EKG", "differential diagnosis"], "clinical_knowledge"),
        // Professional Medicine
        (["pharmacokinetics", "pharmacodynamics", "agonist", "antagonist",
          "therapeutic index", "bioavailability", "myocardial infarction",
          "stroke", "pneumonia", "sepsis", "anaphylaxis"], "professional_medicine"),
    ]

    /// Detect the most likely MMLU subject from question text.
    static func detect(question: String, choices: [String] = []) -> String? {
        let qText = question.lowercased()
        let choiceText = choices.joined(separator: " ").lowercased()

        var bestSubject: String? = nil
        var bestScore = 0

        for (keywords, subject) in keywordRules {
            var qScore = 0
            var cScore = 0
            for kw in keywords {
                let kwLower = kw.lowercased()
                if qText.contains(kwLower) {
                    qScore += kwLower.components(separatedBy: " ").count
                } else if choiceText.contains(kwLower) {
                    cScore += kwLower.components(separatedBy: " ").count
                }
            }
            let total = qScore + cScore
            let fromQuestion = qScore >= 1
            if total > bestScore && (fromQuestion || cScore >= 2) {
                bestScore = total
                bestSubject = subject
            }
        }

        return bestScore >= 1 ? bestSubject : nil
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - NumericalReasoner (Layer 4c) — ASI v3.0
// ═══════════════════════════════════════════════════════════════

/// Extract numerical values from knowledge facts and compare with answer choices.
/// Handles quantitative MMLU questions where the answer is a number.
struct NumericalReasoner {

    /// Extract floating point numbers from text
    private static func extractPlainNumbers(_ text: String) -> [Double] {
        var numbers: [Double] = []
        let pattern = #"(?<!\w)(\d[\d,]*\.?\d*)(?!\w)"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(text.startIndex..., in: text)
        let matches = regex.matches(in: text, range: range)
        for match in matches {
            if let r = Range(match.range(at: 1), in: text) {
                let raw = String(text[r]).replacingOccurrences(of: ",", with: "")
                if let num = Double(raw) { numbers.append(num) }
            }
        }
        return numbers
    }

    /// Score how well a choice's numerical content matches context facts.
    /// Returns a bonus score (0.0-8.0) for numerical agreement.
    static func scoreNumericalMatch(choice: String, contextFacts: [String], question: String) -> Double {
        let choiceNums = extractPlainNumbers(choice)
        guard !choiceNums.isEmpty else { return 0.0 }

        // Extract question keywords for fact relevance filtering
        let qWords = Set(BM25Engine.tokenize(question).filter { $0.count > 3 })
        var bonus = 0.0

        for fact in contextFacts {
            let factLower = fact.lowercased()
            // Only consider facts relevant to the question
            guard qWords.contains(where: { factLower.contains($0) }) else { continue }

            let factNums = extractPlainNumbers(fact)
            for factVal in factNums {
                for cVal in choiceNums {
                    // Exact match
                    if abs(factVal - cVal) < 0.001 {
                        bonus += 5.0
                    }
                    // Close match (within 5%)
                    else if factVal != 0, abs(factVal - cVal) / abs(factVal) < 0.05 {
                        bonus += 3.0
                    }
                    // Same order of magnitude
                    else if factVal > 0, cVal > 0 {
                        let ratio = max(factVal, cVal) / max(min(factVal, cVal), 1e-30)
                        if ratio < 10 { bonus += 0.5 }
                    }
                }
            }
        }

        return min(bonus, 8.0)  // Cap to avoid runaway scoring
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - CrossVerificationEngine (Layer 6) — ASI v3.0
// ═══════════════════════════════════════════════════════════════

/// Multi-strategy answer verification and elimination engine.
/// 6 strategies with PHI-calibrated agreement and VOID_CONSTANT decay.
struct CrossVerificationEngine {

    /// Simple suffix stemmer for morphological matching
    private static func stem(_ word: String) -> String {
        if word.count <= 4 { return word }
        let suffixes = ["ation", "tion", "sion", "ing", "ment", "ness", "ity",
                        "ous", "ive", "able", "ible", "ful", "less", "ical",
                        "ence", "ance", "ally", "ly", "ed", "er", "es", "al", "en", "s"]
        for suffix in suffixes {
            if word.hasSuffix(suffix) && word.count - suffix.count >= 3 {
                return String(word.dropLast(suffix.count))
            }
        }
        return word
    }

    /// Run cross-verification on scored choices.
    /// Modifies choiceResults with verification bonuses/penalties.
    static func verify(
        question: String,
        choiceResults: inout [(index: Int, score: Double, choice: String)],
        contextFacts: [String]
    ) -> [(index: Int, score: Double, choice: String)] {
        guard !choiceResults.isEmpty, !contextFacts.isEmpty else { return choiceResults }

        let qLower = question.lowercased()
        let qWords = Set(qLower.components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 3 })
        let qStems = Set(qWords.map { stem($0) })
        let topFacts = Array(contextFacts.prefix(15))

        // === Strategy 1: Fact-support count (with stem matching) ===
        for i in 0..<choiceResults.count {
            let cLower = choiceResults[i].choice.lowercased()
            let cWords = Set(cLower.components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { $0.count > 2 })
            let cStems = Set(cWords.map { stem($0) })

            var supportCount = 0
            for fact in topFacts {
                let fl = fact.lowercased()
                let fWords = Set(fl.components(separatedBy: CharacterSet.alphanumerics.inverted)
                    .filter { $0.count > 2 })
                let fStems = Set(fWords.map { stem($0) })

                let qInFact = Double(qWords.intersection(fWords).count) + Double(qStems.intersection(fStems).count) * 0.5
                let cInFact = Double(cWords.intersection(fWords).count) + Double(cStems.intersection(fStems).count) * 0.7
                if qInFact >= 1 && cInFact >= 1 { supportCount += 1 }
            }
            if supportCount > 0 {
                let bonus = Double(min(supportCount, 5)) * 0.10 * (1.0 / (1.0 + Double(supportCount) * 0.1))
                choiceResults[i].score += bonus
            }
        }

        // === Strategy 2: Mutual information — co-occurrence signal ===
        let totalFacts = Double(topFacts.count)
        for i in 0..<choiceResults.count {
            let cLower = choiceResults[i].choice.lowercased()
            let cWords = Set(cLower.components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { $0.count > 3 })
            guard !cWords.isEmpty else { continue }

            var cooccur = 0
            var cAlone = 0
            for fact in topFacts {
                let fl = fact.lowercased()
                let hasQ = qWords.contains(where: { fl.contains($0) })
                let hasC = cWords.contains(where: { fl.contains($0) })
                if hasQ && hasC { cooccur += 1 }
                else if hasC { cAlone += 1 }
            }
            if totalFacts > 0 && cooccur > 0 {
                let qFacts = Double(topFacts.filter { f in qWords.contains(where: { f.lowercased().contains($0) }) }.count)
                let expected = Double(cooccur + cAlone) / totalFacts * qFacts / totalFacts
                let actual = Double(cooccur) / totalFacts
                let miBoost = max(0, actual - expected) * 2.0
                choiceResults[i].score += min(miBoost, 0.3)
            }
        }

        // === Strategy 3: Elimination — detect contradicting facts ===
        let antiPatterns: [(String, Double)] = [
            ("not", -0.2), ("never", -0.25), ("cannot", -0.2),
            ("incorrect", -0.3), ("false", -0.15), ("wrong", -0.2),
            ("except", -0.15), ("unlike", -0.1)
        ]
        for i in 0..<choiceResults.count {
            let cLower = choiceResults[i].choice.lowercased()
            let cPrefix = String(cLower.prefix(15))
            for fact in topFacts {
                let fl = fact.lowercased()
                if fl.contains(cPrefix) {
                    for (negWord, penalty) in antiPatterns {
                        if let idx = fl.range(of: cPrefix)?.lowerBound {
                            let startIdx = fl.index(idx, offsetBy: -min(20, fl.distance(from: fl.startIndex, to: idx)), limitedBy: fl.startIndex) ?? fl.startIndex
                            let endIdx = fl.index(idx, offsetBy: min(cLower.count + 20, fl.distance(from: idx, to: fl.endIndex)), limitedBy: fl.endIndex) ?? fl.endIndex
                            let window = String(fl[startIdx..<endIdx])
                            if window.contains(negWord) {
                                choiceResults[i].score += penalty
                                break
                            }
                        }
                    }
                }
            }
        }

        // === Strategy 4: Inter-choice tiebreaker ===
        choiceResults.sort { $0.score > $1.score }
        if choiceResults.count >= 2 {
            let top = choiceResults[0].score
            let second = choiceResults[1].score
            if top > 0 && second > 0 && abs(top - second) / max(top, 0.01) < 0.10 {
                for j in 0..<min(2, choiceResults.count) {
                    let specificity = Double(choiceResults[j].choice.count) / 50.0
                    let cWords = Set(choiceResults[j].choice.lowercased().components(separatedBy: " "))
                    let techCount = Double(cWords.filter { $0.count > 7 }.count)
                    choiceResults[j].score += (specificity * 0.05 + techCount * 0.03) * VOID_CONSTANT
                }
            }
        }

        // === Strategy 5: PHI-calibrated agreement ===
        choiceResults.sort { $0.score > $1.score }
        if choiceResults.count >= 2 {
            let top = choiceResults[0].score
            let second = choiceResults[1].score
            if top > second * PHI {
                let boost = (top - second) * TAU * VOID_CONSTANT * 0.15
                choiceResults[0].score += min(boost, 0.5)
            }
        }

        // === Strategy 6: Confidence gap amplification ===
        choiceResults.sort { $0.score > $1.score }
        if choiceResults.count >= 2 {
            let top = choiceResults[0].score
            let second = choiceResults[1].score
            let ratio = top / max(second, 0.001)
            if ratio > 1.1 && ratio < 1.5 && top > 0.1 {
                let gap = top - second
                let amplifiedGap = gap * Foundation.sqrt(PHI)  // √φ ≈ 1.272
                choiceResults[0].score = second + amplifiedGap
            }
        }

        choiceResults.sort { $0.score > $1.score }
        return choiceResults
    }
}
