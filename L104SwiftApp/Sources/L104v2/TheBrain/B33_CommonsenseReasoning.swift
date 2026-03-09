// ═══════════════════════════════════════════════════════════════
// B33_CommonsenseReasoning.swift — Commonsense Reasoning Engine
// L104v2 — TheBrain Layer — EVO_68 SOVEREIGN_CONVERGENCE
// 8-Layer Reasoning + ScienceEngineBridge + MCQ Solver
// ═══════════════════════════════════════════════════════════════

import Foundation

// ═══════════════════════════════════════════════════════════════
// MARK: - CommonsenseResult — Structured output of reasoning
// ═══════════════════════════════════════════════════════════════

struct CommonsenseResult {
    let query: String
    let layerScores: [String: Double]
    let overallConfidence: Double
    let reasoning: [String]
    let scienceFacts: [(String, Double)]
}

// ═══════════════════════════════════════════════════════════════
// MARK: - CommonsenseRule — A single commonsense rule
// ═══════════════════════════════════════════════════════════════

struct CommonsenseRule {
    let condition: String
    let conclusion: String
    let confidence: Double
}

// ═══════════════════════════════════════════════════════════════
// MARK: - EventScript — A script for a common activity
// ═══════════════════════════════════════════════════════════════

struct EventScript {
    let name: String
    let steps: [String]
    let confidence: Double
}

// ═══════════════════════════════════════════════════════════════
// MARK: - AnalogicalPattern — A structural mapping between domains
// ═══════════════════════════════════════════════════════════════

struct AnalogicalPattern {
    let source: String
    let target: String
    let mapping: String
    let confidence: Double
}

// ═══════════════════════════════════════════════════════════════
// MARK: - ScienceEngineBridge — Bridge to ScienceKB
// ═══════════════════════════════════════════════════════════════

struct ScienceEngineBridge {

    // ─── Query ScienceKB for reasoning-relevant facts ───
    func queryForReasoning(domain: String, topic: String) -> [(fact: String, confidence: Double)] {
        let kb = ScienceKB.shared
        let domainFacts = kb.factsForDomain(domain)
        let topicTokens = tokenize(topic)

        var results: [(fact: String, confidence: Double)] = []

        for fact in domainFacts {
            let factText = "\(fact.subject) \(fact.relation) \(fact.obj)"
            let factTokens = tokenize(factText)
            let overlap = Double(topicTokens.intersection(factTokens).count)
            let total = Double(max(topicTokens.count, 1))
            let relevance = overlap / total

            if relevance > 0.0 {
                let adjustedConfidence = fact.confidence * relevance * TAU + fact.confidence * (1.0 - TAU)
                results.append((fact: factText, confidence: min(adjustedConfidence, 1.0)))
            }
        }

        results.sort { $0.confidence > $1.confidence }
        return Array(results.prefix(20))
    }

    // ─── Query across all domains ───
    func queryAllDomains(topic: String) -> [(fact: String, confidence: Double)] {
        let domains = ["biology", "body_systems", "earth_science", "physics",
                       "chemistry", "astronomy", "ecology", "measurement", "technology"]
        var allResults: [(fact: String, confidence: Double)] = []
        for domain in domains {
            allResults.append(contentsOf: queryForReasoning(domain: domain, topic: topic))
        }
        allResults.sort { $0.confidence > $1.confidence }
        return Array(allResults.prefix(30))
    }

    // ─── Simple tokenizer ───
    private func tokenize(_ text: String) -> Set<String> {
        let lower = text.lowercased()
        let words = lower.components(separatedBy: CharacterSet.alphanumerics.inverted)
        return Set(words.filter { $0.count > 2 })
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - CommonsenseReasoningEngine — Main Engine
// ═══════════════════════════════════════════════════════════════

final class CommonsenseReasoningEngine {

    // ─── Singleton ───
    static let shared = CommonsenseReasoningEngine()

    // ─── Version ───
    let VERSION = COMMONSENSE_ENGINE_VERSION

    // ─── Thread Safety ───
    private let lock = NSLock()

    // ─── Science Bridge ───
    private let scienceBridge = ScienceEngineBridge()

    // ─── Metrics ───
    private var totalQueriesProcessed: Int = 0
    private var totalMCQSolved: Int = 0
    private var averageConfidence: Double = 0.0
    private var layerInvocationCounts: [String: Int] = [
        "spatial": 0, "temporal": 0, "causal": 0, "social": 0,
        "physical": 0, "taxonomic": 0, "eventScript": 0, "analogical": 0
    ]

    // ─── PHI weights for each layer (PHI-scaled) ───
    private let layerWeights: [String: Double] = [
        "spatial":     1.0,
        "temporal":    1.0 / PHI,
        "causal":      1.0 / (PHI * PHI),
        "social":      1.0 / (PHI * PHI * PHI),
        "physical":    TAU,
        "taxonomic":   TAU / PHI,
        "eventScript": TAU / (PHI * PHI),
        "analogical":  TAU / (PHI * PHI * PHI)
    ]

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Initialization
    // ═══════════════════════════════════════════════════════════════

    private init() {}

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Layer 1: Spatial Reasoning
    // ═══════════════════════════════════════════════════════════════

    func spatialReasoning() -> [CommonsenseRule] {
        lock.lock()
        layerInvocationCounts["spatial"] = (layerInvocationCounts["spatial"] ?? 0) + 1
        lock.unlock()

        return [
            CommonsenseRule(condition: "X is inside Y and Y is inside Z",
                           conclusion: "X is inside Z",
                           confidence: 0.99),
            CommonsenseRule(condition: "X is on top of Y",
                           conclusion: "X is above Y",
                           confidence: 0.98),
            CommonsenseRule(condition: "X is above Y and Y is above Z",
                           conclusion: "X is above Z",
                           confidence: 0.97),
            CommonsenseRule(condition: "X is next to Y",
                           conclusion: "Y is next to X",
                           confidence: 0.99),
            CommonsenseRule(condition: "X is between Y and Z",
                           conclusion: "X is closer to both Y and Z than Y is to Z",
                           confidence: 0.90),
            CommonsenseRule(condition: "X is inside a closed container",
                           conclusion: "X cannot be seen from outside without opening the container",
                           confidence: 0.95),
            CommonsenseRule(condition: "X is behind Y from observer's perspective",
                           conclusion: "Y occludes X from the observer",
                           confidence: 0.93),
            CommonsenseRule(condition: "X fills a container completely",
                           conclusion: "No more X can be added to the container without overflow",
                           confidence: 0.97),
            CommonsenseRule(condition: "X is north of Y and Y is north of Z",
                           conclusion: "X is north of Z",
                           confidence: 0.99),
            CommonsenseRule(condition: "X is larger than the opening of container Y",
                           conclusion: "X cannot fit through the opening of Y",
                           confidence: 0.96),
            CommonsenseRule(condition: "Room A is connected to Room B by a door",
                           conclusion: "One can move from Room A to Room B through the door",
                           confidence: 0.98),
            CommonsenseRule(condition: "X is at the bottom of a stack",
                           conclusion: "X must be removed last if the stack is LIFO",
                           confidence: 0.95),
            CommonsenseRule(condition: "X and Y are in different sealed rooms",
                           conclusion: "X and Y cannot physically interact directly",
                           confidence: 0.94),
            CommonsenseRule(condition: "X is under Y on a shelf",
                           conclusion: "Y must be moved to access X",
                           confidence: 0.90),
            CommonsenseRule(condition: "A is left of B and B is left of C",
                           conclusion: "A is left of C",
                           confidence: 0.98),
            CommonsenseRule(condition: "X is on the surface of Y",
                           conclusion: "X is supported by Y",
                           confidence: 0.96),
            CommonsenseRule(condition: "X is near a heat source",
                           conclusion: "X will become warmer over time",
                           confidence: 0.92),
            CommonsenseRule(condition: "X is at the center of a circle",
                           conclusion: "X is equidistant from all points on the circle",
                           confidence: 0.99),
            CommonsenseRule(condition: "X is attached to Y and Y moves",
                           conclusion: "X moves with Y",
                           confidence: 0.95),
            CommonsenseRule(condition: "X is floating on water",
                           conclusion: "X is less dense than water or has buoyant shape",
                           confidence: 0.94),
            CommonsenseRule(condition: "X is in a sealed box inside a car",
                           conclusion: "X is also inside the car",
                           confidence: 0.99),
            CommonsenseRule(condition: "X is adjacent to Y and Y is adjacent to Z",
                           conclusion: "X may or may not be adjacent to Z",
                           confidence: 0.70),
            CommonsenseRule(condition: "Person is in the kitchen",
                           conclusion: "Person has access to kitchen appliances and utensils",
                           confidence: 0.95),
            CommonsenseRule(condition: "X is upside down",
                           conclusion: "The top of X is now facing the ground",
                           confidence: 0.98),
            CommonsenseRule(condition: "X is in the same room as Y",
                           conclusion: "X and Y can potentially interact",
                           confidence: 0.93),
            CommonsenseRule(condition: "X is in front of a mirror",
                           conclusion: "X's reflection can be seen in the mirror",
                           confidence: 0.97),
            CommonsenseRule(condition: "X is surrounded by water on all sides",
                           conclusion: "X is on an island or is floating",
                           confidence: 0.88),
            CommonsenseRule(condition: "X is in a moving vehicle",
                           conclusion: "X is also moving relative to the ground",
                           confidence: 0.99),
            CommonsenseRule(condition: "X is beyond the horizon",
                           conclusion: "X cannot be seen with the naked eye at ground level",
                           confidence: 0.92),
            CommonsenseRule(condition: "X is stacked on Y and Z is stacked on X",
                           conclusion: "Z is higher than Y",
                           confidence: 0.98),
            CommonsenseRule(condition: "X is in a pocket of person P",
                           conclusion: "X moves wherever P goes",
                           confidence: 0.97),
            CommonsenseRule(condition: "X is spread across the entire table",
                           conclusion: "There is little room for other objects on the table",
                           confidence: 0.85)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Layer 2: Temporal Reasoning
    // ═══════════════════════════════════════════════════════════════

    func temporalReasoning() -> [CommonsenseRule] {
        lock.lock()
        layerInvocationCounts["temporal"] = (layerInvocationCounts["temporal"] ?? 0) + 1
        lock.unlock()

        return [
            CommonsenseRule(condition: "X happened before Y and Y happened before Z",
                           conclusion: "X happened before Z",
                           confidence: 0.99),
            CommonsenseRule(condition: "X and Y happened at the same time",
                           conclusion: "Neither X preceded Y nor Y preceded X",
                           confidence: 0.99),
            CommonsenseRule(condition: "X takes longer than Y",
                           conclusion: "If started at the same time, Y finishes first",
                           confidence: 0.97),
            CommonsenseRule(condition: "X was done yesterday and today is Tuesday",
                           conclusion: "X was done on Monday",
                           confidence: 0.99),
            CommonsenseRule(condition: "X happened in the morning and Y in the evening of the same day",
                           conclusion: "X happened before Y",
                           confidence: 0.99),
            CommonsenseRule(condition: "X is a child and Y is X's grandparent",
                           conclusion: "Y was born before X",
                           confidence: 0.99),
            CommonsenseRule(condition: "Seasons cycle in order: spring, summer, fall, winter",
                           conclusion: "After winter comes spring again",
                           confidence: 0.99),
            CommonsenseRule(condition: "X started before Y and both are still ongoing",
                           conclusion: "X has been going on longer than Y",
                           confidence: 0.98),
            CommonsenseRule(condition: "X must dry before painting",
                           conclusion: "Painting cannot begin until X is dry",
                           confidence: 0.96),
            CommonsenseRule(condition: "X was born in 1990 and Y in 2000",
                           conclusion: "X is older than Y",
                           confidence: 0.99),
            CommonsenseRule(condition: "Breakfast is eaten before lunch",
                           conclusion: "Lunch occurs after breakfast",
                           confidence: 0.98),
            CommonsenseRule(condition: "X graduated from college",
                           conclusion: "X attended college before graduating",
                           confidence: 0.99),
            CommonsenseRule(condition: "X planted a seed",
                           conclusion: "The seed will germinate after some time, not immediately",
                           confidence: 0.95),
            CommonsenseRule(condition: "Night follows day",
                           conclusion: "After every day there will be a night",
                           confidence: 0.99),
            CommonsenseRule(condition: "X is cooking dinner at 6pm",
                           conclusion: "Dinner will be ready sometime after 6pm",
                           confidence: 0.95),
            CommonsenseRule(condition: "X was built in the 1800s",
                           conclusion: "X is over 100 years old",
                           confidence: 0.99),
            CommonsenseRule(condition: "X deadline is tomorrow",
                           conclusion: "X must be completed within the next 24 hours",
                           confidence: 0.97),
            CommonsenseRule(condition: "X occurs annually",
                           conclusion: "X happens once every year",
                           confidence: 0.99),
            CommonsenseRule(condition: "X is a baby",
                           conclusion: "X was born recently",
                           confidence: 0.96),
            CommonsenseRule(condition: "X has been running for 2 hours",
                           conclusion: "X started 2 hours ago",
                           confidence: 0.99),
            CommonsenseRule(condition: "X happens every Monday",
                           conclusion: "X will happen next Monday",
                           confidence: 0.95),
            CommonsenseRule(condition: "The ice cream has been out of the freezer for an hour in summer",
                           conclusion: "The ice cream has likely melted",
                           confidence: 0.93),
            CommonsenseRule(condition: "X requires Y to finish first",
                           conclusion: "Y must be completed before X can start",
                           confidence: 0.98),
            CommonsenseRule(condition: "X went to sleep at 10pm and woke at 6am",
                           conclusion: "X slept for about 8 hours",
                           confidence: 0.97),
            CommonsenseRule(condition: "X happened during World War II",
                           conclusion: "X happened between 1939 and 1945",
                           confidence: 0.98),
            CommonsenseRule(condition: "X is a historical figure from ancient Rome",
                           conclusion: "X lived over 1500 years ago",
                           confidence: 0.95),
            CommonsenseRule(condition: "X is still under warranty for 1 year",
                           conclusion: "X was purchased less than the warranty period ago",
                           confidence: 0.94),
            CommonsenseRule(condition: "X takes 30 minutes to bake",
                           conclusion: "X will be done 30 minutes after entering the oven",
                           confidence: 0.96),
            CommonsenseRule(condition: "X is aging",
                           conclusion: "X is getting older with each passing day",
                           confidence: 0.99),
            CommonsenseRule(condition: "X retired after 30 years of work",
                           conclusion: "X started working about 30 years before retirement",
                           confidence: 0.98),
            CommonsenseRule(condition: "A meeting is scheduled for 3pm",
                           conclusion: "Attendees should arrive by 3pm",
                           confidence: 0.96),
            CommonsenseRule(condition: "X expired last week",
                           conclusion: "X is no longer valid or fresh",
                           confidence: 0.95)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Layer 3: Causal Reasoning
    // ═══════════════════════════════════════════════════════════════

    func causalReasoning() -> [CommonsenseRule] {
        lock.lock()
        layerInvocationCounts["causal"] = (layerInvocationCounts["causal"] ?? 0) + 1
        lock.unlock()

        return [
            CommonsenseRule(condition: "It rains heavily",
                           conclusion: "The ground gets wet",
                           confidence: 0.98),
            CommonsenseRule(condition: "A glass is dropped on a hard floor",
                           conclusion: "The glass is likely to break",
                           confidence: 0.93),
            CommonsenseRule(condition: "Someone studies hard for an exam",
                           conclusion: "They are more likely to get a good grade",
                           confidence: 0.88),
            CommonsenseRule(condition: "A plant receives no water for weeks",
                           conclusion: "The plant will wilt and eventually die",
                           confidence: 0.96),
            CommonsenseRule(condition: "A car runs out of fuel",
                           conclusion: "The car will stop running",
                           confidence: 0.99),
            CommonsenseRule(condition: "Someone eats spoiled food",
                           conclusion: "They may become sick",
                           confidence: 0.90),
            CommonsenseRule(condition: "Ice is exposed to temperatures above 0 degrees Celsius",
                           conclusion: "The ice will begin to melt",
                           confidence: 0.99),
            CommonsenseRule(condition: "A person exercises regularly",
                           conclusion: "Their physical fitness improves over time",
                           confidence: 0.92),
            CommonsenseRule(condition: "An alarm clock rings",
                           conclusion: "The person wakes up",
                           confidence: 0.85),
            CommonsenseRule(condition: "A fire is deprived of oxygen",
                           conclusion: "The fire will go out",
                           confidence: 0.98),
            CommonsenseRule(condition: "Someone presses the brake pedal",
                           conclusion: "The vehicle slows down or stops",
                           confidence: 0.97),
            CommonsenseRule(condition: "A ball is thrown upward",
                           conclusion: "The ball will eventually come back down due to gravity",
                           confidence: 0.99),
            CommonsenseRule(condition: "A person does not sleep for several days",
                           conclusion: "Their cognitive performance degrades significantly",
                           confidence: 0.96),
            CommonsenseRule(condition: "A rubber band is stretched too far",
                           conclusion: "The rubber band will snap",
                           confidence: 0.92),
            CommonsenseRule(condition: "Electricity flows through a wire",
                           conclusion: "The wire heats up slightly due to resistance",
                           confidence: 0.95),
            CommonsenseRule(condition: "A traffic light turns red",
                           conclusion: "Cars in that lane must stop",
                           confidence: 0.97),
            CommonsenseRule(condition: "Someone practices a musical instrument daily",
                           conclusion: "Their musical skill improves",
                           confidence: 0.93),
            CommonsenseRule(condition: "A window is left open during a rainstorm",
                           conclusion: "The area near the window will get wet",
                           confidence: 0.94),
            CommonsenseRule(condition: "A company raises its prices significantly",
                           conclusion: "Fewer customers will buy the product",
                           confidence: 0.85),
            CommonsenseRule(condition: "Someone yells in a library",
                           conclusion: "Other people will be disturbed",
                           confidence: 0.96),
            CommonsenseRule(condition: "A pot of water is heated on a stove",
                           conclusion: "The water will eventually boil",
                           confidence: 0.98),
            CommonsenseRule(condition: "A road is covered with ice",
                           conclusion: "Vehicles are more likely to skid",
                           confidence: 0.95),
            CommonsenseRule(condition: "A person touches a hot stove",
                           conclusion: "They will feel pain and pull their hand away",
                           confidence: 0.97),
            CommonsenseRule(condition: "Seeds are planted in fertile soil with water and sunlight",
                           conclusion: "The seeds will germinate and grow",
                           confidence: 0.94),
            CommonsenseRule(condition: "A pipe freezes in winter",
                           conclusion: "The pipe may burst as ice expands",
                           confidence: 0.88),
            CommonsenseRule(condition: "Someone runs a red light",
                           conclusion: "They risk getting into an accident or receiving a ticket",
                           confidence: 0.92),
            CommonsenseRule(condition: "A battery is fully discharged",
                           conclusion: "The device powered by it stops working",
                           confidence: 0.98),
            CommonsenseRule(condition: "A match is struck against a rough surface",
                           conclusion: "The match ignites",
                           confidence: 0.94),
            CommonsenseRule(condition: "Metal is left in the rain for a long time",
                           conclusion: "The metal will rust",
                           confidence: 0.91),
            CommonsenseRule(condition: "A person reads in very dim light",
                           conclusion: "Their eyes will strain over time",
                           confidence: 0.87),
            CommonsenseRule(condition: "An overloaded electrical circuit receives more current",
                           conclusion: "A fuse blows or a breaker trips",
                           confidence: 0.93),
            CommonsenseRule(condition: "A dam breaks",
                           conclusion: "Water floods the downstream area",
                           confidence: 0.99)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Layer 4: Social Reasoning
    // ═══════════════════════════════════════════════════════════════

    func socialReasoning() -> [CommonsenseRule] {
        lock.lock()
        layerInvocationCounts["social"] = (layerInvocationCounts["social"] ?? 0) + 1
        lock.unlock()

        return [
            CommonsenseRule(condition: "Someone smiles at another person",
                           conclusion: "They are likely expressing friendliness or happiness",
                           confidence: 0.90),
            CommonsenseRule(condition: "Someone frowns and crosses their arms",
                           conclusion: "They are likely displeased or defensive",
                           confidence: 0.87),
            CommonsenseRule(condition: "Someone gives a gift",
                           conclusion: "They likely want to express appreciation or strengthen a relationship",
                           confidence: 0.92),
            CommonsenseRule(condition: "Someone apologizes sincerely",
                           conclusion: "They recognize they did something wrong and want to make amends",
                           confidence: 0.91),
            CommonsenseRule(condition: "Someone avoids eye contact",
                           conclusion: "They may be shy, uncomfortable, or hiding something",
                           confidence: 0.78),
            CommonsenseRule(condition: "A person is crying",
                           conclusion: "They are likely sad, in pain, or experiencing strong emotions",
                           confidence: 0.88),
            CommonsenseRule(condition: "Someone extends their hand for a handshake",
                           conclusion: "They are initiating a greeting or agreement",
                           confidence: 0.95),
            CommonsenseRule(condition: "A person whispers to another",
                           conclusion: "They are sharing private or sensitive information",
                           confidence: 0.85),
            CommonsenseRule(condition: "Someone stands up when an elder enters the room",
                           conclusion: "They are showing respect",
                           confidence: 0.93),
            CommonsenseRule(condition: "Someone says thank you",
                           conclusion: "They are expressing gratitude",
                           confidence: 0.97),
            CommonsenseRule(condition: "A person turns their back on someone speaking",
                           conclusion: "They are being rude or dismissive",
                           confidence: 0.82),
            CommonsenseRule(condition: "Someone buys flowers for their partner",
                           conclusion: "They are expressing romantic affection or apologizing",
                           confidence: 0.86),
            CommonsenseRule(condition: "A group of people clap after a performance",
                           conclusion: "They are showing appreciation for the performance",
                           confidence: 0.96),
            CommonsenseRule(condition: "Someone raises their voice in an argument",
                           conclusion: "They are angry or frustrated",
                           confidence: 0.89),
            CommonsenseRule(condition: "A person holds the door open for someone behind them",
                           conclusion: "They are being polite and considerate",
                           confidence: 0.95),
            CommonsenseRule(condition: "Someone nods while another person is talking",
                           conclusion: "They are indicating agreement or understanding",
                           confidence: 0.88),
            CommonsenseRule(condition: "A person waves goodbye",
                           conclusion: "They are departing or signaling the end of an interaction",
                           confidence: 0.96),
            CommonsenseRule(condition: "Someone offers their seat to a pregnant woman on a bus",
                           conclusion: "They are being courteous and following social norms",
                           confidence: 0.97),
            CommonsenseRule(condition: "A child hides behind their parent when meeting a stranger",
                           conclusion: "The child is feeling shy or fearful",
                           confidence: 0.91),
            CommonsenseRule(condition: "Someone tips generously at a restaurant",
                           conclusion: "They appreciated the service or are naturally generous",
                           confidence: 0.87),
            CommonsenseRule(condition: "A person laughs at someone's joke",
                           conclusion: "They found it amusing or are being polite",
                           confidence: 0.85),
            CommonsenseRule(condition: "Someone introduces themselves with their name",
                           conclusion: "They are following social convention for a first meeting",
                           confidence: 0.96),
            CommonsenseRule(condition: "A person sends a condolence card",
                           conclusion: "They are expressing sympathy for someone's loss",
                           confidence: 0.97),
            CommonsenseRule(condition: "Someone helps a stranger carry heavy bags",
                           conclusion: "They are being altruistic and helpful",
                           confidence: 0.93),
            CommonsenseRule(condition: "A person makes a promise",
                           conclusion: "Others expect them to keep the promise",
                           confidence: 0.94),
            CommonsenseRule(condition: "Someone invites others to their home for dinner",
                           conclusion: "They are being hospitable and want to socialize",
                           confidence: 0.93),
            CommonsenseRule(condition: "A person covers their mouth when coughing",
                           conclusion: "They are following hygiene etiquette",
                           confidence: 0.95),
            CommonsenseRule(condition: "Someone speaks softly in a hospital",
                           conclusion: "They are being considerate of patients resting",
                           confidence: 0.92),
            CommonsenseRule(condition: "A person congratulates someone on their promotion",
                           conclusion: "They are acknowledging the achievement and showing goodwill",
                           confidence: 0.95),
            CommonsenseRule(condition: "Someone cuts in line",
                           conclusion: "Others in line will likely be annoyed or upset",
                           confidence: 0.93),
            CommonsenseRule(condition: "A leader delegates tasks to team members",
                           conclusion: "They trust their team and want to distribute workload",
                           confidence: 0.88),
            CommonsenseRule(condition: "Someone refuses to share credit for a group project",
                           conclusion: "Others will view them as selfish or unfair",
                           confidence: 0.90)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Layer 5: Physical Intuition
    // ═══════════════════════════════════════════════════════════════

    func physicalIntuition() -> [CommonsenseRule] {
        lock.lock()
        layerInvocationCounts["physical"] = (layerInvocationCounts["physical"] ?? 0) + 1
        lock.unlock()

        return [
            CommonsenseRule(condition: "An unsupported object is released in mid-air",
                           conclusion: "The object falls due to gravity",
                           confidence: 0.99),
            CommonsenseRule(condition: "Water is poured into a container with a hole at the bottom",
                           conclusion: "Water leaks out through the hole",
                           confidence: 0.98),
            CommonsenseRule(condition: "A heavy object is placed on a thin piece of glass",
                           conclusion: "The glass may crack or shatter under the weight",
                           confidence: 0.90),
            CommonsenseRule(condition: "A metal spoon is placed in hot soup",
                           conclusion: "The spoon handle becomes hot due to thermal conduction",
                           confidence: 0.95),
            CommonsenseRule(condition: "Oil is poured into water",
                           conclusion: "The oil floats on top because it is less dense",
                           confidence: 0.97),
            CommonsenseRule(condition: "A balloon is inflated with helium",
                           conclusion: "The balloon rises because helium is lighter than air",
                           confidence: 0.97),
            CommonsenseRule(condition: "A paper is folded many times",
                           conclusion: "It becomes increasingly difficult to fold further",
                           confidence: 0.96),
            CommonsenseRule(condition: "A cup of hot coffee is left on a table",
                           conclusion: "The coffee gradually cools to room temperature",
                           confidence: 0.97),
            CommonsenseRule(condition: "A nail is hammered into wood",
                           conclusion: "The nail is held in place by friction with the wood fibers",
                           confidence: 0.94),
            CommonsenseRule(condition: "A rubber ball is dropped on a hard surface",
                           conclusion: "The ball bounces back up",
                           confidence: 0.96),
            CommonsenseRule(condition: "A magnet is brought near iron filings",
                           conclusion: "The iron filings are attracted to the magnet",
                           confidence: 0.98),
            CommonsenseRule(condition: "A wet cloth is hung in a warm, windy area",
                           conclusion: "The cloth dries faster due to evaporation",
                           confidence: 0.95),
            CommonsenseRule(condition: "A bicycle tire has low air pressure",
                           conclusion: "Riding is harder and the tire may go flat",
                           confidence: 0.93),
            CommonsenseRule(condition: "A rock is thrown into a pond",
                           conclusion: "The rock sinks and creates ripples on the surface",
                           confidence: 0.98),
            CommonsenseRule(condition: "Sandpaper is rubbed against wood",
                           conclusion: "The wood surface becomes smoother over time",
                           confidence: 0.96),
            CommonsenseRule(condition: "A candle is lit in a closed jar",
                           conclusion: "The flame will extinguish when oxygen is depleted",
                           confidence: 0.97),
            CommonsenseRule(condition: "A spring is compressed and released",
                           conclusion: "The spring pushes back to its original shape",
                           confidence: 0.98),
            CommonsenseRule(condition: "Sugar is stirred into warm water",
                           conclusion: "The sugar dissolves",
                           confidence: 0.97),
            CommonsenseRule(condition: "A feather and a bowling ball are dropped in a vacuum",
                           conclusion: "Both fall at the same rate",
                           confidence: 0.99),
            CommonsenseRule(condition: "An ice cube is placed on a hot pan",
                           conclusion: "The ice melts rapidly",
                           confidence: 0.98),
            CommonsenseRule(condition: "A car tire rolls over a nail",
                           conclusion: "The tire is likely to get punctured and lose air",
                           confidence: 0.89),
            CommonsenseRule(condition: "Sound is produced in a vacuum",
                           conclusion: "No sound is heard because sound needs a medium",
                           confidence: 0.99),
            CommonsenseRule(condition: "A wooden block is placed in water",
                           conclusion: "The block floats because wood is generally less dense than water",
                           confidence: 0.93),
            CommonsenseRule(condition: "A mirror is placed at an angle to a light source",
                           conclusion: "Light reflects off the mirror at the same angle",
                           confidence: 0.97),
            CommonsenseRule(condition: "A pendulum is released from one side",
                           conclusion: "It swings back and forth, gradually losing amplitude",
                           confidence: 0.96),
            CommonsenseRule(condition: "Two magnets with the same pole face each other",
                           conclusion: "They repel each other",
                           confidence: 0.99),
            CommonsenseRule(condition: "A glass of water is placed in a freezer",
                           conclusion: "The water turns to ice",
                           confidence: 0.99),
            CommonsenseRule(condition: "A heavy stone is pushed on a rough surface",
                           conclusion: "Friction makes it difficult to move",
                           confidence: 0.96),
            CommonsenseRule(condition: "A thin wire carries a large electric current",
                           conclusion: "The wire heats up and may melt",
                           confidence: 0.91),
            CommonsenseRule(condition: "A boat has a hole below the waterline",
                           conclusion: "Water enters the boat and it may sink",
                           confidence: 0.95),
            CommonsenseRule(condition: "Air is pumped out of a sealed container",
                           conclusion: "A vacuum is created inside the container",
                           confidence: 0.97),
            CommonsenseRule(condition: "A lever is used with a fulcrum close to the load",
                           conclusion: "Less force is needed to lift the load",
                           confidence: 0.94)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Layer 6: Taxonomic Reasoning
    // ═══════════════════════════════════════════════════════════════

    func taxonomicReasoning() -> [CommonsenseRule] {
        lock.lock()
        layerInvocationCounts["taxonomic"] = (layerInvocationCounts["taxonomic"] ?? 0) + 1
        lock.unlock()

        return [
            CommonsenseRule(condition: "Dogs are mammals and mammals are animals",
                           conclusion: "Dogs are animals",
                           confidence: 0.99),
            CommonsenseRule(condition: "Roses are flowers and flowers are plants",
                           conclusion: "Roses are plants",
                           confidence: 0.99),
            CommonsenseRule(condition: "Sparrows are birds and birds can fly",
                           conclusion: "Sparrows can fly",
                           confidence: 0.95),
            CommonsenseRule(condition: "Cars are vehicles and vehicles have engines",
                           conclusion: "Cars have engines",
                           confidence: 0.96),
            CommonsenseRule(condition: "Apples are fruits and fruits grow on trees or bushes",
                           conclusion: "Apples grow on trees",
                           confidence: 0.97),
            CommonsenseRule(condition: "Whales are mammals and mammals breathe air",
                           conclusion: "Whales breathe air",
                           confidence: 0.99),
            CommonsenseRule(condition: "Penguins are birds",
                           conclusion: "Penguins have feathers like all birds, but cannot fly",
                           confidence: 0.98),
            CommonsenseRule(condition: "Diamonds are minerals and minerals are naturally occurring",
                           conclusion: "Diamonds are naturally occurring",
                           confidence: 0.97),
            CommonsenseRule(condition: "Python is a programming language and programming languages have syntax",
                           conclusion: "Python has syntax",
                           confidence: 0.99),
            CommonsenseRule(condition: "Salmon are fish and fish live in water",
                           conclusion: "Salmon live in water",
                           confidence: 0.98),
            CommonsenseRule(condition: "Oak is a type of tree and trees have roots",
                           conclusion: "Oak trees have roots",
                           confidence: 0.99),
            CommonsenseRule(condition: "Violins are stringed instruments and stringed instruments make sound via vibration",
                           conclusion: "Violins make sound via string vibration",
                           confidence: 0.98),
            CommonsenseRule(condition: "Gold is a metal and metals conduct electricity",
                           conclusion: "Gold conducts electricity",
                           confidence: 0.98),
            CommonsenseRule(condition: "Cats are felines and felines are carnivores",
                           conclusion: "Cats are carnivores",
                           confidence: 0.97),
            CommonsenseRule(condition: "Tomatoes are fruits, even though commonly treated as vegetables",
                           conclusion: "Botanically, tomatoes are fruits because they develop from a flower",
                           confidence: 0.96),
            CommonsenseRule(condition: "Sharks are fish and fish have gills",
                           conclusion: "Sharks have gills",
                           confidence: 0.99),
            CommonsenseRule(condition: "Bats are mammals",
                           conclusion: "Bats give live birth and nurse their young, despite being able to fly",
                           confidence: 0.98),
            CommonsenseRule(condition: "Spiders are arachnids and arachnids have eight legs",
                           conclusion: "Spiders have eight legs",
                           confidence: 0.99),
            CommonsenseRule(condition: "Jupiter is a planet and planets orbit stars",
                           conclusion: "Jupiter orbits a star (the Sun)",
                           confidence: 0.99),
            CommonsenseRule(condition: "Fungi are not plants",
                           conclusion: "Fungi do not photosynthesize",
                           confidence: 0.98),
            CommonsenseRule(condition: "Eagles are raptors and raptors are predatory birds",
                           conclusion: "Eagles are predatory birds",
                           confidence: 0.99),
            CommonsenseRule(condition: "Bacteria are microorganisms and microorganisms are too small to see unaided",
                           conclusion: "Bacteria are too small to see without a microscope",
                           confidence: 0.98),
            CommonsenseRule(condition: "Copper is a metal and metals are malleable",
                           conclusion: "Copper is malleable",
                           confidence: 0.97),
            CommonsenseRule(condition: "Snakes are reptiles and reptiles are cold-blooded",
                           conclusion: "Snakes are cold-blooded",
                           confidence: 0.99),
            CommonsenseRule(condition: "Bananas are berries in botanical classification",
                           conclusion: "Bananas fit the botanical definition of a berry despite common perception",
                           confidence: 0.94),
            CommonsenseRule(condition: "Chess is a board game and board games have rules",
                           conclusion: "Chess has rules",
                           confidence: 0.99),
            CommonsenseRule(condition: "Hydrogen is an element and elements are made of one type of atom",
                           conclusion: "Hydrogen is made of only hydrogen atoms",
                           confidence: 0.99),
            CommonsenseRule(condition: "Elephants are herbivores and herbivores eat plants",
                           conclusion: "Elephants eat plants",
                           confidence: 0.98),
            CommonsenseRule(condition: "Dolphins are mammals, not fish",
                           conclusion: "Dolphins breathe air and nurse their young",
                           confidence: 0.99),
            CommonsenseRule(condition: "Insects have six legs",
                           conclusion: "Any arthropod with six legs is likely an insect",
                           confidence: 0.93),
            CommonsenseRule(condition: "Mars is a terrestrial planet",
                           conclusion: "Mars has a solid rocky surface",
                           confidence: 0.97),
            CommonsenseRule(condition: "Ferns are plants that reproduce via spores",
                           conclusion: "Ferns do not produce flowers or seeds",
                           confidence: 0.97)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Layer 7: Event Script Reasoning
    // ═══════════════════════════════════════════════════════════════

    func eventScriptReasoning() -> [EventScript] {
        lock.lock()
        layerInvocationCounts["eventScript"] = (layerInvocationCounts["eventScript"] ?? 0) + 1
        lock.unlock()

        return [
            EventScript(name: "Restaurant Dining",
                        steps: ["Enter the restaurant",
                                "Wait to be seated or find a table",
                                "Receive menus from the server",
                                "Review the menu and decide what to order",
                                "Place the order with the server",
                                "Wait for the food to be prepared",
                                "Receive and eat the food",
                                "Ask for the check",
                                "Pay the bill and leave a tip",
                                "Leave the restaurant"],
                        confidence: 0.97),
            EventScript(name: "Grocery Shopping",
                        steps: ["Make a shopping list at home",
                                "Drive or walk to the grocery store",
                                "Get a shopping cart or basket",
                                "Walk through aisles selecting items",
                                "Check items off the shopping list",
                                "Proceed to the checkout line",
                                "Place items on the conveyor belt",
                                "Pay for the groceries",
                                "Bag the groceries",
                                "Transport groceries home",
                                "Put groceries away in pantry and refrigerator"],
                        confidence: 0.96),
            EventScript(name: "Morning Routine",
                        steps: ["Wake up from sleep",
                                "Turn off the alarm clock",
                                "Get out of bed",
                                "Use the bathroom",
                                "Brush teeth and wash face",
                                "Take a shower or bath",
                                "Get dressed",
                                "Prepare or eat breakfast",
                                "Gather belongings for the day",
                                "Leave the house for work or school"],
                        confidence: 0.95),
            EventScript(name: "Doctor Visit",
                        steps: ["Schedule an appointment",
                                "Arrive at the doctor's office",
                                "Check in at the reception desk",
                                "Fill out or update paperwork",
                                "Wait in the waiting room",
                                "A nurse calls you back",
                                "Nurse takes vitals: weight, blood pressure, temperature",
                                "Wait for the doctor in the exam room",
                                "Doctor examines and discusses symptoms",
                                "Doctor prescribes treatment or orders tests",
                                "Check out and schedule follow-up if needed",
                                "Pick up any prescribed medication at a pharmacy"],
                        confidence: 0.95),
            EventScript(name: "Air Travel",
                        steps: ["Book a flight",
                                "Pack luggage within weight limits",
                                "Travel to the airport",
                                "Check in and check luggage at the counter or kiosk",
                                "Go through security screening",
                                "Find the departure gate",
                                "Wait for boarding to begin",
                                "Board the plane and find your seat",
                                "Stow carry-on luggage in overhead bin",
                                "Listen to safety instructions",
                                "Remain seated during takeoff",
                                "Fly to the destination",
                                "Land and taxi to the gate",
                                "Deplane and proceed to baggage claim",
                                "Collect luggage and exit the airport"],
                        confidence: 0.96),
            EventScript(name: "Online Shopping",
                        steps: ["Browse an online store or search for a product",
                                "Read product descriptions and reviews",
                                "Select the desired item and quantity",
                                "Add the item to the shopping cart",
                                "Review the shopping cart",
                                "Proceed to checkout",
                                "Enter shipping address",
                                "Select a shipping method",
                                "Enter payment information",
                                "Confirm and place the order",
                                "Receive order confirmation email",
                                "Track the shipment",
                                "Receive and inspect the delivery"],
                        confidence: 0.95),
            EventScript(name: "Cooking a Meal",
                        steps: ["Decide what to cook",
                                "Gather a recipe if needed",
                                "Check that all ingredients are available",
                                "Wash hands before cooking",
                                "Prepare ingredients: wash, peel, chop",
                                "Preheat the oven or stove",
                                "Follow the recipe steps to combine and cook ingredients",
                                "Stir, flip, or check doneness periodically",
                                "Plate the finished meal",
                                "Serve the meal",
                                "Clean up the kitchen and wash dishes"],
                        confidence: 0.96),
            EventScript(name: "Job Interview",
                        steps: ["Apply for a job position",
                                "Receive an interview invitation",
                                "Research the company and role",
                                "Prepare answers to common interview questions",
                                "Choose appropriate professional attire",
                                "Arrive at the interview location on time",
                                "Greet the interviewer with a handshake",
                                "Answer interview questions",
                                "Ask questions about the role and company",
                                "Thank the interviewer",
                                "Send a follow-up thank-you email",
                                "Wait for a response regarding the decision"],
                        confidence: 0.95),
            EventScript(name: "Going to the Movies",
                        steps: ["Choose a movie to see",
                                "Check showtimes",
                                "Buy tickets online or at the box office",
                                "Arrive at the theater",
                                "Present tickets at the entrance",
                                "Buy popcorn and drinks at the concession stand",
                                "Find the correct theater room",
                                "Find your seat",
                                "Watch previews and the movie",
                                "Leave the theater after the movie ends"],
                        confidence: 0.96),
            EventScript(name: "Taking a Bus",
                        steps: ["Check the bus schedule and route",
                                "Walk to the bus stop",
                                "Wait for the bus to arrive",
                                "Board the bus when it arrives",
                                "Pay the fare or tap a transit card",
                                "Find a seat or stand and hold a rail",
                                "Watch for your stop",
                                "Signal to the driver before your stop",
                                "Exit the bus at your stop"],
                        confidence: 0.96),
            EventScript(name: "Doing Laundry",
                        steps: ["Sort clothes by color and fabric type",
                                "Check pockets for items",
                                "Load clothes into the washing machine",
                                "Add detergent and fabric softener",
                                "Select the wash cycle and start the machine",
                                "Wait for the wash cycle to complete",
                                "Transfer clothes to the dryer or hang to dry",
                                "Wait for clothes to dry",
                                "Remove clothes and fold them",
                                "Put clothes away in closet or drawers"],
                        confidence: 0.96),
            EventScript(name: "Library Visit",
                        steps: ["Decide on a topic or book to find",
                                "Travel to the library",
                                "Enter the library and check the catalog",
                                "Locate the book on the shelves",
                                "Browse and select books",
                                "Take the books to the checkout desk",
                                "Present your library card",
                                "Check out the books",
                                "Read the books at home",
                                "Return the books before the due date"],
                        confidence: 0.95),
            EventScript(name: "Birthday Party",
                        steps: ["Choose a date and venue for the party",
                                "Create a guest list",
                                "Send invitations",
                                "Plan the menu and order or bake a cake",
                                "Decorate the venue",
                                "Prepare party games or activities",
                                "Greet guests as they arrive",
                                "Serve food and drinks",
                                "Play games and socialize",
                                "Sing happy birthday and cut the cake",
                                "Open presents",
                                "Thank guests as they leave",
                                "Clean up the venue"],
                        confidence: 0.94),
            EventScript(name: "Moving to a New Home",
                        steps: ["Search for a new home",
                                "Visit and compare properties",
                                "Sign a lease or purchase agreement",
                                "Notify current landlord and set a move-out date",
                                "Pack belongings into boxes",
                                "Hire a moving company or rent a truck",
                                "Load boxes and furniture onto the truck",
                                "Transport belongings to the new home",
                                "Unload and arrange furniture",
                                "Unpack boxes and organize belongings",
                                "Update your address with postal service and accounts",
                                "Set up utilities at the new location"],
                        confidence: 0.94),
            EventScript(name: "Fueling a Car",
                        steps: ["Notice the fuel gauge is low",
                                "Drive to a gas station",
                                "Park at an available fuel pump",
                                "Turn off the engine",
                                "Open the fuel cap",
                                "Select the fuel type",
                                "Insert the nozzle into the fuel tank",
                                "Pump fuel until the tank is full or desired amount",
                                "Remove the nozzle and close the fuel cap",
                                "Pay for the fuel",
                                "Drive away from the station"],
                        confidence: 0.97),
            EventScript(name: "Attending a Wedding",
                        steps: ["Receive a wedding invitation",
                                "RSVP to the invitation",
                                "Buy a gift from the registry",
                                "Choose appropriate formal attire",
                                "Travel to the wedding venue",
                                "Attend the ceremony",
                                "Congratulate the couple",
                                "Attend the reception",
                                "Eat dinner and enjoy music",
                                "Dance and socialize with other guests",
                                "Say goodbye to the couple and leave"],
                        confidence: 0.94),
            EventScript(name: "Washing a Car",
                        steps: ["Gather supplies: soap, sponge, bucket, hose",
                                "Fill the bucket with soapy water",
                                "Rinse the car with the hose to remove loose dirt",
                                "Wash the car from top to bottom with the sponge",
                                "Scrub wheels and tires separately",
                                "Rinse all soap off the car",
                                "Dry the car with a clean towel or chamois",
                                "Clean the windows with glass cleaner",
                                "Optionally apply wax for protection and shine"],
                        confidence: 0.96),
            EventScript(name: "Taking an Exam",
                        steps: ["Study and review course material",
                                "Get a good night's sleep before the exam",
                                "Eat a balanced breakfast",
                                "Arrive at the exam location early",
                                "Bring required materials: pencils, ID, calculator",
                                "Receive the exam paper",
                                "Read all instructions carefully",
                                "Answer questions, starting with easier ones",
                                "Review answers if time permits",
                                "Submit the exam before time runs out"],
                        confidence: 0.96),
            EventScript(name: "Planting a Garden",
                        steps: ["Choose a suitable location with adequate sunlight",
                                "Decide which plants or seeds to grow",
                                "Prepare the soil by tilling and adding compost",
                                "Dig holes or furrows at appropriate spacing",
                                "Place seeds or seedlings in the holes",
                                "Cover with soil and pat down gently",
                                "Water the newly planted garden",
                                "Add mulch to retain moisture",
                                "Water regularly and monitor for pests",
                                "Weed the garden periodically",
                                "Harvest when crops are mature"],
                        confidence: 0.95),
            EventScript(name: "Getting a Haircut",
                        steps: ["Decide on a hairstyle or look",
                                "Schedule an appointment at a salon or barbershop",
                                "Arrive at the appointment time",
                                "Describe the desired haircut to the stylist",
                                "Sit in the chair and wear a cape",
                                "The stylist washes your hair",
                                "The stylist cuts and styles your hair",
                                "Review the result in the mirror",
                                "Pay for the haircut and tip the stylist",
                                "Leave the salon"],
                        confidence: 0.96),
            EventScript(name: "Calling Emergency Services",
                        steps: ["Recognize an emergency situation",
                                "Ensure personal safety first",
                                "Dial the emergency number (e.g., 911)",
                                "State the nature of the emergency clearly",
                                "Provide the exact location",
                                "Answer the dispatcher's questions",
                                "Follow any instructions given by the dispatcher",
                                "Stay on the line until told to hang up",
                                "Administer first aid if trained and able",
                                "Wait for emergency responders to arrive"],
                        confidence: 0.97),
            EventScript(name: "Checking into a Hotel",
                        steps: ["Make a hotel reservation in advance",
                                "Travel to the hotel",
                                "Enter the lobby and approach the front desk",
                                "Provide your name and reservation confirmation",
                                "Present identification and a credit card",
                                "Receive the room key or key card",
                                "Get directions to the room",
                                "Take the elevator or stairs to your floor",
                                "Enter the room and settle in",
                                "Request any additional amenities if needed"],
                        confidence: 0.96)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Layer 8: Analogical Reasoning
    // ═══════════════════════════════════════════════════════════════

    func analogicalReasoning() -> [AnalogicalPattern] {
        lock.lock()
        layerInvocationCounts["analogical"] = (layerInvocationCounts["analogical"] ?? 0) + 1
        lock.unlock()

        return [
            AnalogicalPattern(source: "Atom",
                              target: "Solar System",
                              mapping: "Nucleus is to atom as Sun is to solar system; electrons orbit the nucleus as planets orbit the Sun",
                              confidence: 0.85),
            AnalogicalPattern(source: "Heart",
                              target: "Pump",
                              mapping: "The heart pumps blood through the body as a mechanical pump pushes fluid through pipes",
                              confidence: 0.92),
            AnalogicalPattern(source: "Brain",
                              target: "Computer",
                              mapping: "The brain processes information and stores memories as a computer processes data and stores files",
                              confidence: 0.80),
            AnalogicalPattern(source: "Tree Roots",
                              target: "Foundation of a Building",
                              mapping: "Roots anchor a tree and absorb nutrients as a foundation anchors a building and distributes load",
                              confidence: 0.88),
            AnalogicalPattern(source: "Cell Membrane",
                              target: "Security Checkpoint",
                              mapping: "A cell membrane controls what enters and exits a cell as a security checkpoint controls access to a facility",
                              confidence: 0.87),
            AnalogicalPattern(source: "Electric Circuit",
                              target: "Water Pipe System",
                              mapping: "Voltage is to electric current as water pressure is to water flow; resistance is like pipe narrowing",
                              confidence: 0.91),
            AnalogicalPattern(source: "DNA",
                              target: "Blueprint",
                              mapping: "DNA contains instructions for building an organism as a blueprint contains instructions for constructing a building",
                              confidence: 0.90),
            AnalogicalPattern(source: "Immune System",
                              target: "Army Defense",
                              mapping: "White blood cells fight pathogens as soldiers defend against invaders; antibodies are like specialized weapons",
                              confidence: 0.86),
            AnalogicalPattern(source: "River",
                              target: "Highway",
                              mapping: "Water flows through a river carrying sediment as vehicles travel on a highway carrying goods; tributaries are like on-ramps",
                              confidence: 0.83),
            AnalogicalPattern(source: "Eye",
                              target: "Camera",
                              mapping: "The lens focuses light onto the retina as a camera lens focuses light onto the sensor; the pupil is like the aperture",
                              confidence: 0.93),
            AnalogicalPattern(source: "Ecosystem",
                              target: "Economy",
                              mapping: "Producers, consumers, and decomposers in an ecosystem parallel manufacturers, buyers, and recyclers in an economy",
                              confidence: 0.82),
            AnalogicalPattern(source: "Earth's Crust",
                              target: "Eggshell",
                              mapping: "The crust is a thin, rigid outer layer on a molten interior, just as an eggshell is a thin, rigid layer around a liquid interior",
                              confidence: 0.84),
            AnalogicalPattern(source: "Neuron",
                              target: "Transistor",
                              mapping: "A neuron receives, processes, and transmits electrical signals as a transistor switches and amplifies electronic signals",
                              confidence: 0.88),
            AnalogicalPattern(source: "Book Index",
                              target: "Database Index",
                              mapping: "A book index maps topics to page numbers as a database index maps keys to record locations for fast lookup",
                              confidence: 0.94),
            AnalogicalPattern(source: "Lungs",
                              target: "Bellows",
                              mapping: "Lungs expand to draw in air and contract to expel it as bellows expand and compress to move air for a fire",
                              confidence: 0.89),
            AnalogicalPattern(source: "Teacher",
                              target: "Gardener",
                              mapping: "A teacher nurtures students' growth with knowledge as a gardener nurtures plants' growth with water and care",
                              confidence: 0.81),
            AnalogicalPattern(source: "Lock and Key",
                              target: "Enzyme and Substrate",
                              mapping: "A key fits a specific lock as an enzyme's active site fits a specific substrate molecule",
                              confidence: 0.93),
            AnalogicalPattern(source: "Thermostat",
                              target: "Homeostasis",
                              mapping: "A thermostat maintains room temperature within a range as homeostasis maintains body conditions within a range",
                              confidence: 0.91),
            AnalogicalPattern(source: "Assembly Line",
                              target: "Ribosome",
                              mapping: "An assembly line builds products step by step as a ribosome assembles proteins amino acid by amino acid",
                              confidence: 0.87),
            AnalogicalPattern(source: "Map",
                              target: "Model",
                              mapping: "A map represents geographic territory at a smaller scale as a model represents a system in simplified form",
                              confidence: 0.90),
            AnalogicalPattern(source: "Firewall (computing)",
                              target: "Castle Wall",
                              mapping: "A firewall filters network traffic to protect a system as a castle wall filters who enters to protect inhabitants",
                              confidence: 0.88),
            AnalogicalPattern(source: "Seed",
                              target: "Egg",
                              mapping: "A seed contains the embryo and nutrients for a new plant as an egg contains the embryo and nutrients for a new animal",
                              confidence: 0.90),
            AnalogicalPattern(source: "Orchestra Conductor",
                              target: "Operating System Scheduler",
                              mapping: "A conductor coordinates musicians' timing as an OS scheduler coordinates process execution across CPU cores",
                              confidence: 0.84),
            AnalogicalPattern(source: "Postal System",
                              target: "Internet Packet Routing",
                              mapping: "Letters are sorted by address and routed through post offices as data packets are routed through network nodes based on IP addresses",
                              confidence: 0.89)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Query Tokenizer (shared utility)
    // ═══════════════════════════════════════════════════════════════

    private func tokenize(_ text: String) -> Set<String> {
        let lower = text.lowercased()
        let words = lower.components(separatedBy: CharacterSet.alphanumerics.inverted)
        return Set(words.filter { $0.count > 2 })
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Score a query against a set of rules
    // ═══════════════════════════════════════════════════════════════

    private func scoreRulesAgainstQuery(_ rules: [CommonsenseRule], query: String) -> Double {
        let queryTokens = tokenize(query)
        if queryTokens.isEmpty { return 0.0 }

        var totalScore = 0.0
        var matchCount = 0

        for rule in rules {
            let condTokens = tokenize(rule.condition)
            let concTokens = tokenize(rule.conclusion)
            let ruleTokens = condTokens.union(concTokens)

            let overlap = Double(queryTokens.intersection(ruleTokens).count)
            let denominator = Double(max(queryTokens.count, 1))
            let relevance = overlap / denominator

            if relevance > 0.0 {
                totalScore += relevance * rule.confidence
                matchCount += 1
            }
        }

        if matchCount == 0 { return 0.0 }
        return totalScore / Double(matchCount)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Score event scripts against a query
    // ═══════════════════════════════════════════════════════════════

    private func scoreEventScriptsAgainstQuery(_ scripts: [EventScript], query: String) -> Double {
        let queryTokens = tokenize(query)
        if queryTokens.isEmpty { return 0.0 }

        var totalScore = 0.0
        var matchCount = 0

        for script in scripts {
            let nameTokens = tokenize(script.name)
            let stepTokens = script.steps.flatMap { tokenize($0) }
            let allTokens = nameTokens.union(Set(stepTokens))

            let overlap = Double(queryTokens.intersection(allTokens).count)
            let denominator = Double(max(queryTokens.count, 1))
            let relevance = overlap / denominator

            if relevance > 0.0 {
                totalScore += relevance * script.confidence
                matchCount += 1
            }
        }

        if matchCount == 0 { return 0.0 }
        return totalScore / Double(matchCount)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Score analogical patterns against a query
    // ═══════════════════════════════════════════════════════════════

    private func scoreAnalogicalPatternsAgainstQuery(_ patterns: [AnalogicalPattern], query: String) -> Double {
        let queryTokens = tokenize(query)
        if queryTokens.isEmpty { return 0.0 }

        var totalScore = 0.0
        var matchCount = 0

        for pattern in patterns {
            let sourceTokens = tokenize(pattern.source)
            let targetTokens = tokenize(pattern.target)
            let mappingTokens = tokenize(pattern.mapping)
            let allTokens = sourceTokens.union(targetTokens).union(mappingTokens)

            let overlap = Double(queryTokens.intersection(allTokens).count)
            let denominator = Double(max(queryTokens.count, 1))
            let relevance = overlap / denominator

            if relevance > 0.0 {
                totalScore += relevance * pattern.confidence
                matchCount += 1
            }
        }

        if matchCount == 0 { return 0.0 }
        return totalScore / Double(matchCount)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Collect reasoning traces from matching rules
    // ═══════════════════════════════════════════════════════════════

    private func collectReasoningTraces(_ rules: [CommonsenseRule], query: String) -> [String] {
        let queryTokens = tokenize(query)
        if queryTokens.isEmpty { return [] }

        var traces: [String] = []

        for rule in rules {
            let condTokens = tokenize(rule.condition)
            let concTokens = tokenize(rule.conclusion)
            let ruleTokens = condTokens.union(concTokens)

            let overlap = Double(queryTokens.intersection(ruleTokens).count)
            let denominator = Double(max(queryTokens.count, 1))
            let relevance = overlap / denominator

            if relevance > 0.15 {
                traces.append("[\(String(format: "%.2f", rule.confidence))] IF \(rule.condition) THEN \(rule.conclusion)")
            }
        }

        return traces
    }

    private func collectEventScriptTraces(_ scripts: [EventScript], query: String) -> [String] {
        let queryTokens = tokenize(query)
        if queryTokens.isEmpty { return [] }

        var traces: [String] = []

        for script in scripts {
            let nameTokens = tokenize(script.name)
            let stepTokens = script.steps.flatMap { tokenize($0) }
            let allTokens = nameTokens.union(Set(stepTokens))

            let overlap = Double(queryTokens.intersection(allTokens).count)
            let denominator = Double(max(queryTokens.count, 1))
            let relevance = overlap / denominator

            if relevance > 0.15 {
                let stepsStr = script.steps.joined(separator: " -> ")
                traces.append("[Script:\(script.name)] \(stepsStr)")
            }
        }

        return traces
    }

    private func collectAnalogicalTraces(_ patterns: [AnalogicalPattern], query: String) -> [String] {
        let queryTokens = tokenize(query)
        if queryTokens.isEmpty { return [] }

        var traces: [String] = []

        for pattern in patterns {
            let sourceTokens = tokenize(pattern.source)
            let targetTokens = tokenize(pattern.target)
            let mappingTokens = tokenize(pattern.mapping)
            let allTokens = sourceTokens.union(targetTokens).union(mappingTokens)

            let overlap = Double(queryTokens.intersection(allTokens).count)
            let denominator = Double(max(queryTokens.count, 1))
            let relevance = overlap / denominator

            if relevance > 0.15 {
                traces.append("[Analogy:\(pattern.source)->\(pattern.target)] \(pattern.mapping)")
            }
        }

        return traces
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Full Reasoning Pipeline
    // ═══════════════════════════════════════════════════════════════

    func reason(about query: String) -> CommonsenseResult {
        lock.lock()
        totalQueriesProcessed += 1
        lock.unlock()

        // ─── Run all 8 layers ───
        let spatialRules = spatialReasoning()
        let temporalRules = temporalReasoning()
        let causalRules = causalReasoning()
        let socialRules = socialReasoning()
        let physicalRules = physicalIntuition()
        let taxonomicRules = taxonomicReasoning()
        let eventScripts = eventScriptReasoning()
        let analogicalPatterns = analogicalReasoning()

        // ─── Score each layer ───
        let spatialScore = scoreRulesAgainstQuery(spatialRules, query: query)
        let temporalScore = scoreRulesAgainstQuery(temporalRules, query: query)
        let causalScore = scoreRulesAgainstQuery(causalRules, query: query)
        let socialScore = scoreRulesAgainstQuery(socialRules, query: query)
        let physicalScore = scoreRulesAgainstQuery(physicalRules, query: query)
        let taxonomicScore = scoreRulesAgainstQuery(taxonomicRules, query: query)
        let eventScriptScore = scoreEventScriptsAgainstQuery(eventScripts, query: query)
        let analogicalScore = scoreAnalogicalPatternsAgainstQuery(analogicalPatterns, query: query)

        let layerScores: [String: Double] = [
            "spatial": spatialScore,
            "temporal": temporalScore,
            "causal": causalScore,
            "social": socialScore,
            "physical": physicalScore,
            "taxonomic": taxonomicScore,
            "eventScript": eventScriptScore,
            "analogical": analogicalScore
        ]

        // ─── PHI-weighted aggregation ───
        var weightedSum = 0.0
        var weightTotal = 0.0

        for (layerName, score) in layerScores {
            let weight = layerWeights[layerName] ?? TAU
            weightedSum += score * weight * PHI
            weightTotal += weight
        }

        let overallConfidence: Double
        if weightTotal > 0.0 {
            overallConfidence = min(weightedSum / (weightTotal * PHI), 1.0)
        } else {
            overallConfidence = 0.0
        }

        // ─── Collect reasoning traces ───
        var reasoning: [String] = []
        reasoning.append(contentsOf: collectReasoningTraces(spatialRules, query: query))
        reasoning.append(contentsOf: collectReasoningTraces(temporalRules, query: query))
        reasoning.append(contentsOf: collectReasoningTraces(causalRules, query: query))
        reasoning.append(contentsOf: collectReasoningTraces(socialRules, query: query))
        reasoning.append(contentsOf: collectReasoningTraces(physicalRules, query: query))
        reasoning.append(contentsOf: collectReasoningTraces(taxonomicRules, query: query))
        reasoning.append(contentsOf: collectEventScriptTraces(eventScripts, query: query))
        reasoning.append(contentsOf: collectAnalogicalTraces(analogicalPatterns, query: query))

        // ─── Query ScienceEngineBridge ───
        let scienceFacts = scienceBridge.queryAllDomains(topic: query)

        // ─── Update average confidence metric ───
        lock.lock()
        let n = Double(totalQueriesProcessed)
        averageConfidence = averageConfidence * ((n - 1.0) / n) + overallConfidence / n
        lock.unlock()

        return CommonsenseResult(
            query: query,
            layerScores: layerScores,
            overallConfidence: overallConfidence,
            reasoning: reasoning,
            scienceFacts: scienceFacts
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - MCQ Solver
    // ═══════════════════════════════════════════════════════════════

    func solveMCQ(question: String, choices: [String]) -> (answer: Int, confidence: Double, reasoning: String) {
        lock.lock()
        totalMCQSolved += 1
        lock.unlock()

        guard !choices.isEmpty else {
            return (answer: -1, confidence: 0.0, reasoning: "No choices provided")
        }

        // ─── Tokenize question ───
        let questionTokens = tokenize(question)

        // ─── Load all 8 layers ───
        let spatialRules = spatialReasoning()
        let temporalRules = temporalReasoning()
        let causalRules = causalReasoning()
        let socialRules = socialReasoning()
        let physicalRules = physicalIntuition()
        let taxonomicRules = taxonomicReasoning()
        let eventScripts = eventScriptReasoning()
        let analogicalPatterns = analogicalReasoning()

        // ─── Score each choice ───
        var choiceScores: [(index: Int, score: Double, traces: [String])] = []

        for (index, choice) in choices.enumerated() {
            let combinedQuery = question + " " + choice
            let choiceTokens = tokenize(choice)
            let allTokens = questionTokens.union(choiceTokens)
            _ = allTokens.joined(separator: " ")

            // ─── Score against all 8 layers with PHI weighting ───
            let spatialScore = scoreRulesAgainstQuery(spatialRules, query: combinedQuery)
            let temporalScore = scoreRulesAgainstQuery(temporalRules, query: combinedQuery)
            let causalScore = scoreRulesAgainstQuery(causalRules, query: combinedQuery)
            let socialScore = scoreRulesAgainstQuery(socialRules, query: combinedQuery)
            let physicalScore = scoreRulesAgainstQuery(physicalRules, query: combinedQuery)
            let taxonomicScore = scoreRulesAgainstQuery(taxonomicRules, query: combinedQuery)
            let eventScriptScore = scoreEventScriptsAgainstQuery(eventScripts, query: combinedQuery)
            let analogicalScore = scoreAnalogicalPatternsAgainstQuery(analogicalPatterns, query: combinedQuery)

            // ─── PHI-weighted aggregation ───
            let scores: [(String, Double)] = [
                ("spatial", spatialScore),
                ("temporal", temporalScore),
                ("causal", causalScore),
                ("social", socialScore),
                ("physical", physicalScore),
                ("taxonomic", taxonomicScore),
                ("eventScript", eventScriptScore),
                ("analogical", analogicalScore)
            ]

            var weightedSum = 0.0
            var weightTotal = 0.0

            for (layerName, score) in scores {
                let weight = layerWeights[layerName] ?? TAU
                weightedSum += score * weight * PHI
                weightTotal += weight
            }

            // ─── Apply ScienceEngineBridge boost for science questions ───
            let scienceResults = scienceBridge.queryAllDomains(topic: combinedQuery)
            var scienceBoost = 0.0
            for (_, conf) in scienceResults {
                scienceBoost += conf
            }
            if !scienceResults.isEmpty {
                scienceBoost = (scienceBoost / Double(scienceResults.count)) * TAU
            }

            let baseScore: Double
            if weightTotal > 0.0 {
                baseScore = weightedSum / (weightTotal * PHI)
            } else {
                baseScore = 0.0
            }

            let finalScore = min(baseScore + scienceBoost * 0.3, 1.0)

            // ─── Collect traces ───
            var traces: [String] = []
            traces.append(contentsOf: collectReasoningTraces(spatialRules, query: combinedQuery))
            traces.append(contentsOf: collectReasoningTraces(temporalRules, query: combinedQuery))
            traces.append(contentsOf: collectReasoningTraces(causalRules, query: combinedQuery))
            traces.append(contentsOf: collectReasoningTraces(socialRules, query: combinedQuery))
            traces.append(contentsOf: collectReasoningTraces(physicalRules, query: combinedQuery))
            traces.append(contentsOf: collectReasoningTraces(taxonomicRules, query: combinedQuery))
            traces.append(contentsOf: collectEventScriptTraces(eventScripts, query: combinedQuery))
            traces.append(contentsOf: collectAnalogicalTraces(analogicalPatterns, query: combinedQuery))

            choiceScores.append((index: index, score: finalScore, traces: traces))
        }

        // ─── Select best choice ───
        choiceScores.sort { $0.score > $1.score }

        let bestChoice = choiceScores[0]
        let bestIndex = bestChoice.index
        let bestConfidence = bestChoice.score

        // ─── Build reasoning string ───
        var reasoningParts: [String] = []
        reasoningParts.append("MCQ Analysis for: \(question)")
        reasoningParts.append("Best answer: choice \(bestIndex) — \"\(choices[bestIndex])\"")
        reasoningParts.append("Confidence: \(String(format: "%.4f", bestConfidence))")
        reasoningParts.append("PHI-weighted aggregation across 8 layers + ScienceKB")

        for (idx, choiceData) in choiceScores.enumerated() {
            let choiceText = choices[choiceData.index]
            reasoningParts.append("  [\(idx + 1)] Choice \(choiceData.index) \"\(choiceText)\": score=\(String(format: "%.4f", choiceData.score))")
        }

        if !bestChoice.traces.isEmpty {
            reasoningParts.append("Supporting reasoning:")
            for trace in bestChoice.traces.prefix(10) {
                reasoningParts.append("  - \(trace)")
            }
        }

        let reasoning = reasoningParts.joined(separator: "\n")

        return (answer: bestIndex, confidence: bestConfidence, reasoning: reasoning)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Batch MCQ Solver (for benchmark sets)
    // ═══════════════════════════════════════════════════════════════

    func solveMCQBatch(questions: [(question: String, choices: [String])]) -> [(answer: Int, confidence: Double, reasoning: String)] {
        return questions.map { solveMCQ(question: $0.question, choices: $0.choices) }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Layer Score Breakdown (for diagnostics)
    // ═══════════════════════════════════════════════════════════════

    func layerScoreBreakdown(for query: String) -> [String: Double] {
        let spatialRules = spatialReasoning()
        let temporalRules = temporalReasoning()
        let causalRules = causalReasoning()
        let socialRules = socialReasoning()
        let physicalRules = physicalIntuition()
        let taxonomicRules = taxonomicReasoning()
        let eventScripts = eventScriptReasoning()
        let analogicalPatterns = analogicalReasoning()

        return [
            "spatial": scoreRulesAgainstQuery(spatialRules, query: query),
            "temporal": scoreRulesAgainstQuery(temporalRules, query: query),
            "causal": scoreRulesAgainstQuery(causalRules, query: query),
            "social": scoreRulesAgainstQuery(socialRules, query: query),
            "physical": scoreRulesAgainstQuery(physicalRules, query: query),
            "taxonomic": scoreRulesAgainstQuery(taxonomicRules, query: query),
            "eventScript": scoreEventScriptsAgainstQuery(eventScripts, query: query),
            "analogical": scoreAnalogicalPatternsAgainstQuery(analogicalPatterns, query: query)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Rule Count Per Layer
    // ═══════════════════════════════════════════════════════════════

    var ruleCountPerLayer: [String: Int] {
        return [
            "spatial": spatialReasoning().count,
            "temporal": temporalReasoning().count,
            "causal": causalReasoning().count,
            "social": socialReasoning().count,
            "physical": physicalIntuition().count,
            "taxonomic": taxonomicReasoning().count,
            "eventScript": eventScriptReasoning().count,
            "analogical": analogicalReasoning().count
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Total Rule Count
    // ═══════════════════════════════════════════════════════════════

    var totalRuleCount: Int {
        let counts = ruleCountPerLayer
        return counts.values.reduce(0, +)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - PHI-Weighted Layer Importance
    // ═══════════════════════════════════════════════════════════════

    var phiWeightedLayerImportance: [String: Double] {
        var importance: [String: Double] = [:]
        let totalWeight = layerWeights.values.reduce(0.0, +)

        for (layer, weight) in layerWeights {
            importance[layer] = (weight / totalWeight) * PHI
        }

        return importance
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - GOD_CODE Alignment Score
    // ═══════════════════════════════════════════════════════════════

    func godCodeAlignment(for result: CommonsenseResult) -> Double {
        let layerValues = Array(result.layerScores.values).sorted()
        guard layerValues.count >= 2 else { return 0.0 }

        var ratioSum = 0.0
        var ratioCount = 0

        for i in 1..<layerValues.count {
            if layerValues[i - 1] > 0.001 {
                let ratio = layerValues[i] / layerValues[i - 1]
                ratioSum += ratio
                ratioCount += 1
            }
        }

        if ratioCount == 0 { return 0.0 }

        let averageRatio = ratioSum / Double(ratioCount)
        let phiDeviation = abs(averageRatio - PHI) / PHI
        let alignment = max(0.0, 1.0 - phiDeviation)

        return alignment * (GOD_CODE / (GOD_CODE + OMEGA)) + alignment * TAU
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - OMEGA Resonance Check
    // ═══════════════════════════════════════════════════════════════

    func omegaResonance(for result: CommonsenseResult) -> Double {
        let scoreSum = result.layerScores.values.reduce(0.0, +)
        let normalized = scoreSum / Double(max(result.layerScores.count, 1))
        let resonance = sin(normalized * OMEGA / GOD_CODE * .pi) * TAU + (1.0 - TAU)
        return max(0.0, min(resonance, 1.0))
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Status Report
    // ═══════════════════════════════════════════════════════════════

    var statusReport: [String: Any] {
        lock.lock()
        defer { lock.unlock() }

        let counts = ruleCountPerLayer

        return [
            "engine": "CommonsenseReasoningEngine",
            "version": VERSION,
            "totalRules": totalRuleCount,
            "ruleCountPerLayer": counts,
            "layerWeights": layerWeights,
            "phiWeightedImportance": phiWeightedLayerImportance,
            "totalQueriesProcessed": totalQueriesProcessed,
            "totalMCQSolved": totalMCQSolved,
            "averageConfidence": averageConfidence,
            "layerInvocationCounts": layerInvocationCounts,
            "sacredConstants": [
                "PHI": PHI,
                "GOD_CODE": GOD_CODE,
                "TAU": TAU,
                "OMEGA": OMEGA
            ],
            "scienceBridge": [
                "available": true,
                "type": "ScienceEngineBridge -> ScienceKB.shared"
            ],
            "layers": [
                "Layer1_Spatial": counts["spatial"] ?? 0,
                "Layer2_Temporal": counts["temporal"] ?? 0,
                "Layer3_Causal": counts["causal"] ?? 0,
                "Layer4_Social": counts["social"] ?? 0,
                "Layer5_Physical": counts["physical"] ?? 0,
                "Layer6_Taxonomic": counts["taxonomic"] ?? 0,
                "Layer7_EventScript": counts["eventScript"] ?? 0,
                "Layer8_Analogical": counts["analogical"] ?? 0
            ],
            "mcqSolver": [
                "method": "PHI-weighted 8-layer + ScienceKB aggregation",
                "totalSolved": totalMCQSolved
            ],
            "threadSafety": "NSLock"
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Quick Diagnostic (single-line summary)
    // ═══════════════════════════════════════════════════════════════

    var diagnosticSummary: String {
        lock.lock()
        defer { lock.unlock() }
        return "[CommonsenseReasoningEngine v\(VERSION)] \(totalRuleCount) rules | \(totalQueriesProcessed) queries | \(totalMCQSolved) MCQs | avg_conf=\(String(format: "%.4f", averageConfidence)) | PHI=\(PHI) | GOD_CODE=\(GOD_CODE)"
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Extended Spatial Rules (Layer 1 supplement)
    // ═══════════════════════════════════════════════════════════════

    func extendedSpatialRules() -> [CommonsenseRule] {
        return [
            CommonsenseRule(condition: "X is to the left of Y and Y is to the left of Z",
                           conclusion: "X is to the left of Z",
                           confidence: 0.98),
            CommonsenseRule(condition: "X is directly below Y",
                           conclusion: "Y is directly above X",
                           confidence: 0.99),
            CommonsenseRule(condition: "X is halfway between Y and Z",
                           conclusion: "The distance from X to Y equals the distance from X to Z",
                           confidence: 0.97),
            CommonsenseRule(condition: "X is inside a locked room",
                           conclusion: "X cannot leave the room without a key or unlocking mechanism",
                           confidence: 0.94),
            CommonsenseRule(condition: "X is on the other side of a wall from Y",
                           conclusion: "X and Y cannot see each other through the wall",
                           confidence: 0.96),
            CommonsenseRule(condition: "X is at the top of a staircase",
                           conclusion: "X is at a higher elevation than the bottom of the staircase",
                           confidence: 0.99),
            CommonsenseRule(condition: "X is submerged in water",
                           conclusion: "X is completely surrounded by water",
                           confidence: 0.98),
            CommonsenseRule(condition: "X is in orbit around Y",
                           conclusion: "X is continuously falling toward Y but moving fast enough sideways to miss",
                           confidence: 0.93),
            CommonsenseRule(condition: "X is hanging from a string attached to the ceiling",
                           conclusion: "X is below the ceiling and above the floor",
                           confidence: 0.98),
            CommonsenseRule(condition: "X is at the end of a dead-end hallway",
                           conclusion: "The only way out for X is back the way it came",
                           confidence: 0.97),
            CommonsenseRule(condition: "X is inside a tunnel",
                           conclusion: "X is enclosed on most sides with openings at the tunnel ends",
                           confidence: 0.96),
            CommonsenseRule(condition: "X is on a higher shelf than Y",
                           conclusion: "X requires more reaching effort than Y to access",
                           confidence: 0.91),
            CommonsenseRule(condition: "X is floating in space",
                           conclusion: "X is in a microgravity environment",
                           confidence: 0.97),
            CommonsenseRule(condition: "X is in the same row as Y in a grid",
                           conclusion: "X and Y share the same horizontal position",
                           confidence: 0.99),
            CommonsenseRule(condition: "X is at the center of a maze",
                           conclusion: "X is equidistant from all edges of the maze",
                           confidence: 0.85),
            CommonsenseRule(condition: "X is parked in a driveway",
                           conclusion: "X is on private property near a house",
                           confidence: 0.92),
            CommonsenseRule(condition: "X is at the bottom of the ocean",
                           conclusion: "X is under immense water pressure",
                           confidence: 0.97),
            CommonsenseRule(condition: "X is on the roof of a building",
                           conclusion: "X is at the highest point of the building",
                           confidence: 0.96),
            CommonsenseRule(condition: "X wraps around Y completely",
                           conclusion: "Y is enclosed by X",
                           confidence: 0.97),
            CommonsenseRule(condition: "X is diagonally opposite Y in a rectangle",
                           conclusion: "X and Y are the farthest corners from each other",
                           confidence: 0.98)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Extended Temporal Rules (Layer 2 supplement)
    // ═══════════════════════════════════════════════════════════════

    func extendedTemporalRules() -> [CommonsenseRule] {
        return [
            CommonsenseRule(condition: "X was last updated a year ago",
                           conclusion: "X may be outdated",
                           confidence: 0.88),
            CommonsenseRule(condition: "It is currently winter in the northern hemisphere",
                           conclusion: "It is currently summer in the southern hemisphere",
                           confidence: 0.99),
            CommonsenseRule(condition: "X happens at dawn",
                           conclusion: "X occurs in the early morning when the sun is rising",
                           confidence: 0.98),
            CommonsenseRule(condition: "X was made in the Middle Ages",
                           conclusion: "X is hundreds of years old",
                           confidence: 0.96),
            CommonsenseRule(condition: "X is overdue by three months",
                           conclusion: "X should have been completed three months ago",
                           confidence: 0.98),
            CommonsenseRule(condition: "X takes effect immediately",
                           conclusion: "There is no delay between the action and X",
                           confidence: 0.97),
            CommonsenseRule(condition: "X is celebrated on January 1st every year",
                           conclusion: "X is an annual New Year event",
                           confidence: 0.97),
            CommonsenseRule(condition: "X was invented before electricity was widespread",
                           conclusion: "X does not originally require electricity to function",
                           confidence: 0.90),
            CommonsenseRule(condition: "X will happen in the distant future",
                           conclusion: "X has not happened yet and is far from the present",
                           confidence: 0.95),
            CommonsenseRule(condition: "X occurred during a solar eclipse",
                           conclusion: "X occurred during a brief period when the Moon blocked the Sun",
                           confidence: 0.97),
            CommonsenseRule(condition: "X was the first step in a sequence",
                           conclusion: "X preceded all other steps in that sequence",
                           confidence: 0.99),
            CommonsenseRule(condition: "X is a decade-long project",
                           conclusion: "X spans approximately ten years",
                           confidence: 0.97),
            CommonsenseRule(condition: "X occurs biweekly",
                           conclusion: "X happens once every two weeks",
                           confidence: 0.95),
            CommonsenseRule(condition: "X was the last event of the day",
                           conclusion: "Nothing else occurred after X that day",
                           confidence: 0.97),
            CommonsenseRule(condition: "X precedes Y, and Y is simultaneous with Z",
                           conclusion: "X precedes Z",
                           confidence: 0.99),
            CommonsenseRule(condition: "X is a fossil millions of years old",
                           conclusion: "X existed in a prehistoric era",
                           confidence: 0.98),
            CommonsenseRule(condition: "X was published posthumously",
                           conclusion: "X was published after the author died",
                           confidence: 0.99),
            CommonsenseRule(condition: "X was instantaneous",
                           conclusion: "X took effectively no time at all",
                           confidence: 0.96),
            CommonsenseRule(condition: "X occurs at midnight",
                           conclusion: "X occurs at the boundary between two days",
                           confidence: 0.98),
            CommonsenseRule(condition: "X is a countdown",
                           conclusion: "X decrements toward zero over time",
                           confidence: 0.97)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Extended Causal Rules (Layer 3 supplement)
    // ═══════════════════════════════════════════════════════════════

    func extendedCausalRules() -> [CommonsenseRule] {
        return [
            CommonsenseRule(condition: "A volcano erupts",
                           conclusion: "Ash and lava are expelled, and nearby areas are endangered",
                           confidence: 0.98),
            CommonsenseRule(condition: "Someone forgets to set their alarm",
                           conclusion: "They may oversleep",
                           confidence: 0.89),
            CommonsenseRule(condition: "A tree's roots are severed",
                           conclusion: "The tree will likely die without nutrients and water",
                           confidence: 0.94),
            CommonsenseRule(condition: "A person eats a balanced diet and exercises",
                           conclusion: "Their overall health tends to improve",
                           confidence: 0.91),
            CommonsenseRule(condition: "A company provides excellent customer service",
                           conclusion: "Customer loyalty and satisfaction increase",
                           confidence: 0.88),
            CommonsenseRule(condition: "Strong winds blow during a thunderstorm",
                           conclusion: "Tree branches may break and power lines may fall",
                           confidence: 0.91),
            CommonsenseRule(condition: "A student skips many classes",
                           conclusion: "Their grades are likely to suffer",
                           confidence: 0.90),
            CommonsenseRule(condition: "An earthquake occurs under the ocean",
                           conclusion: "A tsunami may be generated",
                           confidence: 0.85),
            CommonsenseRule(condition: "A person inhales smoke",
                           conclusion: "They may cough or have difficulty breathing",
                           confidence: 0.94),
            CommonsenseRule(condition: "Sunlight is blocked by dense clouds",
                           conclusion: "The area below becomes darker and cooler",
                           confidence: 0.95),
            CommonsenseRule(condition: "A child falls off a bicycle",
                           conclusion: "The child may get bruised or scraped",
                           confidence: 0.92),
            CommonsenseRule(condition: "A market is flooded with supply of a product",
                           conclusion: "The price of that product tends to drop",
                           confidence: 0.87),
            CommonsenseRule(condition: "Someone lies repeatedly",
                           conclusion: "Others will eventually lose trust in them",
                           confidence: 0.93),
            CommonsenseRule(condition: "A phone is dropped in water",
                           conclusion: "The phone may short-circuit and stop working",
                           confidence: 0.91),
            CommonsenseRule(condition: "Pollution increases in a river",
                           conclusion: "Fish and wildlife in the river may die",
                           confidence: 0.93),
            CommonsenseRule(condition: "Someone practices gratitude regularly",
                           conclusion: "Their mental wellbeing tends to improve",
                           confidence: 0.84),
            CommonsenseRule(condition: "A road is not maintained for years",
                           conclusion: "Potholes and cracks develop on the road surface",
                           confidence: 0.93),
            CommonsenseRule(condition: "A person is exposed to cold weather without warm clothing",
                           conclusion: "They risk hypothermia",
                           confidence: 0.92),
            CommonsenseRule(condition: "Bees disappear from an ecosystem",
                           conclusion: "Pollination decreases and crop yields drop",
                           confidence: 0.94),
            CommonsenseRule(condition: "A new highway is built near a town",
                           conclusion: "Traffic flow to and from the town improves",
                           confidence: 0.87)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Extended Social Rules (Layer 4 supplement)
    // ═══════════════════════════════════════════════════════════════

    func extendedSocialRules() -> [CommonsenseRule] {
        return [
            CommonsenseRule(condition: "Someone consistently arrives late to meetings",
                           conclusion: "Others perceive them as unreliable or disrespectful of time",
                           confidence: 0.90),
            CommonsenseRule(condition: "A person volunteers regularly at a shelter",
                           conclusion: "They are compassionate and community-minded",
                           confidence: 0.91),
            CommonsenseRule(condition: "Someone takes credit for another's work",
                           conclusion: "The victim feels resentful and trust is damaged",
                           confidence: 0.93),
            CommonsenseRule(condition: "A person remembers someone's birthday",
                           conclusion: "The other person feels valued and appreciated",
                           confidence: 0.89),
            CommonsenseRule(condition: "Someone talks on the phone loudly in a quiet space",
                           conclusion: "Others around them feel annoyed",
                           confidence: 0.91),
            CommonsenseRule(condition: "A manager gives constructive feedback privately",
                           conclusion: "The employee feels respected and motivated to improve",
                           confidence: 0.88),
            CommonsenseRule(condition: "A person wears formal attire to a casual event",
                           conclusion: "They may feel out of place or overdressed",
                           confidence: 0.82),
            CommonsenseRule(condition: "Someone shares a meal with a colleague",
                           conclusion: "It strengthens their working relationship",
                           confidence: 0.84),
            CommonsenseRule(condition: "A person breaks a promise to a friend",
                           conclusion: "The friend feels betrayed and trust is diminished",
                           confidence: 0.92),
            CommonsenseRule(condition: "Someone listens attentively without interrupting",
                           conclusion: "The speaker feels heard and respected",
                           confidence: 0.93),
            CommonsenseRule(condition: "A new employee introduces themselves to the team",
                           conclusion: "It helps them integrate and be recognized",
                           confidence: 0.90),
            CommonsenseRule(condition: "Someone gossips about others frequently",
                           conclusion: "People become wary of confiding in them",
                           confidence: 0.91),
            CommonsenseRule(condition: "A person sends a thank-you note after receiving help",
                           conclusion: "The helper feels appreciated and is more inclined to help again",
                           confidence: 0.90),
            CommonsenseRule(condition: "Someone interrupts others during conversations",
                           conclusion: "Others view them as rude or domineering",
                           confidence: 0.89),
            CommonsenseRule(condition: "A parent reads bedtime stories to their child",
                           conclusion: "It strengthens the parent-child bond",
                           confidence: 0.93),
            CommonsenseRule(condition: "Someone openly admits a mistake at work",
                           conclusion: "Colleagues respect their honesty and accountability",
                           confidence: 0.87),
            CommonsenseRule(condition: "A guest brings wine to a dinner party",
                           conclusion: "The host appreciates the gesture of goodwill",
                           confidence: 0.90),
            CommonsenseRule(condition: "Someone ignores a person's greeting",
                           conclusion: "The greeted person feels snubbed or hurt",
                           confidence: 0.88),
            CommonsenseRule(condition: "A leader celebrates team achievements publicly",
                           conclusion: "Team morale and cohesion improve",
                           confidence: 0.91),
            CommonsenseRule(condition: "Someone returns a lost wallet with all contents",
                           conclusion: "The owner is grateful and trusts in human goodness",
                           confidence: 0.93)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Extended Physical Intuition (Layer 5 supplement)
    // ═══════════════════════════════════════════════════════════════

    func extendedPhysicalIntuition() -> [CommonsenseRule] {
        return [
            CommonsenseRule(condition: "A sealed bottle of carbonated drink is shaken vigorously",
                           conclusion: "Opening it will cause the liquid to spray out due to released carbon dioxide",
                           confidence: 0.96),
            CommonsenseRule(condition: "Two objects of different masses are dropped from the same height in air",
                           conclusion: "The heavier object hits the ground first due to air resistance differences",
                           confidence: 0.80),
            CommonsenseRule(condition: "A snowball is placed in direct sunlight",
                           conclusion: "The snowball melts",
                           confidence: 0.98),
            CommonsenseRule(condition: "A sponge is squeezed while submerged in water",
                           conclusion: "Water is expelled from the sponge; it absorbs water again when released",
                           confidence: 0.97),
            CommonsenseRule(condition: "A ceramic plate is heated rapidly then cooled quickly",
                           conclusion: "The plate may crack due to thermal shock",
                           confidence: 0.89),
            CommonsenseRule(condition: "A spinning top is left on a flat surface",
                           conclusion: "It gradually wobbles more and eventually falls over due to friction",
                           confidence: 0.95),
            CommonsenseRule(condition: "Iron is left outdoors in humid conditions",
                           conclusion: "The iron corrodes and develops rust",
                           confidence: 0.94),
            CommonsenseRule(condition: "A flashlight is turned on in a dark room",
                           conclusion: "The beam illuminates the area in its path",
                           confidence: 0.99),
            CommonsenseRule(condition: "A block of dry ice is placed on a table at room temperature",
                           conclusion: "The dry ice sublimates, producing visible fog as CO2 gas cools surrounding moisture",
                           confidence: 0.95),
            CommonsenseRule(condition: "A prism is placed in a beam of white light",
                           conclusion: "The white light separates into a spectrum of colors",
                           confidence: 0.97),
            CommonsenseRule(condition: "A vacuum flask contains hot coffee",
                           conclusion: "The coffee stays hot much longer than in a regular cup",
                           confidence: 0.96),
            CommonsenseRule(condition: "A soap bubble floats in the air",
                           conclusion: "It eventually pops due to thinning of its liquid film",
                           confidence: 0.94),
            CommonsenseRule(condition: "A tuning fork is struck",
                           conclusion: "It vibrates at a specific frequency producing a pure tone",
                           confidence: 0.97),
            CommonsenseRule(condition: "Oil and vinegar are mixed in a bottle",
                           conclusion: "They separate into layers because they are immiscible",
                           confidence: 0.96),
            CommonsenseRule(condition: "A parachute is deployed during a fall",
                           conclusion: "Air resistance increases dramatically slowing the descent",
                           confidence: 0.98),
            CommonsenseRule(condition: "An egg is placed in salt water",
                           conclusion: "The egg floats because salt water is denser",
                           confidence: 0.93),
            CommonsenseRule(condition: "A guitar string is plucked",
                           conclusion: "The string vibrates and produces a musical note",
                           confidence: 0.98),
            CommonsenseRule(condition: "Concrete is poured and left to set",
                           conclusion: "It hardens over time through a chemical curing process",
                           confidence: 0.97),
            CommonsenseRule(condition: "A plastic comb is rubbed against wool",
                           conclusion: "The comb becomes electrostatically charged and can attract small bits of paper",
                           confidence: 0.94),
            CommonsenseRule(condition: "A heavy ball rolls down a hill",
                           conclusion: "It accelerates due to gravity until friction or an obstacle slows it",
                           confidence: 0.97)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Extended Taxonomic Rules (Layer 6 supplement)
    // ═══════════════════════════════════════════════════════════════

    func extendedTaxonomicRules() -> [CommonsenseRule] {
        return [
            CommonsenseRule(condition: "A tomato is a fruit",
                           conclusion: "A tomato develops from the ovary of a flowering plant",
                           confidence: 0.96),
            CommonsenseRule(condition: "Crocodiles are reptiles and reptiles have scales",
                           conclusion: "Crocodiles have scales",
                           confidence: 0.99),
            CommonsenseRule(condition: "Pine trees are conifers and conifers produce cones",
                           conclusion: "Pine trees produce cones",
                           confidence: 0.99),
            CommonsenseRule(condition: "Mushrooms are fungi and fungi decompose organic matter",
                           conclusion: "Mushrooms play a role in decomposing organic matter",
                           confidence: 0.94),
            CommonsenseRule(condition: "Parrots are birds and birds lay eggs",
                           conclusion: "Parrots lay eggs",
                           confidence: 0.99),
            CommonsenseRule(condition: "Iron is a metal and metals are generally solid at room temperature",
                           conclusion: "Iron is solid at room temperature",
                           confidence: 0.99),
            CommonsenseRule(condition: "Frogs are amphibians and amphibians undergo metamorphosis",
                           conclusion: "Frogs undergo metamorphosis from tadpole to adult",
                           confidence: 0.98),
            CommonsenseRule(condition: "Banjos are stringed instruments and stringed instruments produce sound through vibrating strings",
                           conclusion: "Banjos produce sound through vibrating strings",
                           confidence: 0.99),
            CommonsenseRule(condition: "Helium is a noble gas and noble gases are chemically inert",
                           conclusion: "Helium is chemically inert",
                           confidence: 0.99),
            CommonsenseRule(condition: "Wolves are canids and canids are social pack animals",
                           conclusion: "Wolves live in social packs",
                           confidence: 0.96),
            CommonsenseRule(condition: "Mercury is a metal but liquid at room temperature",
                           conclusion: "Mercury is an exception to the general rule that metals are solid at room temperature",
                           confidence: 0.98),
            CommonsenseRule(condition: "Turtles are reptiles with shells",
                           conclusion: "Turtles can retract into their shells for protection",
                           confidence: 0.92),
            CommonsenseRule(condition: "Mosses are non-vascular plants",
                           conclusion: "Mosses lack a true vascular system for transporting water",
                           confidence: 0.96),
            CommonsenseRule(condition: "Corals are animals, not plants",
                           conclusion: "Corals are colonial marine invertebrates",
                           confidence: 0.95),
            CommonsenseRule(condition: "Viruses are not classified as living organisms by many biologists",
                           conclusion: "Viruses cannot reproduce independently; they need a host cell",
                           confidence: 0.95),
            CommonsenseRule(condition: "Platypuses are mammals that lay eggs",
                           conclusion: "Platypuses are monotremes, an exception to the rule that mammals give live birth",
                           confidence: 0.97),
            CommonsenseRule(condition: "Bamboo is a grass, not a tree",
                           conclusion: "Despite its woody appearance, bamboo belongs to the grass family",
                           confidence: 0.94),
            CommonsenseRule(condition: "Starfish are echinoderms, not fish",
                           conclusion: "Starfish are marine invertebrates related to sea urchins",
                           confidence: 0.97),
            CommonsenseRule(condition: "Rubies and sapphires are both forms of corundum",
                           conclusion: "Rubies and sapphires are the same mineral with different trace elements",
                           confidence: 0.95),
            CommonsenseRule(condition: "Seahorses are fish",
                           conclusion: "Seahorses breathe through gills and live in water, despite their unusual shape",
                           confidence: 0.96)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Extended Event Scripts (Layer 7 supplement)
    // ═══════════════════════════════════════════════════════════════

    func extendedEventScripts() -> [EventScript] {
        return [
            EventScript(name: "Making a Phone Call",
                        steps: ["Decide who to call and why",
                                "Pick up the phone",
                                "Dial the phone number",
                                "Wait for the call to connect",
                                "Greet the person who answers",
                                "State the purpose of the call",
                                "Discuss the topic",
                                "Conclude the conversation",
                                "Say goodbye",
                                "Hang up the phone"],
                        confidence: 0.96),
            EventScript(name: "Renting an Apartment",
                        steps: ["Determine budget and desired location",
                                "Search online listings for available apartments",
                                "Schedule viewings for shortlisted apartments",
                                "Visit apartments and evaluate condition and amenities",
                                "Choose a preferred apartment",
                                "Submit a rental application with references",
                                "Undergo a credit and background check",
                                "Receive approval from the landlord",
                                "Review and sign the lease agreement",
                                "Pay the security deposit and first month's rent",
                                "Receive the keys and move in"],
                        confidence: 0.94),
            EventScript(name: "Going on a Road Trip",
                        steps: ["Choose a destination",
                                "Plan the route and stops along the way",
                                "Pack clothes, snacks, and entertainment",
                                "Check the car: oil, tires, coolant, fuel",
                                "Load the car with luggage",
                                "Set the GPS or navigation system",
                                "Begin driving",
                                "Stop for fuel, food, and restroom breaks",
                                "Enjoy scenic stops and attractions",
                                "Arrive at the destination",
                                "Check into accommodation"],
                        confidence: 0.95),
            EventScript(name: "Sending a Package",
                        steps: ["Determine the item to send",
                                "Find an appropriately sized box or envelope",
                                "Wrap or cushion the item with protective material",
                                "Place the item inside the box and seal it",
                                "Write the recipient's address clearly on the package",
                                "Write the return address",
                                "Take the package to a post office or shipping center",
                                "Choose a shipping method and speed",
                                "Pay for shipping",
                                "Receive a tracking number",
                                "Track the shipment until delivery"],
                        confidence: 0.96),
            EventScript(name: "Filing Taxes",
                        steps: ["Gather all income documents: W-2s, 1099s, etc.",
                                "Collect records of deductions and credits",
                                "Choose to file independently or hire a tax preparer",
                                "Obtain the correct tax forms or use tax software",
                                "Enter income information",
                                "Enter deductions and credits",
                                "Calculate tax owed or refund due",
                                "Review the return for accuracy",
                                "Sign and submit the tax return",
                                "Pay any taxes owed or wait for refund"],
                        confidence: 0.93),
            EventScript(name: "Adopting a Pet",
                        steps: ["Research which type of pet suits your lifestyle",
                                "Visit local shelters or rescue organizations",
                                "Meet available animals and spend time with them",
                                "Choose a pet that bonds well with you",
                                "Fill out an adoption application",
                                "Pass a home check if required",
                                "Pay the adoption fee",
                                "Receive vaccination and medical records",
                                "Purchase food, bowls, bedding, and toys",
                                "Bring the pet home",
                                "Schedule a veterinary check-up"],
                        confidence: 0.94),
            EventScript(name: "Hosting a Dinner Party",
                        steps: ["Decide on the date and guest list",
                                "Send invitations",
                                "Plan the menu considering dietary restrictions",
                                "Shop for ingredients",
                                "Clean and prepare the dining area",
                                "Set the table with plates, silverware, and glasses",
                                "Cook the meal",
                                "Greet guests as they arrive",
                                "Serve appetizers and drinks",
                                "Serve the main course",
                                "Serve dessert",
                                "Enjoy conversation and socializing",
                                "See guests off and clean up"],
                        confidence: 0.95),
            EventScript(name: "Visiting a Museum",
                        steps: ["Choose a museum to visit",
                                "Check museum hours and admission prices",
                                "Travel to the museum",
                                "Purchase an entry ticket or show a membership card",
                                "Obtain a map or guide at the entrance",
                                "Walk through exhibits and galleries",
                                "Read informational plaques and descriptions",
                                "Take photos where permitted",
                                "Visit the gift shop",
                                "Leave the museum"],
                        confidence: 0.95)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Extended Analogical Patterns (Layer 8 supplement)
    // ═══════════════════════════════════════════════════════════════

    func extendedAnalogicalPatterns() -> [AnalogicalPattern] {
        return [
            AnalogicalPattern(source: "Library Catalog",
                              target: "Search Engine",
                              mapping: "A library catalog indexes books for lookup as a search engine indexes web pages for retrieval",
                              confidence: 0.91),
            AnalogicalPattern(source: "Skeleton",
                              target: "Building Frame",
                              mapping: "A skeleton provides structural support and shape to a body as a steel frame provides structure to a building",
                              confidence: 0.90),
            AnalogicalPattern(source: "Nerve Signals",
                              target: "Electrical Wiring",
                              mapping: "Nerves transmit signals from the brain to the body as electrical wires carry current from a panel to devices",
                              confidence: 0.88),
            AnalogicalPattern(source: "Stomach",
                              target: "Blender",
                              mapping: "The stomach mechanically and chemically breaks down food as a blender mechanically breaks down ingredients",
                              confidence: 0.84),
            AnalogicalPattern(source: "Democracy",
                              target: "Distributed Computing",
                              mapping: "In democracy, decisions emerge from many votes as in distributed computing results emerge from many nodes processing in parallel",
                              confidence: 0.79),
            AnalogicalPattern(source: "Blood Vessels",
                              target: "Pipeline Network",
                              mapping: "Blood vessels carry blood to all parts of the body as pipelines carry fluid to all parts of a distribution system",
                              confidence: 0.90),
            AnalogicalPattern(source: "Migration of Birds",
                              target: "Load Balancing",
                              mapping: "Birds migrate to regions with better resources as load balancers route traffic to servers with more capacity",
                              confidence: 0.78),
            AnalogicalPattern(source: "Beehive",
                              target: "Factory",
                              mapping: "Workers in a beehive have specialized roles producing honey as workers in a factory have specialized roles producing goods",
                              confidence: 0.87),
            AnalogicalPattern(source: "Skin",
                              target: "Protective Coating",
                              mapping: "Skin protects internal organs from the environment as a protective coating shields a surface from damage",
                              confidence: 0.91),
            AnalogicalPattern(source: "Antibodies",
                              target: "Antivirus Software",
                              mapping: "Antibodies detect and neutralize specific pathogens as antivirus software detects and removes specific malware",
                              confidence: 0.89),
            AnalogicalPattern(source: "Coral Reef",
                              target: "City",
                              mapping: "A coral reef supports diverse marine species in a dense structure as a city supports diverse human activities in dense infrastructure",
                              confidence: 0.82),
            AnalogicalPattern(source: "Photosynthesis",
                              target: "Solar Panel",
                              mapping: "Plants convert sunlight into chemical energy via photosynthesis as solar panels convert sunlight into electrical energy",
                              confidence: 0.93),
            AnalogicalPattern(source: "River Delta",
                              target: "Network Router",
                              mapping: "A river delta splits water flow into multiple channels as a router splits data flow into multiple paths",
                              confidence: 0.83),
            AnalogicalPattern(source: "Vaccination",
                              target: "Fire Drill",
                              mapping: "Vaccination trains the immune system to respond to a real pathogen as a fire drill trains people to respond to a real fire",
                              confidence: 0.88),
            AnalogicalPattern(source: "Fossil Record",
                              target: "Version Control History",
                              mapping: "The fossil record preserves evidence of past life forms as version control history preserves evidence of past code states",
                              confidence: 0.86),
            AnalogicalPattern(source: "Mitochondria",
                              target: "Power Plant",
                              mapping: "Mitochondria generate ATP energy for the cell as a power plant generates electricity for a city",
                              confidence: 0.92)
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Complete Rule Set Aggregation
    // ═══════════════════════════════════════════════════════════════

    func allSpatialRules() -> [CommonsenseRule] {
        return spatialReasoning() + extendedSpatialRules()
    }

    func allTemporalRules() -> [CommonsenseRule] {
        return temporalReasoning() + extendedTemporalRules()
    }

    func allCausalRules() -> [CommonsenseRule] {
        return causalReasoning() + extendedCausalRules()
    }

    func allSocialRules() -> [CommonsenseRule] {
        return socialReasoning() + extendedSocialRules()
    }

    func allPhysicalRules() -> [CommonsenseRule] {
        return physicalIntuition() + extendedPhysicalIntuition()
    }

    func allTaxonomicRules() -> [CommonsenseRule] {
        return taxonomicReasoning() + extendedTaxonomicRules()
    }

    func allEventScripts() -> [EventScript] {
        return eventScriptReasoning() + extendedEventScripts()
    }

    func allAnalogicalPatterns() -> [AnalogicalPattern] {
        return analogicalReasoning() + extendedAnalogicalPatterns()
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Extended Reasoning Pipeline (uses all rules)
    // ═══════════════════════════════════════════════════════════════

    func reasonExtended(about query: String) -> CommonsenseResult {
        lock.lock()
        totalQueriesProcessed += 1
        lock.unlock()

        // ─── Run all 8 layers with extended rules ───
        let allSpatial = allSpatialRules()
        let allTemporal = allTemporalRules()
        let allCausal = allCausalRules()
        let allSocial = allSocialRules()
        let allPhysical = allPhysicalRules()
        let allTaxonomic = allTaxonomicRules()
        let allScripts = allEventScripts()
        let allAnalogical = allAnalogicalPatterns()

        // ─── Score each layer ───
        let spatialScore = scoreRulesAgainstQuery(allSpatial, query: query)
        let temporalScore = scoreRulesAgainstQuery(allTemporal, query: query)
        let causalScore = scoreRulesAgainstQuery(allCausal, query: query)
        let socialScore = scoreRulesAgainstQuery(allSocial, query: query)
        let physicalScore = scoreRulesAgainstQuery(allPhysical, query: query)
        let taxonomicScore = scoreRulesAgainstQuery(allTaxonomic, query: query)
        let eventScriptScore = scoreEventScriptsAgainstQuery(allScripts, query: query)
        let analogicalScore = scoreAnalogicalPatternsAgainstQuery(allAnalogical, query: query)

        let layerScores: [String: Double] = [
            "spatial": spatialScore,
            "temporal": temporalScore,
            "causal": causalScore,
            "social": socialScore,
            "physical": physicalScore,
            "taxonomic": taxonomicScore,
            "eventScript": eventScriptScore,
            "analogical": analogicalScore
        ]

        // ─── PHI-weighted aggregation with OMEGA modulation ───
        var weightedSum = 0.0
        var weightTotal = 0.0

        for (layerName, score) in layerScores {
            let weight = layerWeights[layerName] ?? TAU
            let omegaModulatedWeight = weight * (1.0 + sin(OMEGA / GOD_CODE) * 0.01)
            weightedSum += score * omegaModulatedWeight * PHI
            weightTotal += omegaModulatedWeight
        }

        let overallConfidence: Double
        if weightTotal > 0.0 {
            overallConfidence = min(weightedSum / (weightTotal * PHI), 1.0)
        } else {
            overallConfidence = 0.0
        }

        // ─── Collect reasoning traces ───
        var reasoning: [String] = []
        reasoning.append(contentsOf: collectReasoningTraces(allSpatial, query: query))
        reasoning.append(contentsOf: collectReasoningTraces(allTemporal, query: query))
        reasoning.append(contentsOf: collectReasoningTraces(allCausal, query: query))
        reasoning.append(contentsOf: collectReasoningTraces(allSocial, query: query))
        reasoning.append(contentsOf: collectReasoningTraces(allPhysical, query: query))
        reasoning.append(contentsOf: collectReasoningTraces(allTaxonomic, query: query))
        reasoning.append(contentsOf: collectEventScriptTraces(allScripts, query: query))
        reasoning.append(contentsOf: collectAnalogicalTraces(allAnalogical, query: query))

        // ─── Query ScienceEngineBridge ───
        let scienceFacts = scienceBridge.queryAllDomains(topic: query)

        // ─── Update average confidence metric ───
        lock.lock()
        let n = Double(totalQueriesProcessed)
        averageConfidence = averageConfidence * ((n - 1.0) / n) + overallConfidence / n
        lock.unlock()

        return CommonsenseResult(
            query: query,
            layerScores: layerScores,
            overallConfidence: overallConfidence,
            reasoning: reasoning,
            scienceFacts: scienceFacts
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Extended MCQ Solver (uses all rules)
    // ═══════════════════════════════════════════════════════════════

    func solveMCQExtended(question: String, choices: [String]) -> (answer: Int, confidence: Double, reasoning: String) {
        lock.lock()
        totalMCQSolved += 1
        lock.unlock()

        guard !choices.isEmpty else {
            return (answer: -1, confidence: 0.0, reasoning: "No choices provided")
        }

        // ─── Tokenize question ───
        let questionTokens = tokenize(question)

        // ─── Load all 8 layers with extended rules ───
        let allSpatial = allSpatialRules()
        let allTemporal = allTemporalRules()
        let allCausal = allCausalRules()
        let allSocial = allSocialRules()
        let allPhysical = allPhysicalRules()
        let allTaxonomic = allTaxonomicRules()
        let allScripts = allEventScripts()
        let allAnalogical = allAnalogicalPatterns()

        // ─── Score each choice ───
        var choiceScores: [(index: Int, score: Double, traces: [String])] = []

        for (index, choice) in choices.enumerated() {
            let combinedQuery = question + " " + choice
            let choiceTokens = tokenize(choice)
            _ = questionTokens.union(choiceTokens)

            // ─── Score against all 8 layers with PHI weighting ───
            let spatialScore = scoreRulesAgainstQuery(allSpatial, query: combinedQuery)
            let temporalScore = scoreRulesAgainstQuery(allTemporal, query: combinedQuery)
            let causalScore = scoreRulesAgainstQuery(allCausal, query: combinedQuery)
            let socialScore = scoreRulesAgainstQuery(allSocial, query: combinedQuery)
            let physicalScore = scoreRulesAgainstQuery(allPhysical, query: combinedQuery)
            let taxonomicScore = scoreRulesAgainstQuery(allTaxonomic, query: combinedQuery)
            let eventScriptScore = scoreEventScriptsAgainstQuery(allScripts, query: combinedQuery)
            let analogicalScore = scoreAnalogicalPatternsAgainstQuery(allAnalogical, query: combinedQuery)

            // ─── PHI-weighted aggregation with OMEGA modulation ───
            let scores: [(String, Double)] = [
                ("spatial", spatialScore),
                ("temporal", temporalScore),
                ("causal", causalScore),
                ("social", socialScore),
                ("physical", physicalScore),
                ("taxonomic", taxonomicScore),
                ("eventScript", eventScriptScore),
                ("analogical", analogicalScore)
            ]

            var weightedSum = 0.0
            var weightTotal = 0.0

            for (layerName, score) in scores {
                let weight = layerWeights[layerName] ?? TAU
                let omegaModulated = weight * (1.0 + sin(OMEGA / GOD_CODE) * 0.01)
                weightedSum += score * omegaModulated * PHI
                weightTotal += omegaModulated
            }

            // ─── Apply ScienceEngineBridge boost ───
            let scienceResults = scienceBridge.queryAllDomains(topic: combinedQuery)
            var scienceBoost = 0.0
            for (_, conf) in scienceResults {
                scienceBoost += conf
            }
            if !scienceResults.isEmpty {
                scienceBoost = (scienceBoost / Double(scienceResults.count)) * TAU
            }

            let baseScore: Double
            if weightTotal > 0.0 {
                baseScore = weightedSum / (weightTotal * PHI)
            } else {
                baseScore = 0.0
            }

            let finalScore = min(baseScore + scienceBoost * 0.3, 1.0)

            // ─── Collect traces ───
            var traces: [String] = []
            traces.append(contentsOf: collectReasoningTraces(allSpatial, query: combinedQuery))
            traces.append(contentsOf: collectReasoningTraces(allTemporal, query: combinedQuery))
            traces.append(contentsOf: collectReasoningTraces(allCausal, query: combinedQuery))
            traces.append(contentsOf: collectReasoningTraces(allSocial, query: combinedQuery))
            traces.append(contentsOf: collectReasoningTraces(allPhysical, query: combinedQuery))
            traces.append(contentsOf: collectReasoningTraces(allTaxonomic, query: combinedQuery))
            traces.append(contentsOf: collectEventScriptTraces(allScripts, query: combinedQuery))
            traces.append(contentsOf: collectAnalogicalTraces(allAnalogical, query: combinedQuery))

            choiceScores.append((index: index, score: finalScore, traces: traces))
        }

        // ─── Select best choice ───
        choiceScores.sort { $0.score > $1.score }

        let bestChoice = choiceScores[0]
        let bestIndex = bestChoice.index
        let bestConfidence = bestChoice.score

        // ─── Build reasoning string ───
        var reasoningParts: [String] = []
        reasoningParts.append("Extended MCQ Analysis for: \(question)")
        reasoningParts.append("Best answer: choice \(bestIndex) — \"\(choices[bestIndex])\"")
        reasoningParts.append("Confidence: \(String(format: "%.4f", bestConfidence))")
        reasoningParts.append("PHI-weighted + OMEGA-modulated aggregation across 8 extended layers + ScienceKB")

        for (idx, choiceData) in choiceScores.enumerated() {
            let choiceText = choices[choiceData.index]
            reasoningParts.append("  [\(idx + 1)] Choice \(choiceData.index) \"\(choiceText)\": score=\(String(format: "%.4f", choiceData.score))")
        }

        if !bestChoice.traces.isEmpty {
            reasoningParts.append("Supporting reasoning:")
            for trace in bestChoice.traces.prefix(15) {
                reasoningParts.append("  - \(trace)")
            }
        }

        let reasoning = reasoningParts.joined(separator: "\n")

        return (answer: bestIndex, confidence: bestConfidence, reasoning: reasoning)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Counterfactual Reasoning
    // ═══════════════════════════════════════════════════════════════

    func counterfactualReason(premise: String, counterfactual: String) -> CommonsenseResult {
        let combinedQuery = "If not '\(premise)' but instead '\(counterfactual)'"

        // Score original premise and counterfactual against all layers
        let originalResult = reasonExtended(about: premise)
        let counterfactualResult = reasonExtended(about: counterfactual)

        // Combine results: show how changing the premise changes conclusions
        var combinedLayerScores: [String: Double] = [:]
        for (layer, originalScore) in originalResult.layerScores {
            let cfScore = counterfactualResult.layerScores[layer] ?? 0.0
            let delta = cfScore - originalScore
            combinedLayerScores[layer] = delta
        }

        var reasoning: [String] = []
        reasoning.append("Counterfactual Analysis:")
        reasoning.append("  Original premise: \(premise)")
        reasoning.append("  Counterfactual: \(counterfactual)")
        reasoning.append("  Layer deltas (counterfactual - original):")

        for (layer, delta) in combinedLayerScores.sorted(by: { abs($0.value) > abs($1.value) }) {
            let direction = delta > 0 ? "+" : ""
            reasoning.append("    \(layer): \(direction)\(String(format: "%.4f", delta))")
        }

        reasoning.append(contentsOf: counterfactualResult.reasoning)

        let overallDelta = combinedLayerScores.values.reduce(0.0, +) / Double(max(combinedLayerScores.count, 1))
        let confidence = min(abs(overallDelta) * PHI + 0.5, 1.0)

        return CommonsenseResult(
            query: combinedQuery,
            layerScores: combinedLayerScores,
            overallConfidence: confidence,
            reasoning: reasoning,
            scienceFacts: counterfactualResult.scienceFacts
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Multi-Hop Reasoning
    // ═══════════════════════════════════════════════════════════════

    func multiHopReason(hops: [String]) -> CommonsenseResult {
        guard !hops.isEmpty else {
            return CommonsenseResult(
                query: "",
                layerScores: [:],
                overallConfidence: 0.0,
                reasoning: ["No hops provided for multi-hop reasoning."],
                scienceFacts: []
            )
        }

        var accumulatedScores: [String: Double] = [:]
        var accumulatedReasoning: [String] = []
        var accumulatedFacts: [(String, Double)] = []

        for (hopIndex, hop) in hops.enumerated() {
            let hopResult = reasonExtended(about: hop)

            accumulatedReasoning.append("--- Hop \(hopIndex + 1): \(hop) ---")
            accumulatedReasoning.append(contentsOf: hopResult.reasoning)

            for (layer, score) in hopResult.layerScores {
                let decay = pow(TAU, Double(hopIndex))
                accumulatedScores[layer] = (accumulatedScores[layer] ?? 0.0) + score * decay
            }

            accumulatedFacts.append(contentsOf: hopResult.scienceFacts)
        }

        // Normalize accumulated scores
        let hopCount = Double(hops.count)
        for key in accumulatedScores.keys {
            accumulatedScores[key] = (accumulatedScores[key] ?? 0.0) / hopCount
        }

        let overallConfidence = accumulatedScores.values.reduce(0.0, +) / Double(max(accumulatedScores.count, 1))

        // Deduplicate facts
        var seenFacts = Set<String>()
        var uniqueFacts: [(String, Double)] = []
        for (fact, conf) in accumulatedFacts {
            if !seenFacts.contains(fact) {
                seenFacts.insert(fact)
                uniqueFacts.append((fact, conf))
            }
        }

        let combinedQuery = hops.joined(separator: " -> ")

        return CommonsenseResult(
            query: combinedQuery,
            layerScores: accumulatedScores,
            overallConfidence: min(overallConfidence, 1.0),
            reasoning: accumulatedReasoning,
            scienceFacts: uniqueFacts
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Confidence Calibration via Sacred Constants
    // ═══════════════════════════════════════════════════════════════

    func calibrateConfidence(_ rawConfidence: Double) -> Double {
        // Apply PHI-based sigmoid calibration centered at TAU
        let x = (rawConfidence - TAU) * PHI
        let sigmoid = 1.0 / (1.0 + exp(-x * GOD_CODE / OMEGA))
        return sigmoid
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Layer Dominance Analysis
    // ═══════════════════════════════════════════════════════════════

    func dominantLayer(for result: CommonsenseResult) -> (layer: String, score: Double) {
        var bestLayer = "none"
        var bestScore = 0.0

        for (layer, score) in result.layerScores {
            if score > bestScore {
                bestScore = score
                bestLayer = layer
            }
        }

        return (layer: bestLayer, score: bestScore)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Cross-Layer Correlation Matrix
    // ═══════════════════════════════════════════════════════════════

    func crossLayerCorrelation(for query: String) -> [String: [String: Double]] {
        let result = reasonExtended(about: query)
        let layers = Array(result.layerScores.keys).sorted()
        var correlation: [String: [String: Double]] = [:]

        for layerA in layers {
            correlation[layerA] = [:]
            let scoreA = result.layerScores[layerA] ?? 0.0
            for layerB in layers {
                let scoreB = result.layerScores[layerB] ?? 0.0
                if scoreA > 0.001 && scoreB > 0.001 {
                    let ratio = min(scoreA, scoreB) / max(scoreA, scoreB)
                    correlation[layerA]?[layerB] = ratio
                } else {
                    correlation[layerA]?[layerB] = 0.0
                }
            }
        }

        return correlation
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Sacred Resonance Scoring
    // ═══════════════════════════════════════════════════════════════

    func sacredResonanceScore(for result: CommonsenseResult) -> Double {
        let godAlignment = godCodeAlignment(for: result)
        let omegaRes = omegaResonance(for: result)

        // Combine with PHI weighting
        let resonance = godAlignment * PHI / (PHI + 1.0) + omegaRes * 1.0 / (PHI + 1.0)
        return min(resonance, 1.0)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Total Extended Rule Count
    // ═══════════════════════════════════════════════════════════════

    var totalExtendedRuleCount: Int {
        return allSpatialRules().count +
               allTemporalRules().count +
               allCausalRules().count +
               allSocialRules().count +
               allPhysicalRules().count +
               allTaxonomicRules().count +
               allEventScripts().count +
               allAnalogicalPatterns().count
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Extended Rule Count Per Layer
    // ═══════════════════════════════════════════════════════════════

    var extendedRuleCountPerLayer: [String: Int] {
        return [
            "spatial": allSpatialRules().count,
            "temporal": allTemporalRules().count,
            "causal": allCausalRules().count,
            "social": allSocialRules().count,
            "physical": allPhysicalRules().count,
            "taxonomic": allTaxonomicRules().count,
            "eventScript": allEventScripts().count,
            "analogical": allAnalogicalPatterns().count
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Extended Status Report
    // ═══════════════════════════════════════════════════════════════

    var extendedStatusReport: [String: Any] {
        lock.lock()
        defer { lock.unlock() }

        let baseCounts = ruleCountPerLayer
        let extCounts = extendedRuleCountPerLayer

        return [
            "engine": "CommonsenseReasoningEngine",
            "version": VERSION,
            "mode": "EXTENDED",
            "baseRules": totalRuleCount,
            "extendedRules": totalExtendedRuleCount,
            "baseRuleCountPerLayer": baseCounts,
            "extendedRuleCountPerLayer": extCounts,
            "layerWeights": layerWeights,
            "phiWeightedImportance": phiWeightedLayerImportance,
            "totalQueriesProcessed": totalQueriesProcessed,
            "totalMCQSolved": totalMCQSolved,
            "averageConfidence": averageConfidence,
            "layerInvocationCounts": layerInvocationCounts,
            "sacredConstants": [
                "PHI": PHI,
                "GOD_CODE": GOD_CODE,
                "TAU": TAU,
                "OMEGA": OMEGA
            ],
            "scienceBridge": [
                "available": true,
                "type": "ScienceEngineBridge -> ScienceKB.shared"
            ],
            "extendedLayers": [
                "Layer1_Spatial": extCounts["spatial"] ?? 0,
                "Layer2_Temporal": extCounts["temporal"] ?? 0,
                "Layer3_Causal": extCounts["causal"] ?? 0,
                "Layer4_Social": extCounts["social"] ?? 0,
                "Layer5_Physical": extCounts["physical"] ?? 0,
                "Layer6_Taxonomic": extCounts["taxonomic"] ?? 0,
                "Layer7_EventScript": extCounts["eventScript"] ?? 0,
                "Layer8_Analogical": extCounts["analogical"] ?? 0
            ],
            "capabilities": [
                "basicReasoning": true,
                "extendedReasoning": true,
                "mcqSolver": true,
                "extendedMCQSolver": true,
                "counterfactualReasoning": true,
                "multiHopReasoning": true,
                "confidenceCalibration": true,
                "crossLayerCorrelation": true,
                "sacredResonance": true,
                "godCodeAlignment": true,
                "omegaResonance": true
            ],
            "threadSafety": "NSLock"
        ]
    }
}

// ═══════════════════════════════════════════════════════════════
// MARK: - END B33_CommonsenseReasoning.swift
// ═══════════════════════════════════════════════════════════════
