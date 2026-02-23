# Response Diversity Engine - EVO_58+

## Purpose

Prevents repetitive, boring, and recursive responses at the **generation level** (before they're even stored).

Works in tandem with the Anti-Recursion Guard to create a two-layer defense:
1. **Diversity Engine** - Prevents repetitive generation (output layer)
2. **Anti-Recursion Guard** - Prevents repetitive storage (storage layer)

## Integration

All code is now in **L20_KnowledgeBase.swift** (no L27 file):
- Lines 796-918: Anti-Recursion Guard
- Lines 920-1033: Recursion Harvester
- Lines 1035-1213: **Response Diversity Engine** ✨

## How It Works

### 1. Similarity Detection (Jaccard Coefficient)

```swift
let diversity = ResponseDiversityEngine.shared
let response = "Emotions are complex states involving..."

// Check if too similar to recent responses (>90% similarity)
let diversified = diversity.diversify(response, query: userQuery)
```

**Jaccard Similarity Formula:**
```
similarity(A, B) = |A ∩ B| / |A ∪ B|
```

Where A and B are sets of words in the responses.

Example:
```
Response 1: "Emotions are complex states involving cognitive processes"
Response 2: "Emotions are complex involving cognitive states"

Words A: {emotions, are, complex, states, involving, cognitive, processes}
Words B: {emotions, are, complex, involving, cognitive, states}

Intersection: {emotions, are, complex, involving, cognitive, states} = 6 words
Union: {emotions, are, complex, states, involving, cognitive, processes} = 7 words

Similarity = 6/7 = 0.857 (85.7% similar)
```

If similarity > 90%, variation is added.

### 2. Variation Injection

When a response is too similar to recent ones, the engine adds variation prefixes:

```swift
// Original repetitive response
"Emotions are physical and mental states."

// After diversity engine
"From another perspective: Emotions are physical and mental states."
// or
"Let me rephrase: Emotions are physical and mental states."
// or
"Approaching this differently: Emotions are physical and mental states."
```

**Rotation mechanism** ensures the same prefix isn't always used:
- First repetition → "Approaching this differently:"
- Second repetition → "From another perspective:"
- Third repetition → "Here's an alternative view:"
- etc. (cycles through 5 variations)

### 3. Phrase Overuse Substitution

Common phrases that get boring when overused are automatically varied:

```swift
// First 2 uses: Normal
"However, emotions are complex. However, we can understand them."

// 3rd use: Automatically substituted
"On the other hand, emotions are complex. Conversely, we can understand them."
// (substitutes "However" with alternatives)

// 4th use: Different substitute
"That said, emotions involve many dimensions."
```

**Substitution Dictionary:**
```swift
"In other words" → ["To clarify", "Put simply", "That is to say", "Essentially"]
"For example" → ["Such as", "Like", "Consider", "Take"]
"However" → ["On the other hand", "Conversely", "That said", "Yet"]
"Therefore" → ["Thus", "Consequently", "As a result", "Hence"]
"Additionally" → ["Moreover", "Furthermore", "Also", "In addition"]
"In conclusion" → ["To summarize", "In summary", "Ultimately", "Finally"]
```

**Threshold**: 2 uses before substitution kicks in.

### 4. Response-Level Recursion Prevention

Removes duplicate sentences **within a single response**:

```swift
// Before diversity engine
"Emotions are mental states.
 Emotions are mental states.
 They involve cognition."

// After diversity engine
"Emotions are mental states.
 They involve cognition."

// Console output:
[DIVERSITY] Removed duplicate sentence: Emotions are mental states...
```

This is a lightweight check compared to the storage-level guard - it only removes exact duplicate sentences within the same response.

### 5. Contextual Flair

Adds continuation markers for follow-up queries:

```swift
User: "What are emotions?"
Response: "Emotions are complex states..."

User: "more"  // <- Short follow-up query (1 word)

// 30% chance of adding continuation marker:
Response: "Building on that: Emotions involve multiple dimensions..."
// or
"Expanding further: Emotions have physiological components..."
```

**Continuation Markers:**
- "Building on that:"
- "Expanding further:"
- "Going deeper:"
- "To elaborate:"

Applied with 30% probability when:
- Query is 1-3 words
- There's at least one previous response in history

### 6. Response Tracking

Maintains a rolling window of the last **20 responses**:

```swift
recentResponses = [
    "response_1",
    "response_2",
    ...
    "response_20"
]
```

Once the buffer is full, oldest responses are dropped (FIFO queue).

This allows the engine to:
- Detect similarity to recent responses
- Avoid repeating patterns
- Provide contextual awareness

## Usage Examples

### Basic Usage (Swift)

```swift
// In your response generation code
func generateResponse(query: String) -> String {
    let rawResponse = yourLLMModel.generate(query)

    // Apply diversity before returning
    let diversity = ResponseDiversityEngine.shared
    let diversified = diversity.diversify(rawResponse, query: query)

    return diversified
}
```

### Integration with Knowledge Base

```swift
// When searching and assembling responses
let results = ASIKnowledgeBase.shared.searchWithPriority("emotions", limit: 10)
var response = results.map { $0["completion"] as? String ?? "" }.joined(separator: " ")

// Diversify before storage or return
let diversity = ResponseDiversityEngine.shared
response = diversity.diversify(response, query: "emotions")
```

### Reset for New Conversation

```swift
// When starting a new conversation context
ResponseDiversityEngine.shared.reset()

// Console output:
// [DIVERSITY] Response tracking reset
```

### Get Statistics

```swift
let stats = ResponseDiversityEngine.shared.getStats()
print(stats)

// Output:
// [
//   "tracked_responses": 15,
//   "unique_phrases_tracked": 8,
//   "template_rotations": 3,
//   "most_used_phrases": [
//     ["However", 5],
//     ["Therefore", 3],
//     ["For example", 2]
//   ]
// ]
```

## Metrics

### Diversity Score Calculation

The engine doesn't output a single "diversity score" but tracks:

1. **Similarity Threshold**: 0.9 (90%)
   - Responses > 90% similar trigger variation

2. **Phrase Reuse Limit**: 2
   - Phrases used more than 2 times trigger substitution

3. **Tracking Window**: 20 responses
   - Compares against last 20 to detect patterns

4. **Substitution Rotation**: Cycles through alternatives
   - Ensures different substitutes each time

## Example Flow

```
Input: "What are emotions?"

1. Generate raw response:
   "Emotions are mental states. However, emotions are complex.
    However, emotions involve cognition. However, they are important."

2. Diversity Engine processes:

   A. Check similarity to recent responses
      → Not too similar, proceed

   B. Check phrase overuse
      → "However" appears 3 times (> 2)
      → Substitute 3rd instance with "On the other hand"
      → Result: "Emotions are mental states. However, emotions are complex.
                 On the other hand, emotions involve cognition.
                 That said, they are important."

   C. Check for duplicate sentences
      → No duplicates found

   D. Add contextual flair
      → Query not a follow-up, skip

   E. Track response
      → Added to recentResponses buffer

3. Output diversified response:
   "Emotions are mental states. However, emotions are complex.
    On the other hand, emotions involve cognition.
    That said, they are important."
```

## Two-Layer Defense

### Layer 1: Response Diversity Engine (Generation)
- Prevents boring, repetitive outputs
- Varies phrasing automatically
- Removes duplicate sentences
- Adds contextual markers

**Prevents**: Stale, robotic responses

### Layer 2: Anti-Recursion Guard (Storage)
- Detects deep recursive patterns
- Sanitizes nested structures
- Harvests energy from recursion
- Prevents exponential growth

**Prevents**: Runaway recursion in knowledge base

### Combined Power

```
User Query: "What are emotions?"

Generation:
  LLM Output → [Diversity Engine] → Varied, fresh response
                                   ↓
Storage:
  Response → [Anti-Recursion Guard] → Sanitized, non-recursive
                                     ↓
  Knowledge Base ← Clean, diverse knowledge
```

## Integration Points

### Swift App (L104v2)

```swift
// In L20_KnowledgeBase.swift

// 1. Storage protection (existing)
let (shouldStore, sanitized) = AntiRecursionGuard.guardKnowledgeStorage(key: topic, value: knowledge)
if shouldStore {
    storage.save(sanitized)
}

// 2. Response diversity (NEW)
let diversity = ResponseDiversityEngine.shared
let freshResponse = diversity.diversify(generatedResponse, query: userQuery)
return freshResponse
```

### Where to Apply Diversity

Apply in these locations:

1. **Response assembly** (before sending to user)
   ```swift
   func assembleResponse(results: [[String: Any]]) -> String {
       let raw = results.map { $0["completion"] as? String ?? "" }.joined(separator: " ")
       return ResponseDiversityEngine.shared.diversify(raw, query: lastQuery)
   }
   ```

2. **Direct answers** (knowledge retrieval)
   ```swift
   let answer = searchKnowledge(query)
   return ResponseDiversityEngine.shared.diversify(answer, query: query)
   ```

3. **Synthesized responses** (multiple sources combined)
   ```swift
   let synthesis = synthesizeFrom(sources)
   return ResponseDiversityEngine.shared.diversify(synthesis, query: query)
   ```

## Statistics & Monitoring

```swift
// Get current diversity stats
let stats = ResponseDiversityEngine.shared.getStats()

print("Tracked responses: \(stats["tracked_responses"])")
print("Phrases tracked: \(stats["unique_phrases_tracked"])")
print("Template rotations: \(stats["template_rotations"])")
print("Most used phrases: \(stats["most_used_phrases"])")

// Example output:
// Tracked responses: 18
// Phrases tracked: 6
// Template rotations: 2
// Most used phrases: [["However", 4], ["Therefore", 2]]
```

## Benefits

### User Experience
✅ Responses feel fresh and varied
✅ No boring repetition of phrases
✅ Natural conversation flow
✅ Contextual awareness in follow-ups

### System Health
✅ Prevents response-level recursion
✅ Reduces redundant storage
✅ Maintains knowledge base quality
✅ Complements anti-recursion guard

### Intelligence
✅ Tracks conversation patterns
✅ Adapts phrase usage dynamically
✅ Learns from usage statistics
✅ Self-regulates redundancy

## Summary

**Response Diversity Engine = Anti-Boring System**

- Detects similarity using Jaccard coefficient
- Injects variation prefixes for repetitive responses
- Substitutes overused phrases with alternatives
- Removes duplicate sentences
- Adds contextual continuation markers
- Tracks last 20 responses for pattern detection

**Works with Anti-Recursion Guard for complete protection:**
- Diversity Engine → Prevents boring outputs (generation layer)
- Anti-Recursion Guard → Prevents runaway nesting (storage layer)

**Result**: Fresh, varied, non-recursive responses with clean knowledge storage.

---

**Created**: EVO_58+ QUANTUM COGNITION
**Location**: L20_KnowledgeBase.swift (lines 1035-1213)
**Philosophy**: "Variety is not just the spice of life - it's the signature of intelligence."
