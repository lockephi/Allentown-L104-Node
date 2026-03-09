// ═══════════════════════════════════════════════════════════════════
// L31_ScienceKB.swift
// [EVO_68_PIPELINE] SOVEREIGN_NODE_UPGRADE :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 Sovereign Intelligence — Science Knowledge Base
// 500+ RDF-style fact triples across 9 domains for commonsense reasoning
// ARC benchmark MCQ scoring, fact retrieval, and knowledge graph queries
// ═══════════════════════════════════════════════════════════════════

import Foundation
import Accelerate

// ═══════════════════════════════════════════════════════════════════
// MARK: - ScienceFact Triple
// ═══════════════════════════════════════════════════════════════════

struct ScienceFact: Hashable {
    let subject: String
    let relation: String
    let obj: String
    let confidence: Double   // 0.0–1.0
    let domain: String       // biology, body_systems, earth_science, physics, chemistry, astronomy, ecology, measurement, technology

    func hash(into hasher: inout Hasher) {
        hasher.combine(subject)
        hasher.combine(relation)
        hasher.combine(obj)
        hasher.combine(domain)
    }

    static func == (lhs: ScienceFact, rhs: ScienceFact) -> Bool {
        lhs.subject == rhs.subject &&
        lhs.relation == rhs.relation &&
        lhs.obj == rhs.obj &&
        lhs.domain == rhs.domain
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - ScienceKB — Singleton Knowledge Base
// ═══════════════════════════════════════════════════════════════════

final class ScienceKB {
    static let shared = ScienceKB()

    // ─── Triple Indices for O(1) Lookup ───
    private var subjectIndex: [String: [ScienceFact]] = [:]
    private var relationIndex: [String: [ScienceFact]] = [:]
    private var objectIndex: [String: [ScienceFact]] = [:]
    private var allFacts: [ScienceFact] = []
    private let lock = NSLock()

    // ─── Domain Statistics ───
    private var domainCounts: [String: Int] = [:]

    // ─── Sacred Constants (from L01_Constants.swift) ───
    private let phi: Double = PHI           // 1.618033988749895
    private let godCode: Double = GOD_CODE  // 527.5184818492612
    private let tau: Double = TAU           // 0.618033988749895

    // ─── Tokenization ───
    private static let nonAlphanumeric = CharacterSet.alphanumerics.inverted
    private static let stopWords: Set<String> = [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must",
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more", "most",
        "other", "some", "such", "no", "only", "own", "same", "than",
        "too", "very", "just", "because", "if", "when", "where", "how",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "it", "its", "he", "she", "they", "them", "we", "you", "i", "me",
        "my", "your", "his", "her", "our", "their"
    ]

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Initialization
    // ═══════════════════════════════════════════════════════════════════

    private init() {
        loadFacts()
        buildIndices()
        l104Log("ScienceKB initialized: \(allFacts.count) facts across \(domainCounts.count) domains")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Fact Loading (500+ RDF Triples)
    // ═══════════════════════════════════════════════════════════════════

    private func loadFacts() {
        // ─── Biology (~60 facts) ───
        let biologyFacts: [(String, String, String, Double)] = [
            ("cell", "is_basic_unit_of", "life", 1.0),
            ("cells", "contain", "DNA", 1.0),
            ("mitosis", "produces", "two identical cells", 0.98),
            ("meiosis", "produces", "four gametes", 0.98),
            ("photosynthesis", "converts", "sunlight to chemical energy", 1.0),
            ("photosynthesis", "requires", "water and carbon dioxide", 1.0),
            ("photosynthesis", "produces", "oxygen and glucose", 1.0),
            ("chloroplast", "is_site_of", "photosynthesis", 1.0),
            ("mitochondria", "is_site_of", "cellular respiration", 1.0),
            ("DNA", "contains", "genetic information", 1.0),
            ("genes", "determine", "traits", 0.98),
            ("chromosomes", "are_made_of", "DNA", 1.0),
            ("natural_selection", "drives", "evolution", 0.99),
            ("mutations", "cause", "genetic variation", 0.97),
            ("adaptation", "helps_organism", "survive", 0.98),
            ("organisms", "compete_for", "resources", 0.97),
            ("predators", "control", "prey_populations", 0.95),
            ("decomposers", "break_down", "dead_organisms", 0.98),
            ("bacteria", "can_be", "helpful_or_harmful", 0.96),
            ("virus", "requires", "host_cell_to_reproduce", 0.99),
            ("fungi", "decompose", "organic_matter", 0.97),
            ("plants", "produce", "oxygen", 1.0),
            ("animals", "consume", "oxygen", 0.99),
            ("roots", "absorb", "water_and_nutrients", 0.99),
            ("leaves", "are_site_of", "photosynthesis", 0.99),
            ("stems", "transport", "water_and_nutrients", 0.98),
            ("seeds", "develop_from", "fertilized_ovules", 0.97),
            ("pollination", "transfers", "pollen", 0.98),
            ("frogs", "eat", "insects", 0.95),
            ("dark_moths", "increase_in", "polluted_areas", 0.94),
            ("light_moths", "increase_when", "air_becomes_cleaner", 0.94),
            ("peppered_moth", "example_of", "natural_selection", 0.96),
            ("mass_extinction", "followed_by", "speciation", 0.90),
            ("fossils", "provide_evidence_of", "evolution", 0.98),
            ("inherited_traits", "passed_from", "parent_to_offspring", 0.99),
            ("acquired_traits", "not_passed_to", "offspring", 0.98),
            ("dominant_allele", "masks", "recessive_allele", 0.99),
            ("homozygous", "has", "two_identical_alleles", 0.99),
            ("heterozygous", "has", "two_different_alleles", 0.99),
            ("phenotype", "is", "observable_trait", 0.99),
            ("genotype", "is", "genetic_makeup", 0.99),
            ("ribosomes", "synthesize", "proteins", 0.99),
            ("cell_membrane", "controls", "what_enters_and_leaves_cell", 0.98),
            ("nucleus", "contains", "DNA", 1.0),
            ("vacuole", "stores", "water_and_nutrients", 0.96),
            ("cell_wall", "provides", "structure_in_plant_cells", 0.98),
            ("osmosis", "is_movement_of", "water_across_membrane", 0.98),
            ("diffusion", "moves_molecules_from", "high_to_low_concentration", 0.98),
            ("ATP", "is", "energy_currency_of_cells", 0.99),
            ("enzymes", "speed_up", "chemical_reactions_in_cells", 0.98),
            ("fermentation", "produces_energy_without", "oxygen", 0.97),
            ("aerobic_respiration", "requires", "oxygen", 0.99),
            ("anaerobic_respiration", "does_not_require", "oxygen", 0.99),
            ("sexual_reproduction", "requires", "two_parents", 0.98),
            ("asexual_reproduction", "requires", "one_parent", 0.98),
            ("taxonomy", "classifies", "organisms", 0.97),
            ("species", "is_most_specific", "classification_level", 0.97),
            ("kingdom", "is_broadest", "classification_level", 0.96),
            ("vertebrates", "have", "backbone", 0.99),
            ("invertebrates", "lack", "backbone", 0.99),
            ("homeostasis", "maintains", "stable_internal_conditions", 0.98),
            ("xylem", "transports", "water_upward_in_plants", 0.98),
            ("phloem", "transports", "sugars_in_plants", 0.98),
            ("tropism", "is_plant_growth_toward", "stimulus", 0.96),
            ("phototropism", "is_growth_toward", "light", 0.97),
            ("germination", "is", "seed_sprouting", 0.98),
            ("chlorophyll", "absorbs", "light_for_photosynthesis", 0.99),
            ("carbon_fixation", "converts", "CO2_to_organic_molecules", 0.97),
            ("food_vacuole", "digests", "food_in_cells", 0.95),
            ("cilia", "help_cells", "move_or_sweep_particles", 0.96),
            ("flagella", "propel", "cells_through_liquid", 0.96),
            ("endoplasmic_reticulum", "transports", "materials_within_cell", 0.97),
            ("golgi_apparatus", "packages", "proteins_for_export", 0.97),
            ("lysosome", "digests", "waste_in_cells", 0.97),
            ("natural_selection", "requires", "variation_in_population", 0.98),
            ("speciation", "occurs_through", "reproductive_isolation", 0.95),
            ("convergent_evolution", "produces", "similar_traits_in_unrelated_species", 0.94),
            ("divergent_evolution", "produces", "different_traits_from_common_ancestor", 0.94),
            ("coevolution", "is_mutual_evolution_of", "interacting_species", 0.94),
        ]

        // ─── Body Systems (~30 facts) ───
        let bodySystemsFacts: [(String, String, String, Double)] = [
            ("skeletal_system", "provides", "support_and_protection", 0.99),
            ("muscular_system", "enables", "movement", 0.99),
            ("nervous_system", "controls", "body_functions", 0.99),
            ("brain", "is_part_of", "nervous_system", 1.0),
            ("heart", "pumps", "blood", 1.0),
            ("circulatory_system", "transports", "oxygen_and_nutrients", 0.99),
            ("respiratory_system", "exchanges", "gases", 0.99),
            ("lungs", "are_site_of", "gas_exchange", 0.99),
            ("digestive_system", "breaks_down", "food", 0.99),
            ("stomach", "produces", "acid", 0.98),
            ("small_intestine", "absorbs", "nutrients", 0.99),
            ("large_intestine", "absorbs", "water", 0.98),
            ("kidneys", "filter", "blood", 0.99),
            ("liver", "detoxifies", "blood", 0.98),
            ("skin", "is_largest", "organ", 0.99),
            ("immune_system", "fights", "infection", 0.99),
            ("white_blood_cells", "destroy", "pathogens", 0.98),
            ("red_blood_cells", "carry", "oxygen", 0.99),
            ("platelets", "help", "blood_clotting", 0.98),
            ("endocrine_system", "produces", "hormones", 0.99),
            ("insulin", "regulates", "blood_sugar", 0.99),
            ("thyroid", "controls", "metabolism", 0.98),
            ("bones", "store", "calcium", 0.97),
            ("joints", "allow", "movement", 0.98),
            ("tendons", "connect", "muscle_to_bone", 0.99),
            ("ligaments", "connect", "bone_to_bone", 0.99),
            ("neurons", "transmit", "electrical_signals", 0.99),
            ("spinal_cord", "relays", "signals_to_brain", 0.99),
            ("arteries", "carry", "blood_away_from_heart", 0.99),
            ("veins", "carry", "blood_toward_heart", 0.99),
            ("capillaries", "connect", "arteries_and_veins", 0.98),
            ("diaphragm", "controls", "breathing", 0.98),
            ("esophagus", "connects", "mouth_to_stomach", 0.98),
            ("pancreas", "produces", "insulin_and_digestive_enzymes", 0.98),
            ("lymphatic_system", "returns", "fluid_to_bloodstream", 0.97),
            ("adrenal_glands", "produce", "adrenaline", 0.97),
            ("pituitary_gland", "is", "master_gland", 0.97),
            ("cerebellum", "coordinates", "balance_and_movement", 0.98),
            ("cerebrum", "controls", "thinking_and_voluntary_actions", 0.98),
            ("medulla_oblongata", "controls", "involuntary_functions", 0.97),
            ("trachea", "carries", "air_to_lungs", 0.98),
            ("bronchi", "branch_into", "lungs", 0.98),
            ("alveoli", "are_site_of", "oxygen_carbon_dioxide_exchange", 0.98),
        ]

        // ─── Earth Science (~60 facts) ───
        let earthScienceFacts: [(String, String, String, Double)] = [
            ("weathering", "breaks_down", "rocks", 0.99),
            ("erosion", "moves", "sediment", 0.99),
            ("deposition", "builds_up", "landforms", 0.97),
            ("igneous_rock", "forms_from", "cooled_magma", 0.99),
            ("sedimentary_rock", "forms_from", "compressed_sediment", 0.99),
            ("metamorphic_rock", "forms_from", "heat_and_pressure", 0.99),
            ("rock_cycle", "transforms", "rock_types", 0.98),
            ("tectonic_plates", "move_on", "asthenosphere", 0.98),
            ("earthquakes", "caused_by", "tectonic_movement", 0.99),
            ("volcanoes", "form_at", "plate_boundaries", 0.97),
            ("mountains", "form_from", "plate_collision", 0.97),
            ("ocean_trenches", "form_at", "subduction_zones", 0.97),
            ("fossils", "found_in", "sedimentary_rock", 0.99),
            ("minerals", "have", "crystal_structure", 0.98),
            ("soil", "forms_from", "weathered_rock", 0.97),
            ("water_cycle", "involves", "evaporation_condensation_precipitation", 0.99),
            ("evaporation", "changes", "liquid_to_gas", 1.0),
            ("condensation", "changes", "gas_to_liquid", 1.0),
            ("precipitation", "falls_as", "rain_snow_sleet_hail", 0.99),
            ("clouds", "form_from", "condensed_water_vapor", 0.99),
            ("wind", "caused_by", "uneven_heating", 0.98),
            ("ocean_currents", "distribute", "heat", 0.97),
            ("greenhouse_effect", "traps", "heat_in_atmosphere", 0.99),
            ("carbon_dioxide", "is_a", "greenhouse_gas", 0.99),
            ("ozone_layer", "protects_from", "ultraviolet_radiation", 0.99),
            ("seasons", "caused_by", "earth_tilt", 0.99),
            ("earth_rotation", "causes", "day_and_night", 1.0),
            ("earth_revolution", "causes", "seasons", 0.99),
            ("moon_phases", "caused_by", "moon_orbit", 0.98),
            ("tides", "caused_by", "moon_gravity", 0.98),
            ("continental_drift", "proposed_by", "wegener", 0.97),
            ("pangaea", "was_a", "supercontinent", 0.98),
            ("layers_of_earth", "include", "crust_mantle_core", 0.99),
            ("inner_core", "is", "solid_iron", 0.99),
            ("outer_core", "is", "liquid_iron", 0.99),
            ("mantle", "is", "semi_solid_rock", 0.98),
            ("crust", "is", "thin_outer_layer", 0.99),
            ("atmosphere", "contains", "nitrogen_and_oxygen", 0.99),
            ("troposphere", "is_where", "weather_occurs", 0.99),
            ("stratosphere", "contains", "ozone_layer", 0.99),
            ("glaciers", "form_from", "compacted_snow", 0.98),
            ("rivers", "erode", "valleys", 0.97),
            ("deltas", "form_at", "river_mouths", 0.97),
            ("stalactites", "form_from", "dripping_water", 0.96),
            ("stalagmites", "grow_from", "cave_floor", 0.96),
            ("convection_currents", "drive", "plate_tectonics", 0.97),
            ("seismograph", "measures", "earthquake_waves", 0.98),
            ("richter_scale", "measures", "earthquake_magnitude", 0.98),
            ("humus", "is", "organic_component_of_soil", 0.96),
            ("aquifer", "stores", "groundwater", 0.97),
            ("water_table", "is_top_of", "saturated_zone", 0.96),
            ("mesosphere", "is_where", "meteors_burn_up", 0.96),
            ("thermosphere", "contains", "ionosphere", 0.95),
            ("exosphere", "is_outermost", "atmospheric_layer", 0.96),
            ("mid_ocean_ridge", "forms_at", "divergent_boundary", 0.97),
            ("transform_boundary", "causes", "lateral_movement", 0.96),
            ("convergent_boundary", "causes", "subduction_or_collision", 0.97),
            ("divergent_boundary", "causes", "seafloor_spreading", 0.97),
            ("topographic_map", "shows", "elevation", 0.97),
            ("contour_lines", "connect", "equal_elevation_points", 0.97),
            ("relative_age", "determined_by", "rock_layer_position", 0.97),
            ("absolute_age", "determined_by", "radioactive_dating", 0.97),
            ("moraine", "deposited_by", "glacier", 0.95),
            ("sandstone", "is_a", "sedimentary_rock", 0.98),
            ("granite", "is_a", "igneous_rock", 0.98),
            ("marble", "is_a", "metamorphic_rock", 0.98),
            ("limestone", "formed_from", "marine_organisms", 0.96),
            ("obsidian", "is", "volcanic_glass", 0.97),
            ("pumice", "is", "lightweight_volcanic_rock", 0.96),
            ("karst_topography", "formed_by", "dissolving_limestone", 0.95),
            ("sinkhole", "caused_by", "underground_erosion", 0.95),
        ]

        // ─── Physics (~50 facts) ───
        let physicsFacts: [(String, String, String, Double)] = [
            ("gravity", "pulls", "objects_toward_earth", 0.99),
            ("gravity", "causes", "objects_to_fall", 0.99),
            ("mass", "is_amount_of", "matter", 0.99),
            ("weight", "is_force_of", "gravity_on_mass", 0.99),
            ("friction", "opposes", "motion", 0.99),
            ("friction", "produces", "heat", 0.98),
            ("force", "causes", "acceleration", 0.99),
            ("newtons_first_law", "states", "objects_stay_at_rest_or_motion", 0.99),
            ("newtons_second_law", "states", "force_equals_mass_times_acceleration", 0.99),
            ("newtons_third_law", "states", "every_action_has_equal_opposite_reaction", 0.99),
            ("potential_energy", "is_stored", "energy", 0.99),
            ("kinetic_energy", "is_energy_of", "motion", 0.99),
            ("energy", "cannot_be", "created_or_destroyed", 1.0),
            ("energy_transforms", "between", "different_forms", 0.99),
            ("heat", "flows_from", "hot_to_cold", 0.99),
            ("conduction", "transfers_heat_through", "direct_contact", 0.99),
            ("convection", "transfers_heat_through", "fluid_movement", 0.99),
            ("radiation", "transfers_heat_through", "electromagnetic_waves", 0.99),
            ("light", "travels_faster_than", "sound", 1.0),
            ("light", "travels_in", "straight_lines", 0.98),
            ("light", "can_be", "reflected_refracted_absorbed", 0.99),
            ("sound", "requires", "medium_to_travel", 0.99),
            ("sound", "travels_fastest_in", "solids", 0.98),
            ("frequency", "determines", "pitch", 0.99),
            ("amplitude", "determines", "loudness", 0.99),
            ("electromagnetic_spectrum", "includes", "radio_micro_infrared_visible_uv_xray_gamma", 0.99),
            ("speed", "equals", "distance_over_time", 1.0),
            ("acceleration", "is_change_in", "velocity", 0.99),
            ("momentum", "equals", "mass_times_velocity", 0.99),
            ("work", "requires", "force_and_distance", 0.99),
            ("power", "is_rate_of", "doing_work", 0.99),
            ("simple_machines", "reduce", "effort_force", 0.98),
            ("lever", "is_a", "simple_machine", 0.99),
            ("inclined_plane", "is_a", "simple_machine", 0.99),
            ("pulley", "is_a", "simple_machine", 0.99),
            ("magnet", "attracts", "iron", 0.99),
            ("magnetic_field", "surrounds", "magnets", 0.99),
            ("electric_current", "flows_through", "conductors", 0.99),
            ("insulators", "resist", "electric_current", 0.99),
            ("circuits", "require", "complete_path", 0.99),
            ("series_circuit", "has", "one_path", 0.99),
            ("parallel_circuit", "has", "multiple_paths", 0.99),
            ("voltage", "is", "electrical_pressure", 0.98),
            ("resistance", "opposes", "current_flow", 0.99),
            ("potential_energy", "example", "compressed_spring", 0.97),
            ("compressed_spring", "has", "elastic_potential_energy", 0.98),
            ("wavelength", "is_distance_between", "wave_crests", 0.99),
            ("reflection", "bounces_light_off", "surface", 0.98),
            ("refraction", "bends_light_through", "different_medium", 0.98),
            ("prism", "separates", "white_light_into_colors", 0.98),
            ("static_electricity", "caused_by", "charge_imbalance", 0.97),
            ("ohms_law", "states", "voltage_equals_current_times_resistance", 0.99),
            ("wedge", "is_a", "simple_machine", 0.99),
            ("screw", "is_a", "simple_machine", 0.99),
            ("wheel_and_axle", "is_a", "simple_machine", 0.99),
            ("inertia", "is_resistance_to", "change_in_motion", 0.99),
            ("centripetal_force", "keeps_objects_moving_in", "circle", 0.98),
            ("terminal_velocity", "is", "maximum_falling_speed", 0.97),
            ("buoyancy", "is_upward_force_from", "displaced_fluid", 0.98),
            ("archimedes_principle", "states", "buoyant_force_equals_weight_of_displaced_fluid", 0.98),
            ("bernoulli_principle", "states", "faster_fluid_has_lower_pressure", 0.97),
            ("doppler_effect", "causes", "frequency_change_with_motion", 0.97),
            ("electromagnetic_induction", "produces", "current_from_changing_magnetic_field", 0.98),
            ("transformer", "changes", "voltage_level", 0.97),
        ]

        // ─── Chemistry (~50 facts) ───
        let chemistryFacts: [(String, String, String, Double)] = [
            ("atom", "is_basic_unit_of", "matter", 1.0),
            ("proton", "has_charge", "positive", 1.0),
            ("neutron", "has_charge", "neutral", 1.0),
            ("electron", "has_charge", "negative", 1.0),
            ("atomic_number", "equals", "number_of_protons", 1.0),
            ("mass_number", "equals", "protons_plus_neutrons", 1.0),
            ("elements", "organized_in", "periodic_table", 0.99),
            ("metals", "are", "good_conductors", 0.99),
            ("nonmetals", "are", "poor_conductors", 0.98),
            ("noble_gases", "are", "unreactive", 0.98),
            ("chemical_bond", "holds_atoms", "together", 0.99),
            ("ionic_bond", "forms_between", "metal_and_nonmetal", 0.98),
            ("covalent_bond", "shares", "electrons", 0.99),
            ("molecule", "is_group_of", "bonded_atoms", 0.99),
            ("water", "is", "H2O", 1.0),
            ("carbon_dioxide", "is", "CO2", 1.0),
            ("photosynthesis_equation", "is", "6CO2+6H2O_C6H12O6+6O2", 0.99),
            ("pH_scale", "measures", "acidity_or_basicity", 0.99),
            ("acid", "has_pH", "less_than_7", 0.99),
            ("base", "has_pH", "greater_than_7", 0.99),
            ("neutral", "has_pH", "of_7", 0.99),
            ("neutralization", "produces", "water_and_salt", 0.98),
            ("chemical_reaction", "rearranges", "atoms", 0.99),
            ("reactants", "are", "starting_materials", 0.99),
            ("products", "are", "resulting_substances", 0.99),
            ("catalyst", "speeds_up", "reaction", 0.99),
            ("exothermic_reaction", "releases", "heat", 0.99),
            ("endothermic_reaction", "absorbs", "heat", 0.99),
            ("oxidation", "involves", "losing_electrons", 0.99),
            ("reduction", "involves", "gaining_electrons", 0.99),
            ("solution", "is", "homogeneous_mixture", 0.98),
            ("solute", "dissolves_in", "solvent", 0.99),
            ("concentration", "is_amount_of", "solute_in_solution", 0.98),
            ("evaporation", "is_a", "physical_change", 0.99),
            ("rusting", "is_a", "chemical_change", 0.99),
            ("states_of_matter", "include", "solid_liquid_gas", 0.99),
            ("solid", "has", "definite_shape_and_volume", 0.99),
            ("liquid", "has", "definite_volume", 0.99),
            ("gas", "fills", "entire_container", 0.99),
            ("melting", "changes", "solid_to_liquid", 1.0),
            ("boiling", "changes", "liquid_to_gas", 1.0),
            ("freezing", "changes", "liquid_to_solid", 1.0),
            ("sublimation", "changes", "solid_to_gas", 0.99),
            ("calcium", "group", "alkaline_earth_metals", 0.98),
            ("sodium", "group", "alkali_metals", 0.98),
            ("oxygen", "is_a", "nonmetal", 0.99),
            ("iron", "is_a", "transition_metal", 0.99),
            ("hydrogen", "is_lightest", "element", 1.0),
            ("mixture", "can_be_separated_by", "physical_means", 0.98),
            ("compound", "can_be_separated_by", "chemical_means", 0.98),
            ("isotopes", "have_different", "number_of_neutrons", 0.99),
            ("law_of_conservation_of_mass", "states", "mass_is_neither_created_nor_destroyed", 1.0),
            ("metalloids", "have_properties_of", "metals_and_nonmetals", 0.97),
            ("halogens", "are", "highly_reactive_nonmetals", 0.98),
            ("condensation_chemistry", "changes", "gas_to_liquid", 1.0),
            ("plasma", "is", "ionized_gas", 0.97),
            ("valence_electrons", "determine", "chemical_properties", 0.99),
            ("electron_shells", "surround", "nucleus", 0.99),
            ("combustion", "requires", "fuel_oxygen_heat", 0.98),
            ("corrosion", "is", "gradual_destruction_of_metal", 0.97),
            ("alloy", "is", "mixture_of_metals", 0.97),
            ("electrolysis", "uses", "electricity_to_decompose_compound", 0.97),
            ("indicator", "changes_color_in", "acid_or_base", 0.97),
            ("litmus_paper", "tests_for", "acid_or_base", 0.97),
            ("saturated_solution", "cannot_dissolve", "more_solute", 0.98),
            ("precipitate", "forms_when", "insoluble_product_created", 0.96),
            ("mole", "is_unit_of", "amount_of_substance", 0.99),
            ("avogadro_number", "is", "6.022e23_particles_per_mole", 0.99),
        ]

        // ─── Astronomy (~30 facts) ───
        let astronomyFacts: [(String, String, String, Double)] = [
            ("sun", "is_a", "star", 1.0),
            ("earth", "orbits", "sun", 1.0),
            ("moon", "orbits", "earth", 1.0),
            ("solar_system", "contains", "eight_planets", 0.99),
            ("mercury", "is_closest_to", "sun", 1.0),
            ("venus", "is_hottest", "planet", 0.99),
            ("mars", "is_called", "red_planet", 0.99),
            ("jupiter", "is_largest", "planet", 1.0),
            ("saturn", "has", "prominent_rings", 0.99),
            ("neptune", "is_farthest", "planet", 0.99),
            ("asteroid_belt", "between", "mars_and_jupiter", 0.99),
            ("comets", "made_of", "ice_and_dust", 0.98),
            ("light_year", "measures", "distance", 0.99),
            ("stars", "produce_energy_by", "nuclear_fusion", 0.99),
            ("galaxies", "contain", "billions_of_stars", 0.99),
            ("milky_way", "is_our", "galaxy", 1.0),
            ("universe", "is", "expanding", 0.99),
            ("telescope", "used_to", "observe_distant_objects", 0.99),
            ("satellite", "orbits", "larger_body", 0.99),
            ("solar_eclipse", "occurs_when", "moon_blocks_sun", 0.99),
            ("lunar_eclipse", "occurs_when", "earth_blocks_sunlight_to_moon", 0.99),
            ("constellation", "is_pattern_of", "stars", 0.99),
            ("gravity", "keeps_planets_in", "orbit", 0.99),
            ("rotation", "causes", "day_and_night", 1.0),
            ("revolution", "causes", "year", 0.99),
            ("inner_planets", "are", "rocky", 0.99),
            ("outer_planets", "are", "gas_giants", 0.98),
            ("dwarf_planet", "example", "pluto", 0.99),
            ("nebula", "is", "cloud_of_gas_and_dust", 0.98),
            ("supernova", "is", "exploding_star", 0.99),
            ("red_giant", "is", "late_stage_star", 0.97),
            ("white_dwarf", "is", "dense_remnant_star", 0.97),
            ("black_hole", "has", "extreme_gravity", 0.99),
            ("neutron_star", "is", "collapsed_star_core", 0.97),
            ("big_bang", "is_origin_of", "universe", 0.98),
            ("hubble_telescope", "observes", "deep_space", 0.98),
            ("uranus", "rotates_on", "its_side", 0.98),
            ("kuiper_belt", "beyond", "neptune", 0.98),
            ("oort_cloud", "is_source_of", "long_period_comets", 0.95),
            ("aurora", "caused_by", "solar_wind_and_magnetic_field", 0.97),
            ("main_sequence_star", "fuses", "hydrogen_to_helium", 0.98),
            ("hertzsprung_russell_diagram", "classifies", "stars_by_luminosity_and_temperature", 0.96),
            ("meteor", "is", "streak_of_light_in_atmosphere", 0.97),
            ("meteorite", "is", "space_rock_that_hits_earth", 0.97),
            ("asteroid", "is", "rocky_body_orbiting_sun", 0.98),
            ("parsec", "measures", "astronomical_distance", 0.96),
        ]

        // ─── Ecology (~40 facts) ───
        let ecologyFacts: [(String, String, String, Double)] = [
            ("ecosystem", "includes", "living_and_nonliving_things", 0.99),
            ("biome", "is_large", "ecosystem_type", 0.98),
            ("food_chain", "shows", "energy_flow", 0.99),
            ("food_chain", "starts_with", "producers", 0.99),
            ("producers", "make", "own_food", 0.99),
            ("consumers", "eat", "other_organisms", 0.99),
            ("herbivores", "eat", "plants", 0.99),
            ("carnivores", "eat", "animals", 0.99),
            ("omnivores", "eat", "plants_and_animals", 0.99),
            ("decomposers", "recycle", "nutrients", 0.99),
            ("food_web", "shows", "interconnected_food_chains", 0.98),
            ("energy_pyramid", "shows", "energy_decreases_at_each_level", 0.98),
            ("only_10_percent", "of_energy_transfers_to", "next_level", 0.97),
            ("population", "is_group_of", "same_species", 0.99),
            ("community", "is_group_of", "different_populations", 0.99),
            ("habitat", "is_where", "organism_lives", 0.99),
            ("niche", "is_role_of", "organism_in_ecosystem", 0.98),
            ("symbiosis", "is", "close_relationship_between_species", 0.99),
            ("mutualism", "benefits", "both_species", 0.99),
            ("parasitism", "benefits_one", "harms_other", 0.99),
            ("commensalism", "benefits_one", "neutral_to_other", 0.98),
            ("competition", "occurs_when", "organisms_share_resources", 0.98),
            ("predation", "is_when", "predator_eats_prey", 0.99),
            ("keystone_species", "has_large_impact_on", "ecosystem", 0.97),
            ("invasive_species", "can_harm", "native_ecosystems", 0.98),
            ("biodiversity", "is", "variety_of_life", 0.99),
            ("extinction", "is", "loss_of_species", 0.99),
            ("conservation", "aims_to", "protect_species", 0.98),
            ("carbon_cycle", "circulates", "carbon_through_environment", 0.99),
            ("nitrogen_cycle", "circulates", "nitrogen_through_environment", 0.98),
            ("water_cycle", "circulates", "water_through_environment", 0.99),
            ("deforestation", "reduces", "biodiversity", 0.98),
            ("pollution", "harms", "ecosystems", 0.98),
            ("renewable_resources", "can_be", "replenished", 0.99),
            ("nonrenewable_resources", "are", "finite", 0.99),
            ("fossil_fuels", "are", "nonrenewable", 0.99),
            ("solar_energy", "is", "renewable", 0.99),
            ("wind_energy", "is", "renewable", 0.99),
            ("energy_transfer", "flows", "producers_to_consumers", 0.98),
            ("frogs", "compete_for", "food", 0.92),
            ("tundra", "is_a", "cold_treeless_biome", 0.97),
            ("desert", "receives_very_little", "precipitation", 0.98),
            ("rainforest", "has_highest", "biodiversity", 0.97),
            ("grassland", "is_dominated_by", "grasses", 0.97),
            ("wetland", "filters", "water_naturally", 0.96),
            ("coral_reef", "is_a", "marine_ecosystem", 0.98),
            ("succession", "is_gradual_change_in", "ecosystem_over_time", 0.97),
            ("pioneer_species", "colonize", "barren_environments", 0.96),
            ("climax_community", "is", "stable_ecosystem_endpoint", 0.95),
            ("biotic_factors", "are", "living_components", 0.99),
            ("abiotic_factors", "are", "nonliving_components", 0.99),
            ("carrying_capacity", "is_maximum", "population_environment_can_support", 0.98),
            ("limiting_factors", "restrict", "population_growth", 0.97),
            ("bioaccumulation", "increases", "toxin_concentration_up_food_chain", 0.96),
            ("eutrophication", "caused_by", "excess_nutrients_in_water", 0.96),
            ("acid_rain", "caused_by", "sulfur_and_nitrogen_emissions", 0.97),
            ("greenhouse_gases", "contribute_to", "global_warming", 0.98),
            ("endangered_species", "face", "risk_of_extinction", 0.98),
            ("habitat_fragmentation", "reduces", "biodiversity", 0.96),
        ]

        // ─── Measurement (~25 facts) ───
        let measurementFacts: [(String, String, String, Double)] = [
            ("meter", "measures", "length", 1.0),
            ("gram", "measures", "mass", 1.0),
            ("liter", "measures", "volume", 1.0),
            ("second", "measures", "time", 1.0),
            ("kelvin", "measures", "temperature", 0.99),
            ("celsius", "measures", "temperature", 0.99),
            ("fahrenheit", "measures", "temperature", 0.99),
            ("graduated_cylinder", "measures", "liquid_volume", 0.99),
            ("balance", "measures", "mass", 0.99),
            ("thermometer", "measures", "temperature", 0.99),
            ("ruler", "measures", "length", 0.99),
            ("speed", "measured_in", "meters_per_second", 0.99),
            ("density", "equals", "mass_divided_by_volume", 1.0),
            ("density", "determines", "whether_object_sinks_or_floats", 0.98),
            ("volume", "can_be_found_by", "water_displacement", 0.98),
            ("scientific_method", "includes", "hypothesis_experiment_conclusion", 0.99),
            ("hypothesis", "is_a", "testable_prediction", 0.99),
            ("variable", "is_factor_that", "changes", 0.99),
            ("independent_variable", "is", "changed_by_scientist", 0.99),
            ("dependent_variable", "is", "measured_result", 0.99),
            ("control_group", "is", "baseline_for_comparison", 0.99),
            ("metric_system", "based_on", "powers_of_ten", 0.99),
            ("microscope", "magnifies", "small_objects", 0.99),
            ("hardness", "measured_by", "mohs_scale", 0.98),
            ("pH_paper", "measures", "acidity", 0.98),
            ("newton", "measures", "force", 0.99),
            ("joule", "measures", "energy", 0.99),
            ("watt", "measures", "power", 0.99),
            ("ampere", "measures", "electric_current", 0.99),
            ("volt", "measures", "electric_potential", 0.99),
            ("hertz", "measures", "frequency", 0.99),
            ("pascal", "measures", "pressure", 0.99),
            ("spring_scale", "measures", "force", 0.98),
            ("anemometer", "measures", "wind_speed", 0.97),
            ("barometer", "measures", "air_pressure", 0.98),
            ("rain_gauge", "measures", "precipitation", 0.98),
            ("caliper", "measures", "small_distances_precisely", 0.97),
            ("hygrometer", "measures", "humidity", 0.97),
            ("spectrometer", "measures", "light_wavelengths", 0.97),
            ("geiger_counter", "detects", "radiation", 0.98),
        ]

        // ─── Technology (~20 facts) ───
        let technologyFacts: [(String, String, String, Double)] = [
            ("computer", "processes", "information", 0.99),
            ("internet", "connects", "computers_worldwide", 0.99),
            ("binary", "uses", "zeros_and_ones", 0.99),
            ("circuit_board", "connects", "electronic_components", 0.98),
            ("solar_panel", "converts", "sunlight_to_electricity", 0.99),
            ("wind_turbine", "converts", "wind_to_electricity", 0.99),
            ("recycling", "reduces", "waste", 0.98),
            ("engineering", "applies", "science_to_solve_problems", 0.98),
            ("technology", "applies", "knowledge_practically", 0.97),
            ("GPS", "uses", "satellites_for_navigation", 0.99),
            ("MRI", "uses", "magnetic_fields", 0.99),
            ("x_ray", "uses", "electromagnetic_radiation", 0.99),
            ("vaccine", "prevents", "disease", 0.98),
            ("antibiotic", "kills", "bacteria", 0.98),
            ("pasteurization", "kills", "harmful_microorganisms", 0.98),
            ("refrigeration", "slows", "bacterial_growth", 0.97),
            ("telescope", "observes", "distant_objects", 0.99),
            ("microscope", "observes", "tiny_objects", 0.99),
            ("3d_printing", "creates", "objects_layer_by_layer", 0.97),
            ("LED", "is", "energy_efficient_light", 0.97),
            ("battery", "stores", "chemical_energy", 0.98),
            ("generator", "converts", "mechanical_to_electrical_energy", 0.98),
            ("semiconductor", "has_conductivity_between", "conductor_and_insulator", 0.97),
            ("fiber_optics", "transmits", "data_using_light", 0.98),
            ("radar", "uses", "radio_waves_for_detection", 0.98),
            ("sonar", "uses", "sound_waves_for_detection", 0.98),
            ("hydroelectric_dam", "converts", "water_energy_to_electricity", 0.98),
            ("geothermal_energy", "uses", "earth_internal_heat", 0.97),
            ("nuclear_power", "uses", "nuclear_fission", 0.98),
            ("biomass", "is_a", "renewable_energy_source", 0.96),
        ]

        // ─── Register all facts by domain ───
        let domainTuples: [(String, [(String, String, String, Double)])] = [
            ("biology", biologyFacts),
            ("body_systems", bodySystemsFacts),
            ("earth_science", earthScienceFacts),
            ("physics", physicsFacts),
            ("chemistry", chemistryFacts),
            ("astronomy", astronomyFacts),
            ("ecology", ecologyFacts),
            ("measurement", measurementFacts),
            ("technology", technologyFacts),
        ]

        allFacts.reserveCapacity(600)

        for (domain, tuples) in domainTuples {
            var count = 0
            for (subj, rel, obj, conf) in tuples {
                let fact = ScienceFact(
                    subject: subj,
                    relation: rel,
                    obj: obj,
                    confidence: conf,
                    domain: domain
                )
                allFacts.append(fact)
                count += 1
            }
            domainCounts[domain] = count
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Index Building
    // ═══════════════════════════════════════════════════════════════════

    private func buildIndices() {
        subjectIndex.removeAll(keepingCapacity: true)
        relationIndex.removeAll(keepingCapacity: true)
        objectIndex.removeAll(keepingCapacity: true)

        for fact in allFacts {
            let subjKey = fact.subject.lowercased()
            let relKey = fact.relation.lowercased()
            let objKey = fact.obj.lowercased()

            subjectIndex[subjKey, default: []].append(fact)
            relationIndex[relKey, default: []].append(fact)
            objectIndex[objKey, default: []].append(fact)
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Query (Partial Match / Substring Containment)
    // ═══════════════════════════════════════════════════════════════════

    /// Query facts by any combination of subject, relation, and/or object.
    /// Supports partial matches via substring containment.
    /// Pass `nil` for fields you do not wish to filter on.
    func query(subject: String? = nil, relation: String? = nil, obj: String? = nil) -> [ScienceFact] {
        lock.lock()
        defer { lock.unlock() }

        // If a subject is provided, start from the subject index for efficiency
        var candidates: [ScienceFact]

        if let subj = subject?.lowercased() {
            // Try exact match first, then substring scan
            if let exact = subjectIndex[subj] {
                candidates = exact
            } else {
                candidates = allFacts.filter { $0.subject.lowercased().contains(subj) }
            }
        } else if let rel = relation?.lowercased() {
            if let exact = relationIndex[rel] {
                candidates = exact
            } else {
                candidates = allFacts.filter { $0.relation.lowercased().contains(rel) }
            }
        } else if let o = obj?.lowercased() {
            if let exact = objectIndex[o] {
                candidates = exact
            } else {
                candidates = allFacts.filter { $0.obj.lowercased().contains(o) }
            }
        } else {
            // No filters — return all
            return allFacts
        }

        // Apply remaining filters
        if let rel = relation?.lowercased() {
            candidates = candidates.filter { $0.relation.lowercased().contains(rel) }
        }
        if let o = obj?.lowercased() {
            candidates = candidates.filter { $0.obj.lowercased().contains(o) }
        }
        // If we started from relation or object index, also filter by subject
        if subject != nil {
            // Already filtered above
        } else if let subj = subject?.lowercased() {
            candidates = candidates.filter { $0.subject.lowercased().contains(subj) }
        }

        return candidates
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Score MCQ Choice (ARC Benchmark)
    // ═══════════════════════════════════════════════════════════════════

    /// Score a multiple-choice answer option against the knowledge base.
    /// - Parameters:
    ///   - questionKeywords: Tokenized keywords from the question stem
    ///   - choiceText: The text of this particular answer choice
    ///   - allChoices: All answer choice texts (for relative scoring)
    /// - Returns: A score in [0, 1] indicating how well this choice is supported
    func scoreChoice(questionKeywords: Set<String>, choiceText: String, allChoices: [String]) -> Double {
        lock.lock()
        defer { lock.unlock() }

        let choiceTokens = tokenize(choiceText)
        let combinedKeywords = questionKeywords.union(choiceTokens)

        var totalScore: Double = 0.0
        var matchCount: Int = 0

        // Search for facts that relate question keywords to choice keywords
        for keyword in combinedKeywords {
            let keyLower = keyword.lowercased()

            // Check subject index
            if let subjectFacts = subjectIndex[keyLower] {
                for fact in subjectFacts {
                    let objTokens = tokenizeUnderscore(fact.obj)
                    let relTokens = tokenizeUnderscore(fact.relation)

                    // Check if choice tokens overlap with fact object or relation
                    let objOverlap = choiceTokens.intersection(objTokens)
                    let relOverlap = questionKeywords.intersection(relTokens)

                    if !objOverlap.isEmpty {
                        let boost = 1.0 + 0.5 * fact.confidence
                        totalScore += boost * Double(objOverlap.count)
                        matchCount += 1
                    }
                    if !relOverlap.isEmpty && !objOverlap.isEmpty {
                        // Bonus: question matches relation AND choice matches object
                        let boost = 1.0 + 0.5 * fact.confidence
                        totalScore += boost * phi  // PHI-weighted bonus for full triple alignment
                        matchCount += 1
                    }
                }
            }

            // Check object index (reverse lookup)
            if let objectFacts = objectIndex[keyLower] {
                for fact in objectFacts {
                    let subjTokens = tokenizeUnderscore(fact.subject)
                    let subjOverlap = choiceTokens.intersection(subjTokens)

                    if !subjOverlap.isEmpty {
                        let boost = 1.0 + 0.5 * fact.confidence
                        totalScore += boost * Double(subjOverlap.count)
                        matchCount += 1
                    }
                }
            }
        }

        // Apply TAU-decay normalization to prevent score explosion
        if matchCount > 0 {
            totalScore = totalScore / (1.0 + tau * Double(matchCount))
        }

        // Compute relative score against all choices
        if allChoices.count > 1 {
            var allScores: [Double] = []
            for choice in allChoices {
                let ct = tokenize(choice)
                let ck = questionKeywords.union(ct)
                var cs: Double = 0.0
                var cm: Int = 0

                for kw in ck {
                    let kwLower = kw.lowercased()
                    if let subjectFacts = subjectIndex[kwLower] {
                        for fact in subjectFacts {
                            let objTokens = tokenizeUnderscore(fact.obj)
                            let objOverlap = ct.intersection(objTokens)
                            if !objOverlap.isEmpty {
                                cs += (1.0 + 0.5 * fact.confidence) * Double(objOverlap.count)
                                cm += 1
                            }
                        }
                    }
                }
                if cm > 0 {
                    cs = cs / (1.0 + tau * Double(cm))
                }
                allScores.append(cs)
            }

            let maxScore = allScores.max() ?? 1.0
            if maxScore > 0 {
                totalScore = totalScore / maxScore
            }
        }

        return min(max(totalScore, 0.0), 1.0)
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Find Relevant Facts
    // ═══════════════════════════════════════════════════════════════════

    /// Find facts relevant to a given text, returning the top-k by relevance.
    /// - Parameters:
    ///   - text: The input text to find relevant facts for
    ///   - limit: Maximum number of facts to return (default 10)
    /// - Returns: Array of relevant ScienceFacts sorted by relevance score descending
    func findRelevantFacts(text: String, limit: Int = 10) -> [ScienceFact] {
        lock.lock()
        defer { lock.unlock() }

        let keywords = tokenize(text)
        if keywords.isEmpty { return [] }

        // Score each fact by relevance to input text
        var factScores: [(ScienceFact, Double)] = []
        var seen = Set<ScienceFact>()

        for keyword in keywords {
            let keyLower = keyword.lowercased()

            // Subject matches
            if let facts = subjectIndex[keyLower] {
                for fact in facts where !seen.contains(fact) {
                    seen.insert(fact)
                    let score = computeRelevance(fact: fact, keywords: keywords)
                    factScores.append((fact, score))
                }
            }

            // Also try substring matches against all subjects
            for (subj, facts) in subjectIndex {
                if subj.contains(keyLower) || keyLower.contains(subj) {
                    for fact in facts where !seen.contains(fact) {
                        seen.insert(fact)
                        let score = computeRelevance(fact: fact, keywords: keywords) * tau  // Discount partial matches
                        factScores.append((fact, score))
                    }
                }
            }

            // Object matches
            if let facts = objectIndex[keyLower] {
                for fact in facts where !seen.contains(fact) {
                    seen.insert(fact)
                    let score = computeRelevance(fact: fact, keywords: keywords)
                    factScores.append((fact, score))
                }
            }
        }

        // Sort by score descending, return top-k
        factScores.sort { $0.1 > $1.1 }
        let resultCount = min(limit, factScores.count)
        return Array(factScores.prefix(resultCount).map { $0.0 })
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Domain Query
    // ═══════════════════════════════════════════════════════════════════

    /// Return all facts in a specific domain.
    func factsForDomain(_ domain: String) -> [ScienceFact] {
        lock.lock()
        defer { lock.unlock() }
        return allFacts.filter { $0.domain == domain.lowercased() }
    }

    /// Return the count of facts per domain.
    func factCountByDomain() -> [String: Int] {
        lock.lock()
        defer { lock.unlock() }
        return domainCounts
    }

    /// Total number of facts in the knowledge base.
    var totalFacts: Int {
        lock.lock()
        defer { lock.unlock() }
        return allFacts.count
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Status Reporting
    // ═══════════════════════════════════════════════════════════════════

    func getStatus() -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }

        return [
            "subsystem": "ScienceKB",
            "version": SCIENCE_KB_VERSION,
            "total_facts": allFacts.count,
            "domains": domainCounts.count,
            "domain_counts": domainCounts,
            "subject_index_keys": subjectIndex.count,
            "relation_index_keys": relationIndex.count,
            "object_index_keys": objectIndex.count,
            "sacred_alignment": godCode,
            "phi_weight": phi,
            "tau_decay": tau,
            "status": allFacts.count >= 500 ? "FULLY_LOADED" : "PARTIAL",
        ]
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - Private Helpers
    // ═══════════════════════════════════════════════════════════════════

    /// Tokenize text into a set of lowercased, non-stop-word keywords.
    private func tokenize(_ text: String) -> Set<String> {
        let lowered = text.lowercased()
        let components = lowered.components(separatedBy: ScienceKB.nonAlphanumeric)
        var tokens = Set<String>()
        for component in components {
            let trimmed = component.trimmingCharacters(in: .whitespaces)
            if trimmed.count >= 2 && !ScienceKB.stopWords.contains(trimmed) {
                tokens.insert(trimmed)
            }
        }
        // Also split on underscores for compound terms
        for component in lowered.components(separatedBy: "_") {
            let trimmed = component.trimmingCharacters(in: .whitespacesAndNewlines)
                .components(separatedBy: ScienceKB.nonAlphanumeric).joined()
            if trimmed.count >= 2 && !ScienceKB.stopWords.contains(trimmed) {
                tokens.insert(trimmed)
            }
        }
        return tokens
    }

    /// Tokenize an underscore-separated string into a set of keywords.
    private func tokenizeUnderscore(_ text: String) -> Set<String> {
        let lowered = text.lowercased()
        var tokens = Set<String>()
        for part in lowered.components(separatedBy: "_") {
            let trimmed = part.trimmingCharacters(in: .whitespaces)
            if trimmed.count >= 2 && !ScienceKB.stopWords.contains(trimmed) {
                tokens.insert(trimmed)
            }
        }
        // Also add the full string (spaces replaced) for exact match
        let full = lowered.replacingOccurrences(of: "_", with: "")
        if full.count >= 2 {
            tokens.insert(full)
        }
        return tokens
    }

    /// Compute relevance of a single fact against a set of keywords.
    private func computeRelevance(fact: ScienceFact, keywords: Set<String>) -> Double {
        let subjTokens = tokenizeUnderscore(fact.subject)
        let relTokens = tokenizeUnderscore(fact.relation)
        let objTokens = tokenizeUnderscore(fact.obj)

        let allFactTokens = subjTokens.union(relTokens).union(objTokens)
        let overlap = keywords.intersection(allFactTokens)

        if overlap.isEmpty { return 0.0 }

        // Base relevance from token overlap ratio
        let overlapRatio = Double(overlap.count) / Double(max(keywords.count, 1))

        // Weight by confidence and PHI-scaled boost
        let confidenceBoost = 1.0 + 0.5 * fact.confidence
        let relevance = overlapRatio * confidenceBoost

        // Extra boost for subject matches (most specific)
        let subjectOverlap = keywords.intersection(subjTokens)
        let subjectBoost = subjectOverlap.isEmpty ? 1.0 : (1.0 + tau)

        return relevance * subjectBoost
    }
}
