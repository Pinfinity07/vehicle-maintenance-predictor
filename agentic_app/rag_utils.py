"""RAG pipeline — FAISS vector store with vehicle maintenance knowledge base."""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Vehicle Maintenance Knowledge Base
# ---------------------------------------------------------------------------

MAINTENANCE_DOCS = [
    # Engine & Oil
    """ENGINE OIL CHANGE GUIDELINES
    Regular oil changes are critical for engine longevity. For fleet vehicles:
    - Petrol engines: change every 8,000-10,000 km or 6 months (whichever comes first).
    - Diesel engines: change every 6,000-8,000 km or 4 months due to higher soot.
    - Synthetic oil extends intervals to 12,000-15,000 km.
    Signs of overdue oil change: dark/gritty oil, engine noise, reduced fuel efficiency,
    oil warning light. Neglecting oil changes leads to sludge buildup, increased wear,
    and potential engine seizure. Always use manufacturer-recommended viscosity grade.""",

    """ENGINE MAINTENANCE — GENERAL
    Routine engine checks should include: coolant level and condition, serpentine belt
    tension and cracking, spark plug condition (petrol), injector cleaning (diesel),
    air filter inspection, and exhaust system checks. High-mileage vehicles (>100,000 km)
    require closer monitoring of gasket integrity, compression tests, and timing
    chain/belt replacement. Unusual noises (knocking, ticking) warrant immediate
    inspection. Engine overheating must be addressed immediately to prevent head
    gasket failure or block warping.""",

    # Brakes
    """BRAKE SYSTEM MAINTENANCE
    Brake inspection should occur every 20,000 km or when symptoms appear:
    - Squealing or grinding noises during braking.
    - Vibration or pulsation in the brake pedal.
    - Increased stopping distance.
    - Vehicle pulling to one side during braking.
    Brake pads: replace when thickness is below 3mm. Brake rotors: resurface or
    replace when below minimum thickness or if scoring is present. Brake fluid:
    flush every 2 years or 40,000 km — it absorbs moisture over time, reducing
    boiling point. For fleet trucks and buses, check air brake systems, slack
    adjusters, and compressor function regularly. Worn brakes are a critical safety
    hazard and must be addressed immediately.""",

    # Tires
    """TIRE MAINTENANCE AND REPLACEMENT
    Tire inspections should cover tread depth, pressure, alignment, and rotation:
    - Minimum legal tread depth: 1.6mm, but replace at 3mm for safety.
    - Check tire pressure monthly; under-inflation increases fuel consumption by 3-5%
      and causes uneven wear.
    - Rotate tires every 8,000-10,000 km to ensure even wear.
    - Wheel alignment check every 20,000 km or after hitting potholes/curbs.
    - Replace tires older than 6 years regardless of tread depth (rubber degrades).
    For fleet vehicles, mismatched tires affect handling and braking. All-season vs.
    seasonal tires should be selected based on operating climate. Worn-out tires
    significantly increase accident risk in wet conditions.""",

    # Battery
    """BATTERY CARE AND REPLACEMENT
    Vehicle battery lifespan is typically 3-5 years. Warning signs of a failing battery:
    - Slow engine cranking or clicking sound on start.
    - Dim headlights at idle.
    - Battery warning light on dashboard.
    - Swollen or corroded battery case.
    Preventive measures: clean terminals every 6 months, ensure tight connections,
    test voltage regularly (healthy battery: 12.4-12.7V when off, 13.7-14.7V running).
    Extreme temperatures accelerate degradation. For fleet vehicles, maintain a
    replacement schedule and keep jump-start equipment available. Electric/hybrid
    vehicles require specialized high-voltage battery diagnostics.""",

    # Transmission
    """TRANSMISSION SERVICE GUIDELINES
    Automatic transmission: fluid change every 50,000-80,000 km.
    Manual transmission: fluid change every 50,000-60,000 km.
    Signs of transmission issues: delayed or rough shifting, slipping gears, fluid
    leaks (red/brown fluid), burning smell, check engine light.
    Transmission fluid should be checked monthly — it should be red/pink and not
    smell burnt. Dark or burnt-smelling fluid indicates overheating damage.
    For fleet trucks with heavy loads, consider more frequent service intervals.
    Transmission failure is one of the most expensive repairs; preventive
    maintenance is far more cost-effective than replacement.""",

    # Coolant
    """COOLANT SYSTEM MAINTENANCE
    Engine coolant prevents overheating and freezing, and inhibits corrosion.
    - Coolant flush: every 40,000-60,000 km or 2-3 years.
    - Check coolant level weekly for fleet vehicles.
    - Inspect hoses for cracks, bulging, or soft spots quarterly.
    - Test coolant concentration with a refractometer (should be 50/50 mix).
    - Thermostat replacement if engine runs cold or overheats intermittently.
    - Radiator cap: replace if not holding pressure (causes boil-over).
    Overheating can cause catastrophic engine damage within minutes. If temperature
    gauge enters red zone, stop immediately and let the engine cool.""",

    # Fleet scheduling
    """FLEET MAINTENANCE SCHEDULING — BEST PRACTICES
    Effective fleet maintenance follows a tiered schedule:
    - Daily: driver walk-around (lights, tires, fluids, visible damage).
    - Weekly: fluid top-ups, tire pressure, windshield washer.
    - Monthly: battery check, brake visual inspection, belt inspection.
    - Quarterly: comprehensive inspection (all systems).
    - Annually: full service including transmission, coolant, brake fluid flush.
    Mileage-based: minor service every 10,000 km, major service every 40,000 km.
    Track each vehicle's service history in a centralized system. Vehicles with
    higher-than-average repair frequency should be flagged for potential retirement.
    Downtime cost often exceeds repair cost — preventive maintenance reduces both.""",

    # Emergency protocols
    """EMERGENCY MAINTENANCE PROTOCOLS
    Critical issues requiring immediate vehicle removal from service:
    - Brake failure or significant brake degradation.
    - Steering system malfunction.
    - Tire blowout or tread separation.
    - Engine overheating (temperature in red zone).
    - Fluid leaks creating fire hazard (fuel, oil on hot components).
    - Suspension failure affecting vehicle control.
    - Electrical fire or burning smell from wiring.
    Response procedure: 1) Remove vehicle from service immediately. 2) Document
    symptoms and conditions. 3) Tow to maintenance facility (do not drive).
    4) Perform root cause analysis. 5) Verify repair before returning to service.
    Never allow a vehicle with critical safety issues to remain in operation.""",

    # Fuel efficiency
    """FUEL EFFICIENCY OPTIMIZATION
    Poor fuel efficiency often indicates underlying maintenance issues:
    - Dirty air filter reduces efficiency by up to 10%.
    - Under-inflated tires increase rolling resistance and fuel use by 3-5%.
    - Misaligned wheels cause drag and uneven tire wear.
    - Worn spark plugs (petrol) cause incomplete combustion.
    - Clogged fuel injectors reduce atomization and power.
    - Dragging brakes (stuck caliper) waste energy as heat.
    Fleet-wide fuel tracking helps identify vehicles needing attention. A sudden
    drop in fuel efficiency (>15%) for a specific vehicle warrants immediate
    inspection. Regular maintenance keeps fuel costs predictable and minimizes
    environmental impact.""",

    # Seasonal maintenance
    """SEASONAL MAINTENANCE CHECKLIST
    Pre-summer: Check A/C system refrigerant and operation, coolant concentration,
    battery (heat accelerates degradation), tire pressure (increases with heat),
    cabin air filter.
    Pre-winter: Battery load test, antifreeze concentration, tire tread depth
    (consider winter tires), wiper blades, heater/defroster operation, door seal
    condition, emergency kit (blankets, torch, sand/salt).
    Monsoon/rainy season: Wiper condition, tire tread depth, headlight alignment,
    brake inspection (wet conditions demand more), rust-proofing inspection,
    drainage channel clearance.
    Seasonal transitions are when breakdowns spike — proactive checks reduce
    unplanned downtime by up to 40%.""",

    # High-mileage vehicles
    """HIGH-MILEAGE VEHICLE MANAGEMENT (>100,000 KM)
    Vehicles exceeding 100,000 km require enhanced monitoring:
    - Compression test to assess engine health.
    - Transmission fluid and filter change if not done recently.
    - Suspension bushings and shock absorbers — often worn by this mileage.
    - Water pump replacement (often done with timing belt).
    - CV joints/boots inspection (front-wheel drive).
    - Exhaust system inspection for rust and leaks.
    - Power steering fluid flush.
    Consider total cost of ownership: if annual maintenance exceeds 50% of the
    vehicle's value, replacement may be more economical. High-mileage fleet vehicles
    should be evaluated for reliability impact on operations.""",

    # Safety compliance
    """VEHICLE SAFETY COMPLIANCE AND INSPECTION
    Fleet vehicles must meet safety standards at all times:
    - Functional headlights, tail lights, brake lights, and indicators.
    - Working horn and windshield wipers.
    - Seat belts in proper condition for all seating positions.
    - Mirrors: properly adjusted and undamaged.
    - Fire extinguisher (commercial vehicles): charged and accessible.
    - First aid kit: stocked and not expired.
    - Vehicle registration and insurance documentation current.
    Regular safety audits (monthly for commercial fleets) ensure compliance and
    reduce liability. Document all inspections with date, inspector, and findings.
    Non-compliant vehicles must be removed from service until issues are resolved.""",

    # Warranty considerations
    """WARRANTY AND SERVICE RECORD MANAGEMENT
    Maintaining accurate service records is essential for warranty claims and resale:
    - Record all maintenance with date, mileage, work performed, and parts used.
    - Keep receipts for all services and parts purchases.
    - Follow manufacturer-recommended service intervals to maintain warranty.
    - Warranty typically covers powertrain for 5 years/100,000 km.
    - Extended warranties may cover additional components.
    - Warranty may be voided by: unauthorized modifications, missed scheduled
      services, use of non-approved fluids or parts.
    For fleet managers: centralized digital records enable trend analysis, cost
    tracking, and resale value optimization. Expired warranty vehicles should be
    evaluated for self-insurance vs. extended warranty purchase.""",
]


# ---------------------------------------------------------------------------
# Risk Modifier Knowledge Base (for extra info not in the ML model)
# ---------------------------------------------------------------------------

RISK_MODIFIER_DOCS = [
    """DRIVING CONDITIONS — RISK MODIFIERS
    Driving conditions significantly affect vehicle wear and maintenance needs beyond
    what standard telemetry captures:
    - Off-road or unpaved roads: +25% maintenance frequency. Increases wear on
      suspension, tires, undercarriage. Dust ingestion clogs air filters faster.
    - Mountain/hill driving: +20% brake wear, +15% transmission stress. Frequent
      elevation changes strain cooling system and brakes.
    - Stop-and-go city traffic: +30% brake wear, +20% transmission wear vs highway.
      Increases fuel consumption by 20-40%. Battery works harder with frequent starts.
    - Highway/long-distance: Lower brake wear but higher tire and engine hour wear.
      Generally less stressful on drivetrain.
    - Towing/heavy loads: +30-50% wear on transmission, brakes, tires, and engine.
      Requires shorter service intervals across all systems.
    Risk adjustment: For harsh conditions, multiply base risk by 1.2-1.5x.""",

    """CLIMATE AND WEATHER — RISK MODIFIERS
    Environmental conditions impact vehicle degradation rates:
    - Extreme heat (>40C regularly): Battery life reduced by 30-50%. Tire blowout
      risk increases. Cooling system under greater stress. AC compressor wear.
    - Extreme cold (<-10C regularly): Battery capacity drops 30-60%. Oil thickens,
      increasing engine wear on cold starts. Rust from road salt exposure.
    - High humidity/coastal: Accelerated corrosion and rust. Electrical system
      moisture damage. Mold risk in cabin. Brake rotor surface rust.
    - Heavy rain/flooding areas: Water ingestion risk. Hydroplaning risk with worn
      tires. Electrical shorts. Rust on undercarriage.
    - Dusty/sandy environments: Air filter clogging 2-3x faster. Paint and windshield
      abrasion. Seal degradation. Requires frequent cleaning.
    Risk adjustment: Extreme climate adds +15-30% to base maintenance risk.""",

    """ACCIDENT SEVERITY — RISK MODIFIERS
    Not all accidents are equal in terms of ongoing maintenance impact:
    - Minor fender bender (cosmetic): Minimal ongoing risk (+5%). Check alignment.
    - Moderate collision (structural): +25% ongoing risk. Frame damage may cause
      progressive alignment, tire, and suspension issues even after repair.
    - Major/severe accident (airbag deployment): +40-60% ongoing risk. Hidden
      structural damage, electrical system compromise, fluid system integrity.
      Vehicle should be thoroughly inspected even if repaired.
    - Flood damage: +50% risk. Electrical corrosion is progressive and may not
      manifest immediately. Mold in hidden cavities. Mechanical contamination.
    - Rollover: +40% risk. Roof structural integrity, glass seal integrity,
      fluid system contamination, suspension geometry.
    Risk adjustment: Recent severe accident adds 0.3-0.6 to probability score.""",

    """USAGE PATTERNS — RISK MODIFIERS
    How a vehicle is used affects wear patterns beyond simple mileage:
    - Commercial delivery (frequent stops): +25% drivetrain wear. Door hinges,
      cabin entry points wear faster. Higher idle hours.
    - Ride-sharing/taxi: +30% interior wear, suspension, brakes. High idle time.
      More frequent fluid changes needed.
    - Construction site use: +40% wear on suspension, tires, air filters.
      Vibration damage to electronics. Stone chip damage.
    - Emergency vehicle: High-stress driving patterns, rapid acceleration/braking.
      +35% drivetrain wear despite potentially lower mileage.
    - School bus: Frequent stops, door mechanisms, safety systems critical.
      +20% brake wear. Regular safety inspections mandatory.
    - Long-term parked (>30 days): Flat spots on tires, battery discharge,
      fuel degradation, seal dry-out, pest damage risk.
    Risk adjustment: High-stress usage adds +20-40% to base risk.""",

    """MODIFICATION AND AFTERMARKET — RISK MODIFIERS
    Vehicle modifications can significantly affect reliability and maintenance:
    - Lift kit/suspension modification: Changes center of gravity, affects handling.
      Increased CV joint angles, accelerated wear. +20% suspension maintenance.
    - Engine tuning/chip: Increased power stresses drivetrain. May void warranty.
      +15-25% engine maintenance. Cooling system may be inadequate.
    - Aftermarket exhaust: May affect back-pressure and engine tuning. Emissions
      compliance risk. Generally low maintenance impact if properly done.
    - Oversized tires: Speedometer inaccuracy, increased drivetrain stress,
      reduced fuel efficiency. +15% tire and suspension wear.
    - Aftermarket electrical (lights, audio, accessories): Increased battery draw,
      potential wiring fire risk if poorly installed. +10% electrical system risk.
    - Non-OEM parts: Quality varies widely. May not fit precisely, causing
      accelerated wear on mating components. Warranty implications.
    Risk adjustment: Significant modifications add +10-25% to base risk.""",

    """DRIVER BEHAVIOR — RISK MODIFIERS
    Driver habits have a major impact on vehicle wear:
    - Aggressive driving (hard acceleration/braking): +30-40% brake wear,
      +25% tire wear, +20% fuel consumption. Transmission shock loading.
    - Riding the clutch (manual): Premature clutch wear, can cost $1000-2500
      to replace. +50% clutch system maintenance.
    - Ignoring warning lights: Cascading damage — a $50 sensor fix becomes a
      $3000 engine repair. Dramatically increases catastrophic failure risk.
    - Overloading vehicle: Exceeding GVWR stresses every system — suspension,
      brakes, tires, engine, transmission. +30-50% wear across the board.
    - Skipping warm-up in cold weather: Increased engine wear due to poor
      lubrication. Modern engines need 30-60 seconds, not extended idling.
    - Fuel quality (low-grade or contaminated): Injector clogging, fuel system
      corrosion, reduced efficiency. +10-15% engine maintenance.
    Risk adjustment: Poor driving habits add +20-40% to base risk.""",

    """FLUID LEAKS AND WARNING SIGNS — RISK MODIFIERS
    Active symptoms reported by the user should significantly elevate risk:
    - Oil leak (spots under vehicle): +30% engine risk. Monitor level frequently.
      Small leak can become catastrophic if oil runs low.
    - Coolant leak (sweet smell, green/orange fluid): +40% overheating risk.
      Can lead to head gasket failure ($2000+).
    - Transmission fluid leak (red/brown): +35% transmission risk. Low fluid
      causes overheating and gear damage.
    - Brake fluid leak: CRITICAL — +80% brake failure risk. Vehicle should not
      be driven. Immediate repair required.
    - Unusual noises (knocking, squealing, grinding): +25% component failure risk.
      Noise indicates active mechanical degradation.
    - Vibration at speed: +20% risk. Could be wheel balance, warped rotors,
      CV joint, or driveshaft issue.
    - Check engine light on: +15-30% risk depending on code. Should be diagnosed.
    Risk adjustment: Active symptoms add 0.15-0.8 to probability depending on severity.""",

    """VEHICLE BRAND RELIABILITY — RISK MODIFIERS
    Different manufacturers have varying reliability track records:
    - Japanese manufacturers (Toyota, Honda, Suzuki): Generally higher reliability,
      lower maintenance costs. -10% risk modifier.
    - Korean manufacturers (Hyundai, Kia): Good reliability in recent models (2018+),
      older models may have higher maintenance. Neutral to -5%.
    - European manufacturers (BMW, Mercedes, Audi, VW): Higher performance but
      also higher maintenance costs and complexity. +10-20% risk and cost.
    - American manufacturers (Ford, GM, Jeep): Variable by model. Trucks generally
      reliable. Some models have known issues. Neutral to +10%.
    - Chinese manufacturers: Rapidly improving but less long-term reliability data.
      Parts availability may be limited. +10-15% risk.
    - Luxury/performance variants: Higher maintenance cost due to specialized parts
      and labor. Premium fluids and consumables required. +15-25% cost modifier.
    Risk adjustment: Brand reliability modifies base risk by -10% to +25%.""",
]


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

_maintenance_rag = None
_modifier_rag = None


class MaintenanceRAG:
    """FAISS-backed retrieval of maintenance guidelines."""

    def __init__(self, docs: list[str], prefix: str = "guideline"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        documents = [
            Document(page_content=text, metadata={"source": f"{prefix}_{i}"})
            for i, text in enumerate(docs)
        ]
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content.strip() for doc in results]


def get_rag() -> MaintenanceRAG:
    """Singleton accessor for maintenance guidelines RAG."""
    global _maintenance_rag
    if _maintenance_rag is None:
        _maintenance_rag = MaintenanceRAG(MAINTENANCE_DOCS, prefix="guideline")
    return _maintenance_rag


def get_modifier_rag() -> MaintenanceRAG:
    """Singleton accessor for risk modifier RAG."""
    global _modifier_rag
    if _modifier_rag is None:
        _modifier_rag = MaintenanceRAG(RISK_MODIFIER_DOCS, prefix="modifier")
    return _modifier_rag
