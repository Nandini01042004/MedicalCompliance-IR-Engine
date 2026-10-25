"""
Corpus Generator and Ingestion for Medical Compliance Engine.
Generates structured legal compliance clauses across US HIPAA, US FDA 21 CFR Part 11,
and Indian CDSCO Drugs & Cosmetics Rules (Schedules M, H, H1, X).
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.common.config import CORPUS_FILE, CORPUS_DIR


CLAUSES_DATA = [
    # =========================================================================
    # US HIPAA (45 CFR Part 160 & 164)
    # =========================================================================
    {
        "doc_id": "HIPAA_164_502_A",
        "jurisdiction": "US_HIPAA",
        "statute": "45 CFR Part 164 - Security and Privacy",
        "part": "Subpart E - Privacy of Individually Identifiable Health Information",
        "clause_reference": "§ 164.502(a)",
        "subtopic_tag": "phi_disclosure_general",
        "title": "General Rules for Uses and Disclosures of Protected Health Information",
        "text": "A covered entity or business associate may not use or disclose protected health information (PHI), except as permitted or required by this subpart or as required by other law. Disclosures are permitted to the individual upon request, for treatment, payment, or health care operations (TPO), or pursuant to a valid signed authorization.",
        "mandatory_level": "SHALL",
        "entities": ["PHI", "covered entity", "business associate", "treatment", "payment", "health care operations", "authorization"]
    },
    {
        "doc_id": "HIPAA_164_504_E",
        "jurisdiction": "US_HIPAA",
        "statute": "45 CFR Part 164 - Security and Privacy",
        "part": "Subpart E - Privacy of Individually Identifiable Health Information",
        "clause_reference": "§ 164.504(e)",
        "subtopic_tag": "business_associate_agreement",
        "title": "Business Associate Agreement (BAA) Requirements",
        "text": "The contract or other written arrangement between the covered entity and the business associate must establish the permitted and required uses and disclosures of PHI by the business associate. The business associate must implement appropriate administrative, physical, and technical safeguards, report security incidents and breaches, and ensure downstream subcontractors agree to the same restrictions.",
        "mandatory_level": "MUST",
        "entities": ["BAA", "business associate agreement", "subcontractor", "safeguards", "breach reporting", "contractual obligation"]
    },
    {
        "doc_id": "HIPAA_164_508_A",
        "jurisdiction": "US_HIPAA",
        "statute": "45 CFR Part 164 - Security and Privacy",
        "part": "Subpart E - Privacy of Individually Identifiable Health Information",
        "clause_reference": "§ 164.508(a)",
        "subtopic_tag": "patient_authorization",
        "title": "Valid Individual Authorization for Uses and Disclosures",
        "text": "Except as otherwise permitted, a covered entity must obtain a signed written authorization prior to using or disclosing protected health information for marketing, sale of PHI, or research purposes without institutional review board (IRB) waiver. The authorization must contain specific description of information, expiration date, and right to revoke.",
        "mandatory_level": "SHALL",
        "entities": ["authorization", "patient consent", "marketing", "sale of PHI", "IRB waiver", "revocation"]
    },
    {
        "doc_id": "HIPAA_164_512_B",
        "jurisdiction": "US_HIPAA",
        "statute": "45 CFR Part 164 - Security and Privacy",
        "part": "Subpart E - Privacy of Individually Identifiable Health Information",
        "clause_reference": "§ 164.512(b)",
        "subtopic_tag": "public_health_disclosure",
        "title": "Disclosures for Public Health Activities Without Consent",
        "text": "A covered entity may disclose protected health information without individual authorization to a public health authority authorized by law to collect or receive such information for the purpose of preventing or controlling disease, injury, or disability, or reporting vital events such as births and communicable disease outbreaks.",
        "mandatory_level": "PERMITTED",
        "entities": ["public health authority", "disease control", "communicable diseases", "epidemiology", "non-consent disclosure"]
    },
    {
        "doc_id": "HIPAA_164_514_B_SAFE_HARBOR",
        "jurisdiction": "US_HIPAA",
        "statute": "45 CFR Part 164 - Security and Privacy",
        "part": "Subpart E - Privacy of Individually Identifiable Health Information",
        "clause_reference": "§ 164.514(b)(2)",
        "subtopic_tag": "de_identification",
        "title": "Safe Harbor Method for De-Identification of Health Information",
        "text": "Health information is de-identified under the Safe Harbor method if 18 specific direct and indirect identifiers of the individual or of relatives, employers, or household members are removed: names, all geographic subdivisions smaller than a state (except first three digits of zip codes matching criteria), all elements of dates (except year), telephone/fax numbers, email addresses, SSN, medical record numbers, health plan beneficiary numbers, account numbers, certificate/license numbers, vehicle identifiers, device identifiers/serial numbers, URLs, IP addresses, biometric identifiers including finger and voice prints, full-face photographs, and any unique identifying characteristic, and the covered entity has no actual knowledge that the remaining information could be used to identify an individual.",
        "mandatory_level": "MUST",
        "entities": ["safe harbor", "18 identifiers", "de-identification", "anonymization", "names", "SSN", "IP address", "biometric", "dates"]
    },
    {
        "doc_id": "HIPAA_164_514_A_EXPERT",
        "jurisdiction": "US_HIPAA",
        "statute": "45 CFR Part 164 - Security and Privacy",
        "part": "Subpart E - Privacy of Individually Identifiable Health Information",
        "clause_reference": "§ 164.514(a)",
        "subtopic_tag": "de_identification",
        "title": "Expert Determination Method for De-Identification",
        "text": "Health information is not individually identifiable if a person with appropriate knowledge of and experience with generally accepted statistical and scientific principles and methods applies such principles to determine that the risk is very small that the information could be used, alone or in combination with other reasonably available information, by an anticipated recipient to identify an individual who is a subject of the information, and documents the methods and results of the analysis.",
        "mandatory_level": "SHALL",
        "entities": ["expert determination", "statistical disclosure control", "re-identification risk", "scientific principles", "documentation"]
    },
    {
        "doc_id": "HIPAA_164_308_A1",
        "jurisdiction": "US_HIPAA",
        "statute": "45 CFR Part 164 - Security and Privacy",
        "part": "Subpart C - Security Standards for the Protection of Electronic PHI",
        "clause_reference": "§ 164.308(a)(1)",
        "subtopic_tag": "security_risk_analysis",
        "title": "Administrative Safeguards - Security Management Process",
        "text": "A covered entity or business associate must conduct an accurate and thorough assessment of the potential risks and vulnerabilities to the confidentiality, integrity, and availability of electronic protected health information (ePHI) held by the organization. Implement security measures sufficient to reduce risks and vulnerabilities to a reasonable and appropriate level, alongside sanction policies and regular information system activity reviews.",
        "mandatory_level": "SHALL",
        "entities": ["security risk assessment", "vulnerability analysis", "confidentiality", "integrity", "availability", "sanction policy", "audit logs"]
    },
    {
        "doc_id": "HIPAA_164_312_A",
        "jurisdiction": "US_HIPAA",
        "statute": "45 CFR Part 164 - Security and Privacy",
        "part": "Subpart C - Security Standards for the Protection of Electronic PHI",
        "clause_reference": "§ 164.312(a)(1)",
        "subtopic_tag": "technical_safeguards_access",
        "title": "Technical Safeguards - Access Control and Emergency Access",
        "text": "Implement technical policies and procedures for electronic information systems that maintain electronic protected health information to allow access only to those persons or software programs that have been granted access rights. Required specifications include Unique User Identification (assigning a unique name/number for tracking), Emergency Access Procedure (establishing documented emergency access protocol), and addressable specifications for Automatic Logoff and Encryption/Decryption.",
        "mandatory_level": "SHALL",
        "entities": ["access control", "unique user ID", "emergency access protocol", "automatic logoff", "role-based access", "RBAC"]
    },
    {
        "doc_id": "HIPAA_164_312_B",
        "jurisdiction": "US_HIPAA",
        "statute": "45 CFR Part 164 - Security and Privacy",
        "part": "Subpart C - Security Standards for the Protection of Electronic PHI",
        "clause_reference": "§ 164.312(b)",
        "subtopic_tag": "audit_trails",
        "title": "Technical Safeguards - Audit Controls",
        "text": "Implement hardware, software, and/or procedural mechanisms that record and examine activity in information systems that contain or use electronic protected health information. System audit trails must log user logins, access to patient medical charts, record modifications, export operations, and administrative privilege elevation.",
        "mandatory_level": "SHALL",
        "entities": ["audit controls", "audit trail", "system logs", "record modification", "export tracking", "forensic logging"]
    },
    {
        "doc_id": "HIPAA_164_312_E",
        "jurisdiction": "US_HIPAA",
        "statute": "45 CFR Part 164 - Security and Privacy",
        "part": "Subpart C - Security Standards for the Protection of Electronic PHI",
        "clause_reference": "§ 164.312(e)(1)",
        "subtopic_tag": "encryption_transmission",
        "title": "Technical Safeguards - Transmission Security and Encryption",
        "text": "Implement technical security measures to guard against unauthorized access to electronic protected health information that is being transmitted over an electronic communications network. Guard against tampering through integrity controls and implement encryption mechanisms (such as TLS 1.3 for data in transit and AES-256 for data at rest) whenever deemed appropriate.",
        "mandatory_level": "SHALL",
        "entities": ["transmission security", "encryption in transit", "TLS", "AES-256", "data integrity", "network security"]
    },
    {
        "doc_id": "HIPAA_164_404_A",
        "jurisdiction": "US_HIPAA",
        "statute": "45 CFR Part 164 - Security and Privacy",
        "part": "Subpart D - Notification in the Case of Breach of Unsecured PHI",
        "clause_reference": "§ 164.404(a)",
        "subtopic_tag": "breach_notification",
        "title": "Breach Notification to Affected Individuals",
        "text": "A covered entity shall, following the discovery of a breach of unsecured protected health information, notify each individual whose unsecured PHI has been, or is reasonably believed by the covered entity to have been, accessed, acquired, used, or disclosed as a result of such breach without unreasonable delay and in no case later than 60 calendar days after discovery of the breach.",
        "mandatory_level": "SHALL",
        "entities": ["breach notification", "unsecured PHI", "60 days", "incident response", "affected patients", "written notice"]
    },
    {
        "doc_id": "HIPAA_164_406_A",
        "jurisdiction": "US_HIPAA",
        "statute": "45 CFR Part 164 - Security and Privacy",
        "part": "Subpart D - Notification in the Case of Breach of Unsecured PHI",
        "clause_reference": "§ 164.406(a)",
        "subtopic_tag": "breach_notification_media",
        "title": "Notification to Media for Major Health Data Breaches",
        "text": "For a breach of unsecured protected health information that affects more than 500 residents of a State or jurisdiction, the covered entity shall notify prominent media outlets serving the State or jurisdiction without unreasonable delay and in no case later than 60 calendar days after discovery of the breach. Notice shall include breach description, data types involved, and protective steps.",
        "mandatory_level": "SHALL",
        "entities": ["media notification", "500 individuals", "major breach", "press release", "public notice", "HHS reporting"]
    },

    # =========================================================================
    # US FDA 21 CFR Part 11 (Electronic Records & Electronic Signatures)
    # =========================================================================
    {
        "doc_id": "FDA_21_CFR_11_10_A",
        "jurisdiction": "US_FDA",
        "statute": "21 CFR Part 11 - Electronic Records; Electronic Signatures",
        "part": "Subpart B - Electronic Records",
        "clause_reference": "§ 11.10(a)",
        "subtopic_tag": "computer_system_validation",
        "title": "Controls for Closed Systems - Validation",
        "text": "Validation of systems to ensure accuracy, reliability, consistent intended performance, and the ability to discern invalid or altered records. System validation must include documented IQ (Installation Qualification), OQ (Operational Qualification), and PQ (Performance Qualification) protocols for computerized clinical and manufacturing data systems.",
        "mandatory_level": "SHALL",
        "entities": ["validation", "computer system validation", "CSV", "IQ", "OQ", "PQ", "reliability", "altered records"]
    },
    {
        "doc_id": "FDA_21_CFR_11_10_E",
        "jurisdiction": "US_FDA",
        "statute": "21 CFR Part 11 - Electronic Records; Electronic Signatures",
        "part": "Subpart B - Electronic Records",
        "clause_reference": "§ 11.10(e)",
        "subtopic_tag": "audit_trails",
        "title": "Controls for Closed Systems - Secure Time-Stamped Audit Trails",
        "text": "Use of secure, computer-generated, time-stamped audit trails to independently record the date and time of operator entries and actions that create, modify, or delete electronic records. Record changes shall not obscure previously recorded information. Such audit trail documentation shall be retained for a period at least as long as that required for the subject electronic records and shall be available for agency review and copying.",
        "mandatory_level": "SHALL",
        "entities": ["audit trail", "time-stamped", "operator actions", "immutable log", "tamper-proof", "record retention", "regulatory inspection"]
    },
    {
        "doc_id": "FDA_21_CFR_11_10_F",
        "jurisdiction": "US_FDA",
        "statute": "21 CFR Part 11 - Electronic Records; Electronic Signatures",
        "part": "Subpart B - Electronic Records",
        "clause_reference": "§ 11.10(f)",
        "subtopic_tag": "workflow_enforcement",
        "title": "Controls for Closed Systems - Operational Sequencing Checks",
        "text": "Use of operational system checks to enforce permitted sequencing of steps and events, as appropriate. Systems must prevent out-of-order execution of clinical trial data entry, laboratory sample testing, batch record approvals, and electronic signature sign-offs.",
        "mandatory_level": "SHALL",
        "entities": ["operational checks", "sequencing", "workflow enforcement", "batch record approvals", "process control"]
    },
    {
        "doc_id": "FDA_21_CFR_11_10_G",
        "jurisdiction": "US_FDA",
        "statute": "21 CFR Part 11 - Electronic Records; Electronic Signatures",
        "part": "Subpart B - Electronic Records",
        "clause_reference": "§ 11.10(g)",
        "subtopic_tag": "authority_checks",
        "title": "Controls for Closed Systems - Authority Checks",
        "text": "Use of authority checks to ensure that only authorized individuals can use the system, electronically sign a record, access the operation or computer system input or output device, alter a record, or perform the operation at hand through strict role-based access permissions.",
        "mandatory_level": "SHALL",
        "entities": ["authority checks", "authorization", "role-based permissions", "electronic signature authority", "access control"]
    },
    {
        "doc_id": "FDA_21_CFR_11_30",
        "jurisdiction": "US_FDA",
        "statute": "21 CFR Part 11 - Electronic Records; Electronic Signatures",
        "part": "Subpart B - Electronic Records",
        "clause_reference": "§ 11.30",
        "subtopic_tag": "open_systems_controls",
        "title": "Controls for Open Systems - Document Encryption and Digital Signatures",
        "text": "Persons who use open systems to create, modify, maintain, or transmit electronic records shall employ procedures and controls designed to ensure the authenticity, integrity, and, as appropriate, the confidentiality of electronic records from the point of creation to the point of receipt. Such measures include document encryption and the use of appropriate digital signature standards to prevent undetectable alteration.",
        "mandatory_level": "SHALL",
        "entities": ["open systems", "authenticity", "integrity", "digital signatures", "payload encryption", "tampering prevention"]
    },
    {
        "doc_id": "FDA_21_CFR_11_50_A",
        "jurisdiction": "US_FDA",
        "statute": "21 CFR Part 11 - Electronic Records; Electronic Signatures",
        "part": "Subpart C - Electronic Signatures",
        "clause_reference": "§ 11.50(a)",
        "subtopic_tag": "signature_manifestations",
        "title": "Signature Manifestations - Display Requirements",
        "text": "Signed electronic records shall contain information associated with the signing that clearly indicates all of the following: (1) The printed name of the signer; (2) The date and time when the signature was executed (including timezone); and (3) The meaning (such as review, approval, responsibility, or authorship) associated with the signature.",
        "mandatory_level": "SHALL",
        "entities": ["signature manifestation", "printed name", "timestamp", "signing meaning", "authorship", "approval", "review"]
    },
    {
        "doc_id": "FDA_21_CFR_11_70",
        "jurisdiction": "US_FDA",
        "statute": "21 CFR Part 11 - Electronic Records; Electronic Signatures",
        "part": "Subpart C - Electronic Signatures",
        "clause_reference": "§ 11.70",
        "subtopic_tag": "signature_record_linking",
        "title": "Signature and Record Linking Requirements",
        "text": "Electronic signatures and handwritten signatures executed to electronic records shall be linked to their respective electronic records to ensure that the signatures cannot be excised, copied, or otherwise transferred to falsify an electronic record by ordinary means. Cryptographic hashes must bind the signature metadata permanently to the target document payload.",
        "mandatory_level": "SHALL",
        "entities": ["signature linking", "anti-tampering", "cryptographic binding", "falsification prevention", "record excision"]
    },
    {
        "doc_id": "FDA_21_CFR_11_200_A",
        "jurisdiction": "US_FDA",
        "statute": "21 CFR Part 11 - Electronic Records; Electronic Signatures",
        "part": "Subpart C - Electronic Signatures",
        "clause_reference": "§ 11.200(a)",
        "subtopic_tag": "two_component_credentials",
        "title": "Electronic Signature Components - Password and Identification Codes",
        "text": "Electronic signatures that are not based upon biometrics shall employ at least two distinct identification components such as an identification code and password. When an individual executes a series of signings during a single continuous period of controlled system access, the first signing shall be executed using all electronic signature components, and subsequent signings shall be executed using at least one component (typically password).",
        "mandatory_level": "MUST",
        "entities": ["two distinct components", "identification code", "password", "session timeout", "non-biometric signature", "credential security"]
    },
    {
        "doc_id": "FDA_21_CFR_11_300_A",
        "jurisdiction": "US_FDA",
        "statute": "21 CFR Part 11 - Electronic Records; Electronic Signatures",
        "part": "Subpart C - Electronic Signatures",
        "clause_reference": "§ 11.300(a)-(d)",
        "subtopic_tag": "password_management",
        "title": "Controls for Identification Codes and Passwords",
        "text": "Persons who use electronic signatures based upon use of identification codes in combination with passwords shall employ controls to ensure their security and integrity. Controls include: maintaining uniqueness of each combined identification code and password; ensuring credentials are periodically checked, recalled, or revised; following loss management procedures to deauthorize lost or compromised tokens/cards; and use of transaction safeguards to prevent unauthorized password guessing attempts.",
        "mandatory_level": "SHALL",
        "entities": ["password management", "periodic expiration", "brute force protection", "deauthorization", "compromised tokens", "lockout"]
    },

    # =========================================================================
    # INDIAN CDSCO - Drugs and Cosmetics Act & Rules (Schedules M, H, H1, X)
    # =========================================================================
    {
        "doc_id": "CDSCO_SCH_M_PART_I_SEC_3_1",
        "jurisdiction": "IN_CDSCO",
        "statute": "Drugs and Cosmetics Rules, 1945",
        "part": "Schedule M - Good Manufacturing Practices (GMP) and Requirements of Premises",
        "clause_reference": "Schedule M, Part I, Section 3.1",
        "subtopic_tag": "sanitation_hygiene",
        "title": "Sanitation in the Manufacturing Premises",
        "text": "The premises shall be clean and free from accumulated waste, dust, debris, and insects. Proper drainage and ventilation shall be provided. All walls, floors, and ceilings shall be smooth, free from cracks and open joints, non-shedding, and permit easy cleaning and disinfection. Routine microbiological monitoring of air, surfaces, and drains must be conducted and documented in cleanrooms.",
        "mandatory_level": "SHALL",
        "entities": ["sanitation", "cleanroom surfaces", "drainage", "ventilation", "microbiological monitoring", "disinfection", "GMP"]
    },
    {
        "doc_id": "CDSCO_SCH_M_PART_I_SEC_4_1",
        "jurisdiction": "IN_CDSCO",
        "statute": "Drugs and Cosmetics Rules, 1945",
        "part": "Schedule M - Good Manufacturing Practices (GMP) and Requirements of Premises",
        "clause_reference": "Schedule M, Part I, Section 4.1",
        "subtopic_tag": "waste_disposal",
        "title": "Disposal of Hazardous Pharmaceutical Waste and Effluents",
        "text": "Effluent and waste materials shall be collected in dedicated, labeled receptacles and disposed of in accordance with the Environment (Protection) Act, 1986 and local pollution control norms. Biohazard waste and rejected active pharmaceutical ingredients (APIs) shall be destroyed or neutralized under supervision of Quality Assurance with documented destruction certificates.",
        "mandatory_level": "SHALL",
        "entities": ["hazardous waste", "effluent treatment", "ETP", "rejected API", "destruction certificate", "pollution control", "waste disposal"]
    },
    {
        "doc_id": "CDSCO_SCH_M_PART_I_SEC_5_2",
        "jurisdiction": "IN_CDSCO",
        "statute": "Drugs and Cosmetics Rules, 1945",
        "part": "Schedule M - Good Manufacturing Practices (GMP) and Requirements of Premises",
        "clause_reference": "Schedule M, Part I, Section 5.2",
        "subtopic_tag": "personnel_hygiene",
        "title": "Health, Sanitation and Protective Garments of Manufacturing Personnel",
        "text": "All personnel prior to employment and periodically (at least once a year) shall undergo medical examination including visual acuity and communicable disease testing. Personnel with infectious diseases or open lesions on exposed surfaces shall not handle raw materials or in-process pharmaceuticals. Direct contact between operators' hands and exposed products, starting materials, or primary packaging components shall be prohibited.",
        "mandatory_level": "SHALL",
        "entities": ["medical examination", "personnel hygiene", "protective clothing", "cleanroom gowning", "communicable disease", "glove sanitization"]
    },
    {
        "doc_id": "CDSCO_SCH_M_PART_I_SEC_6_1",
        "jurisdiction": "IN_CDSCO",
        "statute": "Drugs and Cosmetics Rules, 1945",
        "part": "Schedule M - Good Manufacturing Practices (GMP) and Requirements of Premises",
        "clause_reference": "Schedule M, Part I, Section 6.1",
        "subtopic_tag": "air_handling_hvac",
        "title": "Air Handling Units (AHU) and Ventilation in Manufacturing Areas",
        "text": "Premises shall be ventilated with air handling systems including air filtration and exhaust systems. Areas where dust is generated (e.g., sampling, weighing, dry mixing) shall have dedicated extraction systems to prevent cross-contamination. Air supply systems shall maintain positive air pressure differentials between clean processing rooms and adjacent corridors with lower cleanliness classifications. HEPA filters of 99.97% efficiency down to 0.3 microns must be installed and integrity tested via DOP/PAO aerosol tests annually.",
        "mandatory_level": "SHALL",
        "entities": ["AHU", "HVAC", "air filtration", "HEPA", "pressure differential", "cross-contamination", "PAO test", "cleanroom"]
    },
    {
        "doc_id": "CDSCO_SCH_M_PART_I_SEC_8_3",
        "jurisdiction": "IN_CDSCO",
        "statute": "Drugs and Cosmetics Rules, 1945",
        "part": "Schedule M - Good Manufacturing Practices (GMP) and Requirements of Premises",
        "clause_reference": "Schedule M, Part I, Section 8.3",
        "subtopic_tag": "record_retention",
        "title": "Documentation - Batch Manufacturing Records Retention",
        "text": "Batch manufacturing records (BMR) and analytical testing records for each batch of drug manufactured shall be retained for at least one year after the expiry date of the batch, or five years from the date of release of the batch, whichever is later. All records must be legible, completed concurrently with operations, signed and dated by the authorized chemist and quality head.",
        "mandatory_level": "SHALL",
        "entities": ["batch manufacturing record", "BMR", "retention period", "expiry date", "quality head", "audit documentation"]
    },
    {
        "doc_id": "CDSCO_SCH_M_PART_IA_STERILE",
        "jurisdiction": "IN_CDSCO",
        "statute": "Drugs and Cosmetics Rules, 1945",
        "part": "Schedule M - Part IA: Specific Requirements for Sterile Products",
        "clause_reference": "Schedule M, Part IA, Section 2.1",
        "subtopic_tag": "sterile_cleanroom_classification",
        "title": "Cleanroom Classifications for Sterile Injectable Products",
        "text": "Manufacturing of sterile products must be carried out in clean areas divided into four environmental grades: Grade A (the local zone for high-risk operations like aseptic filling, stopper bowl, open ampoules with laminar airflow 0.45 m/s); Grade B (the background environment for Grade A zone); Grade C and Grade D (clean areas for less critical stages like preparation of solutions to be filtered). Airborne particulate counts at rest for Grade A must not exceed 3,520 particles/m³ for particles >= 0.5 microns.",
        "mandatory_level": "SHALL",
        "entities": ["Grade A", "Grade B", "Grade C", "Grade D", "cleanroom grade", "aseptic filling", "laminar airflow", "particle count", "sterile"]
    },
    {
        "doc_id": "CDSCO_SCH_M_STORAGE_TEMP",
        "jurisdiction": "IN_CDSCO",
        "statute": "Drugs and Cosmetics Rules, 1945",
        "part": "Schedule M - Good Manufacturing Practices",
        "clause_reference": "Schedule M, Part I, Section 7.4",
        "subtopic_tag": "storage_temperature",
        "title": "Raw Material and Finished Product Storage Temperature Conditions",
        "text": "Storage areas shall have adequate capacity to allow orderly storage of various categories of materials: starting materials, packaging materials, intermediate and finished products. Cold storage facilities for temperature-sensitive drugs (such as vaccines, sera, biologicals, and insulin) must maintain temperatures between 2°C and 8°C. Cold rooms must be equipped with continuous temperature data loggers, auto-dialer temperature alarm systems, and dual redundant power generator backup.",
        "mandatory_level": "SHALL",
        "entities": ["cold storage", "2 to 8 degrees", "temperature data logger", "cold chain", "vaccine storage", "temperature alarm", "backup power"]
    },
    {
        "doc_id": "CDSCO_RULE_65_SCH_H",
        "jurisdiction": "IN_CDSCO",
        "statute": "Drugs and Cosmetics Rules, 1945",
        "part": "Part VI - Conditions of Licences",
        "clause_reference": "Rule 65(9) - Schedule H",
        "subtopic_tag": "prescription_dispensing",
        "title": "Dispensing Conditions for Schedule H Prescription Drugs",
        "text": "Substances specified in Schedule H or Schedule X shall not be sold by retail except on and in accordance with the prescription of a Registered Medical Practitioner (RMP). The supply must be recorded at the time of supply in a prescription register showing: serial number, date of supply, name and address of purchaser, name and address of doctor, name and quantity of drug, and manufacturer batch number. The container must bear the symbol Rx conspicuously displayed on the top left corner.",
        "mandatory_level": "SHALL",
        "entities": ["Schedule H", "Rx", "registered medical practitioner", "RMP", "prescription register", "retail sale", "batch number"]
    },
    {
        "doc_id": "CDSCO_RULE_65_SCH_H1",
        "jurisdiction": "IN_CDSCO",
        "statute": "Drugs and Cosmetics Rules, 1945",
        "part": "Part VI - Conditions of Licences",
        "clause_reference": "Rule 65(3)(h) - Schedule H1",
        "subtopic_tag": "antibiotic_h1_register",
        "title": "Mandatory Separate Register for Schedule H1 Antibiotics and Narcotics",
        "text": "For the supply of drugs specified in Schedule H1 (including cephalosporins, carbapenems, fluoroquinolones, anti-TB drugs, and tramadol), a separate register shall be maintained with pages consecutively numbered. The register shall contain: name and address of the patient, date of dispensing, name and address of prescriber, name of drug and quantity, and batch number. Such records shall be preserved for a minimum of 3 years and shall be open for inspection by Drug Inspectors.",
        "mandatory_level": "MUST",
        "entities": ["Schedule H1", "separate register", "3 years retention", "antibiotics", "carbapenem", "anti-TB", "drug inspector inspection"]
    },
    {
        "doc_id": "CDSCO_RULE_96_SCH_H1_WARNING",
        "jurisdiction": "IN_CDSCO",
        "statute": "Drugs and Cosmetics Rules, 1945",
        "part": "Part IX - Labelling and Packing of Drugs",
        "clause_reference": "Rule 96(1)(iv) - Schedule H1 Labeling",
        "subtopic_tag": "labeling_warning_box",
        "title": "Mandatory Red Box Warning for Schedule H1 Drugs",
        "text": "The package containing a drug specified in Schedule H1 shall be labeled with the symbol Rx in red and conspicuously displayed on the left top corner of the label, and shall also be labeled with the following warning in a box with a red border: 'SCHEDULE H1 PRESCRIPTION DRUG - CAUTION: It is dangerous to take this preparation except in accordance with medical advice and not to be sold by retail without the prescription of a Registered Medical Practitioner.'",
        "mandatory_level": "SHALL",
        "entities": ["red border box", "warning label", "Schedule H1", "Rx symbol in red", "caution warning", "packaging requirements"]
    },
    {
        "doc_id": "CDSCO_RULE_65_SCH_X_DUAL_LOCK",
        "jurisdiction": "IN_CDSCO",
        "statute": "Drugs and Cosmetics Rules, 1945",
        "part": "Part VI - Conditions of Licences",
        "clause_reference": "Rule 65(4) - Schedule X",
        "subtopic_tag": "psychotropic_security",
        "title": "Storage Security and Custody for Schedule X Psychotropic Drugs",
        "text": "Drugs specified in Schedule X (barbiturates, amphetamines, and potent psychotropic substances) shall be kept under lock and key in a cupboard or room where no other drugs are stored, under the personal supervision of a registered pharmacist. The licensee shall retain a carbon copy of every prescription for Schedule X drugs for a period of two years from the date of dispensing.",
        "mandatory_level": "SHALL",
        "entities": ["Schedule X", "psychotropics", "lock and key", "cupboard", "carbon copy prescription", "two years retention", "registered pharmacist"]
    },
    {
        "doc_id": "CDSCO_RULE_74_STABILITY",
        "jurisdiction": "IN_CDSCO",
        "statute": "Drugs and Cosmetics Rules, 1945",
        "part": "Part VII - Manufacture for Sale or Distribution",
        "clause_reference": "Rule 74(o)",
        "subtopic_tag": "stability_testing",
        "title": "Stability Testing and Expiry Dating for Manufactured Drugs",
        "text": "The licensee shall maintain reference samples from each batch of manufactured drugs in an amount sufficient to conduct at least two full analytical re-examinations. Samples shall be stored under recommended storage conditions for at least six months beyond the stated expiry date. Accelerated and real-time stability studies must validate the shelf-life under Zone IVb climatic conditions (30°C / 75% RH).",
        "mandatory_level": "SHALL",
        "entities": ["stability testing", "reference samples", "retention samples", "expiry date", "Zone IVb", "shelf life", "climatic conditions"]
    },
    {
        "doc_id": "HIPAA_164_310_PHYSICAL",
        "jurisdiction": "US_HIPAA",
        "statute": "45 CFR Part 164 - Security and Privacy",
        "part": "Subpart C - Security Standards for the Protection of Electronic PHI",
        "clause_reference": "§ 164.310(a)(1)",
        "subtopic_tag": "physical_safeguards",
        "title": "Physical Safeguards - Facility Access and Server Security",
        "text": "Implement policies and procedures to limit physical access to electronic information systems and the facility or facilities in which they are housed, while ensuring that properly authorized access is allowed. Standards include facility security plans, access control badges, visitor escort logs, and maintenance records of hardware repairs.",
        "mandatory_level": "SHALL",
        "entities": ["physical safeguards", "facility security", "server room", "access badges", "visitor logs", "hardware maintenance"]
    },
    {
        "doc_id": "HIPAA_164_312_C_INTEGRITY",
        "jurisdiction": "US_HIPAA",
        "statute": "45 CFR Part 164 - Security and Privacy",
        "part": "Subpart C - Security Standards for the Protection of Electronic PHI",
        "clause_reference": "§ 164.312(c)(1)",
        "subtopic_tag": "data_integrity",
        "title": "Technical Safeguards - Data Integrity and Checksums",
        "text": "Implement policies and procedures to protect electronic protected health information from improper alteration or destruction. Implement electronic mechanisms such as cryptographic checksums, message authentication codes (HMAC), or digital signatures to corroborate that electronic protected health information has not been altered or destroyed in an unauthorized manner.",
        "mandatory_level": "SHALL",
        "entities": ["data integrity", "checksums", "HMAC", "cryptographic verification", "anti-tampering", "unauthorized alteration"]
    },
    {
        "doc_id": "HIPAA_164_316_DOCUMENTATION",
        "jurisdiction": "US_HIPAA",
        "statute": "45 CFR Part 164 - Security and Privacy",
        "part": "Subpart C - Security Standards for the Protection of Electronic PHI",
        "clause_reference": "§ 164.316(b)(1)",
        "subtopic_tag": "record_retention",
        "title": "Policies and Procedures Retention Period",
        "text": "Maintain the policies and procedures implemented to comply with this subpart in written (which may be electronic) form. Retain the documentation for 6 years from the date of its creation or the date when it last was in effect, whichever is later. Make documentation available to those persons responsible for implementing the procedures.",
        "mandatory_level": "SHALL",
        "entities": ["retention period", "6 years", "compliance documentation", "policies and procedures", "written record"]
    },
    {
        "doc_id": "FDA_21_CFR_11_10_C_RETRIEVAL",
        "jurisdiction": "US_FDA",
        "statute": "21 CFR Part 11 - Electronic Records; Electronic Signatures",
        "part": "Subpart B - Electronic Records",
        "clause_reference": "§ 11.10(c)",
        "subtopic_tag": "record_retention",
        "title": "Protection of Records for Ready Retrieval",
        "text": "Protection of records to enable their accurate and ready retrieval throughout the records retention period. Electronic systems must ensure archived study data, clinical trial trial master files (TMF), and batch records remain readable across software upgrades and media migrations without data loss or semantic corruption.",
        "mandatory_level": "SHALL",
        "entities": ["ready retrieval", "record retention", "archiving", "TMF", "clinical trials", "software migration", "data preservation"]
    },
    {
        "doc_id": "FDA_21_CFR_11_100_IDENTITY",
        "jurisdiction": "US_FDA",
        "statute": "21 CFR Part 11 - Electronic Records; Electronic Signatures",
        "part": "Subpart C - Electronic Signatures",
        "clause_reference": "§ 11.100(b)",
        "subtopic_tag": "electronic_signatures",
        "title": "Verification of Individual Identity for Electronic Signatures",
        "text": "Before an organization establishes, assigns, certifies, or otherwise sanctions an individual's electronic signature, or any element of such electronic signature, the organization shall verify the identity of the individual through formal government identification documents (e.g., passport, national ID) and human resources verification.",
        "mandatory_level": "SHALL",
        "entities": ["identity verification", "electronic signature", "government ID", "credential certification", "HR verification"]
    },
    {
        "doc_id": "CDSCO_SCH_M_WATER_INJECTION",
        "jurisdiction": "IN_CDSCO",
        "statute": "Drugs and Cosmetics Rules, 1945",
        "part": "Schedule M - Good Manufacturing Practices",
        "clause_reference": "Schedule M, Part I, Section 3.2",
        "subtopic_tag": "water_system_validation",
        "title": "Water for Injection (WFI) and Purified Water Standards",
        "text": "Water used for the manufacture of drugs shall be of appropriate quality for its intended use. Water for Injection (WFI) shall be produced from purified water by distillation in a multi-column still or reverse osmosis with continuous recirculation at not less than 80°C to prevent microbial bio-film formation. Microbial limits for WFI shall not exceed 10 CFU/100 ml and bacterial endotoxins shall not exceed 0.25 EU/ml.",
        "mandatory_level": "SHALL",
        "entities": ["WFI", "Water for Injection", "purified water", "endotoxin", "CFU", "microbial limit", "distillation", "80 degrees recirculation"]
    },
    {
        "doc_id": "CDSCO_SCH_M_HVAC_AIR_LOCKS",
        "jurisdiction": "IN_CDSCO",
        "statute": "Drugs and Cosmetics Rules, 1945",
        "part": "Schedule M - Part IA: Sterile Products",
        "clause_reference": "Schedule M, Part IA, Section 3.3",
        "subtopic_tag": "air_handling_hvac",
        "title": "Air Locks and Pressure Cascades in Cleanrooms",
        "text": "Entry of personnel and materials into clean areas shall be through air locks. Airlocks shall be equipped with interlocked door mechanisms so that both doors cannot be opened simultaneously, accompanied by an audible alarm system. A pressure cascade of not less than 10 to 15 Pascals differential between cleanroom grades must be maintained to prevent reverse air influx from lower grade environments.",
        "mandatory_level": "SHALL",
        "entities": ["air lock", "interlocking doors", "pressure cascade", "15 Pascals", "cleanroom boundary", "reverse air influx"]
    },
    {
        "doc_id": "CDSCO_SCH_M_EQUIPMENT_CALIBRATION",
        "jurisdiction": "IN_CDSCO",
        "statute": "Drugs and Cosmetics Rules, 1945",
        "part": "Schedule M - Good Manufacturing Practices",
        "clause_reference": "Schedule M, Part I, Section 7.1",
        "subtopic_tag": "equipment_validation",
        "title": "Calibration and Maintenance of Analytical Equipment",
        "text": "Manufacturing and testing equipment shall be designed, located, maintained, and calibrated at scheduled intervals against certified reference standards traceable to national or international standards. Written calibration records showing dates, standards used, readings, and tolerance limits must be maintained for balances, pH meters, HPLC, and temperature sensors.",
        "mandatory_level": "SHALL",
        "entities": ["calibration", "traceable standard", "HPLC", "analytical balances", "tolerance limits", "maintenance log"]
    }
]


def build_and_save_corpus():
    """Generates and writes the master regulatory corpus to disk."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(CLAUSES_DATA, f, indent=2, ensure_ascii=False)
    print(f"Master corpus successfully generated at: {CORPUS_FILE}")
    print(f"Total structured compliance clauses: {len(CLAUSES_DATA)}")


if __name__ == "__main__":
    build_and_save_corpus()
