import Foundation

/// Which model a benchmark result is *about*, at the granularity the atomic
/// contract defines it.
///
/// A repo id alone is not a model. `proto/community-benchmark/v1` identifies a
/// primary component by its Hugging Face source (`repo_id` plus an optional
/// `subfolder` selecting a nested variant, plus the `resolved_revision` the
/// loader actually read), by the artifact's `quantization` facts, and by the
/// `identity_strength` that says how much of that the producer could vouch
/// for. `run_builder.unresolved_model_identity` fills all of it from the local
/// cache, so two locally distinguishable variants of one repo — a `4bit/`
/// subfolder, or the same repo at a different snapshot — are different models
/// and must not share a count, a median, or a confirmed publication floor.
///
/// Desktop used to decode `repo_id` and drop the rest, which made those
/// variants indistinguishable to every consumer downstream.
///
/// ## What the public feed can and cannot express
///
/// `atomicPublicProjection` publishes `model: {repo_id, identity_strength}`
/// and nothing else, and `atomicValidateModel` *rejects* any submission whose
/// source carries `subfolder` or `resolved_revision`, or whose `quantization`
/// is anything but `{kind: unknown, base_dtype: unknown}`. So on the live
/// service every accepted run is a bare repo id, and a cell is homogeneous by
/// construction.
///
/// That makes two relations necessary rather than one:
///
/// - ``isSameVariant(as:)`` — full structural equality. Used to group feed
///   cells among themselves. Cells that differ on *any* facet are never
///   merged, whatever the server starts publishing.
/// - ``isCompatible(withPublished:)`` — every facet the *published* cell
///   states must equal this identity's. A facet the feed omits is not a
///   disagreement; it is a fact the service does not record, and it cannot be
///   used either to match or to claim a match. Where that leaves more than one
///   candidate, callers withhold the statistic rather than choosing.
struct CommunityModelIdentity: Hashable, Sendable {
    /// `model.components[0].source.repo_id`.
    let repoID: String
    /// `model.identity_strength`. Constant `"unresolved"` in the current
    /// beta — the worker rejects anything else — but it is part of the
    /// contract's identity and is therefore part of the key.
    let identityStrength: String
    /// `source.subfolder`, when the artifact lives in a nested variant
    /// directory. Absent means "the repo root", which is a different thing
    /// from "some subfolder we did not record".
    let subfolder: String?
    /// `source.resolved_revision` — the snapshot the loader actually read.
    let resolvedRevision: String?
    let quantization: Quantization

    /// The artifact's quantization, as `quantization_facts` projects it.
    struct Quantization: Hashable, Sendable {
        /// `none` | `weights` | `mixed` | `unknown`.
        let kind: String
        let baseDType: String
        /// `affine`, `mxfp4`, `mflux`, … Absent for `none`/`unknown`.
        let method: String?
        /// Bits per weight times two, so 3.5 bpw is an integer. The contract
        /// stores `weight_bits_x2` for exactly that reason.
        let weightBitsX2: Int?
        let groupSize: Int?

        /// What both the producer and the worker use when nothing could be
        /// established. Never a claim that the weights are unquantized —
        /// that is `kind: "none"`.
        static let unknown = Quantization(
            kind: "unknown", baseDType: "unknown", method: nil,
            weightBitsX2: nil, groupSize: nil
        )

        var isUnknown: Bool { self == .unknown }

        /// Human-readable, for the technical-details row. Nil when there is
        /// nothing established to show.
        var displayName: String? {
            switch kind {
            case "none": return String(localized: "unquantized")
            case "weights":
                guard let weightBitsX2 else { return String(localized: "quantized") }
                let bits = Double(weightBitsX2) / 2
                let rounded = weightBitsX2 % 2 == 0
                    ? "\(weightBitsX2 / 2)"
                    : String(format: "%.1f", bits)
                return method.map { "\(rounded)-bit \($0)" } ?? "\(rounded)-bit"
            case "mixed": return String(localized: "mixed precision")
            default: return nil
            }
        }
    }

    init(
        repoID: String,
        identityStrength: String = "unresolved",
        subfolder: String? = nil,
        resolvedRevision: String? = nil,
        quantization: Quantization = .unknown
    ) {
        self.repoID = repoID
        self.identityStrength = identityStrength
        // An empty string is not a subfolder and not a revision; it is a
        // field the producer left blank, which is the same as absent.
        self.subfolder = subfolder.flatMap { $0.isEmpty ? nil : $0 }
        self.resolvedRevision = resolvedRevision.flatMap { $0.isEmpty ? nil : $0 }
        self.quantization = quantization
    }

    // MARK: - Relations

    /// Full structural equality — the grouping key.
    ///
    /// Two cells that disagree about revision, subfolder, identity strength or
    /// quantization describe different artifacts, and their samples may never
    /// be added together or their medians averaged.
    func isSameVariant(as other: Self) -> Bool { self == other }

    /// Whether a **published** cell may be the population this identity's run
    /// belongs to.
    ///
    /// Every facet the cell states must agree. Facets the cell omits are not
    /// evidence of anything: the current projection omits subfolder, revision
    /// and quantization for every cell, so requiring them would withhold every
    /// comparison on the live service over a distinction the service does not
    /// make. Callers pair this with ``publishedFacetsAreComplete(_:)`` and
    /// withhold the statistic when several cells remain compatible.
    /// A facet discriminates only when **both** sides state it. One side
    /// knowing something the other does not is missing information, not a
    /// disagreement — a cold cache yields `quantization: unknown` locally, and
    /// the current projection yields it for every published cell.
    func isCompatible(withPublished published: Self) -> Bool {
        guard repoID == published.repoID else { return false }
        guard identityStrength == published.identityStrength else { return false }
        if let mine = subfolder, let theirs = published.subfolder, mine != theirs {
            return false
        }
        if let mine = resolvedRevision, let theirs = published.resolvedRevision, mine != theirs {
            return false
        }
        if !quantization.isUnknown, !published.quantization.isUnknown,
           quantization != published.quantization {
            return false
        }
        return true
    }

    /// Whether both sides state the same facets, so a compatible match is an
    /// *exact* one rather than an under-specified one. Callers pair this with
    /// the number of compatible cells: more than one, or an under-specified
    /// match, means the statistic is withheld rather than chosen.
    func facetsAreFullyDetermined(against published: Self) -> Bool {
        (subfolder == nil) == (published.subfolder == nil)
            && (resolvedRevision == nil) == (published.resolvedRevision == nil)
            && quantization.isUnknown == published.quantization.isUnknown
    }

    /// A stable key over every facet, for dictionary grouping. Mirrors the
    /// worker's sorted-key canonical form in spirit: same facets, one order.
    var canonicalKey: String {
        [
            repoID,
            identityStrength,
            subfolder ?? "-",
            resolvedRevision ?? "-",
            quantization.kind,
            quantization.baseDType,
            quantization.method ?? "-",
            quantization.weightBitsX2.map(String.init) ?? "-",
            quantization.groupSize.map(String.init) ?? "-",
        ].joined(separator: "|")
    }

    /// The facets worth showing a user when two variants must be told apart.
    /// Empty when the identity is just a repo id, which is the common case.
    var distinguishingFacets: [String] {
        var facets: [String] = []
        if let subfolder { facets.append(subfolder) }
        if let quantizationName = quantization.displayName { facets.append(quantizationName) }
        if let resolvedRevision {
            facets.append(String(resolvedRevision.prefix(7)))
        }
        return facets
    }
}

// MARK: - Wire decoding

extension CommunityModelIdentity {
    /// Decodes both shapes the contract uses for a model.
    ///
    /// **Nested**, as a local benchmark record writes it
    /// (`unresolved_model_identity`): the facets live under
    /// `components[0].source` and `components[0].quantization`.
    ///
    /// **Flat**, as `/api/benchmarks/atomic/public` publishes it: `repo_id`,
    /// `identity_strength`, and — when the projection carries them —
    /// `subfolder`, `resolved_revision` and a top-level `quantization`.
    ///
    /// Reading only the nested shape meant every variant facet the public feed
    /// sent was silently dropped on the floor, so two published variants
    /// decoded as the same model and were merged by every consumer downstream.
    /// Each facet is therefore taken from the nested position when present and
    /// from the flat one otherwise.
    struct Wire: Decodable, Sendable {
        struct Quantization: Decodable, Sendable {
            let kind: String?
            let baseDType: String?
            let method: String?
            let weightBitsX2: Int?
            let groupSize: Int?
            enum CodingKeys: String, CodingKey {
                case kind, method
                case baseDType = "base_dtype"
                case weightBitsX2 = "weight_bits_x2"
                case groupSize = "group_size"
            }

            var resolved: CommunityModelIdentity.Quantization {
                CommunityModelIdentity.Quantization(
                    kind: kind ?? "unknown",
                    baseDType: baseDType ?? "unknown",
                    method: method,
                    weightBitsX2: weightBitsX2,
                    groupSize: groupSize
                )
            }
        }

        struct Component: Decodable, Sendable {
            struct Source: Decodable, Sendable {
                let repoID: String?
                let subfolder: String?
                let resolvedRevision: String?
                enum CodingKeys: String, CodingKey {
                    case repoID = "repo_id"
                    case subfolder
                    case resolvedRevision = "resolved_revision"
                }
            }
            let source: Source?
            let quantization: Quantization?
        }

        /// Present on the full nested local identity.
        let components: [Component]?
        let identityStrength: String?
        /// The flat public projection.
        let repoID: String?
        let subfolder: String?
        let resolvedRevision: String?
        let quantization: Quantization?

        enum CodingKeys: String, CodingKey {
            case components, subfolder, quantization
            case identityStrength = "identity_strength"
            case repoID = "repo_id"
            case resolvedRevision = "resolved_revision"
        }

        /// Nil only when neither shape carried a repo id, which is a record
        /// that names no model at all.
        var identity: CommunityModelIdentity? {
            let component = components?.first
            guard let repo = component?.source?.repoID ?? repoID else { return nil }
            // Nested first (it is the producer's own, most specific record),
            // then flat. A facet stated in neither place stays absent, which
            // is what keeps "we don't know" distinct from "it is the default".
            let resolvedQuantization = component?.quantization?.resolved
                ?? quantization?.resolved
                ?? .unknown
            return CommunityModelIdentity(
                repoID: repo,
                // The worker only ever accepts "unresolved" today. A record
                // that omits it is read as that rather than as a distinct
                // third value, so old and new records group together.
                identityStrength: identityStrength ?? "unresolved",
                subfolder: component?.source?.subfolder ?? subfolder,
                resolvedRevision: component?.source?.resolvedRevision ?? resolvedRevision,
                quantization: resolvedQuantization
            )
        }
    }
}
