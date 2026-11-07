import Foundation
import Observation

/// Only authority identifiers survive local deletion. No credentials or source text.
struct OwnerDeletionIntent: Codable, Equatable, Identifiable, Sendable {
    let id: UUID
    var conversationID: String?
    var storeUUID: UUID?
    var turnID: String?
}

@MainActor @Observable
final class OwnerConversationDeletion {
    struct Confirmation: Identifiable {
        let id: UUID
        let storeUUID: UUID
        let title: String
    }
    enum Failure: Error { case unavailable, identityChanged, invalidTurn, stillRunning }
    private(set) var intents: [OwnerDeletionIntent] = []
    private(set) var notice: String?
    private(set) var busy = false
    var confirmation: Confirmation?
    private let url: URL?
    private let clientProvider: @Sendable () async throws -> any ChatTurnClient
    private var unreadable = false
    var persistenceUnavailable: Bool { unreadable }
    private var retryTask: Task<Void, Never>?
    private var retryRequested = false
    var commitLocalRemoval: (@MainActor () async throws -> Void)?

    init(url: URL?, clientProvider: @escaping @Sendable () async throws -> any ChatTurnClient) {
        self.url = url
        self.clientProvider = clientProvider
        if let url, FileManager.default.fileExists(atPath: url.path) {
            do { intents = try JSONDecoder().decode([OwnerDeletionIntent].self, from: Data(contentsOf: url)) }
            catch { unreadable = true; notice = String(localized: "Couldn't read pending server deletions. Local history was kept.") }
        }
    }

    static func companionURL(for store: URL?) -> URL {
        ConversationStore.fileURL(override: store).deletingLastPathComponent().appendingPathComponent("owner-deletions.json")
    }

    /// Write-through and synchronous: failure must prevent loss of the local owner mapping.
    private func replace(_ values: [OwnerDeletionIntent]) throws {
        guard !unreadable else { throw Failure.unavailable }
        if let url {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try OwnerOnlyFile.writeAtomically(JSONEncoder().encode(values), to: url)
        }
        intents = values
    }

    func enqueue(_ intent: OwnerDeletionIntent) throws {
        if intents.contains(where: { $0.id == intent.id }) { return }
        try replace(intents + [intent])
    }

    func capture(localID: UUID, conversationID: String, storeUUID: UUID?, turnID: String?) throws {
        guard let index = intents.firstIndex(where: { $0.id == localID }) else { return }
        var values = intents
        if let old = values[index].storeUUID, old != storeUUID { throw Failure.identityChanged }
        if let old = values[index].conversationID, old != conversationID { throw Failure.identityChanged }
        let newlyCreated = values[index].conversationID == nil
        values[index].conversationID = conversationID
        if newlyCreated || values[index].storeUUID != nil { values[index].storeUUID = storeUUID }
        if let turnID { values[index].turnID = turnID }
        try replace(values)
    }

    /// Deletion has its own lifetime; chat admission never awaits network cleanup.
    /// Coalesce newly added work into one further pass, never an unbounded retry loop.
    func scheduleRetry() {
        retryRequested = true
        guard !busy, retryTask == nil else { return }
        retryTask = Task { [weak self] in
            guard let self else { return }
            for _ in 0..<2 {
                self.retryRequested = false
                await self.retry()
                if !self.retryRequested { break }
            }
            self.retryTask = nil
        }
    }

    func retry() async {
        guard !busy, !intents.isEmpty, !unreadable else { return }
        busy = true
        defer {
            busy = false
            if retryRequested && retryTask == nil { scheduleRetry() }
        }
        let eligible = intents
        notice = nil
        do { try await commitLocalRemoval?() }
        catch {
            notice = String(localized: "Couldn't save local removal. Server deletion remains pending.")
            return
        }
        // One bounded pass per reconnect/relaunch/manual retry, no polling or owner launch.
        for intent in eligible {
            guard let id = intent.conversationID else {
                notice = String(localized: "Removed from history. Server deletion is pending because its conversation ID wasn't received.")
                continue
            }
            guard let expected = intent.storeUUID else {
                notice = String(localized: "Removed from history. Confirm the connected server before deleting older history.")
                continue
            }
            do {
                let client = try await clientProvider()
                guard try await client.ownerStoreUUID() == expected else { throw Failure.identityChanged }
                if let turnID = intent.turnID {
                    do {
                        let turn = try await client.turn(turnID: turnID)
                        let status = try Self.status(turn, conversationID: id)
                        if status == "running" {
                            let cancelled = try await client.cancel(turnID: turnID)
                            guard try Self.status(cancelled, conversationID: id) != "running" else { throw Failure.stillRunning }
                        }
                    } catch ChatTurnClientError.http(status: 404) {
                        // DELETE remains the authority for other active turns.
                    }
                }
                guard try await client.ownerStoreUUID() == expected else { throw Failure.identityChanged }
                do { try await client.deleteConversation(id: id) }
                catch ChatTurnClientError.http(status: 404) { /* Already absent from this authenticated store. */ }
                // A late creation/admission callback cannot replace or erase an updated receipt.
                if intents.first(where: { $0.id == intent.id }) == intent {
                    try replace(intents.filter { $0.id != intent.id })
                }
            } catch Failure.identityChanged {
                notice = String(localized: "Removed from history. Reconnect the original chat server to finish deletion.")
            } catch ChatTurnClientError.http(status: 409) {
                notice = String(localized: "Removed from history. A server turn is still active; stop it before retrying deletion.")
            } catch {
                notice = String(localized: "Removed from history. Server deletion is pending. Reconnect and retry.")
            }
        }
    }

    /// Legacy authority is never adopted by mere capability discovery.
    func inspectLegacy() async {
        guard !busy, let intent = intents.first(where: { $0.storeUUID == nil && $0.conversationID != nil }), let id = intent.conversationID else { return }
        busy = true
        defer { busy = false }
        do {
            let client = try await clientProvider()
            guard let store = try await client.ownerStoreUUID(),
                  case .object(let row) = try await client.conversation(id: id), row["id"] == .string(id),
                  case .string(let title) = row["title"],
                  try await client.ownerStoreUUID() == store else { throw Failure.identityChanged }
            confirmation = Confirmation(id: intent.id, storeUUID: store, title: title)
        } catch { notice = String(localized: "Couldn't inspect the connected server. Deletion remains pending.") }
    }

    func confirmLegacy(_ value: Confirmation) async {
        guard !busy, let original = intents.first(where: { $0.id == value.id && $0.storeUUID == nil }) else { return }
        busy = true
        confirmation = nil
        do {
            let client = try await clientProvider()
            guard try await client.ownerStoreUUID() == value.storeUUID,
                  let index = intents.firstIndex(where: { $0.id == value.id && $0.storeUUID == nil && $0.conversationID == original.conversationID }) else { throw Failure.identityChanged }
            var values = intents
            values[index].storeUUID = value.storeUUID
            try replace(values)
        } catch {
            busy = false
            notice = String(localized: "The connected server changed. Deletion remains pending.")
            if retryRequested && retryTask == nil { scheduleRetry() }
            return
        }
        busy = false
        await retry()
    }

    private static func status(_ value: ChatWireJSON, conversationID: String) throws -> String {
        guard case .object(let row) = value, row["conversation_id"] == .string(conversationID),
              case .string(let status) = row["status"], ["running", "completed", "cancelled", "error"].contains(status) else { throw Failure.invalidTurn }
        return status
    }
}
