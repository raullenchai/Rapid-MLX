import SwiftUI

/// Settings → Memory panel. Users can toggle extraction on/off, browse
/// learned entries, edit or delete individual ones, and clear the store.
@MainActor
struct SettingsMemoryPanel: View {
    @Environment(MemoryStore.self) private var memoryStore
    @State private var editingEntry: MemoryEntry?
    @State private var editDraft = ""
    @State private var confirmingClear = false

    var body: some View {
        @Bindable var store = memoryStore
        VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
            SectionHeader(
                "Memory",
                subtitle: "The assistant learns your preferences across conversations. It never sends data off this Mac.",
                emphasis: .page
            )

            Toggle(isOn: Binding(
                get: { memoryStore.isEnabled },
                set: { memoryStore.isEnabled = $0 }
            )) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Automatic memory")
                        .font(.body)
                    Text("Review completed conversations and save durable preferences. Disabled by default.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .toggleStyle(.switch)
            .accessibilityIdentifier("Settings.Memory.EnableToggle")

            if memoryStore.isEnabled && !memoryStore.entries.isEmpty {
                HStack {
                    Text("\(memoryStore.entries.count) saved memor\(memoryStore.entries.count == 1 ? "y" : "ies")")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Clear All", role: .destructive) {
                        confirmingClear = true
                    }
                    .buttonStyle(.borderless)
                    .font(.callout)
                    .accessibilityIdentifier("Settings.Memory.ClearAll")
                }

                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(memoryStore.entries) { entry in
                            memoryRow(entry)
                        }
                    }
                }
                .frame(minHeight: 200)
            } else if memoryStore.isEnabled {
                Text("No memories yet. They appear here after the assistant completes a conversation.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
        .padding()
        .alert("Clear all memories?", isPresented: $confirmingClear) {
            Button("Cancel", role: .cancel) {}
                .accessibilityIdentifier("Settings.Memory.ClearAlert.Cancel")
            Button("Clear", role: .destructive) {
                memoryStore.removeAll()
            }
            .accessibilityIdentifier("Settings.Memory.ClearAlert.Confirm")
        } message: {
            Text("This removes every learned fact. It cannot be undone.")
        }
        .sheet(item: $editingEntry) { entry in
            editSheet(entry)
        }
    }

    private func memoryRow(_ entry: MemoryEntry) -> some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(entry.content)
                    .font(.callout)
                    .textSelection(.enabled)
                Text("\(entry.evidenceCount)× seen · \(entry.updatedAt, style: .relative)")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            Spacer()
            Button("Edit") {
                editDraft = entry.content
                editingEntry = entry
            }
            .buttonStyle(.borderless)
            .font(.caption)
            .accessibilityIdentifier("Settings.Memory.Row.Edit")
            Button(role: .destructive) {
                memoryStore.remove(id: entry.id)
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.borderless)
            .font(.caption)
            .accessibilityIdentifier("Settings.Memory.Row.Delete")
        }
        .padding(.vertical, 4)
    }

    private func editSheet(_ entry: MemoryEntry) -> some View {
        NavigationStack {
            Form {
                TextField("Memory", text: $editDraft, axis: .vertical)
                    .lineLimit(3...10)
                    .accessibilityIdentifier("Settings.Memory.EditSheet.Field")
            }
            .navigationTitle("Edit Memory")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { editingEntry = nil }
                        .accessibilityIdentifier("Settings.Memory.EditSheet.Cancel")
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        memoryStore.update(id: entry.id, content: editDraft)
                        editingEntry = nil
                    }
                    .accessibilityIdentifier("Settings.Memory.EditSheet.Save")
                }
            }
        }
        .frame(minWidth: 380, minHeight: 200)
    }
}
