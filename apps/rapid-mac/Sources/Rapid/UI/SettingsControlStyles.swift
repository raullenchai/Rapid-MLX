import SwiftUI

/// Settings rows use a stable trailing control column. SwiftUI's stock
/// switch style sizes itself from the label's ideal width, which made a short
/// description place its switch near the middle while longer descriptions
/// happened to push theirs right. This style makes that alignment explicit
/// across every settings panel while retaining the native macOS switch.
struct TrailingSettingsToggleStyle: ToggleStyle {
    func makeBody(configuration: Configuration) -> some View {
        let binding = Binding(
            get: { configuration.isOn },
            set: { configuration.isOn = $0 }
        )

        return Toggle(isOn: binding) {
            configuration.label
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .toggleStyle(.switch)
        .frame(maxWidth: .infinity)
    }
}
