# Rapid-MLX Demo — Speed Comparison
# Run: vhs scripts/demo.tape
# Requires: Rapid-MLX on :8000, Ollama on :11434

Output docs/assets/demo.gif
Output docs/assets/demo.mp4

Set Shell "zsh"
Set FontSize 15
Set Width 1280
Set Height 720
Set Theme "Catppuccin Mocha"
Set Padding 20

# Run the race script
Type "python3.12 scripts/demo_race.py"
Enter
Sleep 45s
