"""Quick TTS diagnostic — tests multiple speech backends."""
import sys
import time

print("=" * 50)
print("TTS Diagnostic")
print("=" * 50)

# Test 1: pyttsx3
print("\n[1] Testing pyttsx3...")
try:
    import pyttsx3
    e = pyttsx3.init()
    e.setProperty("volume", 1.0)
    e.setProperty("rate", 180)
    e.say("Test one. pyttsx3 speaking.")
    e.runAndWait()
    print("    pyttsx3 runAndWait completed (did you hear it?)")
except Exception as ex:
    print(f"    pyttsx3 FAILED: {ex}")

time.sleep(1)

# Test 2: Windows SAPI via win32com
print("\n[2] Testing win32com SAPI...")
try:
    import win32com.client
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Speak("Test two. Windows SAPI speaking.")
    print("    win32com SAPI completed (did you hear it?)")
except Exception as ex:
    print(f"    win32com SAPI FAILED: {ex}")

time.sleep(1)

# Test 3: PowerShell System.Speech
print("\n[3] Testing PowerShell System.Speech...")
try:
    import subprocess
    ps_cmd = (
        'Add-Type -AssemblyName System.Speech; '
        '$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
        '$synth.Speak("Test three. PowerShell speech speaking.")'
    )
    result = subprocess.run(
        ["powershell", "-Command", ps_cmd],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode == 0:
        print("    PowerShell speech completed (did you hear it?)")
    else:
        print(f"    PowerShell FAILED: {result.stderr}")
except Exception as ex:
    print(f"    PowerShell FAILED: {ex}")

print("\n" + "=" * 50)
print("Which test(s) produced sound? (1, 2, 3, or none)")
